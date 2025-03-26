from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import tempfile
from models import db, Question, UserResponse, UserResult, UserProgress
from speech_analyzer import analyze_speech, transcribe_audio
from gemini_analyzer import analyze_with_gemini
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
db.init_app(app)

@app.route('/api/get-questions', methods=['GET'])
def get_questions():
    """
    Get 6 random IELTS speaking questions.
    Returns a JSON array of question objects.
    """
    # Get random questions from each part
    part1_questions = Question.query.filter_by(part=1).order_by(db.func.random()).limit(2).all()
    part2_questions = Question.query.filter_by(part=2).order_by(db.func.random()).limit(2).all()
    part3_questions = Question.query.filter_by(part=3).order_by(db.func.random()).limit(2).all()
    
    # Combine all questions
    questions = part1_questions + part2_questions + part3_questions
    
    # Convert to JSON
    questions_json = [
        {
            'id': q.id,
            'text': q.text,
            'part': q.part,
            'topic': q.topic,
            'isAudioOnly': False  # Default value, frontend can override
        } for q in questions
    ]
    
    return jsonify(questions_json)

@app.route('/api/submit-response', methods=['POST'])
def submit_response():
    """
    Submit a user's audio response for analysis.
    Expects: 
    - audio file in request.files['audio']
    - question_id in request.form
    - question_text in request.form
    - user_id in request.form
    Returns analysis results.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    question_id = request.form.get('question_id')
    question_text = request.form.get('question_text')
    user_id = request.form.get('user_id')
    
    if not question_id or not user_id:
        return jsonify({'error': 'Missing question_id or user_id'}), 400
    
    # Save audio file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    audio_file.save(temp_path)
    
    try:
        # Transcribe audio
        transcript = transcribe_audio(temp_path)
        
        # Get question for context
        question = Question.query.get(question_id)
        if not question:
            # If question not found in DB but text was provided, use that
            if question_text:
                question_context = question_text
            else:
                return jsonify({'error': 'Question not found'}), 404
        else:
            question_context = question.text
        
        # Analyze speech with traditional NLP
        nlp_analysis = analyze_speech(transcript)
        
        # Analyze with Gemini AI for deeper insights
        gemini_analysis = analyze_with_gemini(transcript, question_context)
        
        # Combine analyses for final result
        combined_analysis = combine_analyses(nlp_analysis, gemini_analysis)
        
        # Store response in database
        user_response = UserResponse(
            user_id=user_id,
            question_id=question_id,
            audio_path=temp_path,  # In production, you'd store this in a more permanent location
            transcript=transcript
        )
        db.session.add(user_response)
        
        # Create user result record
        user_result = UserResult(
            user_id=user_id,
            response_id=user_response.id,
            fluency_score=combined_analysis['fluency_score'],
            vocabulary_score=combined_analysis['vocabulary_score'],
            grammar_score=combined_analysis['grammar_score'],
            coherence_score=combined_analysis['coherence_score'],
            overall_score=combined_analysis['overall_score'],
            feedback=combined_analysis['feedback']
        )
        db.session.add(user_result)
        
        # Update user progress
        update_user_progress(user_id, combined_analysis['overall_score'])
        
        db.session.commit()
        
        # Return analysis results
        return jsonify({
            'response_id': user_response.id,
            'transcript': transcript,
            'analysis': combined_analysis
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        # In a production environment, you might want to keep the audio files
        # Here we're removing them after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/user-progress/<user_id>', methods=['GET'])
def get_user_progress(user_id):
    """
    Get progress history for a specific user.
    Returns a JSON object with progress data.
    """
    # Get user progress records
    progress = UserProgress.query.filter_by(user_id=user_id).order_by(UserProgress.created_at).all()
    
    # Get recent test results
    results = UserResult.query.filter_by(user_id=user_id).order_by(UserResult.created_at.desc()).limit(10).all()
    
    results_json = []
    for result in results:
        response = UserResponse.query.get(result.response_id)
        question = Question.query.get(response.question_id)
        
        results_json.append({
            'id': result.id,
            'date': result.created_at.isoformat(),
            'question': {
                'id': question.id if question else None,
                'text': question.text if question else "Unknown question",
                'part': question.part if question else None,
                'topic': question.topic if question else None
            },
            'transcript': response.transcript,
            'fluency_score': result.fluency_score,
            'vocabulary_score': result.vocabulary_score,
            'grammar_score': result.grammar_score,
            'coherence_score': result.coherence_score,
            'overall_score': result.overall_score
        })
    
    # Format progress data for chart display
    progress_data = {
        'dates': [p.created_at.isoformat() for p in progress],
        'scores': [p.average_score for p in progress],
        'recent_results': results_json
    }
    
    return jsonify(progress_data)

def combine_analyses(nlp_analysis, gemini_analysis):
    """
    Combine traditional NLP analysis with Gemini AI analysis.
    
    Args:
        nlp_analysis: Results from traditional NLP analysis
        gemini_analysis: Results from Gemini AI analysis
        
    Returns:
        Combined analysis results
    """
    # Extract scores from both analyses
    fluency_nlp = nlp_analysis['fluency_score']
    vocabulary_nlp = nlp_analysis['vocabulary_score']
    grammar_nlp = nlp_analysis['grammar_score']
    
    fluency_gemini = gemini_analysis['fluency_score']
    vocabulary_gemini = gemini_analysis['vocabulary_score']
    grammar_gemini = gemini_analysis['grammar_score']
    coherence_gemini = gemini_analysis['coherence_score']
    
    # Weighted combination (giving more weight to Gemini for deeper analysis)
    fluency_combined = (fluency_nlp * 0.4) + (fluency_gemini * 0.6)
    vocabulary_combined = (vocabulary_nlp * 0.4) + (vocabulary_gemini * 0.6)
    grammar_combined = (grammar_nlp * 0.4) + (grammar_gemini * 0.6)
    
    # Calculate overall score (including coherence from Gemini)
    overall_score = (
        fluency_combined * 0.3 +
        vocabulary_combined * 0.25 +
        grammar_combined * 0.25 +
        coherence_gemini * 0.2
    )
    
    # Combine feedback
    nlp_feedback = nlp_analysis.get('feedback', '{}')
    gemini_feedback = gemini_analysis.get('feedback', '{}')
    
    # For simplicity, we'll use Gemini's feedback as it's more comprehensive
    # In a production system, you might want to merge the feedback more intelligently
    
    return {
        'fluency_score': round(fluency_combined, 1),
        'vocabulary_score': round(vocabulary_combined, 1),
        'grammar_score': round(grammar_combined, 1),
        'coherence_score': round(coherence_gemini, 1),
        'overall_score': round(overall_score, 1),
        'feedback': gemini_feedback,
        'nlp_analysis': nlp_analysis,
        'gemini_analysis': gemini_analysis
    }

def update_user_progress(user_id, score):
    """
    Update user progress with new test score.
    
    Args:
        user_id: User identifier
        score: New test score
    """
    # Get existing progress or create new
    progress = UserProgress.query.filter_by(user_id=user_id).first()
    
    if progress:
        # Update existing progress
        progress.test_count += 1
        progress.total_score += score
        progress.average_score = progress.total_score / progress.test_count
        progress.latest_score = score
    else:
        # Create new progress record
        progress = UserProgress(
            user_id=user_id,
            test_count=1,
            total_score=score,
            average_score=score,
            latest_score=score
        )
        db.session.add(progress)

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple endpoint to test if the API is running."""
    return jsonify({'status': 'API is running'})

@app.before_request
def create_tables():
    """Create database tables before first request."""
    db.create_all()
    
    # Seed the database with sample questions if it's empty
    if Question.query.count() == 0:
        seed_database()

def seed_database():
    """Seed the database with sample IELTS speaking questions."""
    sample_questions = [
        # Part 1 questions (simple, direct questions about familiar topics)
        {'text': 'Tell me about your hometown and what you like about it.', 'part': 1, 'topic': 'Hometown'},
        {'text': 'What kind of accommodation do you live in?', 'part': 1, 'topic': 'Accommodation'},
        {'text': 'Do you work or study? Tell me about it.', 'part': 1, 'topic': 'Work/Study'},
        {'text': 'What do you enjoy doing in your free time?', 'part': 1, 'topic': 'Hobbies'},
        {'text': 'How often do you use public transportation?', 'part': 1, 'topic': 'Transportation'},
        {'text': 'What types of food do you enjoy eating?', 'part': 1, 'topic': 'Food'},
        {'text': 'Do you prefer to spend time alone or with friends?', 'part': 1, 'topic': 'Social Life'},
        {'text': 'What kind of music do you like to listen to?', 'part': 1, 'topic': 'Music'},
        
        # Part 2 questions (longer, descriptive responses)
        {'text': 'Describe a skill you would like to learn and explain why.', 'part': 2, 'topic': 'Skills'},
        {'text': 'Describe a memorable trip you have taken in the past.', 'part': 2, 'topic': 'Travel'},
        {'text': 'Describe a person who has had a significant influence on your life.', 'part': 2, 'topic': 'People'},
        {'text': 'Describe a book or movie that made a strong impression on you.', 'part': 2, 'topic': 'Entertainment'},
        {'text': 'Describe a time when you helped someone.', 'part': 2, 'topic': 'Experiences'},
        {'text': 'Describe a place you would like to visit in the future.', 'part': 2, 'topic': 'Travel'},
        {'text': 'Describe an important decision you have made in your life.', 'part': 2, 'topic': 'Life Choices'},
        {'text': 'Describe a traditional festival or celebration in your country.', 'part': 2, 'topic': 'Culture'},
        
        # Part 3 questions (abstract, opinion-based questions)
        {'text': 'What changes would you like to see in your country in the next ten years?', 'part': 3, 'topic': 'Society'},
        {'text': 'Do you think social media has a positive or negative impact on society?', 'part': 3, 'topic': 'Technology'},
        {'text': 'How do you think education will change in the future?', 'part': 3, 'topic': 'Education'},
        {'text': 'What are the advantages and disadvantages of living in a big city?', 'part': 3, 'topic': 'Urban Life'},
        {'text': 'How important is it to preserve traditional cultures in a globalized world?', 'part': 3, 'topic': 'Culture'},
        {'text': 'What role should governments play in protecting the environment?', 'part': 3, 'topic': 'Environment'},
        {'text': 'Do you think technology makes people more or less creative?', 'part': 3, 'topic': 'Technology'},
        {'text': 'How might climate change affect future generations?', 'part': 3, 'topic': 'Environment'}
    ]
    
    for q in sample_questions:
        question = Question(text=q['text'], part=q['part'], topic=q['topic'])
        db.session.add(question)
    
    db.session.commit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)

