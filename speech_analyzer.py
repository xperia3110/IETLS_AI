import whisper
import spacy
import numpy as np
import json
from textstat import flesch_kincaid_grade, syllable_count
import language_tool_python

# Load models
whisper_model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm")
language_tool = language_tool_python.LanguageTool('en-US')

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Whisper AI.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def analyze_speech(transcript):
    """
    Analyze speech transcript for fluency, vocabulary, and grammar.
    
    Args:
        transcript: Transcribed text from audio
        
    Returns:
        Dictionary with analysis results
    """
    # Process text with spaCy
    doc = nlp(transcript)
    
    # Analyze fluency
    fluency_score = analyze_fluency(transcript, doc)
    
    # Analyze vocabulary
    vocabulary_score = analyze_vocabulary(doc)
    
    # Analyze grammar
    grammar_score = analyze_grammar(transcript)
    
    # Calculate overall score (weighted average)
    overall_score = calculate_overall_score(fluency_score, vocabulary_score, grammar_score)
    
    # Generate feedback
    feedback = generate_feedback(transcript, doc, fluency_score, vocabulary_score, grammar_score)
    
    return {
        'fluency_score': round(fluency_score, 1),
        'vocabulary_score': round(vocabulary_score, 1),
        'grammar_score': round(grammar_score, 1),
        'overall_score': round(overall_score, 1),
        'feedback': feedback
    }

def analyze_fluency(transcript, doc):
    """
    Analyze speech fluency based on:
    - Speech rate (words per minute)
    - Sentence length and variation
    - Filler words and hesitations
    - Reading ease
    
    Returns a score from 0-9 (IELTS scale)
    """
    # Count words and sentences
    word_count = len([token for token in doc if not token.is_punct and not token.is_space])
    sentence_count = len(list(doc.sents))
    
    # Assume average speaking rate is about 150 words per minute
    # This is a simplification - in a real system, you'd use the audio length
    estimated_speech_rate = 150
    
    # Check for filler words
    filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of']
    filler_count = sum(transcript.lower().count(filler) for filler in filler_words)
    
    # Calculate reading ease
    fk_grade = flesch_kincaid_grade(transcript)
    
    # Calculate average sentence length and variation
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
        sentence_lengths = [len([token for token in sent if not token.is_punct and not token.is_space]) 
                           for sent in doc.sents]
        sentence_length_variation = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
    else:
        avg_sentence_length = 0
        sentence_length_variation = 0
    
    # Score components (each from 0-9)
    speech_rate_score = min(9, max(0, 4.5 + (estimated_speech_rate - 120) / 20))
    
    filler_ratio = filler_count / max(1, word_count)
    filler_score = min(9, max(0, 9 - filler_ratio * 100))
    
    complexity_score = min(9, max(0, 4.5 + (fk_grade - 7) * 0.5))
    
    sentence_variation_score = min(9, max(0, 4.5 + sentence_length_variation * 0.5))
    
    # Weighted average for final fluency score
    fluency_score = (
        speech_rate_score * 0.3 +
        filler_score * 0.3 +
        complexity_score * 0.2 +
        sentence_variation_score * 0.2
    )
    
    return fluency_score

def analyze_vocabulary(doc):
    """
    Analyze vocabulary based on:
    - Lexical diversity
    - Word rarity/complexity
    - Appropriate collocations
    - Topic-specific vocabulary
    
    Returns a score from 0-9 (IELTS scale)
    """
    # Count total and unique words
    all_words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    unique_words = set(all_words)
    
    # Calculate lexical diversity (type-token ratio)
    if len(all_words) > 0:
        lexical_diversity = len(unique_words) / len(all_words)
    else:
        lexical_diversity = 0
    
    # Calculate average word length and syllable count
    avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0
    avg_syllables = np.mean([syllable_count(word) for word in all_words]) if all_words else 0
    
    # Calculate word rarity using spaCy's frequency ranks
    # Lower rank means more common word
    word_ranks = [token.rank if hasattr(token, 'rank') else 0 for token in doc 
                 if not token.is_punct and not token.is_space]
    avg_word_rank = np.mean(word_ranks) if word_ranks else 0
    
    # Score components (each from 0-9)
    diversity_score = min(9, max(0, lexical_diversity * 15))
    
    length_score = min(9, max(0, 4.5 + (avg_word_length - 4.5) * 1.5))
    
    syllable_score = min(9, max(0, 4.5 + (avg_syllables - 1.5) * 3))
    
    rarity_score = min(9, max(0, 4.5 - avg_word_rank / 10000))
    
    # Weighted average for final vocabulary score
    vocabulary_score = (
        diversity_score * 0.4 +
        length_score * 0.2 +
        syllable_score * 0.2 +
        rarity_score * 0.2
    )
    
    return vocabulary_score

def analyze_grammar(transcript):
    """
    Analyze grammar based on:
    - Grammatical errors
    - Sentence structure complexity
    - Tense usage
    - Agreement errors
    
    Returns a score from 0-9 (IELTS scale)
    """
    # Check for grammar errors using LanguageTool
    matches = language_tool.check(transcript)
    error_count = len(matches)
    
    # Calculate error density (errors per 100 words)
    word_count = len(transcript.split())
    error_density = (error_count / max(1, word_count)) * 100
    
    # Score based on error density
    # Fewer errors = higher score
    grammar_score = min(9, max(0, 9 - error_density))
    
    return grammar_score

def calculate_overall_score(fluency_score, vocabulary_score, grammar_score):
    """
    Calculate overall IELTS speaking score based on component scores.
    
    IELTS speaking is scored on a 9-band scale.
    """
    # Weighted average of component scores
    # Weights based on IELTS speaking assessment criteria
    overall_score = (
        fluency_score * 0.4 +
        vocabulary_score * 0.3 +
        grammar_score * 0.3
    )
    
    return overall_score

def generate_feedback(transcript, doc, fluency_score, vocabulary_score, grammar_score):
    """
    Generate detailed feedback based on analysis.
    
    Returns a JSON string with structured feedback.
    """
    feedback = {
        'strengths': [],
        'weaknesses': [],
        'suggestions': []
    }
    
    # Fluency feedback
    if fluency_score >= 7.0:
        feedback['strengths'].append("Good flow of speech with effective use of connective phrases")
    elif fluency_score >= 5.0:
        feedback['weaknesses'].append("Some hesitations when expressing complex ideas")
        feedback['suggestions'].append("Practice speaking about unfamiliar topics to improve fluency")
    else:
        feedback['weaknesses'].append("Frequent hesitations and difficulty maintaining flow")
        feedback['suggestions'].append("Record yourself speaking and identify points of hesitation")
    
    # Vocabulary feedback
    if vocabulary_score >= 7.0:
        feedback['strengths'].append("Good range of vocabulary with some effective use of idiomatic expressions")
    elif vocabulary_score >= 5.0:
        feedback['weaknesses'].append("Limited vocabulary range with some repetition")
        feedback['suggestions'].append("Learn topic-specific vocabulary for common IELTS themes")
    else:
        feedback['weaknesses'].append("Basic vocabulary with frequent repetition")
        feedback['suggestions'].append("Build vocabulary by reading articles on diverse topics")
    
    # Grammar feedback
    if grammar_score >= 7.0:
        feedback['strengths'].append("Good control of complex grammatical structures")
    elif grammar_score >= 5.0:
        feedback['weaknesses'].append("Some grammatical errors in complex structures")
        feedback['suggestions'].append("Practice using a variety of tenses and complex sentences")
    else:
        feedback['weaknesses'].append("Frequent basic grammatical errors")
        feedback['suggestions'].append("Review basic grammar rules and practice with simple sentences first")
    
    # Add specific vocabulary suggestions
    rare_words = [token.text for token in doc if token.rank and token.rank < 30000 
                 and not token.is_stop and not token.is_punct]
    if rare_words:
        feedback['strengths'].append(f"Good use of advanced vocabulary such as: {', '.join(rare_words[:3])}")
    
    # Add specific grammar error examples
    matches = language_tool.check(transcript)
    if matches:
        error_examples = [match.context for match in matches[:2]]
        feedback['weaknesses'].append(f"Grammar errors in phrases like: {'; '.join(error_examples)}")
    
    return json.dumps(feedback)

