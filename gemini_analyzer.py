import google.generativeai as genai
import json
import os
from config import Config

# Configure the Gemini API
genai.configure(api_key=os.environ.get('AIzaSyBdlN7r024n13FZOZEYhEbqtzJI1Z-xPB8', Config.GEMINI_API_KEY))

# Set up the model
model = genai.GenerativeModel('gemini-pro')

def analyze_with_gemini(transcript, question):
    """
    Analyze speech transcript using Gemini AI for deeper insights.
    
    Args:
        transcript: Transcribed text from audio
        question: The IELTS question that was asked
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create prompt for Gemini
        prompt = f"""
        You are an expert IELTS speaking examiner. Analyze the following response to an IELTS speaking question.
        
        IELTS Question: {question}
        
        Response: {transcript}
        
        Provide a detailed analysis of the response based on the IELTS speaking assessment criteria:
        1. Fluency and Coherence
        2. Lexical Resource (Vocabulary)
        3. Grammatical Range and Accuracy
        4. Pronunciation (though we can't assess this from text)
        
        For each criterion, provide:
        - A score on the IELTS band scale (0-9, with 0.5 increments)
        - Specific strengths
        - Specific weaknesses
        - Suggestions for improvement
        
        Also provide an overall band score and a summary of the key strengths and areas for improvement.
        
        Format your response as a JSON object with the following structure:
        {{
            "fluency_score": float,
            "vocabulary_score": float,
            "grammar_score": float,
            "coherence_score": float,
            "overall_score": float,
            "feedback": {{
                "strengths": [list of strings],
                "weaknesses": [list of strings],
                "suggestions": [list of strings]
            }}
        }}
        
        Only return the JSON object, nothing else.
        """
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            response_text = response.text
            
            # Sometimes Gemini might include markdown code blocks, so we need to extract just the JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            analysis = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ['fluency_score', 'vocabulary_score', 'grammar_score', 'coherence_score', 
                              'overall_score', 'feedback']
            for field in required_fields:
                if field not in analysis:
                    if field == 'coherence_score':
                        # If coherence is missing, estimate it from fluency
                        analysis['coherence_score'] = analysis.get('fluency_score', 6.0)
                    else:
                        # Default values for missing fields
                        analysis[field] = 6.0  # Default to band 6
            
            if 'feedback' not in analysis or not isinstance(analysis['feedback'], dict):
                analysis['feedback'] = {
                    'strengths': [],
                    'weaknesses': [],
                    'suggestions': []
                }
            
            return analysis
            
        except Exception as e:
            # If JSON parsing fails, return a default analysis
            print(f"Error parsing Gemini response: {e}")
            return {
                'fluency_score': 6.0,
                'vocabulary_score': 6.0,
                'grammar_score': 6.0,
                'coherence_score': 6.0,
                'overall_score': 6.0,
                'feedback': json.dumps({
                    'strengths': ["The response addresses the question"],
                    'weaknesses': ["Unable to perform detailed analysis"],
                    'suggestions': ["Try to speak more clearly and at a moderate pace"]
                })
            }
    
    except Exception as e:
        # If Gemini API call fails, return a default analysis
        print(f"Error calling Gemini API: {e}")
        return {
            'fluency_score': 6.0,
            'vocabulary_score': 6.0,
            'grammar_score': 6.0,
            'coherence_score': 6.0,
            'overall_score': 6.0,
            'feedback': json.dumps({
                'strengths': ["The response addresses the question"],
                'weaknesses': ["Unable to perform detailed analysis"],
                'suggestions': ["Try to speak more clearly and at a moderate pace"]
            })
        }

