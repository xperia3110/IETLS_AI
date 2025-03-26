from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Question(db.Model):
    """IELTS speaking test questions."""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    part = db.Column(db.Integer, nullable=False)  # 1, 2, or 3 for IELTS speaking parts
    topic = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Question {self.id}: {self.text[:30]}...>'

class UserResponse(db.Model):
    """User's audio responses to questions."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)  # External user ID from your auth system
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    audio_path = db.Column(db.String(255), nullable=True)  # Path to stored audio file
    transcript = db.Column(db.Text, nullable=True)  # Transcribed text from audio
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    question = db.relationship('Question', backref=db.backref('responses', lazy=True))
    
    def __repr__(self):
        return f'<UserResponse {self.id} from user {self.user_id}>'

class UserResult(db.Model):
    """Analysis results for user responses."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    response_id = db.Column(db.Integer, db.ForeignKey('user_response.id'), nullable=False)
    fluency_score = db.Column(db.Float, nullable=False)
    vocabulary_score = db.Column(db.Float, nullable=False)
    grammar_score = db.Column(db.Float, nullable=False)
    coherence_score = db.Column(db.Float, nullable=False)  # Added coherence score
    overall_score = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=True)  # JSON string with detailed feedback
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    response = db.relationship('UserResponse', backref=db.backref('results', lazy=True))
    
    def __repr__(self):
        return f'<UserResult {self.id} for response {self.response_id}>'

class UserProgress(db.Model):
    """User progress tracking over time."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False, unique=True)
    test_count = db.Column(db.Integer, default=0)
    total_score = db.Column(db.Float, default=0.0)
    average_score = db.Column(db.Float, default=0.0)
    latest_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserProgress for user {self.user_id}, avg: {self.average_score}>'

