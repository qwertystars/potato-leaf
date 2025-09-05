"""
User Profile model for farmer chat participants.
"""
from sqlalchemy import Column, String, DateTime, Integer, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name = Column(String(100), nullable=False)
    farming_context = Column(String(200))  # Location, crops, experience
    join_date = Column(DateTime, default=func.now())
    reputation_score = Column(Integer, default=0)
    is_verified_farmer = Column(Boolean, default=False)
    last_active = Column(DateTime, default=func.now())
    messages_sent = Column(Integer, default=0)
    helpful_votes = Column(Integer, default=0)
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'user_id': self.user_id,
            'display_name': self.display_name,
            'farming_context': self.farming_context,
            'join_date': self.join_date.isoformat() if self.join_date else None,
            'reputation_score': self.reputation_score,
            'is_verified_farmer': self.is_verified_farmer,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'messages_sent': self.messages_sent,
            'helpful_votes': self.helpful_votes
        }
    
    def update_activity(self):
        """Update last active timestamp."""
        self.last_active = datetime.utcnow()
    
    def increment_message_count(self):
        """Increment the number of messages sent."""
        self.messages_sent += 1
    
    def add_helpful_vote(self):
        """Add a helpful vote and update reputation."""
        self.helpful_votes += 1
        self.reputation_score += 5  # 5 points per helpful vote
    
    def verify_farmer(self):
        """Mark user as verified farmer."""
        self.is_verified_farmer = True
        self.reputation_score += 50  # Bonus for verification
