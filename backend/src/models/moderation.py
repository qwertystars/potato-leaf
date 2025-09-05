"""
Moderation model for chat content management.
"""
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

Base = declarative_base()

class ActionType(enum.Enum):
    REPORT = "report"
    HIDE = "hide"
    DELETE = "delete"
    MUTE_USER = "mute_user"
    BAN_USER = "ban_user"
    APPROVE = "approve"
    REJECT = "reject"

class ModerationAction(Base):
    __tablename__ = "moderation_actions"
    
    action_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String)  # Remove foreign key constraint
    reported_user_id = Column(String(50))  # User being reported
    moderator_id = Column(String(50))  # User taking action (admin or community moderator)
    action_type = Column(String(20), nullable=False)  # Store as string
    reason = Column(String(500))
    action_taken_at = Column(DateTime, default=func.now())
    automated = Column(String(10), default='false')  # 'true' if AI-moderated, 'false' if human
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'action_id': self.action_id,
            'message_id': self.message_id,
            'reported_user_id': self.reported_user_id,
            'moderator_id': self.moderator_id,
            'action_type': self.action_type,
            'reason': self.reason,
            'action_taken_at': self.action_taken_at.isoformat() if self.action_taken_at else None,
            'automated': self.automated
        }
