"""
Message model for farmer chat system.
"""
from sqlalchemy import Column, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

Base = declarative_base()

class MessageType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    ANALYSIS_SHARE = "analysis_share"
    SYSTEM = "system"

class ModerationStatus(enum.Enum):
    APPROVED = "approved"
    PENDING = "pending"
    HIDDEN = "hidden"
    FLAGGED = "flagged"

class Message(Base):
    __tablename__ = "messages"
    
    message_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    room_id = Column(String, nullable=False)  # Remove foreign key constraint for simplicity
    user_id = Column(String(50), nullable=False)  # Foreign key to User.user_id
    content = Column(Text, nullable=False)
    message_type = Column(String(20), default="text")  # Store as string
    sent_at = Column(DateTime, default=func.now())
    edited_at = Column(DateTime)
    is_deleted = Column(Boolean, default=False)
    parent_message_id = Column(String)  # Remove foreign key constraint
    moderation_status = Column(String(20), default="approved")  # Store as string
    analysis_reference_id = Column(String)  # Optional link to disease analysis
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'message_id': self.message_id,
            'room_id': self.room_id,
            'user_id': self.user_id,
            'content': self.content,
            'message_type': self.message_type,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'edited_at': self.edited_at.isoformat() if self.edited_at else None,
            'is_deleted': self.is_deleted,
            'parent_message_id': self.parent_message_id,
            'moderation_status': self.moderation_status,
            'analysis_reference_id': self.analysis_reference_id
        }
    
    def edit_content(self, new_content):
        """Edit message content and update timestamp."""
        self.content = new_content
        self.edited_at = datetime.utcnow()
    
    def mark_deleted(self):
        """Soft delete the message."""
        self.is_deleted = True
        
    def update_moderation_status(self, status: str):
        """Update the moderation status of the message."""
        self.moderation_status = status
