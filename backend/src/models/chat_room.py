"""
Chat Room model for farmer communication system.
"""
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Enum as SQLEnum
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

Base = declarative_base()

class RoomType(enum.Enum):
    DISEASE_SPECIFIC = "disease_specific"
    GEOGRAPHIC = "geographic"
    GENERAL = "general"
    PRIVATE = "private"

class ModerationLevel(enum.Enum):
    OPEN = "open"
    MODERATED = "moderated"
    RESTRICTED = "restricted"

class ChatRoom(Base):
    __tablename__ = "chat_rooms"
    
    room_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    room_type = Column(String(20), nullable=False)  # Store as string instead of enum for simplicity
    topic_category = Column(String(50))
    created_by = Column(String(50), nullable=False)  # Foreign key to User.user_id
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    participant_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)
    last_activity_at = Column(DateTime, default=func.now())
    moderation_level = Column(String(20), default="open")  # Store as string
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'room_id': self.room_id,
            'name': self.name,
            'description': self.description,
            'room_type': self.room_type,
            'topic_category': self.topic_category,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active,
            'participant_count': self.participant_count,
            'message_count': self.message_count,
            'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None,
            'moderation_level': self.moderation_level
        }
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
