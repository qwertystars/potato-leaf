"""
Chat service for managing chat rooms, messages, and user interactions.
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from models.chat_room import ChatRoom
from models.message import Message
from models.user_profile import UserProfile
from models.moderation import ModerationAction
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import uuid

class ChatService:
    def __init__(self, db: Session):
        self.db = db
    
    # Chat Room Management
    def get_active_rooms(self, limit: int = 50) -> List[Dict]:
        """Get all active chat rooms."""
        rooms = self.db.query(ChatRoom).filter(
            ChatRoom.is_active == True
        ).order_by(desc(ChatRoom.last_activity_at)).limit(limit).all()
        
        return [room.to_dict() for room in rooms]
    
    def get_room_by_id(self, room_id: str) -> Optional[Dict]:
        """Get a specific room by ID."""
        room = self.db.query(ChatRoom).filter(ChatRoom.room_id == room_id).first()
        return room.to_dict() if room else None
    
    def create_room(self, name: str, description: str, room_type: str, 
                   topic_category: str, created_by: str) -> Dict:
        """Create a new chat room."""
        room = ChatRoom(
            name=name,
            description=description,
            room_type=room_type,
            topic_category=topic_category,
            created_by=created_by
        )
        
        self.db.add(room)
        self.db.commit()
        self.db.refresh(room)
        
        return room.to_dict()
    
    def get_rooms_by_category(self, category: str) -> List[Dict]:
        """Get rooms by topic category."""
        rooms = self.db.query(ChatRoom).filter(
            and_(ChatRoom.topic_category == category, ChatRoom.is_active == True)
        ).order_by(desc(ChatRoom.last_activity_at)).all()
        
        return [room.to_dict() for room in rooms]
    
    # Message Management
    def send_message(self, room_id: str, user_id: str, content: str, 
                    message_type: str = "text", parent_message_id: str = None) -> Dict:
        """Send a new message to a chat room."""
        # Validate room exists and is active
        room = self.db.query(ChatRoom).filter(ChatRoom.room_id == room_id).first()
        if not room or not room.is_active:
            raise ValueError("Room not found or inactive")
        
        # Create message
        message = Message(
            room_id=room_id,
            user_id=user_id,
            content=content,
            message_type=message_type,
            parent_message_id=parent_message_id
        )
        
        self.db.add(message)
        
        # Update room statistics
        room.message_count += 1
        room.update_activity()
        
        # Update user statistics
        user = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if user:
            user.increment_message_count()
            user.update_activity()
        
        self.db.commit()
        self.db.refresh(message)
        
        return message.to_dict()
    
    def get_room_messages(self, room_id: str, limit: int = 50, 
                         offset: int = 0) -> List[Dict]:
        """Get messages from a chat room."""
        messages = self.db.query(Message).filter(
            and_(
                Message.room_id == room_id,
                Message.is_deleted == False,
                Message.moderation_status == "approved"
            )
        ).order_by(desc(Message.sent_at)).limit(limit).offset(offset).all()
        
        return [msg.to_dict() for msg in messages]
    
    def get_recent_messages(self, room_id: str, since: datetime) -> List[Dict]:
        """Get messages since a specific timestamp."""
        messages = self.db.query(Message).filter(
            and_(
                Message.room_id == room_id,
                Message.sent_at > since,
                Message.is_deleted == False,
                Message.moderation_status == "approved"
            )
        ).order_by(Message.sent_at).all()
        
        return [msg.to_dict() for msg in messages]
    
    def edit_message(self, message_id: str, user_id: str, new_content: str) -> bool:
        """Edit a user's own message."""
        message = self.db.query(Message).filter(
            and_(Message.message_id == message_id, Message.user_id == user_id)
        ).first()
        
        if not message:
            return False
        
        message.edit_content(new_content)
        self.db.commit()
        return True
    
    def delete_message(self, message_id: str, user_id: str) -> bool:
        """Soft delete a user's own message."""
        message = self.db.query(Message).filter(
            and_(Message.message_id == message_id, Message.user_id == user_id)
        ).first()
        
        if not message:
            return False
        
        message.mark_deleted()
        self.db.commit()
        return True
    
    # User Management
    def create_user_profile(self, user_id: str, display_name: str, 
                           farming_context: str = None) -> Dict:
        """Create a new user profile."""
        # Check if user already exists
        existing_user = self.db.query(UserProfile).filter(
            UserProfile.user_id == user_id
        ).first()
        
        if existing_user:
            return existing_user.to_dict()
        
        user = UserProfile(
            user_id=user_id,
            display_name=display_name,
            farming_context=farming_context
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        return user.to_dict()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile by ID."""
        user = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        return user.to_dict() if user else None
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        user = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if user:
            user.update_activity()
            self.db.commit()
    
    # Moderation Functions
    def report_message(self, message_id: str, reported_by: str, reason: str) -> Dict:
        """Report a message for moderation."""
        message = self.db.query(Message).filter(Message.message_id == message_id).first()
        if not message:
            raise ValueError("Message not found")
        
        moderation_action = ModerationAction(
            message_id=message_id,
            reported_user_id=message.user_id,
            moderator_id=reported_by,
            action_type="report",
            reason=reason
        )
        
        # Flag the message for review
        message.update_moderation_status("flagged")
        
        self.db.add(moderation_action)
        self.db.commit()
        self.db.refresh(moderation_action)
        
        return moderation_action.to_dict()
    
    def moderate_message(self, message_id: str, moderator_id: str, 
                        action: str, reason: str) -> bool:
        """Take moderation action on a message."""
        message = self.db.query(Message).filter(Message.message_id == message_id).first()
        if not message:
            return False
        
        # Create moderation record
        moderation_action = ModerationAction(
            message_id=message_id,
            reported_user_id=message.user_id,
            moderator_id=moderator_id,
            action_type=action,
            reason=reason
        )
        
        # Apply the action
        if action == "hide":
            message.update_moderation_status("hidden")
        elif action == "delete":
            message.mark_deleted()
        elif action == "approve":
            message.update_moderation_status("approved")
        
        self.db.add(moderation_action)
        self.db.commit()
        return True
    
    def get_room_participants(self, room_id: str) -> List[Dict]:
        """Get active participants in a room (users who sent messages recently)."""
        recent_threshold = datetime.utcnow() - timedelta(hours=1)
        
        participants = self.db.query(UserProfile).join(
            Message, UserProfile.user_id == Message.user_id
        ).filter(
            and_(
                Message.room_id == room_id,
                Message.sent_at > recent_threshold
            )
        ).distinct().all()
        
        return [user.to_dict() for user in participants]
