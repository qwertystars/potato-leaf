"""
FastAPI routes for chat functionality.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from services.chat_service import ChatService
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Pydantic models for request/response
class RoomCreate(BaseModel):
    name: str
    description: str
    room_type: str
    topic_category: str

class MessageCreate(BaseModel):
    content: str
    message_type: str = "text"
    parent_message_id: Optional[str] = None

class MessageEdit(BaseModel):
    content: str

class UserProfileCreate(BaseModel):
    display_name: str
    farming_context: Optional[str] = None

class ModerationReport(BaseModel):
    reason: str

class ModerationAction(BaseModel):
    action: str  # "hide", "delete", "approve"
    reason: str

@router.get("/rooms")
async def get_chat_rooms(
    category: Optional[str] = Query(None, description="Filter by topic category"),
    limit: int = Query(50, description="Maximum number of rooms to return"),
    db: Session = Depends(get_db)
):
    """Get list of available chat rooms."""
    chat_service = ChatService(db)
    
    if category:
        rooms = chat_service.get_rooms_by_category(category)
    else:
        rooms = chat_service.get_active_rooms(limit)
    
    return {"rooms": rooms}

@router.post("/rooms")
async def create_chat_room(
    room_data: RoomCreate,
    user_id: str = Query(..., description="ID of the user creating the room"),
    db: Session = Depends(get_db)
):
    """Create a new chat room."""
    chat_service = ChatService(db)
    
    try:
        room = chat_service.create_room(
            name=room_data.name,
            description=room_data.description,
            room_type=room_data.room_type,
            topic_category=room_data.topic_category,
            created_by=user_id
        )
        return room
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rooms/{room_id}")
async def get_chat_room(room_id: str, db: Session = Depends(get_db)):
    """Get details of a specific chat room."""
    chat_service = ChatService(db)
    
    room = chat_service.get_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return room

@router.get("/rooms/{room_id}/messages")
async def get_room_messages(
    room_id: str,
    limit: int = Query(50, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Number of messages to skip"),
    since: Optional[str] = Query(None, description="ISO timestamp to get messages since"),
    db: Session = Depends(get_db)
):
    """Get messages from a chat room."""
    chat_service = ChatService(db)
    
    # Check if room exists
    room = chat_service.get_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if since:
        try:
            since_datetime = datetime.fromisoformat(since.replace('Z', '+00:00'))
            messages = chat_service.get_recent_messages(room_id, since_datetime)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    else:
        messages = chat_service.get_room_messages(room_id, limit, offset)
    
    return {"messages": messages}

@router.post("/rooms/{room_id}/messages")
async def send_message(
    room_id: str,
    message_data: MessageCreate,
    user_id: str = Query(..., description="ID of the user sending the message"),
    db: Session = Depends(get_db)
):
    """Send a message to a chat room."""
    chat_service = ChatService(db)
    
    try:
        message = chat_service.send_message(
            room_id=room_id,
            user_id=user_id,
            content=message_data.content,
            message_type=message_data.message_type,
            parent_message_id=message_data.parent_message_id
        )
        return message
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to send message")

@router.put("/messages/{message_id}")
async def edit_message(
    message_id: str,
    message_data: MessageEdit,
    user_id: str = Query(..., description="ID of the user editing the message"),
    db: Session = Depends(get_db)
):
    """Edit a user's own message."""
    chat_service = ChatService(db)
    
    success = chat_service.edit_message(message_id, user_id, message_data.content)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found or not owned by user")
    
    return {"message": "Message updated successfully"}

@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str,
    user_id: str = Query(..., description="ID of the user deleting the message"),
    db: Session = Depends(get_db)
):
    """Delete a user's own message."""
    chat_service = ChatService(db)
    
    success = chat_service.delete_message(message_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found or not owned by user")
    
    return {"message": "Message deleted successfully"}

@router.get("/rooms/{room_id}/participants")
async def get_room_participants(room_id: str, db: Session = Depends(get_db)):
    """Get active participants in a room."""
    chat_service = ChatService(db)
    
    # Check if room exists
    room = chat_service.get_room_by_id(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    participants = chat_service.get_room_participants(room_id)
    return {"participants": participants}

@router.post("/users/profile")
async def create_user_profile(
    profile_data: UserProfileCreate,
    user_id: str = Query(..., description="User ID"),
    db: Session = Depends(get_db)
):
    """Create or update user profile."""
    chat_service = ChatService(db)
    
    profile = chat_service.create_user_profile(
        user_id=user_id,
        display_name=profile_data.display_name,
        farming_context=profile_data.farming_context
    )
    return profile

@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Get user profile by ID."""
    chat_service = ChatService(db)
    
    profile = chat_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    return profile

@router.post("/messages/{message_id}/report")
async def report_message(
    message_id: str,
    report_data: ModerationReport,
    user_id: str = Query(..., description="ID of the user reporting the message"),
    db: Session = Depends(get_db)
):
    """Report a message for moderation."""
    chat_service = ChatService(db)
    
    try:
        moderation_action = chat_service.report_message(
            message_id=message_id,
            reported_by=user_id,
            reason=report_data.reason
        )
        return moderation_action
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/messages/{message_id}/moderate")
async def moderate_message(
    message_id: str,
    action_data: ModerationAction,
    moderator_id: str = Query(..., description="ID of the moderator"),
    db: Session = Depends(get_db)
):
    """Take moderation action on a message (admin only)."""
    # Note: In a real implementation, you'd check if the user has moderator privileges
    chat_service = ChatService(db)
    
    success = chat_service.moderate_message(
        message_id=message_id,
        moderator_id=moderator_id,
        action=action_data.action,
        reason=action_data.reason
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"message": f"Moderation action '{action_data.action}' applied successfully"}
