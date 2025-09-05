"""
WebSocket handler for real-time chat functionality.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from services.websocket_service import websocket_manager
from services.chat_service import ChatService
import json

router = APIRouter()

@router.websocket("/ws/chat/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: str,
    user_id: str = Query(..., description="User ID for authentication"),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time chat."""
    chat_service = ChatService(db)
    
    # Verify room exists
    room = chat_service.get_room_by_id(room_id)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return
    
    # Verify or create user profile
    user_profile = chat_service.get_user_profile(user_id)
    if not user_profile:
        # Create a basic profile if user doesn't exist
        user_profile = chat_service.create_user_profile(
            user_id=user_id,
            display_name=f"User_{user_id[:8]}"
        )
    
    try:
        # Connect user to the room
        await websocket_manager.connect(websocket, room_id, user_id)
        
        # Update user activity
        chat_service.update_user_activity(user_id)
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "room_id": room_id,
            "user_id": user_id,
            "message": f"Connected to {room.get('name', 'Chat Room')}"
        }))
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            
            # Parse the message
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type")
                
                # Handle different message types
                if message_type == "chat_message":
                    # Save message to database
                    content = message_data.get("content", "").strip()
                    if content:  # Only save non-empty messages
                        saved_message = chat_service.send_message(
                            room_id=room_id,
                            user_id=user_id,
                            content=content,
                            message_type="text"
                        )
                        
                        # Broadcast the saved message with database ID
                        await websocket_manager.broadcast_to_room(room_id, {
                            "type": "chat_message",
                            "message_id": saved_message["message_id"],
                            "user_id": user_id,
                            "content": content,
                            "timestamp": saved_message["sent_at"],
                            "room_id": room_id,
                            "user_profile": user_profile
                        })
                
                elif message_type == "typing":
                    # Handle typing indicators (don't save to database)
                    await websocket_manager.handle_message(websocket, data)
                
                elif message_type == "stop_typing":
                    # Handle stop typing indicators
                    await websocket_manager.handle_message(websocket, data)
                
                elif message_type == "ping":
                    # Handle ping for connection keepalive
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message_data.get("timestamp")
                    }))
                
                else:
                    # Handle other message types
                    await websocket_manager.handle_message(websocket, data)
                    
                # Update user activity for any interaction
                chat_service.update_user_activity(user_id)
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to process message"
                }))
                
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected from room {room_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        # Clean up connection
        await websocket_manager.disconnect(websocket)

@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket server status."""
    return {
        "active_rooms": websocket_manager.get_active_rooms(),
        "total_connections": sum(
            websocket_manager.get_room_user_count(room_id) 
            for room_id in websocket_manager.get_active_rooms()
        )
    }
