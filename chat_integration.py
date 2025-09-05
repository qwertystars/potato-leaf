"""
Flask-SocketIO integration for real-time chat functionality.
This integrates the chat system directly into the Flask app on port 5000.
"""
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import uuid
from datetime import datetime
import sys
import os

# Add backend src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

try:
    from backend.src.database import SessionLocal, init_database
    from backend.src.services.chat_service import ChatService
    from backend.src.services.moderation_service import moderation_service
    CHAT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Chat functionality not available: {e}")
    CHAT_AVAILABLE = False

class ChatManager:
    def __init__(self):
        self.active_users = {}  # user_id -> session_id
        self.room_users = {}    # room_id -> set of user_ids
        
    def user_join_room(self, user_id, room_id, session_id):
        """Track user joining a room."""
        self.active_users[user_id] = session_id
        if room_id not in self.room_users:
            self.room_users[room_id] = set()
        self.room_users[room_id].add(user_id)
        
    def user_leave_room(self, user_id, room_id):
        """Track user leaving a room."""
        if user_id in self.active_users:
            del self.active_users[user_id]
        if room_id in self.room_users:
            self.room_users[room_id].discard(user_id)
            if not self.room_users[room_id]:
                del self.room_users[room_id]
    
    def get_room_users(self, room_id):
        """Get list of users in a room."""
        return list(self.room_users.get(room_id, set()))

# Global chat manager
chat_manager = ChatManager()

def init_socketio(app):
    """Initialize SocketIO with the Flask app."""
    if not CHAT_AVAILABLE:
        return None
        
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    @socketio.on('connect')
    def handle_connect(auth=None):
        """Handle client connection."""
        print(f"Client connected: {request.sid}")
        emit('connected', {'message': 'Connected to chat server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"Client disconnected: {request.sid}")
        # Clean up user from all rooms
        for room_id in list(chat_manager.room_users.keys()):
            for user_id in list(chat_manager.room_users[room_id]):
                if chat_manager.active_users.get(user_id) == request.sid:
                    chat_manager.user_leave_room(user_id, room_id)
                    socketio.emit('user_left', {
                        'user_id': user_id,
                        'timestamp': datetime.utcnow().isoformat()
                    }, room=room_id)
    
    @socketio.on('join_room')
    def handle_join_room(data):
        """Handle user joining a chat room."""
        try:
            room_id = data['room_id']
            user_id = data['user_id']
            
            # Join the SocketIO room
            join_room(room_id)
            
            # Track in chat manager
            chat_manager.user_join_room(user_id, room_id, request.sid)
            
            # Get/create user profile
            db = SessionLocal()
            try:
                chat_service = ChatService(db)
                user_profile = chat_service.get_user_profile(user_id)
                if not user_profile:
                    user_profile = chat_service.create_user_profile(
                        user_id=user_id,
                        display_name=f"Farmer_{user_id[-4:]}"
                    )
                
                # Update user activity
                chat_service.update_user_activity(user_id)
                
            finally:
                db.close()
            
            # Notify room about new user
            emit('user_joined', {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'user_profile': user_profile
            }, room=room_id)
            
            # Send welcome message to user
            emit('room_joined', {
                'room_id': room_id,
                'message': f'You joined the chat room',
                'participants': chat_manager.get_room_users(room_id)
            })
            
        except Exception as e:
            print(f"Error in join_room: {e}")
            emit('error', {'message': 'Failed to join room'})
    
    @socketio.on('leave_room')
    def handle_leave_room(data):
        """Handle user leaving a chat room."""
        try:
            room_id = data['room_id']
            user_id = data['user_id']
            
            # Leave the SocketIO room
            leave_room(room_id)
            
            # Track in chat manager
            chat_manager.user_leave_room(user_id, room_id)
            
            # Notify room about user leaving
            emit('user_left', {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, room=room_id)
            
        except Exception as e:
            print(f"Error in leave_room: {e}")
            emit('error', {'message': 'Failed to leave room'})
    
    @socketio.on('send_message')
    def handle_send_message(data):
        """Handle sending a chat message."""
        try:
            room_id = data['room_id']
            user_id = data['user_id']
            content = data['content'].strip()
            
            if not content:
                return
            
            # Check content moderation
            moderation_result = moderation_service.check_content(content)
            if not moderation_result['approved']:
                emit('message_rejected', {
                    'reason': 'Message flagged by moderation',
                    'suggestions': moderation_service.suggest_improvements(content)
                })
                return
            
            # Save message to database
            db = SessionLocal()
            try:
                chat_service = ChatService(db)
                saved_message = chat_service.send_message(
                    room_id=room_id,
                    user_id=user_id,
                    content=content,
                    message_type="text"
                )
                
                # Get user profile for display
                user_profile = chat_service.get_user_profile(user_id)
                
            finally:
                db.close()
            
            # Broadcast message to room
            message_data = {
                'type': 'chat_message',
                'message_id': saved_message['message_id'],
                'user_id': user_id,
                'content': content,
                'timestamp': saved_message['sent_at'],
                'room_id': room_id,
                'user_profile': user_profile
            }
            
            socketio.emit('new_message', message_data, room=room_id)
            
        except Exception as e:
            print(f"Error sending message: {e}")
            emit('error', {'message': 'Failed to send message'})
    
    @socketio.on('typing')
    def handle_typing(data):
        """Handle typing indicator."""
        try:
            room_id = data['room_id']
            user_id = data['user_id']
            
            # Broadcast typing indicator to others in room
            emit('user_typing', {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, room=room_id, include_self=False)
            
        except Exception as e:
            print(f"Error in typing: {e}")
    
    @socketio.on('stop_typing')
    def handle_stop_typing(data):
        """Handle stop typing indicator."""
        try:
            room_id = data['room_id']
            user_id = data['user_id']
            
            # Broadcast stop typing to others in room
            emit('user_stop_typing', {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, room=room_id, include_self=False)
            
        except Exception as e:
            print(f"Error in stop_typing: {e}")
    
    return socketio

def setup_chat_routes(app):
    """Setup Flask routes for chat functionality."""
    if not CHAT_AVAILABLE:
        return
    
    @app.route("/api/chat/rooms")
    def get_chat_rooms():
        """Get list of available chat rooms."""
        try:
            db = SessionLocal()
            try:
                chat_service = ChatService(db)
                rooms = chat_service.get_active_rooms()
                return {"rooms": rooms}
            finally:
                db.close()
        except Exception as e:
            return {"error": str(e)}, 500
    
    @app.route("/api/chat/rooms/<room_id>/messages")
    def get_room_messages(room_id):
        """Get messages from a chat room."""
        try:
            db = SessionLocal()
            try:
                chat_service = ChatService(db)
                messages = chat_service.get_room_messages(room_id, limit=50)
                return {"messages": messages}
            finally:
                db.close()
        except Exception as e:
            return {"error": str(e)}, 500
    
    @app.route("/api/chat/users/<user_id>/profile", methods=["GET", "POST"])
    def user_profile(user_id):
        """Get or create user profile."""
        try:
            db = SessionLocal()
            try:
                chat_service = ChatService(db)
                
                if request.method == "POST":
                    data = request.get_json()
                    profile = chat_service.create_user_profile(
                        user_id=user_id,
                        display_name=data.get('display_name', f'Farmer_{user_id[-4:]}'),
                        farming_context=data.get('farming_context', 'Potato farmer')
                    )
                else:
                    profile = chat_service.get_user_profile(user_id)
                    if not profile:
                        return {"error": "Profile not found"}, 404
                
                return profile
            finally:
                db.close()
        except Exception as e:
            return {"error": str(e)}, 500

def init_chat_system():
    """Initialize the chat system."""
    if CHAT_AVAILABLE:
        print("🚀 Initializing chat system...")
        init_database()
        print("✅ Chat system ready!")
    else:
        print("⚠️ Chat system not available")
