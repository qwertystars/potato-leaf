"""
WebSocket service for real-time chat functionality.
"""
import json
import redis
from typing import Dict, List, Set
from datetime import datetime
import asyncio
import uuid

class WebSocketManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize WebSocket manager with Redis for pub/sub."""
        self.active_connections: Dict[str, Set[object]] = {}  # room_id -> set of websockets
        self.user_connections: Dict[str, object] = {}  # user_id -> websocket
        self.connection_rooms: Dict[object, str] = {}  # websocket -> room_id
        self.connection_users: Dict[object, str] = {}  # websocket -> user_id
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            print("✅ Redis connection established for WebSocket manager")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
            self.pubsub = None
    
    async def connect(self, websocket, room_id: str, user_id: str):
        """Connect a user to a chat room."""
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Add to room connections
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
        
        # Track user and room mappings
        self.user_connections[user_id] = websocket
        self.connection_rooms[websocket] = room_id
        self.connection_users[websocket] = user_id
        
        # Subscribe to Redis channel for this room if Redis is available
        if self.redis_client:
            try:
                # Subscribe to room-specific channel
                await self._subscribe_to_room(room_id)
            except Exception as e:
                print(f"⚠️ Redis subscription failed: {e}")
        
        # Notify room about new user joining
        await self.broadcast_to_room(room_id, {
            "type": "user_joined",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"User {user_id} joined the chat"
        }, exclude_user=user_id)
        
        print(f"✅ User {user_id} connected to room {room_id}")
    
    async def disconnect(self, websocket):
        """Disconnect a user from their room."""
        room_id = self.connection_rooms.get(websocket)
        user_id = self.connection_users.get(websocket)
        
        if room_id and websocket in self.active_connections.get(room_id, set()):
            self.active_connections[room_id].discard(websocket)
            
            # Clean up empty room connections
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
        
        # Clean up user mappings
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
        
        if websocket in self.connection_rooms:
            del self.connection_rooms[websocket]
        
        if websocket in self.connection_users:
            del self.connection_users[websocket]
        
        # Notify room about user leaving
        if room_id and user_id:
            await self.broadcast_to_room(room_id, {
                "type": "user_left",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"User {user_id} left the chat"
            })
        
        print(f"✅ User {user_id} disconnected from room {room_id}")
    
    async def broadcast_to_room(self, room_id: str, message: Dict, exclude_user: str = None):
        """Broadcast a message to all users in a room."""
        if room_id in self.active_connections:
            message_json = json.dumps(message)
            disconnected_websockets = []
            
            for websocket in self.active_connections[room_id]:
                # Skip the user who sent the message if exclude_user is specified
                user_id = self.connection_users.get(websocket)
                if exclude_user and user_id == exclude_user:
                    continue
                
                try:
                    await websocket.send_text(message_json)
                except Exception as e:
                    print(f"❌ Failed to send message to websocket: {e}")
                    disconnected_websockets.append(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected_websockets:
                await self.disconnect(websocket)
        
        # Also publish to Redis for scaling across multiple server instances
        if self.redis_client:
            try:
                await self._publish_to_redis(room_id, message)
            except Exception as e:
                print(f"⚠️ Redis publish failed: {e}")
    
    async def send_to_user(self, user_id: str, message: Dict):
        """Send a direct message to a specific user."""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"❌ Failed to send message to user {user_id}: {e}")
                await self.disconnect(websocket)
    
    async def handle_message(self, websocket, data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            room_id = self.connection_rooms.get(websocket)
            user_id = self.connection_users.get(websocket)
            
            if not room_id or not user_id:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Not connected to a room"
                }))
                return
            
            if message_type == "chat_message":
                # Broadcast chat message to room
                await self.broadcast_to_room(room_id, {
                    "type": "chat_message",
                    "message_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "content": message.get("content", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "room_id": room_id
                })
            
            elif message_type == "typing":
                # Notify others that user is typing
                await self.broadcast_to_room(room_id, {
                    "type": "typing",
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, exclude_user=user_id)
            
            elif message_type == "stop_typing":
                # Notify others that user stopped typing
                await self.broadcast_to_room(room_id, {
                    "type": "stop_typing",
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, exclude_user=user_id)
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
                
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid JSON format"
            }))
        except Exception as e:
            print(f"❌ Error handling message: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Internal server error"
            }))
    
    async def _subscribe_to_room(self, room_id: str):
        """Subscribe to Redis channel for a room."""
        if self.pubsub:
            channel = f"chat_room_{room_id}"
            self.pubsub.subscribe(channel)
    
    async def _publish_to_redis(self, room_id: str, message: Dict):
        """Publish message to Redis for other server instances."""
        if self.redis_client:
            channel = f"chat_room_{room_id}"
            self.redis_client.publish(channel, json.dumps(message))
    
    def get_room_user_count(self, room_id: str) -> int:
        """Get the number of active users in a room."""
        return len(self.active_connections.get(room_id, set()))
    
    def get_active_rooms(self) -> List[str]:
        """Get list of rooms with active connections."""
        return list(self.active_connections.keys())
    
    def is_user_online(self, user_id: str) -> bool:
        """Check if a user is currently online."""
        return user_id in self.user_connections

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
