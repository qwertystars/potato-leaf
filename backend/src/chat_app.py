"""
FastAPI application for chat functionality.
This runs alongside the existing Flask app.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.chat_routes import router as chat_router
from api.websocket_handler import router as websocket_router
from database import init_database
import uvicorn
import os

# Create FastAPI app
app = FastAPI(
    title="Potato Disease Analyzer - Chat API",
    description="Real-time chat system for farmer communication",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(websocket_router)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    print("🚀 Starting Potato Disease Analyzer Chat API...")
    init_database()
    print("✅ Chat API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down Chat API...")

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Potato Disease Analyzer Chat API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "chat-api",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Get port from environment or default to 8001
    port = int(os.getenv("CHAT_API_PORT", 8001))
    
    print(f"🚀 Starting Chat API on port {port}")
    uvicorn.run(
        "chat_app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
