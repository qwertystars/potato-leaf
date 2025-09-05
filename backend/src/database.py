"""
Database configuration and session management for chat system.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL from environment or default to SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_database.db")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a single Base that includes all models
Base = declarative_base()

def create_tables():
    """Create all database tables."""
    try:
        # Import all models to ensure they're registered
        from models import chat_room, message, user_profile, moderation, language
        
        # Create all tables
        chat_room.Base.metadata.create_all(bind=engine)
        message.Base.metadata.create_all(bind=engine)
        user_profile.Base.metadata.create_all(bind=engine)
        moderation.Base.metadata.create_all(bind=engine)
        language.Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with default data."""
    create_tables()
    
    # Add default chat rooms and language data
    db = SessionLocal()
    try:
        from models.chat_room import ChatRoom
        from models.language import Language
        from services.language_service import seed_languages, seed_translations
        
        # Check if rooms already exist
        existing_rooms = db.query(ChatRoom).count()
        if existing_rooms == 0:
            default_rooms = [
                ChatRoom(
                    name="Early Blight Discussion",
                    description="Discuss early blight symptoms, treatments, and prevention",
                    room_type="disease_specific",
                    topic_category="early_blight",
                    created_by="system"
                ),
                ChatRoom(
                    name="Late Blight Discussion",
                    description="Share experiences with late blight management",
                    room_type="disease_specific",
                    topic_category="late_blight",
                    created_by="system"
                ),
                ChatRoom(
                    name="General Farming Chat",
                    description="General discussion about potato farming and agriculture",
                    room_type="general",
                    topic_category="general",
                    created_by="system"
                ),
                ChatRoom(
                    name="Healthy Potato Growing",
                    description="Tips for maintaining healthy potato crops",
                    room_type="disease_specific",
                    topic_category="healthy",
                    created_by="system"
                )
            ]
            
            for room in default_rooms:
                db.add(room)
            
            db.commit()
            print("✅ Default chat rooms created")
        else:
            print("ℹ️ Chat rooms already initialized")
        
        # Initialize language data
        existing_languages = db.query(Language).count()
        if existing_languages == 0:
            seed_languages(db)
            seed_translations(db)
            print("✅ Multi-language support initialized")
        else:
            print("ℹ️ Language data already initialized")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
