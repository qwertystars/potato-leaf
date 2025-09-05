"""
Database initialization script for multi-language support.
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(__file__))

from database import engine, SessionLocal, Base
from models.language import Language, UserLanguagePreference, Translation, TranslationSession, AITranslationCache, TranslationFeedback
from services.language_service import seed_languages, seed_translations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_language_tables():
    """Create all language-related database tables."""
    try:
        # Import all models to ensure they're registered with Base
        from models import language
        
        # Create tables
        language.Base.metadata.create_all(bind=engine)
        logger.info("✅ Language tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating language tables: {e}")
        raise


def initialize_language_data():
    """Initialize database with language data."""
    try:
        db = SessionLocal()
        
        # Seed languages
        seed_languages(db)
        logger.info("✅ Languages seeded successfully")
        
        # Seed translations
        seed_translations(db)
        logger.info("✅ Translations seeded successfully")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Error initializing language data: {e}")
        raise


def main():
    """Main initialization function."""
    logger.info("🚀 Starting language database initialization...")
    
    try:
        # Create tables
        create_language_tables()
        
        # Initialize data
        initialize_language_data()
        
        logger.info("✅ Language database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Language database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
