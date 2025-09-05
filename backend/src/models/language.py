"""
Language model for multi-language support.
"""
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, Enum, Float
from sqlalchemy.sql import func
from database import Base
import enum


class Language(Base):
    """Represents supported languages in the system."""
    __tablename__ = "languages"

    code = Column(String(5), primary_key=True)  # ISO 639-1 language code
    name = Column(String(100), nullable=False)  # Native language name
    english_name = Column(String(100), nullable=False)  # English name for admin interface
    rtl_direction = Column(Boolean, default=False, nullable=False)  # Right-to-left languages
    enabled = Column(Boolean, default=False, nullable=False)  # Currently available
    flag_icon = Column(String(10))  # Unicode flag emoji or icon identifier
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<Language(code='{self.code}', name='{self.name}', enabled={self.enabled})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "code": self.code,
            "name": self.name,
            "englishName": self.english_name,
            "rtlDirection": self.rtl_direction,
            "enabled": self.enabled,
            "flagIcon": self.flag_icon
        }


class PreferenceSource(enum.Enum):
    """Source of language preference."""
    USER_SELECTION = "user_selection"
    BROWSER_LOCALE = "browser_locale"
    IP_GEOLOCATION = "ip_geolocation"
    DEFAULT = "default"


class UserLanguagePreference(Base):
    """Stores user language preferences with fallback hierarchy."""
    __tablename__ = "user_language_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=True)  # User identifier (null for anonymous)
    session_id = Column(String(128), nullable=True)  # Browser session identifier
    language_code = Column(String(5), nullable=False)  # Foreign key to Language.code
    preference_source = Column(Enum(PreferenceSource), nullable=False)
    selected_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)  # null for permanent preferences
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<UserLanguagePreference(user_id='{self.user_id}', language='{self.language_code}', active={self.is_active})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "languageCode": self.language_code,
            "preferenceSource": self.preference_source.value,
            "selectedAt": self.selected_at.isoformat() if self.selected_at else None,
            "isActive": self.is_active
        }


class Translation(Base):
    """Static translation content for UI elements."""
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), nullable=False)  # Hierarchical translation key
    language_code = Column(String(5), nullable=False)  # Foreign key to Language.code
    value = Column(Text, nullable=False)  # Translated content
    context = Column(String(500), nullable=True)  # Optional context for translators
    namespace = Column(String(50), nullable=False)  # Logical grouping
    is_reviewed = Column(Boolean, default=False, nullable=False)  # Expert-reviewed
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<Translation(key='{self.key}', language='{self.language_code}', namespace='{self.namespace}')>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "key": self.key,
            "value": self.value,
            "languageCode": self.language_code,
            "namespace": self.namespace,
            "isReviewed": self.is_reviewed
        }


class TranslationSession(Base):
    """Tracks active language sessions for analytics and debugging."""
    __tablename__ = "translation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False)  # Browser session identifier
    language_code = Column(String(5), nullable=False)  # Current active language
    started_at = Column(DateTime, default=func.now(), nullable=False)
    last_activity_at = Column(DateTime, default=func.now(), nullable=False)
    page_views = Column(Integer, default=0, nullable=False)
    ai_translations_requested = Column(Integer, default=0, nullable=False)
    ended_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<TranslationSession(session_id='{self.session_id}', language='{self.language_code}')>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "sessionId": self.session_id,
            "languageCode": self.language_code,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "lastActivityAt": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "pageViews": self.page_views,
            "aiTranslationsRequested": self.ai_translations_requested
        }


class TranslationService(enum.Enum):
    """AI translation service providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    AWS = "aws"


class ReviewStatus(enum.Enum):
    """Review status for AI translations."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class AITranslationCache(Base):
    """Caches AI-generated translations for performance."""
    __tablename__ = "ai_translation_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_hash = Column(String(64), nullable=False)  # SHA-256 hash of original content
    source_language = Column(String(5), nullable=False)  # Original content language code
    target_language = Column(String(5), nullable=False)  # Translation target language code
    original_content = Column(Text, nullable=False)  # Original content to translate
    translated_content = Column(Text, nullable=False)  # AI-generated translation
    translation_service = Column(Enum(TranslationService), nullable=False)  # AI service used
    confidence_score = Column(Float, nullable=False)  # Translation confidence (0.0-1.0)
    is_reviewed = Column(Boolean, default=False, nullable=False)  # Human-reviewed
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)  # Cache expiration time
    usage_count = Column(Integer, default=0, nullable=False)  # Times this translation was served

    def __repr__(self):
        return f"<AITranslationCache(hash='{self.content_hash[:8]}...', {self.source_language}->{self.target_language})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "translatedContent": self.translated_content,
            "confidenceScore": self.confidence_score,
            "cached": True,
            "translationService": self.translation_service.value,
            "reviewStatus": self.review_status.value
        }


class FeedbackType(enum.Enum):
    """Types of user feedback on translations."""
    HELPFUL = "helpful"
    UNHELPFUL = "unhelpful"
    INCORRECT = "incorrect"
    OFFENSIVE = "offensive"


class TranslationType(enum.Enum):
    """Types of translations."""
    STATIC = "static"
    AI_GENERATED = "ai_generated"


class TranslationFeedback(Base):
    """User feedback on translation quality."""
    __tablename__ = "translation_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    translation_id = Column(Integer, nullable=False)  # Reference to Translation or AITranslationCache
    translation_type = Column(Enum(TranslationType), nullable=False)
    user_id = Column(String(128), nullable=True)  # Optional user identifier
    session_id = Column(String(128), nullable=True)  # Browser session identifier
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    suggested_improvement = Column(Text, nullable=True)  # Optional user suggestion (max 500 chars)
    reported_at = Column(DateTime, default=func.now(), nullable=False)
    is_processed = Column(Boolean, default=False, nullable=False)  # Reviewed by team

    def __repr__(self):
        return f"<TranslationFeedback(translation_id={self.translation_id}, type={self.feedback_type.value})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "translationId": self.translation_id,
            "translationType": self.translation_type.value,
            "feedbackType": self.feedback_type.value,
            "suggestedImprovement": self.suggested_improvement,
            "reportedAt": self.reported_at.isoformat() if self.reported_at else None,
            "isProcessed": self.is_processed
        }
