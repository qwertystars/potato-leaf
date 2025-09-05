"""
Language API endpoints for multi-language support.
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import logging

from ..database import get_db
from ..services.language_service import LanguageService, TranslationService
from ..models.language import PreferenceSource, FeedbackType, TranslationType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Language"])


# Pydantic models for API
class LanguagePreferenceRequest(BaseModel):
    language_code: str = Field(..., description="Language code to set as preference")
    source: Optional[str] = Field("user_selection", description="Source of the preference")


class TranslationRequest(BaseModel):
    content: str = Field(..., max_length=10000, description="Content to translate")
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field("en", description="Source language code")
    context: Optional[str] = Field("general", description="Context for better translation")
    use_cache: Optional[bool] = Field(True, description="Whether to use cached translations")


class TranslationFeedbackRequest(BaseModel):
    translation_id: int = Field(..., description="ID of the translation")
    translation_type: str = Field(..., description="Type of translation (static or ai_generated)")
    feedback_type: str = Field(..., description="Type of feedback")
    suggested_improvement: Optional[str] = Field(None, max_length=500, description="Suggested improvement")


def get_session_id(request: Request, session_id: Optional[str] = Header(None)) -> str:
    """Get or generate session ID for anonymous users."""
    if session_id:
        return session_id
    
    # Generate new session ID
    new_session_id = str(uuid.uuid4())
    logger.info(f"Generated new session ID: {new_session_id}")
    return new_session_id


@router.get("/languages")
async def get_languages(db: Session = Depends(get_db)):
    """Get available languages."""
    try:
        language_service = LanguageService(db)
        languages = language_service.get_available_languages()
        default_language = language_service.get_default_language()
        
        return {
            "languages": languages,
            "defaultLanguage": default_language
        }
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user/language-preference")
async def get_user_language_preference(
    request: Request,
    session_id: str = Depends(get_session_id),
    user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Get user's current language preference."""
    try:
        language_service = LanguageService(db)
        preference = language_service.get_user_language_preference(user_id, session_id)
        
        if not preference:
            # Return default language if no preference set
            default_language = language_service.get_default_language()
            return {
                "languageCode": default_language,
                "preferenceSource": "default",
                "selectedAt": None,
                "isActive": True
            }
        
        return preference
    except Exception as e:
        logger.error(f"Error getting user language preference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/user/language-preference")
async def set_user_language_preference(
    preference_request: LanguagePreferenceRequest,
    request: Request,
    session_id: str = Depends(get_session_id),
    user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Set user's language preference."""
    try:
        language_service = LanguageService(db)
        
        # Convert string to enum
        source = PreferenceSource.USER_SELECTION
        if preference_request.source:
            source = PreferenceSource(preference_request.source)
        
        preference = language_service.set_user_language_preference(
            preference_request.language_code,
            user_id,
            session_id,
            source
        )
        
        # Start translation session for analytics
        language_service.start_translation_session(session_id, preference_request.language_code)
        
        return preference
    except ValueError as e:
        logger.warning(f"Invalid language preference request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting user language preference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/translations/{language_code}")
async def get_translations(
    language_code: str,
    namespace: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get translations for a specific language."""
    try:
        language_service = LanguageService(db)
        translations = language_service.get_translations(language_code, namespace)
        
        return {
            "languageCode": language_code,
            "namespace": namespace,
            "translations": translations
        }
    except Exception as e:
        logger.error(f"Error getting translations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai-translate")
async def translate_content(
    translation_request: TranslationRequest,
    request: Request,
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    """Translate AI-generated content."""
    try:
        translation_service = TranslationService(db)
        language_service = LanguageService(db)
        
        # Update session activity for analytics
        language_service.update_session_activity(session_id, ai_translation=True)
        
        result = translation_service.translate_content(
            translation_request.content,
            translation_request.target_language,
            translation_request.source_language,
            translation_request.context,
            translation_request.use_cache
        )
        
        # Add translation time for response
        result["translationTime"] = 45  # Mock value - replace with actual timing
        
        return result
    except Exception as e:
        logger.error(f"Error translating content: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/translation-feedback")
async def submit_translation_feedback(
    feedback_request: TranslationFeedbackRequest,
    request: Request,
    session_id: str = Depends(get_session_id),
    user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Submit feedback on translation quality."""
    try:
        translation_service = TranslationService(db)
        
        # Convert strings to enums
        translation_type = TranslationType(feedback_request.translation_type)
        feedback_type = FeedbackType(feedback_request.feedback_type)
        
        feedback = translation_service.submit_translation_feedback(
            feedback_request.translation_id,
            translation_type,
            feedback_type,
            user_id,
            session_id,
            feedback_request.suggested_improvement
        )
        
        return feedback
    except ValueError as e:
        logger.warning(f"Invalid feedback request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting translation feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/translation-session/activity")
async def update_session_activity(
    request: Request,
    session_id: str = Depends(get_session_id),
    page_view: bool = False,
    ai_translation: bool = False,
    db: Session = Depends(get_db)
):
    """Update session activity for analytics."""
    try:
        language_service = LanguageService(db)
        language_service.update_session_activity(session_id, page_view, ai_translation)
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error updating session activity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Middleware-like function to detect language from request
def detect_language_from_request(request: Request) -> str:
    """Detect preferred language from request headers."""
    # Check Accept-Language header
    accept_language = request.headers.get("accept-language", "")
    
    # Simple parsing of Accept-Language header
    if accept_language:
        languages = []
        for lang_part in accept_language.split(","):
            lang = lang_part.strip().split(";")[0].strip()
            if "-" in lang:
                lang = lang.split("-")[0]  # Take just language code, ignore region
            languages.append(lang.lower())
        
        # Check if any of the requested languages are supported
        supported_languages = ["en", "es", "fr", "hi", "zh"]
        for lang in languages:
            if lang in supported_languages:
                return lang
    
    # Default to English
    return "en"


# Helper function to get user's current language for responses
async def get_current_language(
    request: Request,
    session_id: str = Depends(get_session_id),
    user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> str:
    """Get user's current language for API responses."""
    try:
        language_service = LanguageService(db)
        preference = language_service.get_user_language_preference(user_id, session_id)
        
        if preference:
            return preference["languageCode"]
        
        # Try to detect from request headers
        detected = detect_language_from_request(request)
        return detected
    except Exception:
        # Fallback to default
        return "en"
