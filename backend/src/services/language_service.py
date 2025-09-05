"""
Language service for multi-language support functionality.
"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from models.language import (
    Language, UserLanguagePreference, Translation, TranslationSession,
    AITranslationCache, TranslationFeedback, PreferenceSource,
    TranslationType, FeedbackType, TranslationService as TranslationServiceEnum, ReviewStatus
)

logger = logging.getLogger(__name__)


class LanguageService:
    """Service for managing language preferences and translations."""

    def __init__(self, db: Session):
        self.db = db

    def get_available_languages(self) -> List[Dict[str, Any]]:
        """Get all enabled languages."""
        languages = self.db.query(Language).filter(Language.enabled == True).all()
        return [lang.to_dict() for lang in languages]

    def get_default_language(self) -> str:
        """Get the default language code."""
        return "en"  # English as default

    def get_user_language_preference(self, user_id: Optional[str] = None, 
                                   session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get user's current language preference."""
        query = self.db.query(UserLanguagePreference).filter(
            UserLanguagePreference.is_active == True
        )
        
        if user_id:
            query = query.filter(UserLanguagePreference.user_id == user_id)
        elif session_id:
            query = query.filter(UserLanguagePreference.session_id == session_id)
        else:
            return None
            
        preference = query.first()
        return preference.to_dict() if preference else None

    def set_user_language_preference(self, language_code: str, user_id: Optional[str] = None,
                                   session_id: Optional[str] = None,
                                   source: PreferenceSource = PreferenceSource.USER_SELECTION) -> Dict[str, Any]:
        """Set user's language preference."""
        # Validate language exists and is enabled
        language = self.db.query(Language).filter(
            and_(Language.code == language_code, Language.enabled == True)
        ).first()
        
        if not language:
            raise ValueError(f"Language '{language_code}' is not available")

        # Deactivate existing preferences
        if user_id:
            self.db.query(UserLanguagePreference).filter(
                and_(UserLanguagePreference.user_id == user_id,
                     UserLanguagePreference.is_active == True)
            ).update({"is_active": False})
        elif session_id:
            self.db.query(UserLanguagePreference).filter(
                and_(UserLanguagePreference.session_id == session_id,
                     UserLanguagePreference.is_active == True)
            ).update({"is_active": False})

        # Create new preference
        preference = UserLanguagePreference(
            user_id=user_id,
            session_id=session_id,
            language_code=language_code,
            preference_source=source,
            is_active=True
        )
        
        self.db.add(preference)
        self.db.commit()
        
        logger.info(f"Set language preference to {language_code} for user_id={user_id}, session_id={session_id}")
        
        return preference.to_dict()

    def get_translations(self, language_code: str, namespace: Optional[str] = None) -> Dict[str, str]:
        """Get translations for a specific language and optionally namespace."""
        query = self.db.query(Translation).filter(Translation.language_code == language_code)
        
        if namespace:
            query = query.filter(Translation.namespace == namespace)
            
        translations = query.all()
        
        return {trans.key: trans.value for trans in translations}

    def get_translation(self, key: str, language_code: str, fallback: str = None) -> str:
        """Get a specific translation with fallback to English."""
        translation = self.db.query(Translation).filter(
            and_(Translation.key == key, Translation.language_code == language_code)
        ).first()
        
        if translation:
            return translation.value
            
        # Fallback to English
        if language_code != "en":
            english_translation = self.db.query(Translation).filter(
                and_(Translation.key == key, Translation.language_code == "en")
            ).first()
            
            if english_translation:
                return english_translation.value
        
        # Final fallback
        return fallback or key

    def start_translation_session(self, session_id: str, language_code: str) -> Dict[str, Any]:
        """Start a new translation session for analytics."""
        # End any existing active session for this session_id
        existing_session = self.db.query(TranslationSession).filter(
            and_(TranslationSession.session_id == session_id,
                 TranslationSession.ended_at.is_(None))
        ).first()
        
        if existing_session:
            existing_session.ended_at = datetime.utcnow()
            
        # Create new session
        session = TranslationSession(
            session_id=session_id,
            language_code=language_code
        )
        
        self.db.add(session)
        self.db.commit()
        
        logger.info(f"Started translation session for {session_id} in {language_code}")
        
        return session.to_dict()

    def update_session_activity(self, session_id: str, page_view: bool = False, ai_translation: bool = False):
        """Update session activity metrics."""
        session = self.db.query(TranslationSession).filter(
            and_(TranslationSession.session_id == session_id,
                 TranslationSession.ended_at.is_(None))
        ).first()
        
        if session:
            session.last_activity_at = datetime.utcnow()
            if page_view:
                session.page_views += 1
            if ai_translation:
                session.ai_translations_requested += 1
            self.db.commit()


class TranslationService:
    """Service for managing AI translations and caching."""

    def __init__(self, db: Session):
        self.db = db

    def _generate_content_hash(self, content: str, source_lang: str, target_lang: str) -> str:
        """Generate hash for content caching."""
        combined = f"{content}|{source_lang}|{target_lang}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_cached_translation(self, content: str, source_language: str, target_language: str) -> Optional[Dict[str, Any]]:
        """Get cached AI translation if available."""
        content_hash = self._generate_content_hash(content, source_language, target_language)
        
        cached = self.db.query(AITranslationCache).filter(
            and_(AITranslationCache.content_hash == content_hash,
                 AITranslationCache.source_language == source_language,
                 AITranslationCache.target_language == target_language,
                 AITranslationCache.expires_at > datetime.utcnow())
        ).first()
        
        if cached:
            # Update usage count
            cached.usage_count += 1
            self.db.commit()
            
            return cached.to_dict()
        
        return None

    def cache_translation(self, content: str, translated_content: str, source_language: str,
                         target_language: str, service: TranslationServiceEnum, confidence_score: float,
                         cache_duration_hours: int = 168) -> Dict[str, Any]:  # 1 week default
        """Cache AI translation result."""
        content_hash = self._generate_content_hash(content, source_language, target_language)
        expires_at = datetime.utcnow() + timedelta(hours=cache_duration_hours)
        
        cached = AITranslationCache(
            content_hash=content_hash,
            source_language=source_language,
            target_language=target_language,
            original_content=content,
            translated_content=translated_content,
            translation_service=service,
            confidence_score=confidence_score,
            expires_at=expires_at,
            usage_count=1
        )
        
        self.db.add(cached)
        self.db.commit()
        
        logger.info(f"Cached translation {content_hash[:8]}... for {source_language}->{target_language}")
        
        return cached.to_dict()

    def translate_content(self, content: str, target_language: str, source_language: str = "en",
                         context: str = "general", use_cache: bool = True) -> Dict[str, Any]:
        """Translate content using AI with caching."""
        # Check cache first if enabled
        if use_cache:
            cached = self.get_cached_translation(content, source_language, target_language)
            if cached:
                return cached

        # For now, return a mock translation (replace with actual AI service)
        # This is where you would integrate with OpenAI, Google Translate, etc.
        translated_content = self._mock_translate(content, source_language, target_language, context)
        confidence_score = 0.85  # Mock confidence score
        
        # Cache the result
        if use_cache:
            result = self.cache_translation(
                content, translated_content, source_language, target_language,
                TranslationServiceEnum.OPENAI, confidence_score
            )
        else:
            result = {
                "translatedContent": translated_content,
                "confidenceScore": confidence_score,
                "cached": False,
                "translationService": "openai"
            }
        
        return result

    def _mock_translate(self, content: str, source_lang: str, target_lang: str, context: str) -> str:
        """Mock translation function - replace with actual AI service."""
        # Simple mock translations for common phrases
        translations = {
            "es": {  # Spanish
                "Early blight disease detected": "Enfermedad de tizón temprano detectada",
                "Late blight disease detected": "Enfermedad de tizón tardío detectada",
                "Healthy potato leaf": "Hoja de papa saludable",
                "Upload Image": "Subir Imagen",
                "Analyze": "Analizar",
                "Results": "Resultados",
                "Disease Analysis": "Análisis de Enfermedad",
                "Treatment Recommendations": "Recomendaciones de Tratamiento"
            },
            "fr": {  # French
                "Early blight disease detected": "Maladie de la brûlure précoce détectée",
                "Late blight disease detected": "Maladie de la brûlure tardive détectée",
                "Healthy potato leaf": "Feuille de pomme de terre saine",
                "Upload Image": "Télécharger l'image",
                "Analyze": "Analyser",
                "Results": "Résultats",
                "Disease Analysis": "Analyse des maladies",
                "Treatment Recommendations": "Recommandations de traitement"
            },
            "hi": {  # Hindi
                "Early blight disease detected": "प्रारंभिक अंगमारी रोग का पता चला",
                "Late blight disease detected": "देर से अंगमारी रोग का पता चला",
                "Healthy potato leaf": "स्वस्थ आलू का पत्ता",
                "Upload Image": "छवि अपलोड करें",
                "Analyze": "विश्लेषण करें",
                "Results": "परिणाम",
                "Disease Analysis": "रोग विश्लेषण",
                "Treatment Recommendations": "उपचार की सिफारिशें"
            },
            "zh": {  # Chinese (Simplified)
                "Early blight disease detected": "检测到早疫病",
                "Late blight disease detected": "检测到晚疫病",
                "Healthy potato leaf": "健康的马铃薯叶",
                "Upload Image": "上传图片",
                "Analyze": "分析",
                "Results": "结果",
                "Disease Analysis": "疾病分析",
                "Treatment Recommendations": "治疗建议"
            }
        }
        
        if target_lang in translations and content in translations[target_lang]:
            return translations[target_lang][content]
        
        # Fallback: return original content with language indicator
        return f"[{target_lang.upper()}] {content}"

    def submit_translation_feedback(self, translation_id: int, translation_type: TranslationType,
                                  feedback_type: FeedbackType, user_id: Optional[str] = None,
                                  session_id: Optional[str] = None, 
                                  suggested_improvement: Optional[str] = None) -> Dict[str, Any]:
        """Submit user feedback on translation quality."""
        feedback = TranslationFeedback(
            translation_id=translation_id,
            translation_type=translation_type,
            user_id=user_id,
            session_id=session_id,
            feedback_type=feedback_type,
            suggested_improvement=suggested_improvement
        )
        
        self.db.add(feedback)
        self.db.commit()
        
        logger.info(f"Received translation feedback: {feedback_type.value} for translation {translation_id}")
        
        return feedback.to_dict()


def seed_languages(db: Session):
    """Seed the database with initial supported languages."""
    languages_data = [
        {
            "code": "en",
            "name": "English",
            "english_name": "English",
            "rtl_direction": False,
            "enabled": True,
            "flag_icon": "🇺🇸"
        },
        {
            "code": "es",
            "name": "Español",
            "english_name": "Spanish",
            "rtl_direction": False,
            "enabled": True,
            "flag_icon": "🇪🇸"
        },
        {
            "code": "fr",
            "name": "Français",
            "english_name": "French",
            "rtl_direction": False,
            "enabled": True,
            "flag_icon": "🇫🇷"
        },
        {
            "code": "hi",
            "name": "हिन्दी",
            "english_name": "Hindi",
            "rtl_direction": False,
            "enabled": True,
            "flag_icon": "🇮🇳"
        },
        {
            "code": "zh",
            "name": "中文",
            "english_name": "Chinese (Simplified)",
            "rtl_direction": False,
            "enabled": True,
            "flag_icon": "🇨🇳"
        }
    ]
    
    for lang_data in languages_data:
        existing = db.query(Language).filter(Language.code == lang_data["code"]).first()
        if not existing:
            language = Language(**lang_data)
            db.add(language)
    
    db.commit()
    logger.info("Seeded languages successfully")


def seed_translations(db: Session):
    """Seed the database with initial UI translations."""
    translations_data = [
        # English (base language)
        {"key": "ui.title", "language_code": "en", "value": "Advanced Leaf Disease Analyzer", "namespace": "ui"},
        {"key": "ui.upload.title", "language_code": "en", "value": "Upload Leaf Image", "namespace": "ui"},
        {"key": "ui.upload.description", "language_code": "en", "value": "Select a potato leaf image for analysis", "namespace": "ui"},
        {"key": "ui.analyze.button", "language_code": "en", "value": "Analyze Leaf Disease", "namespace": "ui"},
        {"key": "ui.analyze.another", "language_code": "en", "value": "Analyze Another Leaf", "namespace": "ui"},
        {"key": "ui.results.title", "language_code": "en", "value": "Advanced Leaf Disease Analysis Result", "namespace": "ui"},
        {"key": "ui.language.selector", "language_code": "en", "value": "Select Language", "namespace": "ui"},
        {"key": "ui.model.selector", "language_code": "en", "value": "Select AI Model", "namespace": "ui"},
        {"key": "ui.disease.classification", "language_code": "en", "value": "Disease Classification", "namespace": "ui"},
        {"key": "disease.early_blight.name", "language_code": "en", "value": "Early Blight", "namespace": "disease"},
        {"key": "disease.late_blight.name", "language_code": "en", "value": "Late Blight", "namespace": "disease"},
        {"key": "disease.healthy.name", "language_code": "en", "value": "Healthy", "namespace": "disease"},
        
        # Spanish translations
        {"key": "ui.title", "language_code": "es", "value": "Analizador Avanzado de Enfermedades de Hojas", "namespace": "ui"},
        {"key": "ui.upload.title", "language_code": "es", "value": "Subir Imagen de Hoja", "namespace": "ui"},
        {"key": "ui.upload.description", "language_code": "es", "value": "Seleccione una imagen de hoja de papa para análisis", "namespace": "ui"},
        {"key": "ui.analyze.button", "language_code": "es", "value": "Analizar Enfermedad de Hoja", "namespace": "ui"},
        {"key": "ui.analyze.another", "language_code": "es", "value": "Analizar Otra Hoja", "namespace": "ui"},
        {"key": "ui.results.title", "language_code": "es", "value": "Resultado del Análisis Avanzado de Enfermedades", "namespace": "ui"},
        {"key": "ui.language.selector", "language_code": "es", "value": "Seleccionar Idioma", "namespace": "ui"},
        {"key": "ui.model.selector", "language_code": "es", "value": "Seleccionar Modelo de IA", "namespace": "ui"},
        {"key": "ui.disease.classification", "language_code": "es", "value": "Clasificación de Enfermedad", "namespace": "ui"},
        {"key": "disease.early_blight.name", "language_code": "es", "value": "Tizón Temprano", "namespace": "disease"},
        {"key": "disease.late_blight.name", "language_code": "es", "value": "Tizón Tardío", "namespace": "disease"},
        {"key": "disease.healthy.name", "language_code": "es", "value": "Saludable", "namespace": "disease"},
        
        # French translations
        {"key": "ui.title", "language_code": "fr", "value": "Analyseur Avancé de Maladies des Feuilles", "namespace": "ui"},
        {"key": "ui.upload.title", "language_code": "fr", "value": "Télécharger l'image de feuille", "namespace": "ui"},
        {"key": "ui.upload.description", "language_code": "fr", "value": "Sélectionnez une image de feuille de pomme de terre pour l'analyse", "namespace": "ui"},
        {"key": "ui.analyze.button", "language_code": "fr", "value": "Analyser la Maladie des Feuilles", "namespace": "ui"},
        {"key": "ui.analyze.another", "language_code": "fr", "value": "Analyser une Autre Feuille", "namespace": "ui"},
        {"key": "ui.results.title", "language_code": "fr", "value": "Résultat de l'Analyse Avancée des Maladies", "namespace": "ui"},
        {"key": "ui.language.selector", "language_code": "fr", "value": "Sélectionner la langue", "namespace": "ui"},
        {"key": "ui.model.selector", "language_code": "fr", "value": "Sélectionner le Modèle IA", "namespace": "ui"},
        {"key": "ui.disease.classification", "language_code": "fr", "value": "Classification des Maladies", "namespace": "ui"},
        {"key": "disease.early_blight.name", "language_code": "fr", "value": "Brûlure précoce", "namespace": "disease"},
        {"key": "disease.late_blight.name", "language_code": "fr", "value": "Brûlure tardive", "namespace": "disease"},
        {"key": "disease.healthy.name", "language_code": "fr", "value": "Sain", "namespace": "disease"},
        
        # Hindi translations
        {"key": "ui.title", "language_code": "hi", "value": "उन्नत पत्ती रोग विश्लेषक", "namespace": "ui"},
        {"key": "ui.upload.title", "language_code": "hi", "value": "पत्ती की छवि अपलोड करें", "namespace": "ui"},
        {"key": "ui.upload.description", "language_code": "hi", "value": "विश्लेषण के लिए आलू के पत्ते की छवि चुनें", "namespace": "ui"},
        {"key": "ui.analyze.button", "language_code": "hi", "value": "पत्ती रोग का विश्लेषण करें", "namespace": "ui"},
        {"key": "ui.analyze.another", "language_code": "hi", "value": "दूसरे पत्ते का विश्लेषण करें", "namespace": "ui"},
        {"key": "ui.results.title", "language_code": "hi", "value": "उन्नत पत्ती रोग विश्लेषण परिणाम", "namespace": "ui"},
        {"key": "ui.language.selector", "language_code": "hi", "value": "भाषा चुनें", "namespace": "ui"},
        {"key": "ui.model.selector", "language_code": "hi", "value": "AI मॉडल चुनें", "namespace": "ui"},
        {"key": "ui.disease.classification", "language_code": "hi", "value": "रोग वर्गीकरण", "namespace": "ui"},
        {"key": "disease.early_blight.name", "language_code": "hi", "value": "प्रारंभिक अंगमारी", "namespace": "disease"},
        {"key": "disease.late_blight.name", "language_code": "hi", "value": "देर से अंगमारी", "namespace": "disease"},
        {"key": "disease.healthy.name", "language_code": "hi", "value": "स्वस्थ", "namespace": "disease"},
        
        # Chinese translations
        {"key": "ui.title", "language_code": "zh", "value": "高级叶片疾病分析仪", "namespace": "ui"},
        {"key": "ui.upload.title", "language_code": "zh", "value": "上传叶片图片", "namespace": "ui"},
        {"key": "ui.upload.description", "language_code": "zh", "value": "选择马铃薯叶片图像进行分析", "namespace": "ui"},
        {"key": "ui.analyze.button", "language_code": "zh", "value": "分析叶片疾病", "namespace": "ui"},
        {"key": "ui.analyze.another", "language_code": "zh", "value": "分析另一片叶子", "namespace": "ui"},
        {"key": "ui.results.title", "language_code": "zh", "value": "高级叶片疾病分析结果", "namespace": "ui"},
        {"key": "ui.language.selector", "language_code": "zh", "value": "选择语言", "namespace": "ui"},
        {"key": "ui.model.selector", "language_code": "zh", "value": "选择AI模型", "namespace": "ui"},
        {"key": "ui.disease.classification", "language_code": "zh", "value": "疾病分类", "namespace": "ui"},
        {"key": "disease.early_blight.name", "language_code": "zh", "value": "早疫病", "namespace": "disease"},
        {"key": "disease.late_blight.name", "language_code": "zh", "value": "晚疫病", "namespace": "disease"},
        {"key": "disease.healthy.name", "language_code": "zh", "value": "健康", "namespace": "disease"},
    ]
    
    for trans_data in translations_data:
        existing = db.query(Translation).filter(
            and_(Translation.key == trans_data["key"],
                 Translation.language_code == trans_data["language_code"])
        ).first()
        
        if not existing:
            translation = Translation(**trans_data)
            db.add(translation)
    
    db.commit()
    logger.info("Seeded translations successfully")
