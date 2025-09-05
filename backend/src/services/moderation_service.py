"""
Simple moderation service for chat content filtering.
"""
import re
from typing import List, Dict, Tuple

class ModerationService:
    def __init__(self):
        """Initialize moderation service with basic filtering rules."""
        # Basic inappropriate words list (expand as needed)
        self.inappropriate_words = {
            "spam", "scam", "fraud", "fake", "cheat", "hack", "virus",
            # Add more words as needed for agricultural context
        }
        
        # Farming-related spam patterns
        self.spam_patterns = [
            r"buy now",
            r"click here",
            r"limited time",
            r"guaranteed results",
            r"miracle cure",
            r"instant solution"
        ]
        
        # Patterns for URLs (basic detection)
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
    def check_content(self, content: str, user_id: str = None) -> Dict:
        """
        Check content for moderation issues.
        
        Returns:
            Dict with moderation result:
            {
                "approved": bool,
                "issues": List[str],
                "confidence": float,
                "action": str  # "approve", "flag", "hide"
            }
        """
        content_lower = content.lower().strip()
        issues = []
        confidence = 1.0
        
        # Check for empty content
        if not content_lower:
            return {
                "approved": False,
                "issues": ["Empty content"],
                "confidence": 1.0,
                "action": "hide"
            }
        
        # Check for inappropriate words
        for word in self.inappropriate_words:
            if word in content_lower:
                issues.append(f"Inappropriate word: {word}")
                confidence -= 0.3
        
        # Check for spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"Spam pattern detected: {pattern}")
                confidence -= 0.4
        
        # Check for excessive URLs
        urls = re.findall(self.url_pattern, content)
        if len(urls) > 2:
            issues.append("Too many URLs")
            confidence -= 0.3
        elif len(urls) > 0:
            issues.append("Contains URL")
            confidence -= 0.1
        
        # Check for excessive capitalization
        if len(content) > 10:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.7:
                issues.append("Excessive capitalization")
                confidence -= 0.2
        
        # Check for repetitive content
        if self._is_repetitive(content):
            issues.append("Repetitive content")
            confidence -= 0.3
        
        # Check message length
        if len(content) > 2000:
            issues.append("Message too long")
            confidence -= 0.2
        
        # Determine action based on confidence and issues
        if confidence >= 0.8 and not issues:
            action = "approve"
            approved = True
        elif confidence >= 0.5:
            action = "flag"  # Flag for human review
            approved = False
        else:
            action = "hide"  # Automatically hide
            approved = False
        
        return {
            "approved": approved,
            "issues": issues,
            "confidence": max(0.0, confidence),
            "action": action
        }
    
    def _is_repetitive(self, content: str) -> bool:
        """Check if content is repetitive."""
        words = content.lower().split()
        if len(words) < 3:
            return False
        
        # Check for repeated words
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 3 times in a short message, flag as repetitive
        max_count = max(word_counts.values())
        if max_count > 3 and len(words) < 20:
            return True
        
        # Check for repeated characters
        for i in range(len(content) - 3):
            char_sequence = content[i:i+4]
            if char_sequence == char_sequence[0] * 4:  # Same character repeated 4+ times
                return True
        
        return False
    
    def is_farming_related(self, content: str) -> bool:
        """Check if content is related to farming/agriculture."""
        farming_keywords = {
            "potato", "crop", "plant", "disease", "blight", "fungus", "pest",
            "fertilizer", "soil", "harvest", "seed", "farming", "agriculture",
            "irrigation", "pesticide", "yield", "growth", "leaf", "root",
            "field", "farm", "farmer", "cultivation", "organic", "treatment"
        }
        
        content_lower = content.lower()
        for keyword in farming_keywords:
            if keyword in content_lower:
                return True
        
        return False
    
    def suggest_improvements(self, content: str) -> List[str]:
        """Suggest improvements for flagged content."""
        suggestions = []
        
        moderation_result = self.check_content(content)
        
        if "Inappropriate word" in str(moderation_result["issues"]):
            suggestions.append("Please use appropriate language for professional farming discussions")
        
        if "Spam pattern detected" in str(moderation_result["issues"]):
            suggestions.append("Focus on sharing farming knowledge rather than promotional content")
        
        if "Contains URL" in str(moderation_result["issues"]):
            suggestions.append("Consider describing the resource instead of posting links")
        
        if "Excessive capitalization" in str(moderation_result["issues"]):
            suggestions.append("Please avoid using excessive capital letters")
        
        if "Repetitive content" in str(moderation_result["issues"]):
            suggestions.append("Try to vary your language and avoid repetition")
        
        if "Message too long" in str(moderation_result["issues"]):
            suggestions.append("Consider breaking your message into smaller parts")
        
        if not self.is_farming_related(content):
            suggestions.append("Keep discussions focused on farming and agriculture topics")
        
        return suggestions

# Global moderation service instance
moderation_service = ModerationService()
