/**
 * Enhanced Language support utilities for the frontend
 * Integrates with Google Translate and provides comprehensive language management
 */

class LanguageManager {
    constructor() {
        this.apiBase = '/api';
        this.currentLanguage = 'en';
        this.translations = {};
        this.sessionId = this.getOrCreateSessionId();
        this.googleTranslateInstance = null;
        this.config = {
            cookieName: 'googtrans',
            cookieDays: 30,
            storageKey: 'preferredLanguage',
            debug: false
        };
        this.init();
    }

    /**
     * Initialize language manager with Google Translate integration
     */
    async init() {
        try {
            // Initialize Google Translate integration
            await this.initGoogleTranslate();
            
            // Load available languages
            await this.loadAvailableLanguages();
            
            // Get user's current preference from multiple sources
            await this.loadUserPreference();
            
            // Load translations for current language
            await this.loadTranslations(this.currentLanguage);
            
            // Update UI
            this.updateUI();
            
            // Setup language persistence monitoring
            this.setupLanguagePersistence();
            
        } catch (error) {
            console.error('Error initializing language manager:', error);
        }
    }

    /**
     * Initialize Google Translate integration
     */
    async initGoogleTranslate() {
        if (window.persistentGoogleTranslate) {
            this.googleTranslateInstance = window.persistentGoogleTranslate;
            
            // Listen for Google Translate language changes
            window.addEventListener('languageChanged', (event) => {
                this.handleGoogleTranslateChange(event.detail.language);
            });
        }
    }

    /**
     * Handle Google Translate language changes
     */
    async handleGoogleTranslateChange(language) {
        if (language !== this.currentLanguage) {
            await this.changeLanguage(language);
        }
    }

    /**
     * Setup language persistence monitoring
     */
    setupLanguagePersistence() {
        // Monitor for manual language changes
        setInterval(() => {
            const savedLanguage = this.getSavedLanguageFromStorage();
            if (savedLanguage && savedLanguage !== this.currentLanguage) {
                this.changeLanguage(savedLanguage);
            }
        }, 1000);
    }

    /**
     * Get saved language from storage (priority: sessionStorage > localStorage > cookie)
     */
    getSavedLanguageFromStorage() {
        try {
            // Check sessionStorage first
            let language = sessionStorage.getItem('currentLanguage');
            if (language) return language;
            
            // Check localStorage
            language = localStorage.getItem(this.config.storageKey);
            if (language) return language;
            
            // Check cookie
            language = this.getCookie(this.config.cookieName);
            if (language) return language;
            
        } catch (error) {
            console.error('Error getting saved language:', error);
        }
        
        return null;
    }

    /**
     * Save language preference to all storage methods
     */
    saveLanguagePreference(language) {
        try {
            sessionStorage.setItem('currentLanguage', language);
            localStorage.setItem(this.config.storageKey, language);
            this.setCookie(this.config.cookieName, language, this.config.cookieDays);
            
            // Also save to backend session
            this.persistLanguageToSession(language);
            
        } catch (error) {
            console.error('Error saving language preference:', error);
        }
    }

    /**
     * Persist language preference to backend session
     */
    async persistLanguageToSession(language) {
        try {
            const response = await fetch(`${this.apiBase}/language/session/persist`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'session-id': this.sessionId
                },
                body: JSON.stringify({
                    language: language,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                console.warn('Failed to persist language to backend session');
            }
        } catch (error) {
            console.error('Error persisting language to session:', error);
        }
    }

    /**
     * Get or create session ID
     */
    getOrCreateSessionId() {
        let sessionId = localStorage.getItem('session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('session_id', sessionId);
        }
        return sessionId;
    }

    /**
     * Load available languages from API
     */
    async loadAvailableLanguages() {
        try {
            const response = await fetch(`${this.apiBase}/languages`);
            const data = await response.json();
            this.availableLanguages = data.languages;
            this.defaultLanguage = data.defaultLanguage;
        } catch (error) {
            console.error('Error loading available languages:', error);
            // Fallback to basic languages
            this.availableLanguages = [
                { code: 'en', name: 'English', flagIcon: '🇺🇸' },
                { code: 'es', name: 'Español', flagIcon: '🇪🇸' },
                { code: 'fr', name: 'Français', flagIcon: '🇫🇷' },
                { code: 'hi', name: 'हिन्दी', flagIcon: '🇮🇳' },
                { code: 'zh', name: '中文', flagIcon: '🇨🇳' }
            ];
            this.defaultLanguage = 'en';
        }
    }

    /**
     * Load user's language preference
     */
    async loadUserPreference() {
        try {
            const response = await fetch(`${this.apiBase}/user/language-preference`, {
                headers: {
                    'session-id': this.sessionId
                }
            });
            const preference = await response.json();
            this.currentLanguage = preference.languageCode;
        } catch (error) {
            console.error('Error loading user preference:', error);
            this.currentLanguage = this.defaultLanguage || 'en';
        }
    }

    /**
     * Set user's language preference
     */
    async setLanguagePreference(languageCode) {
        try {
            const response = await fetch(`${this.apiBase}/user/language-preference`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'session-id': this.sessionId
                },
                body: JSON.stringify({
                    language_code: languageCode,
                    source: 'user_selection'
                })
            });

            if (response.ok) {
                this.currentLanguage = languageCode;
                await this.loadTranslations(languageCode);
                this.updateUI();
                
                // Store in localStorage for immediate access
                localStorage.setItem('preferred_language', languageCode);
                
                // Trigger custom event for other components
                window.dispatchEvent(new CustomEvent('languageChanged', {
                    detail: { language: languageCode }
                }));
                
                return true;
            } else {
                throw new Error('Failed to set language preference');
            }
        } catch (error) {
            console.error('Error setting language preference:', error);
            return false;
        }
    }

    /**
     * Load translations for a specific language
     */
    async loadTranslations(languageCode) {
        try {
            const response = await fetch(`${this.apiBase}/translations/${languageCode}`);
            const data = await response.json();
            this.translations[languageCode] = data.translations;
        } catch (error) {
            console.error(`Error loading translations for ${languageCode}:`, error);
            this.translations[languageCode] = {};
        }
    }

    /**
     * Get translation for a key
     */
    t(key, fallback = null) {
        const translation = this.translations[this.currentLanguage] && 
                          this.translations[this.currentLanguage][key];
        
        if (translation) {
            return translation;
        }
        
        // Fallback to English if available
        if (this.currentLanguage !== 'en' && this.translations['en'] && this.translations['en'][key]) {
            return this.translations['en'][key];
        }
        
        // Final fallback
        return fallback || key;
    }

    /**
     * Translate AI-generated content
     */
    async translateContent(content, context = 'general') {
        try {
            if (this.currentLanguage === 'en') {
                return content; // No translation needed
            }

            const response = await fetch(`${this.apiBase}/ai-translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'session-id': this.sessionId
                },
                body: JSON.stringify({
                    content: content,
                    target_language: this.currentLanguage,
                    source_language: 'en',
                    context: context,
                    use_cache: true
                })
            });

            if (response.ok) {
                const data = await response.json();
                return data.translatedContent;
            } else {
                throw new Error('Translation failed');
            }
        } catch (error) {
            console.error('Error translating content:', error);
            return content; // Return original content on error
        }
    }

    /**
     * Update UI elements with current language
     */
    updateUI() {
        // Update all elements with data-translate attribute
        const translatableElements = document.querySelectorAll('[data-translate]');
        translatableElements.forEach(element => {
            const key = element.getAttribute('data-translate');
            const translation = this.t(key);
            
            if (element.tagName.toLowerCase() === 'input' && element.type === 'submit') {
                element.value = translation;
            } else if (element.hasAttribute('placeholder')) {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });

        // Update language selector if present
        this.updateLanguageSelector();

        // Update document language attribute
        document.documentElement.lang = this.currentLanguage;
    }

    /**
     * Create language selector HTML
     */
    createLanguageSelector() {
        const currentLang = this.availableLanguages.find(lang => lang.code === this.currentLanguage);
        
        return `
            <div class="language-selector">
                <button class="language-button" onclick="languageManager.toggleLanguageDropdown()">
                    <span class="flag">${currentLang ? currentLang.flagIcon : '🌐'}</span>
                    <span class="language-name">${currentLang ? currentLang.name : 'Language'}</span>
                    <span class="dropdown-arrow">▼</span>
                </button>
                <div class="language-dropdown" id="languageDropdown" style="display: none;">
                    ${this.availableLanguages.map(lang => `
                        <div class="language-option ${lang.code === this.currentLanguage ? 'active' : ''}" 
                             onclick="languageManager.selectLanguage('${lang.code}')">
                            <span class="flag">${lang.flagIcon}</span>
                            <span class="language-name">${lang.name}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Update existing language selector
     */
    updateLanguageSelector() {
        const selector = document.getElementById('languageSelector');
        if (selector) {
            selector.innerHTML = this.createLanguageSelector();
        }
    }

    /**
     * Toggle language dropdown
     */
    toggleLanguageDropdown() {
        const dropdown = document.getElementById('languageDropdown');
        if (dropdown) {
            dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
        }
    }

    /**
     * Select a language
     */
    async selectLanguage(languageCode) {
        if (languageCode !== this.currentLanguage) {
            const success = await this.setLanguagePreference(languageCode);
            if (success) {
                // Hide dropdown
                this.toggleLanguageDropdown();
            }
        } else {
            // Just hide dropdown if same language selected
            this.toggleLanguageDropdown();
        }
    }

    /**
     * Track page view for analytics
     */
    trackPageView() {
        fetch(`${this.apiBase}/translation-session/activity`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'session-id': this.sessionId
            },
            body: JSON.stringify({
                page_view: true
            })
        }).catch(error => {
            console.error('Error tracking page view:', error);
        });
    }

    /**
     * Cookie utilities for language persistence
     */
    setCookie(name, value, days) {
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        const expires = `expires=${date.toUTCString()}`;
        document.cookie = `${name}=${value};${expires};path=/;SameSite=Lax`;
    }

    getCookie(name) {
        const nameEQ = `${name}=`;
        const cookies = document.cookie.split(';');
        
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.indexOf(nameEQ) === 0) {
                return cookie.substring(nameEQ.length);
            }
        }
        return null;
    }

    /**
     * Clear all language preferences
     */
    clearLanguagePreferences() {
        try {
            // Clear all storage
            sessionStorage.removeItem('currentLanguage');
            localStorage.removeItem(this.config.storageKey);
            this.setCookie(this.config.cookieName, '', -1);
            
            // Reset to default language
            this.currentLanguage = 'en';
            document.documentElement.lang = 'en';
            
            // Clear Google Translate if available
            if (this.googleTranslateInstance) {
                this.googleTranslateInstance.clearLanguagePreference();
            }
            
            console.log('All language preferences cleared');
            return true;
            
        } catch (error) {
            console.error('Error clearing language preferences:', error);
            return false;
        }
    }

    /**
     * Get current language with fallback logic
     */
    getCurrentLanguage() {
        return this.currentLanguage || this.getSavedLanguageFromStorage() || 'en';
    }

    /**
     * Manual language change with full integration
     */
    async changeLanguage(language) {
        if (language === this.currentLanguage) {
            return; // No change needed
        }

        try {
            // Update internal state
            this.currentLanguage = language;
            
            // Save to all storage methods
            this.saveLanguagePreference(language);
            
            // Update Google Translate if available
            if (this.googleTranslateInstance) {
                this.googleTranslateInstance.changeLanguage(language);
            }
            
            // Load new translations
            await this.loadTranslations(language);
            
            // Update UI
            this.updateUI();
            
            // Update document language attribute
            document.documentElement.lang = language.split('-')[0];
            
            // Dispatch custom event
            this.dispatchLanguageChangeEvent(language);
            
            console.log(`Language changed to: ${language}`);
            
        } catch (error) {
            console.error('Error changing language:', error);
        }
    }

    /**
     * Dispatch language change event
     */
    dispatchLanguageChangeEvent(language) {
        const event = new CustomEvent('appLanguageChanged', {
            detail: {
                language: language,
                timestamp: new Date().toISOString(),
                source: 'LanguageManager'
            }
        });
        
        window.dispatchEvent(event);
    }
}

// Initialize language manager when DOM is loaded
let languageManager;
document.addEventListener('DOMContentLoaded', function() {
    languageManager = new LanguageManager();
    
    // Track page view
    setTimeout(() => {
        if (languageManager) {
            languageManager.trackPageView();
        }
    }, 1000);
});

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const dropdown = document.getElementById('languageDropdown');
    const selector = document.querySelector('.language-selector');
    
    if (dropdown && selector && !selector.contains(event.target)) {
        dropdown.style.display = 'none';
    }
});

// CSS styles for language selector
const languageSelectorStyles = `
<style>
.language-selector {
    position: relative;
    display: inline-block;
    font-family: inherit;
}

.language-button {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: border-color 0.2s;
}

.language-button:hover {
    border-color: #4CAF50;
}

.language-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    z-index: 1000;
    margin-top: 4px;
}

.language-option {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    cursor: pointer;
    transition: background-color 0.2s;
    border-bottom: 1px solid #f0f0f0;
}

.language-option:last-child {
    border-bottom: none;
}

.language-option:hover {
    background-color: #f5f5f5;
}

.language-option.active {
    background-color: #e8f5e8;
    color: #2e7d32;
}

.flag {
    font-size: 16px;
}

.language-name {
    font-size: 14px;
}

.dropdown-arrow {
    font-size: 10px;
    color: #666;
}

/* RTL support for future languages */
[dir="rtl"] .language-selector {
    direction: rtl;
}

/* Responsive design */
@media (max-width: 768px) {
    .language-button {
        padding: 6px 10px;
        font-size: 13px;
    }
    
    .language-dropdown {
        min-width: 150px;
    }
}
</style>
`;

// Inject styles into document head
document.head.insertAdjacentHTML('beforeend', languageSelectorStyles);
