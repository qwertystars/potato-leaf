/**
 * Enhanced Google Translate Manager with Persistent Language Selection
 * Supports both cookies and localStorage for maximum compatibility
 */

class PersistentGoogleTranslate {
    constructor(options = {}) {
        this.options = {
            pageLanguage: options.pageLanguage || 'en',
            autoDisplay: options.autoDisplay || false,
            includedLanguages: options.includedLanguages || null,
            cookieName: options.cookieName || 'googtrans',
            cookieDays: options.cookieDays || 30,
            storageKey: options.storageKey || 'preferredLanguage',
            debug: options.debug || false,
            ...options
        };
        
        this.isInitialized = false;
        this.translateElement = null;
        this.currentLanguage = null;
        
        this.init();
    }

    /**
     * Initialize the Google Translate functionality
     */
    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    /**
     * Setup Google Translate widget and persistence
     */
    setup() {
        // Create the translate element if it doesn't exist
        this.createTranslateElement();
        
        // Initialize Google Translate
        window.googleTranslateElementInit = () => this.initializeTranslateWidget();
        
        // Load Google Translate script if not already loaded
        this.loadGoogleTranslateScript();
        
        // Setup language persistence
        this.setupLanguagePersistence();
    }

    /**
     * Create the Google Translate element container
     */
    createTranslateElement() {
        let translateElement = document.getElementById('google_translate_element');
        
        if (!translateElement) {
            translateElement = document.createElement('div');
            translateElement.id = 'google_translate_element';
            translateElement.style.display = 'none'; // Hidden by default
            
            // Insert at the beginning of body
            if (document.body.firstChild) {
                document.body.insertBefore(translateElement, document.body.firstChild);
            } else {
                document.body.appendChild(translateElement);
            }
        }
        
        return translateElement;
    }

    /**
     * Initialize the Google Translate widget
     */
    initializeTranslateWidget() {
        try {
            const config = {
                pageLanguage: this.options.pageLanguage,
                autoDisplay: this.options.autoDisplay
            };

            // Add included languages if specified
            if (this.options.includedLanguages) {
                config.includedLanguages = this.options.includedLanguages;
            }

            this.translateElement = new google.translate.TranslateElement(
                config,
                'google_translate_element'
            );

            this.isInitialized = true;
            this.log('Google Translate initialized successfully');
            
            // Apply saved language after initialization
            setTimeout(() => this.applySavedLanguage(), 1000);
            
        } catch (error) {
            console.error('Failed to initialize Google Translate:', error);
        }
    }

    /**
     * Load Google Translate script
     */
    loadGoogleTranslateScript() {
        // Check if script is already loaded
        if (document.querySelector('script[src*="translate.google.com"]')) {
            return;
        }

        const script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
        script.async = true;
        script.defer = true;
        
        script.onerror = () => {
            console.error('Failed to load Google Translate script');
        };
        
        document.head.appendChild(script);
    }

    /**
     * Setup language persistence monitoring
     */
    setupLanguagePersistence() {
        // Monitor for translate select element
        const checkForSelect = () => {
            const select = document.querySelector('select.goog-te-combo');
            
            if (select && !select.hasAttribute('data-persistence-setup')) {
                this.setupSelectMonitoring(select);
                select.setAttribute('data-persistence-setup', 'true');
                this.log('Language select monitoring setup complete');
            } else if (!select) {
                // Keep checking until select is available
                setTimeout(checkForSelect, 500);
            }
        };

        setTimeout(checkForSelect, 1000);
    }

    /**
     * Setup monitoring for the translate select element
     */
    setupSelectMonitoring(select) {
        select.addEventListener('change', (event) => {
            const selectedLanguage = event.target.value;
            this.saveLanguagePreference(selectedLanguage);
            this.currentLanguage = selectedLanguage;
            this.log(`Language changed to: ${selectedLanguage}`);
            
            // Dispatch custom event for other scripts to listen
            this.dispatchLanguageChangeEvent(selectedLanguage);
        });
    }

    /**
     * Save language preference to both cookie and localStorage
     */
    saveLanguagePreference(language) {
        try {
            // Save to cookie
            this.setCookie(this.options.cookieName, language, this.options.cookieDays);
            
            // Save to localStorage
            localStorage.setItem(this.options.storageKey, language);
            
            // Save to sessionStorage for current session
            sessionStorage.setItem('currentLanguage', language);
            
            this.log(`Language preference saved: ${language}`);
        } catch (error) {
            console.error('Failed to save language preference:', error);
        }
    }

    /**
     * Apply saved language preference
     */
    applySavedLanguage() {
        const savedLanguage = this.getSavedLanguage();
        
        if (savedLanguage && savedLanguage !== this.options.pageLanguage) {
            this.applyLanguage(savedLanguage);
        }
    }

    /**
     * Get saved language preference (priority: sessionStorage > localStorage > cookie)
     */
    getSavedLanguage() {
        try {
            // Check sessionStorage first (current session)
            let language = sessionStorage.getItem('currentLanguage');
            if (language) return language;
            
            // Check localStorage (persistent across sessions)
            language = localStorage.getItem(this.options.storageKey);
            if (language) return language;
            
            // Check cookie (fallback)
            language = this.getCookie(this.options.cookieName);
            if (language) return language;
            
        } catch (error) {
            console.error('Failed to get saved language:', error);
        }
        
        return null;
    }

    /**
     * Apply a specific language
     */
    applyLanguage(language) {
        const select = document.querySelector('select.goog-te-combo');
        
        if (select) {
            select.value = language;
            select.dispatchEvent(new Event('change'));
            this.currentLanguage = language;
            this.log(`Applied language: ${language}`);
        } else {
            this.log(`Cannot apply language ${language}: select element not found`);
            // Retry after a delay
            setTimeout(() => this.applyLanguage(language), 500);
        }
    }

    /**
     * Manually change language
     */
    changeLanguage(language) {
        this.saveLanguagePreference(language);
        this.applyLanguage(language);
    }

    /**
     * Get current language
     */
    getCurrentLanguage() {
        return this.currentLanguage || this.getSavedLanguage() || this.options.pageLanguage;
    }

    /**
     * Clear saved language preferences
     */
    clearLanguagePreference() {
        try {
            // Clear cookie
            this.setCookie(this.options.cookieName, '', -1);
            
            // Clear localStorage
            localStorage.removeItem(this.options.storageKey);
            
            // Clear sessionStorage
            sessionStorage.removeItem('currentLanguage');
            
            // Reset to default language
            this.applyLanguage(this.options.pageLanguage);
            
            this.log('Language preferences cleared');
        } catch (error) {
            console.error('Failed to clear language preferences:', error);
        }
    }

    /**
     * Dispatch custom language change event
     */
    dispatchLanguageChangeEvent(language) {
        const event = new CustomEvent('languageChanged', {
            detail: {
                language: language,
                previousLanguage: this.currentLanguage,
                timestamp: new Date().toISOString()
            }
        });
        
        window.dispatchEvent(event);
    }

    /**
     * Set cookie utility
     */
    setCookie(name, value, days) {
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        const expires = `expires=${date.toUTCString()}`;
        document.cookie = `${name}=${value};${expires};path=/;SameSite=Lax`;
    }

    /**
     * Get cookie utility
     */
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
     * Debug logging
     */
    log(message) {
        if (this.options.debug) {
            console.log(`[PersistentGoogleTranslate] ${message}`);
        }
    }
}

// Auto-initialize if not manually configured
window.addEventListener('DOMContentLoaded', () => {
    if (!window.persistentGoogleTranslate) {
        window.persistentGoogleTranslate = new PersistentGoogleTranslate({
            debug: false // Set to true for debugging
        });
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PersistentGoogleTranslate;
}
