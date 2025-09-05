/**
 * Google Translate Integration for Potato Disease Analyzer
 * Provides automatic page translation using Google Translate Widget
 */

class GoogleTranslateManager {
    constructor() {
        this.isInitialized = false;
        this.currentLanguage = this.getSavedLanguage() || 'en';
        this.supportedLanguages = [
            { code: 'en', name: 'English' },
            { code: 'es', name: 'Español' },
            { code: 'fr', name: 'Français' },
            { code: 'de', name: 'Deutsch' },
            { code: 'it', name: 'Italiano' },
            { code: 'pt', name: 'Português' },
            { code: 'ru', name: 'Русский' },
            { code: 'zh', name: '中文' },
            { code: 'ja', name: '日本語' },
            { code: 'ko', name: '한국어' },
            { code: 'ar', name: 'العربية' },
            { code: 'hi', name: 'हिन्दी' },
            { code: 'bn', name: 'বাংলা' },
            { code: 'ur', name: 'اردو' },
            { code: 'ta', name: 'தமிழ்' },
            { code: 'te', name: 'తెలుగు' },
            { code: 'mr', name: 'मराठी' },
            { code: 'gu', name: 'ગુજરાતી' },
            { code: 'kn', name: 'ಕನ್ನಡ' },
            { code: 'ml', name: 'മലയാളം' }
        ];
        this.init();
    }

    init() {
        // Load Google Translate script if not already loaded
        if (!window.google || !window.google.translate) {
            this.loadGoogleTranslateScript();
        } else {
            this.initializeWidget();
        }

        // Don't create language selector - it's now integrated in the form
        // this.createLanguageSelector();
        
        // Handle dynamic content translation
        this.setupDynamicTranslation();
    }

    loadGoogleTranslateScript() {
        const script = document.createElement('script');
        script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
        script.async = true;
        document.head.appendChild(script);

        // Define global callback for Google Translate
        window.googleTranslateElementInit = () => {
            this.initializeWidget();
        };
    }

    initializeWidget() {
        if (this.isInitialized) return;

        try {
            new google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: this.supportedLanguages.map(lang => lang.code).join(','),
                layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
                autoDisplay: false,
                multilanguagePage: true
            }, 'google_translate_element');

            this.isInitialized = true;
            
            // Restore saved language after initialization
            setTimeout(() => {
                this.restoreSavedLanguage();
            }, 1000);

        } catch (error) {
            console.error('Failed to initialize Google Translate:', error);
        }
    }

    createLanguageSelector() {
        const selector = document.createElement('div');
        selector.className = 'language-selector';
        selector.innerHTML = `
            <select id="language-dropdown" class="language-dropdown">
                <option value="">Select Language</option>
                ${this.supportedLanguages.map(lang => 
                    `<option value="${lang.code}" ${lang.code === this.currentLanguage ? 'selected' : ''}>
                        ${lang.name}
                    </option>`
                ).join('')}
            </select>
        `;

        // Create the translate header
        let header = document.querySelector('.translate-header');
        if (!header) {
            header = document.createElement('div');
            header.className = 'translate-header';
            header.style.transform = 'translateY(-100%)';
            header.style.transition = 'transform 0.3s ease-in-out';
            document.body.insertBefore(header, document.body.firstChild);
            
            // Animate header in
            setTimeout(() => {
                header.style.transform = 'translateY(0)';
            }, 100);
        }
        
        header.appendChild(selector);

        // Add event listener for language selection
        document.getElementById('language-dropdown').addEventListener('change', (e) => {
            const selectedLang = e.target.value;
            if (selectedLang) {
                this.translateTo(selectedLang);
            }
        });
    }

    translateTo(languageCode) {
        if (!this.isInitialized) {
            console.warn('Google Translate not initialized yet');
            return;
        }

        try {
            // Find the Google Translate select element
            const translateSelect = document.querySelector('.goog-te-combo');
            if (translateSelect) {
                translateSelect.value = languageCode;
                translateSelect.dispatchEvent(new Event('change'));
                
                // Save language preference
                this.saveLanguage(languageCode);
                this.currentLanguage = languageCode;
            }
        } catch (error) {
            console.error('Failed to translate to', languageCode, error);
        }
    }

    setupDynamicTranslation() {
        // Observer for dynamically added content
        const observer = new MutationObserver((mutations) => {
            let shouldRetranslate = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            // Check if it's significant content that needs translation
                            const hasText = node.textContent && node.textContent.trim().length > 0;
                            const isRelevant = node.matches && (
                                node.matches('.ai-response') ||
                                node.matches('.result-content') ||
                                node.matches('.chat-message') ||
                                node.matches('.disease-info') ||
                                node.matches('.recommendation')
                            );
                            
                            if (hasText && isRelevant) {
                                shouldRetranslate = true;
                            }
                        }
                    });
                }
            });

            if (shouldRetranslate && this.currentLanguage !== 'en') {
                // Delay retranslation to allow content to settle
                setTimeout(() => {
                    this.triggerRetranslation();
                }, 500);
            }
        });

        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    triggerRetranslation() {
        if (!this.isInitialized || this.currentLanguage === 'en') return;

        try {
            // Trigger re-translation by changing language momentarily
            const translateSelect = document.querySelector('.goog-te-combo');
            if (translateSelect) {
                const currentValue = translateSelect.value;
                translateSelect.value = 'en';
                translateSelect.dispatchEvent(new Event('change'));
                
                setTimeout(() => {
                    translateSelect.value = currentValue;
                    translateSelect.dispatchEvent(new Event('change'));
                }, 100);
            }
        } catch (error) {
            console.error('Failed to trigger retranslation:', error);
        }
    }

    saveLanguage(languageCode) {
        try {
            localStorage.setItem('preferredLanguage', languageCode);
        } catch (error) {
            console.warn('Failed to save language preference:', error);
        }
    }

    getSavedLanguage() {
        try {
            return localStorage.getItem('preferredLanguage');
        } catch (error) {
            console.warn('Failed to get saved language:', error);
            return null;
        }
    }

    restoreSavedLanguage() {
        if (this.currentLanguage && this.currentLanguage !== 'en') {
            this.translateTo(this.currentLanguage);
        }
    }

    // Public method to translate specific content (for AJAX responses)
    translateContent(element) {
        if (!this.isInitialized || this.currentLanguage === 'en') return;

        // Add a marker to help Google Translate identify new content
        element.setAttribute('data-translate', 'true');
        
        // Trigger retranslation
        setTimeout(() => {
            this.triggerRetranslation();
        }, 100);
    }

    // Method to get current language
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    // Method to check if translation is active
    isTranslationActive() {
        return this.currentLanguage !== 'en';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.googleTranslateManager = new GoogleTranslateManager();
});

// Also expose for manual initialization
window.GoogleTranslateManager = GoogleTranslateManager;
