/**
 * Dynamic Content Translation Helper
 * Handles translation of dynamically loaded content like AI responses, chat messages, etc.
 */

class DynamicTranslationHelper {
    constructor() {
        this.isInitialized = false;
        this.translationObserver = null;
        this.pendingTranslations = new Set();
        this.init();
    }

    init() {
        // Wait for Google Translate to be available
        this.waitForGoogleTranslate(() => {
            this.isInitialized = true;
            this.setupObserver();
        });
    }

    waitForGoogleTranslate(callback, maxAttempts = 50) {
        let attempts = 0;
        const checkInterval = setInterval(() => {
            attempts++;
            if (window.googleTranslateManager && window.googleTranslateManager.isInitialized) {
                clearInterval(checkInterval);
                callback();
            } else if (attempts >= maxAttempts) {
                clearInterval(checkInterval);
                console.warn('Google Translate not available after maximum attempts');
            }
        }, 200);
    }

    setupObserver() {
        // Create mutation observer for dynamic content
        this.translationObserver = new MutationObserver((mutations) => {
            this.handleMutations(mutations);
        });

        // Start observing
        this.translationObserver.observe(document.body, {
            childList: true,
            subtree: true,
            characterData: true
        });
    }

    handleMutations(mutations) {
        if (!window.googleTranslateManager || !window.googleTranslateManager.isTranslationActive()) {
            return;
        }

        let hasNewContent = false;

        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach((node) => {
                    if (this.shouldTranslateNode(node)) {
                        hasNewContent = true;
                        this.markForTranslation(node);
                    }
                });
            } else if (mutation.type === 'characterData') {
                const parentNode = mutation.target.parentNode;
                if (this.shouldTranslateNode(parentNode)) {
                    hasNewContent = true;
                    this.markForTranslation(parentNode);
                }
            }
        });

        if (hasNewContent) {
            // Debounce translation requests
            this.debounceTranslation();
        }
    }

    shouldTranslateNode(node) {
        if (!node || node.nodeType !== Node.ELEMENT_NODE) {
            return false;
        }

        // Check if it's a relevant element that should be translated
        const relevantSelectors = [
            '.ai-response',
            '.result-content', 
            '.chat-message',
            '.disease-info',
            '.recommendation',
            '.analysis-result',
            '.confidence-score',
            '.treatment-suggestion',
            '.farmer-message',
            '.bot-message',
            '.prediction-result',
            '.error-message',
            '.success-message',
            '.warning-message'
        ];

        return relevantSelectors.some(selector => {
            try {
                return node.matches && node.matches(selector);
            } catch (e) {
                return false;
            }
        }) || this.hasTranslatableText(node);
    }

    hasTranslatableText(node) {
        if (!node || !node.textContent) return false;
        
        const text = node.textContent.trim();
        const minLength = 10; // Minimum text length to consider for translation
        
        // Check if it has meaningful text (not just symbols or numbers)
        const hasLetters = /[a-zA-Z]/.test(text);
        const isLongEnough = text.length >= minLength;
        
        return hasLetters && isLongEnough;
    }

    markForTranslation(node) {
        // Add translation marker
        if (!node.hasAttribute('data-translate-marked')) {
            node.setAttribute('data-translate-marked', 'true');
            this.pendingTranslations.add(node);
        }
    }

    debounceTranslation() {
        // Clear existing timeout
        if (this.translationTimeout) {
            clearTimeout(this.translationTimeout);
        }

        // Set new timeout
        this.translationTimeout = setTimeout(() => {
            this.triggerTranslation();
        }, 300); // Wait 300ms for more content to settle
    }

    triggerTranslation() {
        if (this.pendingTranslations.size === 0) return;

        try {
            // Trigger retranslation via Google Translate Manager
            if (window.googleTranslateManager) {
                window.googleTranslateManager.triggerRetranslation();
            }

            // Clear pending translations
            this.pendingTranslations.clear();
        } catch (error) {
            console.error('Failed to trigger translation:', error);
        }
    }

    // Method to manually trigger translation for specific elements
    translateElement(element) {
        if (!element || !this.isInitialized) return;

        this.markForTranslation(element);
        this.debounceTranslation();
    }

    // Method to translate AI responses specifically
    translateAIResponse(responseElement) {
        if (!responseElement) return;

        // Add specific class for AI responses
        responseElement.classList.add('ai-response');
        
        // Trigger translation
        this.translateElement(responseElement);
    }

    // Method to translate chat messages
    translateChatMessage(messageElement) {
        if (!messageElement) return;

        // Add specific class for chat messages
        messageElement.classList.add('chat-message');
        
        // Trigger translation
        this.translateElement(messageElement);
    }

    // Method to handle AJAX content updates
    handleAjaxContent(container) {
        if (!container) return;

        // Find all translatable elements in the container
        const translatableElements = container.querySelectorAll([
            '.ai-response',
            '.result-content',
            '.chat-message',
            '.disease-info',
            '.recommendation',
            '.analysis-result'
        ].join(','));

        translatableElements.forEach(element => {
            this.translateElement(element);
        });

        // Also check the container itself
        this.translateElement(container);
    }

    // Cleanup method
    destroy() {
        if (this.translationObserver) {
            this.translationObserver.disconnect();
        }
        if (this.translationTimeout) {
            clearTimeout(this.translationTimeout);
        }
        this.pendingTranslations.clear();
    }
}

// Global helper functions for easy access
window.translateAIResponse = function(element) {
    if (window.dynamicTranslationHelper) {
        window.dynamicTranslationHelper.translateAIResponse(element);
    }
};

window.translateChatMessage = function(element) {
    if (window.dynamicTranslationHelper) {
        window.dynamicTranslationHelper.translateChatMessage(element);
    }
};

window.handleAjaxContent = function(container) {
    if (window.dynamicTranslationHelper) {
        window.dynamicTranslationHelper.handleAjaxContent(container);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dynamicTranslationHelper = new DynamicTranslationHelper();
});

// Also expose class for manual initialization
window.DynamicTranslationHelper = DynamicTranslationHelper;
