/**
 * Weather UI Module
 * Handles user interface interactions for weather-based disease prediction
 */

class WeatherUI {
    constructor() {
        this.weatherData = null;
        this.predictionData = null;
        this.diseaseType = 'Early_Blight';
        this.charts = new WeatherCharts();
        this.apiBaseUrl = '/api';
        
        this.init();
    }
    
    init() {
        // Get disease type from URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        this.diseaseType = urlParams.get('disease') || 'Early_Blight';
        
        // Update disease name in header
        this.updateDiseaseDisplay();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize UI state
        this.resetUI();
    }
    
    updateDiseaseDisplay() {
        const diseaseNameElement = document.getElementById('disease-name');
        if (diseaseNameElement) {
            const displayName = this.diseaseType.replace('_', ' ');
            diseaseNameElement.textContent = displayName;
        }
    }
    
    setupEventListeners() {
        // Location input enter key
        const locationInput = document.getElementById('location-input');
        if (locationInput) {
            locationInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.getWeatherForecast();
                }
            });
        }
        
        // Get forecast button
        const getForecastBtn = document.getElementById('get-forecast-btn');
        if (getForecastBtn) {
            getForecastBtn.addEventListener('click', () => {
                this.getWeatherForecast();
            });
        }
        
        // Use IP location button
        const useIPLocationBtn = document.getElementById('use-ip-location-btn');
        if (useIPLocationBtn) {
            useIPLocationBtn.addEventListener('click', () => {
                this.useIPLocation();
            });
        }
        
        // Window resize handler for chart responsiveness
        window.addEventListener('resize', this.debounce(() => {
            this.charts.destroyAllCharts();
            if (this.weatherData && this.predictionData) {
                this.createCharts();
            }
        }, 300));
    }
    
    resetUI() {
        this.hideMessages();
        this.hideLoading();
        this.hideResults();
    }
    
    // Message handling
    showError(message) {
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        this.hideMessage('success-message');
    }
    
    showSuccess(message) {
        const successDiv = document.getElementById('success-message');
        if (successDiv) {
            successDiv.textContent = message;
            successDiv.style.display = 'block';
        }
        this.hideMessage('error-message');
    }
    
    hideMessage(messageId) {
        const messageDiv = document.getElementById(messageId);
        if (messageDiv) {
            messageDiv.style.display = 'none';
        }
    }
    
    hideMessages() {
        this.hideMessage('error-message');
        this.hideMessage('success-message');
    }
    
    // Loading state management
    showLoading() {
        const loadingSection = document.getElementById('loading-section');
        if (loadingSection) {
            loadingSection.style.display = 'block';
        }
        
        this.hideResults();
        this.disableButtons(true);
    }
    
    hideLoading() {
        const loadingSection = document.getElementById('loading-section');
        if (loadingSection) {
            loadingSection.style.display = 'none';
        }
        
        this.disableButtons(false);
    }
    
    hideResults() {
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
    }
    
    showResults() {
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
    }
    
    disableButtons(disabled) {
        const buttons = [
            'get-forecast-btn',
            'use-ip-location-btn'
        ];
        
        buttons.forEach(buttonId => {
            const button = document.getElementById(buttonId);
            if (button) {
                button.disabled = disabled;
            }
        });
    }
    
    // Location handling
    async useIPLocation() {
        try {
            this.showLoading();
            this.hideMessages();
            
            // Use IP geolocation service
            const response = await fetch('https://ipapi.co/json/');
            const data = await response.json();
            
            if (data.city && data.country) {
                const location = `${data.city}, ${data.country}`;
                const locationInput = document.getElementById('location-input');
                if (locationInput) {
                    locationInput.value = location;
                }
                
                // Proceed with weather forecast
                await this.getWeatherForecastForCoords(data.latitude, data.longitude, location);
            } else {
                throw new Error('Could not determine location from IP');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Failed to detect location automatically. Please enter your location manually.');
            console.error('IP location error:', error);
        }
    }
    
    async getWeatherForecast() {
        const locationInput = document.getElementById('location-input');
        const location = locationInput?.value?.trim();
        
        if (!location) {
            this.showError('Please enter a location');
            return;
        }
        
        try {
            this.showLoading();
            this.hideMessages();
            
            // First geocode the location
            const geocodeResponse = await fetch(`${this.apiBaseUrl}/weather/geocode?location=${encodeURIComponent(location)}`);
            const geocodeData = await geocodeResponse.json();
            
            if (!geocodeData.success) {
                throw new Error(geocodeData.error || 'Failed to find location');
            }
            
            const { lat, lon, name } = geocodeData.location;
            
            // Update input with formatted location name
            if (locationInput) {
                locationInput.value = name;
            }
            
            // Get weather forecast
            await this.getWeatherForecastForCoords(lat, lon, name);
            
        } catch (error) {
            this.hideLoading();
            this.showError(error.message || 'Failed to get weather forecast');
            console.error('Weather forecast error:', error);
        }
    }
    
    async getWeatherForecastForCoords(lat, lon, locationName) {
        try {
            // Fetch weather forecast
            const weatherResponse = await fetch(
                `${this.apiBaseUrl}/weather/forecast?lat=${lat}&lon=${lon}&location_name=${encodeURIComponent(locationName)}`
            );
            const weatherResult = await weatherResponse.json();
            
            if (!weatherResult.success) {
                throw new Error(weatherResult.error || 'Failed to get weather data');
            }
            
            this.weatherData = weatherResult;
            
            // Calculate disease spread prediction
            const predictionResponse = await fetch(`${this.apiBaseUrl}/prediction/disease-spread`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    disease_type: this.diseaseType,
                    lat: lat,
                    lon: lon,
                    weather_data: weatherResult.forecast
                })
            });
            
            const predictionResult = await predictionResponse.json();
            
            if (!predictionResult.success) {
                throw new Error(predictionResult.error || 'Failed to calculate disease prediction');
            }
            
            this.predictionData = predictionResult.prediction;
            
            // Show results
            this.displayResults();
            this.showSuccess(`Weather forecast and disease prediction loaded for ${locationName}`);
            
        } catch (error) {
            this.hideLoading();
            this.showError(error.message || 'Failed to process weather data');
            console.error('Weather processing error:', error);
        }
    }
    
    // Results display
    displayResults() {
        this.hideLoading();
        this.showResults();
        
        // Update risk summary
        this.updateRiskSummary();
        
        // Create charts
        this.createCharts();
        
        // Display preventive measures
        this.displayPreventiveMeasures();
        
        // Scroll to results
        this.scrollToResults();
    }
    
    updateRiskSummary() {
        if (!this.predictionData) return;
        
        const riskScore = this.predictionData.risk_score;
        const riskLevel = this.predictionData.risk_level;
        const analysis = this.predictionData.analysis;
        
        // Update risk score
        const riskScoreElement = document.getElementById('risk-score');
        if (riskScoreElement) {
            riskScoreElement.textContent = `${riskScore}%`;
        }
        
        // Update risk level with appropriate styling
        const riskLevelElement = document.getElementById('risk-level');
        if (riskLevelElement) {
            riskLevelElement.textContent = `${riskLevel} Risk`;
            riskLevelElement.className = `risk-level risk-${riskLevel.toLowerCase()}`;
        }
        
        // Update analysis text
        const riskAnalysisElement = document.getElementById('risk-analysis');
        if (riskAnalysisElement) {
            riskAnalysisElement.textContent = analysis;
        }
    }
    
    createCharts() {
        if (!this.weatherData || !this.predictionData) return;
        
        // Destroy existing charts
        this.charts.destroyAllCharts();
        
        // Create weather trends chart
        this.charts.createWeatherTrendsChart('weather-trends-chart', this.weatherData.forecast);
        
        // Create risk breakdown chart
        this.charts.createRiskBreakdownChart('risk-breakdown-chart', this.predictionData);
        
        // Create risk gauge chart
        this.charts.createRiskGaugeChart(
            'risk-gauge-chart', 
            this.predictionData.risk_score, 
            this.predictionData.risk_level
        );
    }
    
    displayPreventiveMeasures() {
        const container = document.getElementById('preventive-measures-list');
        if (!container || !this.predictionData) return;
        
        const measures = this.predictionData.preventive_measures || [];
        
        if (measures.length === 0) {
            container.innerHTML = '<p class="text-muted">No specific preventive measures found for this condition.</p>';
            return;
        }
        
        container.innerHTML = measures.map(measure => this.createMeasureHTML(measure)).join('');
        
        // Animate measure items
        this.animateMeasures();
    }
    
    createMeasureHTML(measure) {
        const effectivenessPercent = Math.round(measure.effectiveness * 100);
        
        return `
            <div class="measure-item">
                <div class="measure-header">
                    <div class="measure-title">${this.escapeHtml(measure.title)}</div>
                    <div class="measure-badges">
                        <span class="badge badge-${measure.measure_type.toLowerCase()}">${measure.measure_type}</span>
                        <span class="badge badge-cost">${measure.estimated_cost}</span>
                    </div>
                </div>
                <div class="measure-description">${this.escapeHtml(measure.description)}</div>
                <div class="measure-footer">
                    <span>⏱️ Time: ${this.escapeHtml(measure.time_to_implement)}</span>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span>🎯 Effectiveness:</span>
                        <div class="effectiveness-bar">
                            <div class="effectiveness-fill" style="width: ${effectivenessPercent}%"></div>
                        </div>
                        <span>${effectivenessPercent}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    animateMeasures() {
        const measureItems = document.querySelectorAll('.measure-item');
        measureItems.forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                item.style.transition = 'all 0.5s ease';
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }
    
    scrollToResults() {
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) {
            resultsSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    }
    
    // Utility functions
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Public API for external access
    getDiseaseType() {
        return this.diseaseType;
    }
    
    setDiseaseType(diseaseType) {
        this.diseaseType = diseaseType;
        this.updateDiseaseDisplay();
    }
    
    getWeatherData() {
        return this.weatherData;
    }
    
    getPredictionData() {
        return this.predictionData;
    }
    
    refresh() {
        this.resetUI();
        this.weatherData = null;
        this.predictionData = null;
        this.charts.destroyAllCharts();
    }
}

// Initialize weather UI when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Make WeatherUI available globally
    window.weatherUI = new WeatherUI();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WeatherUI;
} else {
    window.WeatherUI = WeatherUI;
}
