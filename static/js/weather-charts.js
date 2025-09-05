/**
 * Weather Charts Module
 * Handles Chart.js visualizations for weather data and disease predictions
 */

class WeatherCharts {
    constructor() {
        this.charts = {};
        this.initializeChartDefaults();
    }
    
    initializeChartDefaults() {
        // Set global Chart.js defaults
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
            Chart.defaults.color = '#666';
            Chart.defaults.plugins.legend.labels.usePointStyle = true;
        }
    }
    
    /**
     * Create weather trends line chart
     * @param {string} canvasId - Canvas element ID
     * @param {Object} weatherData - Weather forecast data
     */
    createWeatherTrendsChart(canvasId, weatherData) {
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Process weather data
        const chartData = this.processWeatherData(weatherData);
        
        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.dates,
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: chartData.temperatures,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Humidity (%)',
                        data: chartData.humidities,
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Precipitation (mm)',
                        data: chartData.precipitations,
                        borderColor: '#45b7d1',
                        backgroundColor: 'rgba(69, 183, 209, 0.1)',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Temperature (°C) / Precipitation (mm)'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Humidity (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: '3-Day Weather Forecast',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const datasetLabel = context.dataset.label;
                                if (datasetLabel.includes('Temperature')) {
                                    return 'Affects disease development rate';
                                } else if (datasetLabel.includes('Humidity')) {
                                    return 'Higher humidity increases disease risk';
                                } else if (datasetLabel.includes('Precipitation')) {
                                    return 'Wet conditions favor disease spread';
                                }
                                return '';
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[canvasId];
    }
    
    /**
     * Create risk breakdown doughnut chart
     * @param {string} canvasId - Canvas element ID
     * @param {Object} predictionData - Disease prediction data
     */
    createRiskBreakdownChart(canvasId, predictionData) {
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Temperature Risk', 'Humidity Risk', 'Precipitation Risk'],
                datasets: [{
                    data: [
                        predictionData.temperature_score,
                        predictionData.humidity_score,
                        predictionData.precipitation_score
                    ],
                    backgroundColor: [
                        '#ff6b6b',
                        '#4ecdc4',
                        '#45b7d1'
                    ],
                    borderWidth: 3,
                    borderColor: '#fff',
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '50%',
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk Factor Contributions',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                return `${label}: ${value.toFixed(1)}%`;
                            },
                            afterLabel: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `Contribution: ${percentage}% of total risk`;
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[canvasId];
    }
    
    /**
     * Create risk gauge chart
     * @param {string} canvasId - Canvas element ID
     * @param {number} riskScore - Risk score (0-100)
     * @param {string} riskLevel - Risk level (Low/Medium/High)
     */
    createRiskGaugeChart(canvasId, riskScore, riskLevel) {
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Determine color based on risk level
        let gaugeColor;
        switch(riskLevel) {
            case 'Low':
                gaugeColor = '#28a745';
                break;
            case 'Medium':
                gaugeColor = '#ffc107';
                break;
            case 'High':
                gaugeColor = '#dc3545';
                break;
            default:
                gaugeColor = '#6c757d';
        }
        
        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [riskScore, 100 - riskScore],
                    backgroundColor: [gaugeColor, '#e9ecef'],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            },
            plugins: [{
                id: 'gaugeText',
                beforeDraw: function(chart) {
                    const ctx = chart.ctx;
                    const centerX = chart.chartArea.left + (chart.chartArea.right - chart.chartArea.left) / 2;
                    const centerY = chart.chartArea.top + (chart.chartArea.bottom - chart.chartArea.top) / 2 + 20;
                    
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.font = 'bold 24px sans-serif';
                    ctx.fillStyle = gaugeColor;
                    ctx.fillText(`${riskScore}%`, centerX, centerY);
                    ctx.restore();
                }
            }]
        });
        
        return this.charts[canvasId];
    }
    
    /**
     * Create daily risk comparison bar chart
     * @param {string} canvasId - Canvas element ID
     * @param {Array} dailyRisks - Array of daily risk scores
     */
    createDailyRiskChart(canvasId, dailyRisks) {
        const ctx = document.getElementById(canvasId)?.getContext('2d');
        if (!ctx) return null;
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        const dates = dailyRisks.map((_, index) => `Day ${index + 1}`);
        const riskScores = dailyRisks.map(risk => risk.score);
        
        // Color bars based on risk level
        const barColors = riskScores.map(score => {
            if (score <= 30) return '#28a745';
            if (score <= 60) return '#ffc107';
            return '#dc3545';
        });
        
        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Daily Risk Score',
                    data: riskScores,
                    backgroundColor: barColors,
                    borderColor: barColors.map(color => color),
                    borderWidth: 2,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Risk Score (%)'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Forecast Period'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Daily Disease Risk Forecast',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const score = context.parsed.y;
                                let riskLevel;
                                if (score <= 30) riskLevel = 'Low Risk';
                                else if (score <= 60) riskLevel = 'Medium Risk';
                                else riskLevel = 'High Risk';
                                return `Risk Level: ${riskLevel}`;
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[canvasId];
    }
    
    /**
     * Process weather data for chart visualization
     * @param {Array} weatherForecast - Weather forecast data
     * @returns {Object} Processed chart data
     */
    processWeatherData(weatherForecast) {
        // Group data by date
        const dailyData = {};
        
        weatherForecast.forEach(item => {
            const date = item.date;
            if (!dailyData[date]) {
                dailyData[date] = {
                    temperatures: [],
                    humidities: [],
                    precipitations: []
                };
            }
            
            dailyData[date].temperatures.push(parseFloat(item.temperature));
            dailyData[date].humidities.push(parseFloat(item.humidity));
            dailyData[date].precipitations.push(parseFloat(item.precipitation));
        });
        
        // Calculate daily averages
        const dates = [];
        const temperatures = [];
        const humidities = [];
        const precipitations = [];
        
        Object.keys(dailyData).sort().forEach(date => {
            const data = dailyData[date];
            
            dates.push(new Date(date).toLocaleDateString('en-US', { 
                weekday: 'short', 
                month: 'short', 
                day: 'numeric' 
            }));
            
            temperatures.push(
                Math.round((data.temperatures.reduce((a, b) => a + b, 0) / data.temperatures.length) * 10) / 10
            );
            
            humidities.push(
                Math.round((data.humidities.reduce((a, b) => a + b, 0) / data.humidities.length) * 10) / 10
            );
            
            precipitations.push(
                Math.round(data.precipitations.reduce((a, b) => a + b, 0) * 10) / 10
            );
        });
        
        return {
            dates,
            temperatures,
            humidities,
            precipitations
        };
    }
    
    /**
     * Destroy all charts
     */
    destroyAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }
    
    /**
     * Destroy specific chart
     * @param {string} canvasId - Canvas element ID
     */
    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }
    
    /**
     * Update chart with new data
     * @param {string} canvasId - Canvas element ID
     * @param {Object} newData - New chart data
     */
    updateChart(canvasId, newData) {
        const chart = this.charts[canvasId];
        if (chart) {
            chart.data = newData;
            chart.update();
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WeatherCharts;
} else {
    window.WeatherCharts = WeatherCharts;
}
