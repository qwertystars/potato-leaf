"""
Disease Spread Prediction Engine

This module calculates disease spread risk based on weather conditions
using scientific correlation models for potato diseases.
"""

import sqlite3
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import statistics

# Configure logging
logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Disease spread prediction engine using weather-based correlation models.
    
    Implements rule-based algorithms for Early Blight and Late Blight diseases
    based on temperature, humidity, and precipitation conditions.
    """
    
    def __init__(self, db_path: str = "weather_cache.db"):
        """
        Initialize prediction engine.
        
        Args:
            db_path: Path to SQLite database for storing predictions
        """
        self.db_path = db_path
        
        # Disease-specific correlation weights (based on research)
        self.disease_weights = {
            'Early_Blight': {
                'temperature': 0.40,
                'humidity': 0.35,
                'precipitation': 0.25
            },
            'Late_Blight': {
                'temperature': 0.30,
                'humidity': 0.30,
                'precipitation': 0.40
            },
            'Healthy': {
                'temperature': 0.0,
                'humidity': 0.0,
                'precipitation': 0.0
            }
        }
        
        # Optimal conditions for disease development
        self.disease_conditions = {
            'Early_Blight': {
                'temperature_range': (24, 29),  # °C
                'humidity_threshold': 90,       # %
                'precipitation_factor': 0.5     # mm/day weight
            },
            'Late_Blight': {
                'temperature_range': (15, 20),  # °C
                'humidity_threshold': 80,       # %
                'precipitation_factor': 1.0     # mm/day weight
            }
        }
        
        # Initialize database
        self.init_prediction_db()
    
    def init_prediction_db(self):
        """Initialize SQLite database with disease prediction tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create disease_spread_predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disease_spread_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_type TEXT NOT NULL,
                    location_lat REAL NOT NULL,
                    location_lon REAL NOT NULL,
                    prediction_date DATE NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    temperature_score REAL NOT NULL,
                    humidity_score REAL NOT NULL,
                    precipitation_score REAL NOT NULL,
                    confidence_level REAL NOT NULL,
                    created_at DATETIME NOT NULL,
                    weather_forecast_id INTEGER
                )
            ''')
            
            # Create preventive_measures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preventive_measures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    measure_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    estimated_cost TEXT NOT NULL,
                    effectiveness REAL NOT NULL,
                    time_to_implement TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prediction_location_date 
                ON disease_spread_predictions(location_lat, location_lon, prediction_date)
            ''')
            
            # Insert default preventive measures
            self._insert_default_preventive_measures(cursor)
            
            conn.commit()
            conn.close()
            logger.info("Prediction database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction database: {e}")
            raise
    
    def _insert_default_preventive_measures(self, cursor):
        """Insert default preventive measures for different diseases and risk levels."""
        
        # Check if measures already exist
        cursor.execute('SELECT COUNT(*) FROM preventive_measures')
        if cursor.fetchone()[0] > 0:
            return  # Already populated
        
        measures = [
            # Early Blight - High Risk
            ('Early_Blight', 'High', 'Immediate', 'Apply Fungicide Treatment',
             'Apply copper-based fungicide spray immediately to infected and surrounding plants. Focus on leaf undersides and stems.',
             1, 'Medium', 0.85, '2-3 hours', True),
            
            ('Early_Blight', 'High', 'Immediate', 'Remove Infected Leaves',
             'Carefully remove and destroy all visibly infected leaves and plant debris. Use clean tools and dispose of material away from crops.',
             2, 'Free', 0.70, '1-2 hours', True),
            
            ('Early_Blight', 'High', 'Preventive', 'Reduce Irrigation Frequency',
             'Decrease watering frequency and avoid overhead irrigation. Water at soil level to reduce leaf wetness duration.',
             3, 'Free', 0.60, '30 minutes', True),
            
            # Early Blight - Medium Risk
            ('Early_Blight', 'Medium', 'Preventive', 'Improve Air Circulation',
             'Increase plant spacing and remove lower leaves to improve air flow around plants. This reduces humidity around foliage.',
             1, 'Free', 0.65, '1-2 hours', True),
            
            ('Early_Blight', 'Medium', 'Monitoring', 'Daily Plant Inspection',
             'Inspect plants daily for early signs of disease. Look for dark spots with concentric rings on leaves.',
             2, 'Free', 0.50, '15 minutes daily', True),
            
            # Late Blight - High Risk
            ('Late_Blight', 'High', 'Immediate', 'Emergency Fungicide Application',
             'Apply systemic fungicide containing metalaxyl or dimethomorph immediately. This is critical for Late Blight control.',
             1, 'High', 0.90, '2-3 hours', True),
            
            ('Late_Blight', 'High', 'Immediate', 'Harvest Early if Possible',
             'If tubers are mature enough, consider emergency harvest to prevent total crop loss. Late Blight spreads very rapidly.',
             2, 'Low', 0.80, '1-2 days', True),
            
            ('Late_Blight', 'High', 'Immediate', 'Destroy Infected Plants',
             'Remove and burn all infected plants immediately. Late Blight can destroy entire crops within days.',
             3, 'Free', 0.75, '3-4 hours', True),
            
            # Late Blight - Medium Risk
            ('Late_Blight', 'Medium', 'Preventive', 'Preventive Fungicide Spray',
             'Apply preventive fungicide containing copper compounds or mancozeb before disease symptoms appear.',
             1, 'Medium', 0.75, '1-2 hours', True),
            
            ('Late_Blight', 'Medium', 'Preventive', 'Ensure Proper Drainage',
             'Improve field drainage and avoid waterlogging. Late Blight thrives in wet conditions.',
             2, 'Low', 0.60, '2-3 hours', True),
            
            # Low Risk measures (applicable to both diseases)
            ('Early_Blight', 'Low', 'Monitoring', 'Weekly Health Checks',
             'Conduct weekly plant health assessments. Document any changes in plant appearance or growth.',
             1, 'Free', 0.40, '30 minutes weekly', True),
            
            ('Late_Blight', 'Low', 'Monitoring', 'Weather Monitoring',
             'Monitor weather forecasts for conditions favoring Late Blight (cool, wet weather).',
             1, 'Free', 0.45, '5 minutes daily', True),
            
            # General measures
            ('Early_Blight', 'Low', 'Preventive', 'Maintain Plant Nutrition',
             'Ensure balanced nutrition with adequate potassium and avoid excess nitrogen which increases susceptibility.',
             2, 'Low', 0.50, '1 hour weekly', True),
            
            ('Late_Blight', 'Low', 'Preventive', 'Use Resistant Varieties',
             'Consider planting Late Blight resistant potato varieties in future seasons.',
             2, 'Low', 0.70, 'Next season', True)
        ]
        
        cursor.executemany('''
            INSERT INTO preventive_measures (
                disease_type, risk_level, measure_type, title, description,
                priority, estimated_cost, effectiveness, time_to_implement, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', measures)
    
    def calculate_temperature_score(self, temperature: float, disease_type: str) -> float:
        """
        Calculate temperature contribution to disease risk.
        
        Args:
            temperature: Temperature in Celsius
            disease_type: Disease type ('Early_Blight' or 'Late_Blight')
            
        Returns:
            Score between 0.0 and 100.0
        """
        if disease_type not in self.disease_conditions:
            return 0.0
        
        optimal_range = self.disease_conditions[disease_type]['temperature_range']
        min_temp, max_temp = optimal_range
        
        if min_temp <= temperature <= max_temp:
            # Within optimal range - maximum risk
            return 100.0
        elif temperature < min_temp:
            # Below optimal - risk decreases linearly
            if temperature < min_temp - 10:
                return 0.0
            return max(0.0, 100.0 * (temperature - (min_temp - 10)) / 10)
        else:
            # Above optimal - risk decreases linearly
            if temperature > max_temp + 10:
                return 0.0
            return max(0.0, 100.0 * ((max_temp + 10) - temperature) / 10)
    
    def calculate_humidity_score(self, humidity: float, disease_type: str) -> float:
        """
        Calculate humidity contribution to disease risk.
        
        Args:
            humidity: Relative humidity percentage (0-100)
            disease_type: Disease type ('Early_Blight' or 'Late_Blight')
            
        Returns:
            Score between 0.0 and 100.0
        """
        if disease_type not in self.disease_conditions:
            return 0.0
        
        threshold = self.disease_conditions[disease_type]['humidity_threshold']
        
        if humidity >= threshold:
            # Above threshold - high risk
            return 100.0
        elif humidity >= threshold - 20:
            # Moderate humidity - linear risk increase
            return 100.0 * (humidity - (threshold - 20)) / 20
        else:
            # Low humidity - minimal risk
            return 0.0
    
    def calculate_precipitation_score(self, precipitation: float, disease_type: str) -> float:
        """
        Calculate precipitation contribution to disease risk.
        
        Args:
            precipitation: Precipitation in mm/day
            disease_type: Disease type ('Early_Blight' or 'Late_Blight')
            
        Returns:
            Score between 0.0 and 100.0
        """
        if disease_type not in self.disease_conditions:
            return 0.0
        
        factor = self.disease_conditions[disease_type]['precipitation_factor']
        
        # Risk increases with precipitation, different rates for different diseases
        if precipitation <= 0:
            return 0.0
        elif precipitation <= 5:
            return min(100.0, precipitation * 10 * factor)
        elif precipitation <= 15:
            return min(100.0, 50 + (precipitation - 5) * 3 * factor)
        else:
            # Heavy precipitation - maximum risk
            return 100.0
    
    def calculate_disease_risk(self, weather_data: List[Dict], disease_type: str) -> Dict:
        """
        Calculate disease spread risk based on weather forecast.
        
        Args:
            weather_data: List of weather forecast dictionaries
            disease_type: Disease type ('Early_Blight', 'Late_Blight', or 'Healthy')
            
        Returns:
            Dictionary with risk scores and analysis
        """
        if disease_type == 'Healthy':
            return {
                'risk_score': 0.0,
                'risk_level': 'Low',
                'temperature_score': 0.0,
                'humidity_score': 0.0,
                'precipitation_score': 0.0,
                'confidence_level': 1.0,
                'analysis': 'No disease detected - no spread risk calculated'
            }
        
        if not weather_data or disease_type not in self.disease_weights:
            return {
                'risk_score': 0.0,
                'risk_level': 'Unknown',
                'temperature_score': 0.0,
                'humidity_score': 0.0,
                'precipitation_score': 0.0,
                'confidence_level': 0.0,
                'analysis': 'Insufficient data for risk calculation'
            }
        
        # Calculate individual scores for each weather parameter
        temp_scores = []
        humidity_scores = []
        precip_scores = []
        
        for forecast in weather_data:
            temp_score = self.calculate_temperature_score(forecast['temperature'], disease_type)
            humidity_score = self.calculate_humidity_score(forecast['humidity'], disease_type)
            precip_score = self.calculate_precipitation_score(forecast['precipitation'], disease_type)
            
            temp_scores.append(temp_score)
            humidity_scores.append(humidity_score)
            precip_scores.append(precip_score)
        
        # Calculate average scores
        avg_temp_score = statistics.mean(temp_scores)
        avg_humidity_score = statistics.mean(humidity_scores)
        avg_precip_score = statistics.mean(precip_scores)
        
        # Calculate weighted risk score
        weights = self.disease_weights[disease_type]
        risk_score = (
            weights['temperature'] * avg_temp_score +
            weights['humidity'] * avg_humidity_score +
            weights['precipitation'] * avg_precip_score
        )
        
        # Determine risk level
        if risk_score <= 30:
            risk_level = 'Low'
        elif risk_score <= 60:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Calculate confidence based on data consistency
        temp_variance = statistics.variance(temp_scores) if len(temp_scores) > 1 else 0
        humidity_variance = statistics.variance(humidity_scores) if len(humidity_scores) > 1 else 0
        precip_variance = statistics.variance(precip_scores) if len(precip_scores) > 1 else 0
        
        # Lower variance = higher confidence
        avg_variance = (temp_variance + humidity_variance + precip_variance) / 3
        confidence_level = max(0.5, 1.0 - (avg_variance / 10000))  # Normalize variance
        
        return {
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'temperature_score': round(avg_temp_score, 1),
            'humidity_score': round(avg_humidity_score, 1),
            'precipitation_score': round(avg_precip_score, 1),
            'confidence_level': round(confidence_level, 2),
            'analysis': f'{disease_type} spread risk: {risk_level} ({risk_score:.1f}%)'
        }
    
    def save_prediction(self, prediction_data: Dict, lat: float, lon: float) -> int:
        """
        Save disease spread prediction to database.
        
        Args:
            prediction_data: Prediction results dictionary
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            Prediction ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO disease_spread_predictions (
                    disease_type, location_lat, location_lon, prediction_date,
                    risk_score, risk_level, temperature_score, humidity_score,
                    precipitation_score, confidence_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data['disease_type'],
                lat, lon,
                date.today().isoformat(),
                prediction_data['risk_score'],
                prediction_data['risk_level'],
                prediction_data['temperature_score'],
                prediction_data['humidity_score'],
                prediction_data['precipitation_score'],
                prediction_data['confidence_level'],
                datetime.now()
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Saved prediction {prediction_id} for {prediction_data['disease_type']}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            raise
    
    def get_preventive_measures(self, disease_type: str, risk_level: str) -> List[Dict]:
        """
        Get preventive measures for specific disease and risk level.
        
        Args:
            disease_type: Disease type
            risk_level: Risk level ('Low', 'Medium', 'High')
            
        Returns:
            List of preventive measure dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, description, measure_type, priority,
                       estimated_cost, effectiveness, time_to_implement
                FROM preventive_measures
                WHERE disease_type = ? AND risk_level = ? AND is_active = 1
                ORDER BY priority
            ''', (disease_type, risk_level))
            
            rows = cursor.fetchall()
            conn.close()
            
            measures = []
            for row in rows:
                measures.append({
                    'title': row[0],
                    'description': row[1],
                    'measure_type': row[2],
                    'priority': row[3],
                    'estimated_cost': row[4],
                    'effectiveness': row[5],
                    'time_to_implement': row[6]
                })
            
            return measures
            
        except Exception as e:
            logger.error(f"Failed to get preventive measures: {e}")
            return []
    
    def predict_disease_spread(self, weather_data: List[Dict], disease_type: str, 
                             lat: float, lon: float) -> Dict:
        """
        Complete disease spread prediction workflow.
        
        Args:
            weather_data: Weather forecast data
            disease_type: Detected disease type
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            Complete prediction results with recommendations
        """
        # Calculate risk scores
        risk_analysis = self.calculate_disease_risk(weather_data, disease_type)
        
        # Add metadata
        risk_analysis['disease_type'] = disease_type
        risk_analysis['location'] = {'lat': lat, 'lon': lon}
        risk_analysis['prediction_date'] = date.today().isoformat()
        
        # Save prediction to database
        if risk_analysis['risk_score'] > 0:
            prediction_id = self.save_prediction(risk_analysis, lat, lon)
            risk_analysis['prediction_id'] = prediction_id
        
        # Get preventive measures
        measures = self.get_preventive_measures(disease_type, risk_analysis['risk_level'])
        risk_analysis['preventive_measures'] = measures
        
        return risk_analysis
