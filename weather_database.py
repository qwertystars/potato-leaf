"""
Weather Database Module

Handles database operations for weather forecasts and disease predictions.
Provides utilities for database management and data validation.
"""

import sqlite3
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)

class WeatherDatabase:
    """
    Database management class for weather and prediction data.
    
    Handles SQLite database operations including table creation,
    data validation, and maintenance tasks.
    """
    
    def __init__(self, db_path: str = "weather_cache.db"):
        """
        Initialize weather database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize all required database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute('PRAGMA foreign_keys = ON')
            
            # Create weather_forecasts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_lat REAL NOT NULL CHECK(location_lat >= -90 AND location_lat <= 90),
                    location_lon REAL NOT NULL CHECK(location_lon >= -180 AND location_lon <= 180),
                    location_name TEXT,
                    forecast_date DATE NOT NULL,
                    forecast_hour INTEGER NOT NULL CHECK(forecast_hour >= 0 AND forecast_hour <= 23),
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL CHECK(humidity >= 0 AND humidity <= 100),
                    precipitation REAL NOT NULL DEFAULT 0.0 CHECK(precipitation >= 0),
                    weather_condition TEXT,
                    weather_description TEXT,
                    wind_speed REAL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            ''')
            
            # Create disease_spread_predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disease_spread_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_type TEXT NOT NULL CHECK(disease_type IN ('Early_Blight', 'Late_Blight', 'Healthy')),
                    location_lat REAL NOT NULL CHECK(location_lat >= -90 AND location_lat <= 90),
                    location_lon REAL NOT NULL CHECK(location_lon >= -180 AND location_lon <= 180),
                    prediction_date DATE NOT NULL,
                    risk_score REAL NOT NULL CHECK(risk_score >= 0.0 AND risk_score <= 100.0),
                    risk_level TEXT NOT NULL CHECK(risk_level IN ('Low', 'Medium', 'High')),
                    temperature_score REAL NOT NULL CHECK(temperature_score >= 0.0 AND temperature_score <= 100.0),
                    humidity_score REAL NOT NULL CHECK(humidity_score >= 0.0 AND humidity_score <= 100.0),
                    precipitation_score REAL NOT NULL CHECK(precipitation_score >= 0.0 AND precipitation_score <= 100.0),
                    confidence_level REAL NOT NULL CHECK(confidence_level >= 0.0 AND confidence_level <= 1.0),
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    weather_forecast_id INTEGER,
                    FOREIGN KEY (weather_forecast_id) REFERENCES weather_forecasts(id) ON DELETE SET NULL
                )
            ''')
            
            # Create preventive_measures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS preventive_measures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL CHECK(risk_level IN ('Low', 'Medium', 'High')),
                    measure_type TEXT NOT NULL CHECK(measure_type IN ('Immediate', 'Preventive', 'Monitoring')),
                    title TEXT NOT NULL CHECK(LENGTH(title) <= 100),
                    description TEXT NOT NULL,
                    priority INTEGER NOT NULL CHECK(priority > 0),
                    estimated_cost TEXT NOT NULL CHECK(estimated_cost IN ('Free', 'Low', 'Medium', 'High')),
                    effectiveness REAL NOT NULL CHECK(effectiveness >= 0.0 AND effectiveness <= 1.0),
                    time_to_implement TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_weather_location_date 
                ON weather_forecasts(location_lat, location_lon, forecast_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_weather_expires_at 
                ON weather_forecasts(expires_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_prediction_location_date 
                ON disease_spread_predictions(location_lat, location_lon, prediction_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_preventive_measures_lookup
                ON preventive_measures(disease_type, risk_level, is_active)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Weather database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def validate_weather_data(self, weather_data: Dict) -> bool:
        """
        Validate weather forecast data before insertion.
        
        Args:
            weather_data: Weather data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'location_lat', 'location_lon', 'forecast_date', 'forecast_hour',
            'temperature', 'humidity', 'precipitation'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in weather_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate coordinate ranges
        lat = weather_data['location_lat']
        lon = weather_data['location_lon']
        if not (-90 <= lat <= 90):
            logger.error(f"Invalid latitude: {lat}")
            return False
        if not (-180 <= lon <= 180):
            logger.error(f"Invalid longitude: {lon}")
            return False
        
        # Validate humidity range
        humidity = weather_data['humidity']
        if not (0 <= humidity <= 100):
            logger.error(f"Invalid humidity: {humidity}")
            return False
        
        # Validate precipitation
        precipitation = weather_data['precipitation']
        if precipitation < 0:
            logger.error(f"Invalid precipitation: {precipitation}")
            return False
        
        # Validate forecast hour
        hour = weather_data['forecast_hour']
        if not (0 <= hour <= 23):
            logger.error(f"Invalid forecast hour: {hour}")
            return False
        
        return True
    
    def validate_prediction_data(self, prediction_data: Dict) -> bool:
        """
        Validate disease prediction data before insertion.
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'disease_type', 'location_lat', 'location_lon', 'risk_score',
            'risk_level', 'temperature_score', 'humidity_score',
            'precipitation_score', 'confidence_level'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in prediction_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate disease type
        valid_diseases = ['Early_Blight', 'Late_Blight', 'Healthy']
        if prediction_data['disease_type'] not in valid_diseases:
            logger.error(f"Invalid disease type: {prediction_data['disease_type']}")
            return False
        
        # Validate risk level
        valid_risk_levels = ['Low', 'Medium', 'High']
        if prediction_data['risk_level'] not in valid_risk_levels:
            logger.error(f"Invalid risk level: {prediction_data['risk_level']}")
            return False
        
        # Validate score ranges
        score_fields = ['risk_score', 'temperature_score', 'humidity_score', 'precipitation_score']
        for field in score_fields:
            score = prediction_data[field]
            if not (0.0 <= score <= 100.0):
                logger.error(f"Invalid {field}: {score} (must be 0-100)")
                return False
        
        # Validate confidence level
        confidence = prediction_data['confidence_level']
        if not (0.0 <= confidence <= 1.0):
            logger.error(f"Invalid confidence level: {confidence}")
            return False
        
        return True
    
    def get_weather_forecast_by_location(self, lat: float, lon: float, 
                                       hours: int = 72) -> List[Dict]:
        """
        Retrieve weather forecast for specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            hours: Number of hours to retrieve (default 72 for 3 days)
            
        Returns:
            List of weather forecast dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM weather_forecasts 
                WHERE ABS(location_lat - ?) < 0.1 
                AND ABS(location_lon - ?) < 0.1
                AND expires_at > CURRENT_TIMESTAMP
                ORDER BY forecast_date, forecast_hour
                LIMIT ?
            ''', (lat, lon, hours))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            forecasts = []
            for row in rows:
                forecast = dict(zip(columns, row))
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Failed to get weather forecast: {e}")
            return []
    
    def get_prediction_history(self, lat: float, lon: float, 
                             days: int = 30) -> List[Dict]:
        """
        Get disease prediction history for location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            days: Number of days back to retrieve
            
        Returns:
            List of prediction dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM disease_spread_predictions 
                WHERE ABS(location_lat - ?) < 0.1 
                AND ABS(location_lon - ?) < 0.1
                AND prediction_date >= DATE('now', '-{} days')
                ORDER BY prediction_date DESC, created_at DESC
            '''.format(days), (lat, lon))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            predictions = []
            for row in rows:
                prediction = dict(zip(columns, row))
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get prediction history: {e}")
            return []
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired weather forecasts and old predictions.
        
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean expired weather forecasts
            cursor.execute('''
                DELETE FROM weather_forecasts 
                WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            expired_forecasts = cursor.rowcount
            
            # Clean old predictions (older than 90 days)
            cursor.execute('''
                DELETE FROM disease_spread_predictions 
                WHERE prediction_date < DATE('now', '-90 days')
            ''')
            old_predictions = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            stats = {
                'expired_forecasts': expired_forecasts,
                'old_predictions': old_predictions
            }
            
            logger.info(f"Cleaned up {expired_forecasts} expired forecasts and {old_predictions} old predictions")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            return {'expired_forecasts': 0, 'old_predictions': 0}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and health information.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count records in each table
            cursor.execute('SELECT COUNT(*) FROM weather_forecasts')
            weather_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM disease_spread_predictions')
            prediction_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM preventive_measures')
            measures_count = cursor.fetchone()[0]
            
            # Get database file size
            file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            # Check for expired forecasts
            cursor.execute('''
                SELECT COUNT(*) FROM weather_forecasts 
                WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            expired_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'weather_forecasts': weather_count,
                'disease_predictions': prediction_count,
                'preventive_measures': measures_count,
                'database_size_bytes': file_size,
                'expired_forecasts': expired_count,
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

def init_weather_database(db_path: str = "weather_cache.db"):
    """
    Initialize weather database with all required tables.
    
    Args:
        db_path: Path to database file
        
    Returns:
        WeatherDatabase instance
    """
    return WeatherDatabase(db_path)
