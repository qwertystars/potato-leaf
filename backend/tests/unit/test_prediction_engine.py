"""
Unit tests for Prediction Engine
Tests the disease spread prediction functionality
"""

import pytest
import os
import tempfile
import sqlite3
from datetime import datetime, date

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from prediction_engine import PredictionEngine


class TestPredictionEngine:
    """Unit tests for PredictionEngine class"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create prediction engine instance
        self.prediction_engine = PredictionEngine(db_path=self.temp_db_path)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Remove temporary database
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def get_sample_weather_data(self):
        """Get sample weather data for testing"""
        return [
            {
                'date': '2024-01-01',
                'hour': 12,
                'temperature': 26.0,
                'humidity': 85.0,
                'precipitation': 5.0,
                'weather_condition': 'Rain',
                'weather_description': 'light rain',
                'wind_speed': 3.5
            },
            {
                'date': '2024-01-02',
                'hour': 12,
                'temperature': 27.0,
                'humidity': 90.0,
                'precipitation': 8.0,
                'weather_condition': 'Rain',
                'weather_description': 'moderate rain',
                'wind_speed': 4.0
            }
        ]
    
    def test_initialization(self):
        """Test PredictionEngine initialization"""
        engine = PredictionEngine(db_path=self.temp_db_path)
        
        # Check disease weights configuration
        assert 'Early_Blight' in engine.disease_weights
        assert 'Late_Blight' in engine.disease_weights
        assert 'Healthy' in engine.disease_weights
        
        # Check Early Blight weights
        early_weights = engine.disease_weights['Early_Blight']
        assert early_weights['temperature'] == 0.40
        assert early_weights['humidity'] == 0.35
        assert early_weights['precipitation'] == 0.25
        
        # Check Late Blight weights
        late_weights = engine.disease_weights['Late_Blight']
        assert late_weights['temperature'] == 0.30
        assert late_weights['humidity'] == 0.30
        assert late_weights['precipitation'] == 0.40
        
        # Check Healthy weights (should be zero)
        healthy_weights = engine.disease_weights['Healthy']
        assert all(weight == 0.0 for weight in healthy_weights.values())
    
    def test_database_initialization(self):
        """Test database tables are properly initialized"""
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Check disease_spread_predictions table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='disease_spread_predictions'
        """)
        assert cursor.fetchone() is not None
        
        # Check preventive_measures table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='preventive_measures'
        """)
        assert cursor.fetchone() is not None
        
        # Check if preventive measures are populated
        cursor.execute('SELECT COUNT(*) FROM preventive_measures')
        measures_count = cursor.fetchone()[0]
        assert measures_count > 0
        
        conn.close()
    
    def test_calculate_temperature_score_early_blight(self):
        """Test temperature scoring for Early Blight"""
        # Test optimal temperature (24-29°C)
        score = self.prediction_engine.calculate_temperature_score(26.5, 'Early_Blight')
        assert score == 100.0
        
        # Test temperature below optimal range
        score = self.prediction_engine.calculate_temperature_score(20.0, 'Early_Blight')
        assert 0.0 < score < 100.0
        
        # Test temperature above optimal range
        score = self.prediction_engine.calculate_temperature_score(35.0, 'Early_Blight')
        assert 0.0 < score < 100.0
        
        # Test temperature far outside range
        score = self.prediction_engine.calculate_temperature_score(10.0, 'Early_Blight')
        assert score == 0.0
        
        score = self.prediction_engine.calculate_temperature_score(45.0, 'Early_Blight')
        assert score == 0.0
    
    def test_calculate_temperature_score_late_blight(self):
        """Test temperature scoring for Late Blight"""
        # Test optimal temperature (15-20°C)
        score = self.prediction_engine.calculate_temperature_score(17.5, 'Late_Blight')
        assert score == 100.0
        
        # Test temperature below optimal range
        score = self.prediction_engine.calculate_temperature_score(12.0, 'Late_Blight')
        assert 0.0 < score < 100.0
        
        # Test temperature above optimal range
        score = self.prediction_engine.calculate_temperature_score(25.0, 'Late_Blight')
        assert 0.0 < score < 100.0
    
    def test_calculate_humidity_score(self):
        """Test humidity scoring"""
        # Test Early Blight (threshold 90%)
        score = self.prediction_engine.calculate_humidity_score(95.0, 'Early_Blight')
        assert score == 100.0
        
        score = self.prediction_engine.calculate_humidity_score(85.0, 'Early_Blight')
        assert 0.0 < score < 100.0
        
        score = self.prediction_engine.calculate_humidity_score(65.0, 'Early_Blight')
        assert score == 0.0
        
        # Test Late Blight (threshold 80%)
        score = self.prediction_engine.calculate_humidity_score(85.0, 'Late_Blight')
        assert score == 100.0
        
        score = self.prediction_engine.calculate_humidity_score(75.0, 'Late_Blight')
        assert 0.0 < score < 100.0
        
        score = self.prediction_engine.calculate_humidity_score(55.0, 'Late_Blight')
        assert score == 0.0
    
    def test_calculate_precipitation_score(self):
        """Test precipitation scoring"""
        # Test no precipitation
        score = self.prediction_engine.calculate_precipitation_score(0.0, 'Early_Blight')
        assert score == 0.0
        
        # Test light precipitation
        score = self.prediction_engine.calculate_precipitation_score(2.0, 'Early_Blight')
        assert 0.0 < score < 100.0
        
        # Test moderate precipitation
        score = self.prediction_engine.calculate_precipitation_score(8.0, 'Early_Blight')
        assert 0.0 < score < 100.0
        
        # Test heavy precipitation
        score = self.prediction_engine.calculate_precipitation_score(20.0, 'Early_Blight')
        assert score == 100.0
        
        # Test Late Blight (higher precipitation factor)
        score_early = self.prediction_engine.calculate_precipitation_score(5.0, 'Early_Blight')
        score_late = self.prediction_engine.calculate_precipitation_score(5.0, 'Late_Blight')
        assert score_late > score_early
    
    def test_calculate_disease_risk_early_blight(self):
        """Test disease risk calculation for Early Blight"""
        weather_data = self.get_sample_weather_data()
        
        result = self.prediction_engine.calculate_disease_risk(weather_data, 'Early_Blight')
        
        # Validate result structure
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'temperature_score' in result
        assert 'humidity_score' in result
        assert 'precipitation_score' in result
        assert 'confidence_level' in result
        assert 'analysis' in result
        
        # Validate value ranges
        assert 0.0 <= result['risk_score'] <= 100.0
        assert result['risk_level'] in ['Low', 'Medium', 'High']
        assert 0.0 <= result['temperature_score'] <= 100.0
        assert 0.0 <= result['humidity_score'] <= 100.0
        assert 0.0 <= result['precipitation_score'] <= 100.0
        assert 0.0 <= result['confidence_level'] <= 1.0
        
        # Check that analysis contains disease type
        assert 'Early_Blight' in result['analysis']
    
    def test_calculate_disease_risk_late_blight(self):
        """Test disease risk calculation for Late Blight"""
        weather_data = self.get_sample_weather_data()
        
        result = self.prediction_engine.calculate_disease_risk(weather_data, 'Late_Blight')
        
        assert result['risk_level'] in ['Low', 'Medium', 'High']
        assert 'Late_Blight' in result['analysis']
    
    def test_calculate_disease_risk_healthy(self):
        """Test disease risk calculation for healthy plants"""
        weather_data = self.get_sample_weather_data()
        
        result = self.prediction_engine.calculate_disease_risk(weather_data, 'Healthy')
        
        # Healthy plants should have zero risk
        assert result['risk_score'] == 0.0
        assert result['risk_level'] == 'Low'
        assert result['temperature_score'] == 0.0
        assert result['humidity_score'] == 0.0
        assert result['precipitation_score'] == 0.0
        assert result['confidence_level'] == 1.0
    
    def test_calculate_disease_risk_empty_weather_data(self):
        """Test disease risk calculation with empty weather data"""
        result = self.prediction_engine.calculate_disease_risk([], 'Early_Blight')
        
        assert result['risk_score'] == 0.0
        assert result['risk_level'] == 'Unknown'
        assert result['confidence_level'] == 0.0
    
    def test_calculate_disease_risk_invalid_disease(self):
        """Test disease risk calculation with invalid disease type"""
        weather_data = self.get_sample_weather_data()
        
        result = self.prediction_engine.calculate_disease_risk(weather_data, 'InvalidDisease')
        
        assert result['risk_score'] == 0.0
        assert result['confidence_level'] == 0.0
    
    def test_risk_level_mapping(self):
        """Test risk level mapping based on score"""
        weather_data = [
            {
                'date': '2024-01-01',
                'hour': 12,
                'temperature': 10.0,  # Low risk conditions
                'humidity': 30.0,
                'precipitation': 0.0,
                'weather_condition': 'Clear',
                'weather_description': 'clear sky',
                'wind_speed': 2.0
            }
        ]
        
        result = self.prediction_engine.calculate_disease_risk(weather_data, 'Early_Blight')
        # Should be low risk due to poor conditions for disease
        assert result['risk_level'] == 'Low'
        
        # Test high risk conditions
        high_risk_weather = [
            {
                'date': '2024-01-01',
                'hour': 12,
                'temperature': 26.0,  # Optimal for Early Blight
                'humidity': 95.0,     # High humidity
                'precipitation': 10.0, # Significant precipitation
                'weather_condition': 'Rain',
                'weather_description': 'heavy rain',
                'wind_speed': 1.0
            }
        ]
        
        result = self.prediction_engine.calculate_disease_risk(high_risk_weather, 'Early_Blight')
        # Should be high risk due to optimal conditions
        assert result['risk_level'] in ['Medium', 'High']
    
    def test_save_prediction(self):
        """Test saving prediction to database"""
        prediction_data = {
            'disease_type': 'Early_Blight',
            'risk_score': 75.5,
            'risk_level': 'High',
            'temperature_score': 80.0,
            'humidity_score': 90.0,
            'precipitation_score': 60.0,
            'confidence_level': 0.85
        }
        
        prediction_id = self.prediction_engine.save_prediction(
            prediction_data, 28.6139, 77.2090
        )
        
        assert prediction_id is not None
        assert isinstance(prediction_id, int)
        
        # Verify data was saved correctly
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM disease_spread_predictions WHERE id = ?',
            (prediction_id,)
        )
        row = cursor.fetchone()
        
        assert row is not None
        assert row[1] == 'Early_Blight'  # disease_type
        assert row[5] == 75.5  # risk_score
        assert row[6] == 'High'  # risk_level
        
        conn.close()
    
    def test_get_preventive_measures(self):
        """Test retrieving preventive measures"""
        # Test Early Blight high risk measures
        measures = self.prediction_engine.get_preventive_measures('Early_Blight', 'High')
        
        assert len(measures) > 0
        
        # Validate measure structure
        measure = measures[0]
        required_fields = [
            'title', 'description', 'measure_type', 'priority',
            'estimated_cost', 'effectiveness', 'time_to_implement'
        ]
        
        for field in required_fields:
            assert field in measure
        
        # Check measure types
        measure_types = [m['measure_type'] for m in measures]
        valid_types = ['Immediate', 'Preventive', 'Monitoring']
        for measure_type in measure_types:
            assert measure_type in valid_types
    
    def test_get_preventive_measures_different_risk_levels(self):
        """Test preventive measures for different risk levels"""
        high_measures = self.prediction_engine.get_preventive_measures('Early_Blight', 'High')
        medium_measures = self.prediction_engine.get_preventive_measures('Early_Blight', 'Medium')
        low_measures = self.prediction_engine.get_preventive_measures('Early_Blight', 'Low')
        
        # High risk should have immediate measures
        if high_measures:
            immediate_measures = [m for m in high_measures if m['measure_type'] == 'Immediate']
            assert len(immediate_measures) > 0
        
        # All risk levels should have some measures
        assert len(high_measures) >= 0
        assert len(medium_measures) >= 0
        assert len(low_measures) >= 0
    
    def test_predict_disease_spread_complete_workflow(self):
        """Test complete disease spread prediction workflow"""
        weather_data = self.get_sample_weather_data()
        lat, lon = 28.6139, 77.2090
        
        result = self.prediction_engine.predict_disease_spread(
            weather_data, 'Early_Blight', lat, lon
        )
        
        # Validate complete result structure
        required_fields = [
            'disease_type', 'location', 'prediction_date', 'risk_score',
            'risk_level', 'temperature_score', 'humidity_score', 
            'precipitation_score', 'confidence_level', 'analysis',
            'preventive_measures'
        ]
        
        for field in required_fields:
            assert field in result
        
        # Validate location
        assert result['location']['lat'] == lat
        assert result['location']['lon'] == lon
        
        # Validate prediction was saved (should have prediction_id)
        if result['risk_score'] > 0:
            assert 'prediction_id' in result
        
        # Validate preventive measures
        assert isinstance(result['preventive_measures'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
