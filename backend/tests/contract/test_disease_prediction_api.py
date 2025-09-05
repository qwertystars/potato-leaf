"""
Contract tests for Disease Prediction API
Tests the disease spread prediction endpoints
"""

import pytest
import requests
import json
from datetime import datetime, date


class TestDiseaseSpreadPredictionAPI:
    """Test disease spread prediction API endpoints"""
    
    BASE_URL = "http://localhost:5000/api"
    
    def get_sample_weather_data(self):
        """Get sample weather data for testing"""
        return [
            {
                'date': '2024-01-01',
                'hour': 12,
                'temperature': 26.5,
                'humidity': 85.0,
                'precipitation': 5.2,
                'weather_condition': 'Rain',
                'weather_description': 'light rain',
                'wind_speed': 3.5
            },
            {
                'date': '2024-01-02',
                'hour': 12,
                'temperature': 28.0,
                'humidity': 90.0,
                'precipitation': 8.1,
                'weather_condition': 'Rain',
                'weather_description': 'moderate rain',
                'wind_speed': 4.2
            },
            {
                'date': '2024-01-03',
                'hour': 12,
                'temperature': 25.5,
                'humidity': 88.0,
                'precipitation': 3.5,
                'weather_condition': 'Clouds',
                'weather_description': 'overcast clouds',
                'wind_speed': 2.8
            }
        ]
    
    def test_disease_prediction_early_blight_success(self):
        """Test successful Early Blight disease prediction"""
        payload = {
            'disease_type': 'Early_Blight',
            'lat': 28.6139,
            'lon': 77.2090,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert data['success'] is True
        assert 'prediction' in data
        
        prediction = data['prediction']
        
        # Validate prediction fields
        required_fields = [
            'disease_type', 'location', 'prediction_date', 'risk_score',
            'risk_level', 'temperature_score', 'humidity_score', 
            'precipitation_score', 'confidence_level', 'analysis',
            'preventive_measures'
        ]
        
        for field in required_fields:
            assert field in prediction
        
        # Validate data types and ranges
        assert prediction['disease_type'] == 'Early_Blight'
        assert 0.0 <= prediction['risk_score'] <= 100.0
        assert prediction['risk_level'] in ['Low', 'Medium', 'High']
        assert 0.0 <= prediction['temperature_score'] <= 100.0
        assert 0.0 <= prediction['humidity_score'] <= 100.0
        assert 0.0 <= prediction['precipitation_score'] <= 100.0
        assert 0.0 <= prediction['confidence_level'] <= 1.0
        
        # Validate location
        assert prediction['location']['lat'] == payload['lat']
        assert prediction['location']['lon'] == payload['lon']
        
        # Validate preventive measures
        assert isinstance(prediction['preventive_measures'], list)
        
        if prediction['preventive_measures']:
            measure = prediction['preventive_measures'][0]
            measure_fields = [
                'title', 'description', 'measure_type', 'priority',
                'estimated_cost', 'effectiveness', 'time_to_implement'
            ]
            for field in measure_fields:
                assert field in measure
    
    def test_disease_prediction_late_blight_success(self):
        """Test successful Late Blight disease prediction"""
        payload = {
            'disease_type': 'Late_Blight',
            'lat': 40.7128,
            'lon': -74.0060,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        prediction = data['prediction']
        assert prediction['disease_type'] == 'Late_Blight'
        assert prediction['risk_level'] in ['Low', 'Medium', 'High']
    
    def test_disease_prediction_healthy_plants(self):
        """Test disease prediction for healthy plants"""
        payload = {
            'disease_type': 'Healthy',
            'lat': 28.6139,
            'lon': 77.2090,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] is True
        prediction = data['prediction']
        assert prediction['disease_type'] == 'Healthy'
        assert prediction['risk_score'] == 0.0
        assert prediction['risk_level'] == 'Low'
    
    def test_disease_prediction_invalid_disease_type(self):
        """Test disease prediction with invalid disease type"""
        payload = {
            'disease_type': 'InvalidDisease',
            'lat': 28.6139,
            'lon': 77.2090,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'invalid disease type' in data['error'].lower()
    
    def test_disease_prediction_invalid_coordinates(self):
        """Test disease prediction with invalid coordinates"""
        payload = {
            'disease_type': 'Early_Blight',
            'lat': 91,  # Invalid latitude
            'lon': 77.2090,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'coordinates' in data['error'].lower()
    
    def test_disease_prediction_missing_fields(self):
        """Test disease prediction with missing required fields"""
        # Missing disease_type
        payload = {
            'lat': 28.6139,
            'lon': 77.2090,
            'weather_data': self.get_sample_weather_data()
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'missing' in data['error'].lower()
    
    def test_disease_prediction_empty_weather_data(self):
        """Test disease prediction with empty weather data"""
        payload = {
            'disease_type': 'Early_Blight',
            'lat': 28.6139,
            'lon': 77.2090,
            'weather_data': []
        }
        
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'weather data' in data['error'].lower()
    
    def test_disease_prediction_no_json_data(self):
        """Test disease prediction without JSON data"""
        response = requests.post(
            f"{self.BASE_URL}/prediction/disease-spread",
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'json' in data['error'].lower()


class TestPredictionVisualizationAPI:
    """Test prediction visualization API endpoints"""
    
    BASE_URL = "http://localhost:5000/api"
    
    def test_prediction_visualize_success(self):
        """Test successful visualization data retrieval"""
        params = {
            'lat': 28.6139,
            'lon': 77.2090
        }
        
        response = requests.get(f"{self.BASE_URL}/prediction/visualize", params=params)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert data['success'] is True
        assert 'data' in data
        
        viz_data = data['data']
        
        # Validate visualization data structure
        required_sections = ['weather_trends', 'risk_history', 'current_conditions']
        for section in required_sections:
            assert section in viz_data
        
        # Validate weather trends structure
        weather_trends = viz_data['weather_trends']
        trend_fields = ['dates', 'temperature', 'humidity', 'precipitation']
        for field in trend_fields:
            assert field in weather_trends
            assert isinstance(weather_trends[field], list)
        
        # Validate risk history structure
        risk_history = viz_data['risk_history']
        history_fields = ['dates', 'risk_scores', 'risk_levels']
        for field in history_fields:
            assert field in risk_history
            assert isinstance(risk_history[field], list)
    
    def test_prediction_visualize_missing_coordinates(self):
        """Test visualization with missing coordinates"""
        # Missing latitude
        params = {'lon': 77.2090}
        response = requests.get(f"{self.BASE_URL}/prediction/visualize", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'latitude' in data['error'].lower()
        
        # Missing longitude
        params = {'lat': 28.6139}
        response = requests.get(f"{self.BASE_URL}/prediction/visualize", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'longitude' in data['error'].lower()
    
    def test_prediction_visualize_no_data_available(self):
        """Test visualization when no data is available for location"""
        # Use coordinates with no cached data
        params = {
            'lat': 0.0,
            'lon': 0.0
        }
        
        response = requests.get(f"{self.BASE_URL}/prediction/visualize", params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        
        # Should return empty data structures
        viz_data = data['data']
        assert len(viz_data['weather_trends']['dates']) == 0
        assert len(viz_data['risk_history']['dates']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])