"""
Contract tests for Weather Forecast API
Tests the API endpoints according to the OpenAPI specification
"""

import pytest
import requests
import json
from datetime import datetime


class TestWeatherForecastAPI:
    """Test weather forecast API endpoints"""
    
    BASE_URL = "http://localhost:5000/api"
    
    def test_weather_forecast_success(self):
        """Test successful weather forecast retrieval"""
        # Test coordinates for New Delhi
        params = {
            'lat': 28.6139,
            'lon': 77.2090,
            'location_name': 'New Delhi, India'
        }
        
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert data['success'] is True
        assert 'location' in data
        assert 'forecast' in data
        assert 'cached' in data
        
        # Validate location data
        location = data['location']
        assert location['lat'] == params['lat']
        assert location['lon'] == params['lon']
        assert location['name'] == params['location_name']
        
        # Validate forecast data
        forecast = data['forecast']
        assert isinstance(forecast, list)
        assert len(forecast) > 0
        
        # Validate forecast item structure
        for item in forecast[:3]:  # Check first 3 items
            assert 'date' in item
            assert 'hour' in item
            assert 'temperature' in item
            assert 'humidity' in item
            assert 'precipitation' in item
            assert 'weather_condition' in item
            assert 'weather_description' in item
            assert 'wind_speed' in item
            
            # Validate data types and ranges
            assert isinstance(item['temperature'], (int, float))
            assert 0 <= item['humidity'] <= 100
            assert item['precipitation'] >= 0
            assert 0 <= item['hour'] <= 23
    
    def test_weather_forecast_invalid_coordinates(self):
        """Test weather forecast with invalid coordinates"""
        # Test invalid latitude
        params = {'lat': 91, 'lon': 0}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'latitude' in data['error'].lower()
        
        # Test invalid longitude
        params = {'lat': 0, 'lon': 181}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'longitude' in data['error'].lower()
    
    def test_weather_forecast_missing_parameters(self):
        """Test weather forecast with missing required parameters"""
        # Missing latitude
        params = {'lon': 77.2090}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'latitude' in data['error'].lower()
        
        # Missing longitude
        params = {'lat': 28.6139}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'longitude' in data['error'].lower()
    
    def test_weather_geocode_success(self):
        """Test successful location geocoding"""
        params = {'location': 'New Delhi, India'}
        response = requests.get(f"{self.BASE_URL}/weather/geocode", params=params)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert data['success'] is True
        assert 'location' in data
        
        # Validate location data
        location = data['location']
        assert 'lat' in location
        assert 'lon' in location
        assert 'name' in location
        assert 'query' in location
        
        # Validate coordinate ranges
        assert -90 <= location['lat'] <= 90
        assert -180 <= location['lon'] <= 180
        assert location['query'] == params['location']
    
    def test_weather_geocode_location_not_found(self):
        """Test geocoding with non-existent location"""
        params = {'location': 'NonExistentCity12345XYZ'}
        response = requests.get(f"{self.BASE_URL}/weather/geocode", params=params)
        
        assert response.status_code == 404
        data = response.json()
        assert data['success'] is False
        assert 'not found' in data['error'].lower()
    
    def test_weather_geocode_missing_location(self):
        """Test geocoding with missing location parameter"""
        response = requests.get(f"{self.BASE_URL}/weather/geocode")
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'location' in data['error'].lower()
    
    def test_weather_geocode_long_location_name(self):
        """Test geocoding with overly long location name"""
        long_location = 'A' * 201  # Exceeds 200 character limit
        params = {'location': long_location}
        response = requests.get(f"{self.BASE_URL}/weather/geocode", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
        assert 'too long' in data['error'].lower()
    
    def test_weather_forecast_caching(self):
        """Test weather forecast caching behavior"""
        params = {
            'lat': 28.6139,
            'lon': 77.2090,
            'location_name': 'New Delhi, India'
        }
        
        # First request
        response1 = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request (should be cached)
        response2 = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Second request should indicate cached data
        assert data2['cached'] is True
        
        # Forecast data should be identical
        assert len(data1['forecast']) == len(data2['forecast'])


class TestWeatherAPIErrorHandling:
    """Test error handling scenarios"""
    
    BASE_URL = "http://localhost:5000/api"
    
    @pytest.mark.skip(reason="Requires API service to be down")
    def test_weather_service_unavailable(self):
        """Test behavior when weather service is unavailable"""
        # This test would require mocking or temporarily disabling the service
        params = {'lat': 28.6139, 'lon': 77.2090}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 503
        data = response.json()
        assert data['success'] is False
        assert 'unavailable' in data['error'].lower()
    
    def test_api_response_format_consistency(self):
        """Test that API responses follow consistent format"""
        # Test successful response format
        params = {'lat': 28.6139, 'lon': 77.2090}
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ['success', 'location', 'forecast', 'cached']
            for field in required_fields:
                assert field in data
        
        # Test error response format
        params = {'lat': 91, 'lon': 0}  # Invalid coordinates
        response = requests.get(f"{self.BASE_URL}/weather/forecast", params=params)
        
        assert response.status_code == 400
        data = response.json()
        assert 'success' in data
        assert data['success'] is False
        assert 'error' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])