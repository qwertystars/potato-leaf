"""
Unit tests for Weather Service
Tests the weather service functionality in isolation
"""

import pytest
import os
import tempfile
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from weather_service import WeatherService


class TestWeatherService:
    """Unit tests for WeatherService class"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Mock API key
        self.api_key = "test_api_key"
        
        # Create weather service instance
        self.weather_service = WeatherService(
            api_key=self.api_key,
            db_path=self.temp_db_path
        )
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Remove temporary database
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_init_with_api_key(self):
        """Test WeatherService initialization with API key"""
        service = WeatherService(api_key="test_key", db_path=self.temp_db_path)
        assert service.api_key == "test_key"
        assert service.db_path == self.temp_db_path
        assert service.cache_expiry_minutes == 30
    
    def test_init_without_api_key_raises_error(self):
        """Test WeatherService initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenWeatherMap API key required"):
                WeatherService(db_path=self.temp_db_path)
    
    def test_init_with_env_api_key(self):
        """Test WeatherService initialization with environment API key"""
        with patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'env_api_key'}):
            service = WeatherService(db_path=self.temp_db_path)
            assert service.api_key == 'env_api_key'
    
    def test_database_initialization(self):
        """Test database is properly initialized"""
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Check weather_forecasts table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='weather_forecasts'
        """)
        assert cursor.fetchone() is not None
        
        # Check table structure
        cursor.execute("PRAGMA table_info(weather_forecasts)")
        columns = [row[1] for row in cursor.fetchall()]
        
        expected_columns = [
            'id', 'location_lat', 'location_lon', 'location_name',
            'forecast_date', 'forecast_hour', 'temperature', 'humidity',
            'precipitation', 'weather_condition', 'weather_description',
            'wind_speed', 'created_at', 'expires_at'
        ]
        
        for column in expected_columns:
            assert column in columns
        
        conn.close()
    
    @patch('weather_service.requests.get')
    def test_geocode_location_success(self, mock_get):
        """Test successful location geocoding"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'name': 'New Delhi',
                'lat': 28.6139,
                'lon': 77.2090,
                'country': 'IN',
                'state': 'Delhi'
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test geocoding
        result = self.weather_service.geocode_location("New Delhi, India")
        
        assert result is not None
        lat, lon, formatted_name = result
        assert lat == 28.6139
        assert lon == 77.2090
        assert "New Delhi" in formatted_name
        assert "Delhi" in formatted_name
        assert "IN" in formatted_name
        
        # Verify API call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert 'q=New+Delhi%2C+India' in args[0] or kwargs['params']['q'] == 'New Delhi, India'
    
    @patch('weather_service.requests.get')
    def test_geocode_location_not_found(self, mock_get):
        """Test geocoding when location is not found"""
        # Mock empty API response
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.weather_service.geocode_location("NonExistentCity")
        assert result is None
    
    @patch('weather_service.requests.get')
    def test_geocode_location_api_error(self, mock_get):
        """Test geocoding when API request fails"""
        # Mock API error
        mock_get.side_effect = Exception("API Error")
        
        result = self.weather_service.geocode_location("New Delhi")
        assert result is None
    
    def test_cache_and_retrieve_forecast(self):
        """Test caching and retrieving weather forecast"""
        # Sample forecast data
        forecast_data = [
            {
                'date': '2024-01-01',
                'hour': 12,
                'temperature': 25.5,
                'humidity': 80.0,
                'precipitation': 2.5,
                'weather_condition': 'Rain',
                'weather_description': 'light rain',
                'wind_speed': 3.5
            },
            {
                'date': '2024-01-01',
                'hour': 15,
                'temperature': 26.0,
                'humidity': 75.0,
                'precipitation': 1.0,
                'weather_condition': 'Clouds',
                'weather_description': 'few clouds',
                'wind_speed': 4.0
            }
        ]
        
        lat, lon = 28.6139, 77.2090
        location_name = "New Delhi, India"
        
        # Cache forecast data
        self.weather_service.cache_forecast(lat, lon, location_name, forecast_data)
        
        # Retrieve cached data
        cached_data = self.weather_service.get_cached_forecast(lat, lon)
        
        assert cached_data is not None
        assert len(cached_data) == 2
        
        # Verify data integrity
        assert cached_data[0]['temperature'] == 25.5
        assert cached_data[0]['humidity'] == 80.0
        assert cached_data[1]['temperature'] == 26.0
        assert cached_data[1]['humidity'] == 75.0
    
    def test_get_cached_forecast_expired(self):
        """Test that expired cached forecasts are not returned"""
        # Create expired forecast data
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        expired_time = datetime.now() - timedelta(hours=1)
        
        cursor.execute('''
            INSERT INTO weather_forecasts (
                location_lat, location_lon, location_name,
                forecast_date, forecast_hour, temperature, humidity,
                precipitation, weather_condition, weather_description,
                wind_speed, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            28.6139, 77.2090, "New Delhi",
            '2024-01-01', 12, 25.0, 80.0, 0.0,
            'Clear', 'clear sky', 2.5,
            expired_time, expired_time
        ))
        
        conn.commit()
        conn.close()
        
        # Try to retrieve expired data
        cached_data = self.weather_service.get_cached_forecast(28.6139, 77.2090)
        assert cached_data is None or len(cached_data) == 0
    
    def test_get_cached_forecast_location_tolerance(self):
        """Test location tolerance for cached data retrieval"""
        # Cache data at specific coordinates
        forecast_data = [{
            'date': '2024-01-01',
            'hour': 12,
            'temperature': 25.0,
            'humidity': 80.0,
            'precipitation': 0.0,
            'weather_condition': 'Clear',
            'weather_description': 'clear sky',
            'wind_speed': 2.5
        }]
        
        exact_lat, exact_lon = 28.6139, 77.2090
        self.weather_service.cache_forecast(exact_lat, exact_lon, "Test Location", forecast_data)
        
        # Try to retrieve with slightly different coordinates (within tolerance)
        nearby_lat, nearby_lon = 28.6140, 77.2091
        cached_data = self.weather_service.get_cached_forecast(nearby_lat, nearby_lon)
        assert cached_data is not None
        assert len(cached_data) == 1
        
        # Try to retrieve with coordinates outside tolerance (>0.1 degrees away)
        far_lat, far_lon = 28.8000, 77.4000  # More than 0.1 degrees away
        cached_data = self.weather_service.get_cached_forecast(far_lat, far_lon)
        assert cached_data is None or len(cached_data) == 0
    
    @patch('weather_service.requests.get')
    def test_fetch_weather_forecast_success(self, mock_get):
        """Test successful weather forecast fetching"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'hourly': [
                {
                    'dt': int(datetime.now().timestamp()),
                    'temp': 25.5,
                    'humidity': 80,
                    'wind_speed': 3.5,
                    'weather': [{'main': 'Rain', 'description': 'light rain'}],
                    'rain': {'1h': 2.5}
                },
                {
                    'dt': int((datetime.now() + timedelta(hours=1)).timestamp()),
                    'temp': 26.0,
                    'humidity': 75,
                    'wind_speed': 4.0,
                    'weather': [{'main': 'Clouds', 'description': 'few clouds'}]
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test weather forecast fetching
        result = self.weather_service.fetch_weather_forecast(
            28.6139, 77.2090, "New Delhi, India"
        )
        
        assert result['success'] is True
        assert 'location' in result
        assert 'forecast' in result
        assert result['cached'] is False
        
        # Verify location data
        assert result['location']['lat'] == 28.6139
        assert result['location']['lon'] == 77.2090
        assert result['location']['name'] == "New Delhi, India"
        
        # Verify forecast data
        assert len(result['forecast']) == 2
        assert result['forecast'][0]['temperature'] == 25.5
        assert result['forecast'][0]['precipitation'] == 2.5
        assert result['forecast'][1]['temperature'] == 26.0
        assert result['forecast'][1]['precipitation'] == 0.0  # No rain data
    
    def test_fetch_weather_forecast_invalid_coordinates(self):
        """Test weather forecast with invalid coordinates"""
        # Test invalid latitude
        with pytest.raises(ValueError, match="Invalid latitude"):
            self.weather_service.fetch_weather_forecast(91, 77.2090)
        
        # Test invalid longitude
        with pytest.raises(ValueError, match="Invalid longitude"):
            self.weather_service.fetch_weather_forecast(28.6139, 181)
    
    @patch('weather_service.requests.get')
    def test_fetch_weather_forecast_api_error(self, mock_get):
        """Test weather forecast when API request fails"""
        # Mock API error
        mock_get.side_effect = Exception("API Error")
        
        result = self.weather_service.fetch_weather_forecast(28.6139, 77.2090)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['forecast'] == []
    
    def test_cleanup_expired_cache(self):
        """Test cleanup of expired cache entries"""
        # Add some expired and valid cache entries
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now()
        expired_time = current_time - timedelta(hours=1)
        future_time = current_time + timedelta(hours=1)
        
        # Add expired entry
        cursor.execute('''
            INSERT INTO weather_forecasts (
                location_lat, location_lon, location_name,
                forecast_date, forecast_hour, temperature, humidity,
                precipitation, weather_condition, weather_description,
                wind_speed, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            28.6139, 77.2090, "Test Location",
            '2024-01-01', 12, 25.0, 80.0, 0.0,
            'Clear', 'clear sky', 2.5,
            expired_time, expired_time
        ))
        
        # Add valid entry
        cursor.execute('''
            INSERT INTO weather_forecasts (
                location_lat, location_lon, location_name,
                forecast_date, forecast_hour, temperature, humidity,
                precipitation, weather_condition, weather_description,
                wind_speed, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            28.6139, 77.2090, "Test Location",
            '2024-01-01', 15, 26.0, 75.0, 0.0,
            'Clear', 'clear sky', 3.0,
            current_time, future_time
        ))
        
        conn.commit()
        
        # Count entries before cleanup
        cursor.execute('SELECT COUNT(*) FROM weather_forecasts')
        count_before = cursor.fetchone()[0]
        assert count_before == 2
        
        conn.close()
        
        # Run cleanup
        self.weather_service.cleanup_expired_cache()
        
        # Count entries after cleanup
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM weather_forecasts')
        count_after = cursor.fetchone()[0]
        conn.close()
        
        # Should have removed only the expired entry
        assert count_after == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
