"""
Weather Service Module for Open-Meteo API Integration

This module handles fetching weather forecast data from Open-Meteo API
and provides caching functionality for performance optimization.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import openmeteo_requests
import requests_cache
import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherService:
    """
    Weather service for fetching and caching weather forecast data.
    
    Integrates with Open-Meteo API to fetch 3-day weather
    forecasts including temperature, humidity, and precipitation data.
    """
    
    def __init__(self, db_path: str = "weather_cache.db"):
        """
        Initialize weather service.
        
        Args:
            db_path: Path to SQLite database for caching
        """
        self.db_path = db_path
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = openmeteo_requests.Client(session = cache_session)
        self.client = retry_session
        
        self.cache_expiry_minutes = 30
        
        # Initialize database
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database with weather forecast table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create weather_forecasts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weather_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_lat REAL NOT NULL,
                    location_lon REAL NOT NULL,
                    location_name TEXT,
                    forecast_date DATE NOT NULL,
                    forecast_hour INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL,
                    precipitation REAL NOT NULL DEFAULT 0.0,
                    weather_condition TEXT,
                    weather_description TEXT,
                    wind_speed REAL,
                    created_at DATETIME NOT NULL,
                    expires_at DATETIME NOT NULL
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_location_date 
                ON weather_forecasts(location_lat, location_lon, forecast_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON weather_forecasts(expires_at)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Weather database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def geocode_location(self, location_name: str) -> Optional[Tuple[float, float, str]]:
        """
        Convert location name to coordinates using OpenStreetMap Nominatim API.
        
        Args:
            location_name: Location string (e.g., "New Delhi, India")
            
        Returns:
            Tuple of (latitude, longitude, formatted_name) or None if not found
        """
        try:
            # Use OpenStreetMap Nominatim API for geocoding (free, no API key required)
            nominatim_url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': location_name,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'PotatoDiseaseAnalyzer/1.0'  # Required by Nominatim
            }
            
            response = requests.get(nominatim_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                logger.warning(f"No location found for: {location_name}")
                return None
            
            location = data[0]
            lat = float(location['lat'])
            lon = float(location['lon'])
            
            # Format location name with country
            address = location.get('address', {})
            formatted_name = location.get('display_name', location_name)
            
            # Try to create a shorter, more readable name
            if address:
                city = address.get('city') or address.get('town') or address.get('village')
                state = address.get('state')
                country = address.get('country')
                
                if city:
                    formatted_name = city
                    if state:
                        formatted_name += f", {state}"
                    if country:
                        formatted_name += f", {country}"
            
            logger.info(f"Geocoded '{location_name}' to ({lat}, {lon}): {formatted_name}")
            return lat, lon, formatted_name
            
        except requests.RequestException as e:
            logger.error(f"Failed to geocode location '{location_name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in geocoding: {e}")
            return None
    
    def get_cached_forecast(self, lat: float, lon: float) -> Optional[List[Dict]]:
        """
        Retrieve cached weather forecast if available and not expired.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            List of forecast dictionaries or None if no valid cache
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for non-expired forecasts within 0.1 degree radius
            current_time = datetime.now()
            cursor.execute('''
                SELECT * FROM weather_forecasts 
                WHERE ABS(location_lat - ?) < 0.1 
                AND ABS(location_lon - ?) < 0.1
                AND expires_at > ?
                ORDER BY forecast_date, forecast_hour
            ''', (lat, lon, current_time))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            # Convert rows to dictionaries
            columns = [
                'id', 'location_lat', 'location_lon', 'location_name',
                'forecast_date', 'forecast_hour', 'temperature', 'humidity',
                'precipitation', 'weather_condition', 'weather_description',
                'wind_speed', 'created_at', 'expires_at'
            ]
            
            forecasts = []
            for row in rows:
                forecast = dict(zip(columns, row))
                forecasts.append(forecast)
            
            logger.info(f"Retrieved {len(forecasts)} cached forecasts for ({lat}, {lon})")
            return forecasts
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached forecast: {e}")
            return None
    
    def cache_forecast(self, lat: float, lon: float, location_name: str, forecast_data: List[Dict]):
        """
        Cache weather forecast data in SQLite database.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate  
            location_name: Human-readable location name
            forecast_data: List of forecast dictionaries from API
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now()
            expires_at = current_time + timedelta(minutes=self.cache_expiry_minutes)
            
            # Clear old forecasts for this location
            cursor.execute('''
                DELETE FROM weather_forecasts 
                WHERE ABS(location_lat - ?) < 0.1 
                AND ABS(location_lon - ?) < 0.1
            ''', (lat, lon))
            
            # Insert new forecast data
            for forecast in forecast_data:
                cursor.execute('''
                    INSERT INTO weather_forecasts (
                        location_lat, location_lon, location_name,
                        forecast_date, forecast_hour, temperature, humidity,
                        precipitation, weather_condition, weather_description,
                        wind_speed, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    lat, lon, location_name,
                    forecast['date'], forecast['hour'],
                    forecast['temperature'], forecast['humidity'],
                    forecast['precipitation'], forecast['weather_condition'],
                    forecast['weather_description'], forecast['wind_speed'],
                    current_time, expires_at
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cached {len(forecast_data)} forecasts for {location_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache forecast data: {e}")
            raise
    
    def fetch_weather_forecast(self, lat: float, lon: float, location_name: str = None) -> Dict:
        """
        Fetch 3-day weather forecast from Open-Meteo API.
        
        Args:
            lat: Latitude coordinate (-90 to 90)
            lon: Longitude coordinate (-180 to 180)
            location_name: Optional human-readable location name
            
        Returns:
            Dictionary with success status, location info, and forecast data
        """
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
        
        # Check cache first
        cached_forecast = self.get_cached_forecast(lat, lon)
        if cached_forecast:
            logger.info(f"Using cached forecast for ({lat}, {lon})")
            return {
                'success': True,
                'location': {
                    'lat': lat,
                    'lon': lon,
                    'name': cached_forecast[0]['location_name'] or location_name
                },
                'forecast': cached_forecast,
                'cached': True
            }
        
        # Fetch from Open-Meteo API
        try:
            logger.info(f"Fetching weather forecast from Open-Meteo API for ({lat}, {lon})")
            
            # Open-Meteo API URL
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m"],
                "forecast_days": 3,
                "timezone": "auto"
            }
            
            # Make request
            responses = self.client.weather_api(url, params=params)
            
            # Process first response (should be the only one)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
            hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
            hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
            hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
            hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
            
            # Create hourly dataframe
            hourly_data = {
                "date": pd.date_range(
                    start = pd.to_datetime(hourly.Time(), unit = "s"),
                    end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
                    freq = pd.Timedelta(seconds = hourly.Interval()),
                    inclusive = "left"
                )
            }
            hourly_data["temperature_2m"] = hourly_temperature_2m
            hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
            hourly_data["precipitation"] = hourly_precipitation
            hourly_data["weather_code"] = hourly_weather_code
            hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
            
            hourly_dataframe = pd.DataFrame(data = hourly_data)
            
            # Parse forecast data
            forecast_list = []
            
            # Process hourly data for next 72 hours (3 days)
            for i, row in hourly_dataframe.iterrows():
                if i >= 72:  # Limit to 72 hours
                    break
                    
                forecast_datetime = row['date']
                
                # Get weather description from weather code
                weather_code = int(row['weather_code']) if pd.notna(row['weather_code']) else 0
                weather_condition, weather_description = self._get_weather_from_code(weather_code)
                
                forecast_item = {
                    'date': forecast_datetime.date().isoformat(),
                    'hour': forecast_datetime.hour,
                    'temperature': float(row['temperature_2m']) if pd.notna(row['temperature_2m']) else 20.0,
                    'humidity': int(row['relative_humidity_2m']) if pd.notna(row['relative_humidity_2m']) else 60,
                    'precipitation': float(row['precipitation']) if pd.notna(row['precipitation']) else 0.0,
                    'weather_condition': weather_condition,
                    'weather_description': weather_description,
                    'wind_speed': float(row['wind_speed_10m']) if pd.notna(row['wind_speed_10m']) else 0.0
                }
                forecast_list.append(forecast_item)
            
            # Cache the forecast data
            if forecast_list:
                self.cache_forecast(lat, lon, location_name, forecast_list)
            
            return {
                'success': True,
                'location': {
                    'lat': lat,
                    'lon': lon,
                    'name': location_name
                },
                'forecast': forecast_list,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Open-Meteo API request failed: {e}")
            return {
                'success': False,
                'error': f"Weather service unavailable: {str(e)}",
                'location': {'lat': lat, 'lon': lon, 'name': location_name},
                'forecast': [],
                'cached': False
            }
    
    def _get_weather_from_code(self, weather_code: int) -> Tuple[str, str]:
        """
        Convert Open-Meteo weather code to condition and description.
        
        Args:
            weather_code: Open-Meteo weather code
            
        Returns:
            Tuple of (weather_condition, weather_description)
        """
        # Open-Meteo weather codes mapping
        weather_codes = {
            0: ("Clear", "Clear sky"),
            1: ("Clouds", "Mainly clear"),
            2: ("Clouds", "Partly cloudy"),
            3: ("Clouds", "Overcast"),
            45: ("Fog", "Fog"),
            48: ("Fog", "Depositing rime fog"),
            51: ("Drizzle", "Light drizzle"),
            53: ("Drizzle", "Moderate drizzle"),
            55: ("Drizzle", "Dense drizzle"),
            56: ("Drizzle", "Light freezing drizzle"),
            57: ("Drizzle", "Dense freezing drizzle"),
            61: ("Rain", "Slight rain"),
            63: ("Rain", "Moderate rain"),
            65: ("Rain", "Heavy rain"),
            66: ("Rain", "Light freezing rain"),
            67: ("Rain", "Heavy freezing rain"),
            71: ("Snow", "Slight snow fall"),
            73: ("Snow", "Moderate snow fall"),
            75: ("Snow", "Heavy snow fall"),
            77: ("Snow", "Snow grains"),
            80: ("Rain", "Slight rain showers"),
            81: ("Rain", "Moderate rain showers"),
            82: ("Rain", "Violent rain showers"),
            85: ("Snow", "Slight snow showers"),
            86: ("Snow", "Heavy snow showers"),
            95: ("Thunderstorm", "Thunderstorm"),
            96: ("Thunderstorm", "Thunderstorm with slight hail"),
            99: ("Thunderstorm", "Thunderstorm with heavy hail")
        }
        
        return weather_codes.get(weather_code, ("Unknown", "Unknown weather condition"))
    
    def cleanup_expired_cache(self):
        """Remove expired weather forecast entries from cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = datetime.now()
            cursor.execute('DELETE FROM weather_forecasts WHERE expires_at < ?', (current_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} expired weather cache entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
