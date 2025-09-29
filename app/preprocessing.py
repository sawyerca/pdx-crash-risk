# ==============================================================================
# Real-Time Weather Data Pipeline for Crash Prediction
# ==============================================================================
# Purpose: Fetch real-time weather data and generate ML-ready features for
#          crash prediction model with temporal encoding and station assignment
# 
# Input Files:
#   - ../data/id_lookup.csv: Weather station locations and IDs
#   - ../data/street_seg.parquet: Street segments with station assignments
#   - ../data/segment_stats.parquet: Historical segment crash statistics
#
# Output:
#   - Generator yielding data chunks with weather and street features
# ==============================================================================

# ================= IMPORTS =================

import openmeteo_requests
import pandas as pd
from retry_requests import retry
import pytz
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging
import requests
from urllib3.exceptions import NewConnectionError, MaxRetryError
import psutil
from datetime import datetime
from config import PERFORMANCE_CONFIG

# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# Memory management and chunking configuration
MEMORY_THRESHOLD = PERFORMANCE_CONFIG['memory_warning_threshold']  # RAM percentage threshold for reducing chunk size
DEFAULT_CHUNK_HOURS = PERFORMANCE_CONFIG['chunk_size_hours']  # Standard number of hours per processing chunk
MIN_CHUNK_HOURS = 2      # Minimum chunk size for high memory scenarios
MAX_CHUNK_HOURS = 12     # Maximum chunk size for low memory scenarios

# Optimized data types for memory efficiency
WEATHER_DTYPES = {
    'temp': 'float16', 'humidity': 'float16', 'rain': 'float16', 
    'snow': 'float16', 'snow_depth': 'float16', 'cloud_cover': 'float16',
    'wind_speed': 'float16', 'wind_gusts': 'float16', 'rain_3hr': 'float16'
}

CATEGORICAL_DTYPES = {'is_day': 'int8', 'location_id': 'int32'}

# ================= UTILITY FUNCTIONS =================

def street_encode(street_seg_path):
    """Load street segments and one-hot encode street type classifications"""

    # Load street segment data from parquet file
    street_seg = pd.read_parquet(street_seg_path)
    
    # Define street type classifications
    street_types = [
        '1_fwy',   
        '2_hwy',   
        '3_art1',  
        '4_art2',  
        '5_art3',  
        '6_res',  
        '7_oth'    
    ]
    
    # Configure one-hot encoder for categorical street types
    encoder = OneHotEncoder(
        categories=[street_types],
        sparse_output=False,
        dtype=int,
        handle_unknown='ignore'  
    )
    
    # Transform street types into binary feature columns
    encoded = encoder.fit_transform(street_seg[['type']])
    street_cols = [f'type_{cat}' for cat in street_types]
    
    # Add encoded columns to street segment dataframe
    for i, col in enumerate(street_cols):
        street_seg[col] = encoded[:, i]
    
    # Remove original categorical column (now redundant)
    street_seg = street_seg.drop(columns=['type'])

    return street_seg

def optimize_dtypes(df):
    """Optimize DataFrame data types for memory efficiency"""

    df = df.copy()
    
    # Apply memory-optimized data types based on column characteristics
    for col in df.columns:
        if col in WEATHER_DTYPES:
            df[col] = df[col].astype(WEATHER_DTYPES[col])
        elif col in CATEGORICAL_DTYPES:
            df[col] = df[col].astype(CATEGORICAL_DTYPES[col])
        elif col.startswith(('type_', 'month_', 'day_', 'hour_')):
            df[col] = df[col].astype('int8')
    
    return df

# ================= WEATHER CLIENT =================

class WeatherClient:
    """Handles weather data fetching from Open-Meteo API"""
    
    def __init__(self):
        """Initialize weather client with API configuration and caching"""

        retry_session = retry(requests.Session(), retries=3, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = "https://api.open-meteo.com/v1/forecast"
    
    def fetch_all_stations(self, id_lookup):
        """Fetch weather data for all weather stations with error handling"""

        logger.info(f"Fetching weather data for {len(id_lookup)} stations...")
        
        # Track successful and failed station requests
        successful_data = []
        failed_stations = []
        
        # Process each weather station sequentially
        for idx, station in id_lookup.iterrows():
            try:
                logger.info(f"Fetching station {station['location_id']} ({idx+1}/{len(id_lookup)})")
                df = self.fetch_single_station(
                    station['latitude'], 
                    station['longitude'], 
                    station['location_id']
                )
                successful_data.append(df)
            except Exception as e:
                # Log failures but continue processing other stations
                failed_stations.append(station['location_id'])
                logger.warning(f"Failed station {station['location_id']}: {e}")
        
        # Ensure at least some weather data was retrieved
        if not successful_data:
            raise RuntimeError("No weather data retrieved for any stations")
        
        # Combine all successful station data
        combined_weather = pd.concat(successful_data, ignore_index=True)
        
        # Generate fallback data for failed stations using regional averages
        if failed_stations:
            logger.info(f"Creating fallback data for {len(failed_stations)} failed stations")
            fallback_data = self.create_fallback_weather(combined_weather, failed_stations)
            combined_weather = pd.concat([combined_weather, fallback_data], ignore_index=True)
        
        # Apply memory optimizations immediately after data collection
        return optimize_dtypes(combined_weather)
    
    def fetch_single_station(self, lat, lon, location_id):
        """Fetch weather forecast data for a single weather station"""

        try:
            # Define forecast time window
            now_pst = datetime.now(pytz.timezone('America/Los_Angeles'))
            start_time = now_pst - pd.Timedelta(hours=3)   # 3 hours historical 
            end_time = now_pst + pd.Timedelta(hours=25)    # 25 hours forecast 
            
            # Configure API request parameters 
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["temperature_2m", "relative_humidity_2m",
                          "rain", "snowfall", "snow_depth", "cloud_cover", 
                          "wind_speed_10m", "wind_gusts_10m", "is_day"],
                "timezone": "America/Los_Angeles",
                "start_hour": start_time.strftime('%Y-%m-%dT%H:00'),
                "end_hour": end_time.strftime('%Y-%m-%dT%H:00'),
                "wind_speed_unit": "mph",          
                "temperature_unit": "fahrenheit", 
                "precipitation_unit": "inch",   
            }
            
            # Execute API request 
            responses = self.openmeteo.weather_api(self.url, params=params)
            
            # Validate API response structure
            if not responses:
                raise RuntimeError(f"No response from API for station {location_id}")
            
            response = responses[0]
            hourly = response.Hourly()
            
            if not hourly:
                raise RuntimeError(f"No hourly data for station {location_id}")
            
            # Construct datetime index for time series data
            hourly_data = {
                "datetime": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ).tz_convert('America/Los_Angeles').tz_localize(None)  # Convert to local time
            }
            
            # Extract weather vars
            weather_vars = ["temp", "humidity", "rain", "snow", "snow_depth",
                           "cloud_cover", "wind_speed", "wind_gusts", "is_day"]
            
            # Map API response variables to standardized column names
            for i, var_name in enumerate(weather_vars):
                hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
            
            # Create station-specific weather dataframe
            df = pd.DataFrame(hourly_data)
            df['location_id'] = location_id  # Add station identifier
            
            return df
        
        # Error handling
        except (requests.exceptions.ConnectionError, NewConnectionError, MaxRetryError) as e:
            raise RuntimeError(f"Network error for station {location_id}: {e}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"API error for station {location_id}: {e}")
        except Exception as e:
            raise RuntimeError(f"Station {location_id} failed: {e}")
    
    def create_fallback_weather(self, successful_weather, failed_station_ids):
        """Create fallback weather data using regional averages for failed stations"""

        # Ensure we have baseline data to work with
        if successful_weather.empty:
            raise ValueError("Cannot create fallback: no successful weather data")
        
        # Define core weather variables for regional averaging
        weather_vars = ['temp', 'humidity', 'rain', 'snow', 'snow_depth', 
                    'cloud_cover', 'wind_speed', 'wind_gusts', 'is_day']
        
        # Calculate regional weather averages across all successful stations
        regional_avg = successful_weather.groupby('datetime')[weather_vars].mean().reset_index()
        
        # Generate station-specific fallback data using vectorized assign
        fallback_dfs = [regional_avg.assign(location_id=failed_id) for failed_id in failed_station_ids]
        
        return pd.concat(fallback_dfs, ignore_index=True) if fallback_dfs else pd.DataFrame()

# ================= FEATURE ENGINEERING =================

class FeatureEngineer:
    """Handles temporal feature encoding and rolling weather calculations"""
    
    def __init__(self):
        """Initialize feature engineering pipeline with temporal encoders"""

        # Configure timezone for consistent temporal processing
        self.timezone = pytz.timezone('America/Los_Angeles')

        # Set up one-hot encoder for temporal features (month, day of week, hour)
        self.enc = OneHotEncoder(
            categories=[
                np.arange(1, 13),  # Months: 1-12
                np.arange(1, 8),   # Days of week: 1-7 (Monday=1, Sunday=7)
                np.arange(0, 24)   # Hours: 0-23 
            ],
            sparse_output=False,
            dtype=int,
            handle_unknown='ignore'
        )
        self._encoder_fitted = False  # Track encoder training state
    
    def process_weather_features(self, weather_data):
        """Process weather data with temporal encoding and rolling calculations"""

        # Sort data for proper time series processing
        weather_data = weather_data.sort_values(['location_id', 'datetime'])
        
        # Calculate 3-hour rolling precipitation totals per station
        weather_data['rain_3hr'] = (weather_data.groupby('location_id')['rain']
                                   .rolling(window=3, min_periods=1)
                                   .sum()
                                   .reset_index(0, drop=True))
        
        # Filter to current and future time periods (exclude historical data)
        current_hour = pd.Timestamp.now(tz='America/Los_Angeles').floor('h').replace(tzinfo=None)
        weather_data = weather_data[weather_data['datetime'] >= current_hour]
        
        # Extract temporal components for categorical encoding
        weather_data['month'] = weather_data['datetime'].dt.month
        weather_data['day'] = (weather_data['datetime'].dt.weekday + 1) % 7 + 1  # Sunday = 1 logic
        weather_data['hour'] = weather_data['datetime'].dt.hour
        
        # One-hot encode temporal features for model compatibility
        if not self._encoder_fitted:
            self.enc.fit(weather_data[['month','day','hour']])
            self._encoder_fitted = True
        
        # Transform temporal categories to binary feature columns
        encoded = self.enc.transform(weather_data[['month','day','hour']])
        encoded_cols = self.enc.get_feature_names_out(['month', 'day', 'hour'])
        
        # Replace original temporal columns with encoded versions
        weather_data = weather_data.drop(columns=['month','day','hour'])
        weather_data[encoded_cols] = encoded
        
        return weather_data
    
    def validate_features(self, df):
        """Validate weather feature data for completeness and consistency"""

        # Define essential columns required for crash prediction model
        required_cols = ['datetime', 'temp', 'humidity', 'rain', 'location_id']
        
        # Check for empty dataset
        if df.empty:
            raise ValueError("Weather features DataFrame is empty!")
        
        # Verify all required columns are present
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

# ================= DATA PROCESSOR =================

class DataProcessor:
    """Handles data merging and adaptive chunking with memory management"""
    
    def calculate_optimal_chunk_size(self):
        """Calculate optimal processing chunk size based on current memory usage"""
        
        # Hardcoded 2GB container limit
        CONTAINER_LIMIT_GB = 2.0
        
        # Get current process memory usage
        import psutil
        process = psutil.Process()
        current_memory_gb = process.memory_info().rss / (1024**3)
        memory_percent = (current_memory_gb / CONTAINER_LIMIT_GB) * 100
        
        # More aggressive thresholds for 2GB limit
        if memory_percent > 75:  # 1.5 GB used
            chunk_hours = MIN_CHUNK_HOURS  # 2 hours
            logger.info(f"High memory usage ({memory_percent:.1f}%) - using minimum chunk size: {chunk_hours} hours")
        elif memory_percent > 60:  # 1.2 GB used  
            chunk_hours = DEFAULT_CHUNK_HOURS - 2  # 4 hours
            logger.info(f"Moderate memory usage ({memory_percent:.1f}%) - reducing chunk size to {chunk_hours} hours")
        else:
            chunk_hours = DEFAULT_CHUNK_HOURS  # 6 hours
        
        return chunk_hours
    
    def merge_street_weather(self, street_seg, weather_features):
        """Generator yielding merged street-weather data in chunks"""

        # Apply memory optimizations to input datasets
        street_seg = optimize_dtypes(street_seg)
        weather_features = optimize_dtypes(weather_features)
        
        # Determine temporal scope for processing
        unique_hours = sorted(weather_features['datetime'].unique())
        logger.info(f"Processing {len(unique_hours)} hours of data")
        
        # Pre-sort street segments for efficient merge operations
        street_seg_sorted = street_seg.sort_values('location_id')
        
        # Process data in adaptive chunks to manage memory usage
        i = 0
        while i < len(unique_hours):
            # Dynamically calculate chunk size based on current memory state
            chunk_hours = self.calculate_optimal_chunk_size()
            chunk_hours_list = unique_hours[i:i + chunk_hours]
            
            logger.info(f"Processing hours {i+1}-{min(i+chunk_hours, len(unique_hours))} of {len(unique_hours)}")
            
            # Extract weather data for current time chunk
            weather_chunk = weather_features[
                weather_features['datetime'].isin(chunk_hours_list)
            ].copy()
            
            # Sort weather chunk for efficient merging
            weather_chunk_sorted = weather_chunk.sort_values('location_id')
            
            # Merge street segments with weather data by station assignment
            merged_chunk = street_seg_sorted.merge(
                weather_chunk_sorted,
                on='location_id',  # Join on weather station assignment
                how='inner',       # Only keep segments with weather data
                sort=False         # Maintain existing sort order
            )
            
            # Handle missing weather data (data quality check)
            missing_weather = merged_chunk['datetime'].isna().sum()
            if missing_weather > 0:
                logger.warning(f"{missing_weather} street points missing weather - dropping")
                merged_chunk = merged_chunk.dropna(subset=['datetime'])
            
            # Remove location_id column (no longer needed after merge)
            merged_chunk = merged_chunk.drop(columns=['location_id'])
            
            # Explicit memory cleanup to prevent accumulation
            del weather_chunk, weather_chunk_sorted
            import gc 
            gc.collect()  
            
            yield merged_chunk
            i += chunk_hours

# ================= MAIN PIPELINE =================

class DataPipeline:
    """Main data pipeline orchestrating weather fetching, feature engineering, and merging"""
    
    def __init__(self):
        """Initialize pipeline with component instances"""

        self.weather_client = WeatherClient()
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
    
    def model_input(self, id_lookup, street_seg, seg_stats):
        """Generate complete model-ready input data through pipeline"""

        # Fetch real-time weather data from API
        weather_data = self.weather_client.fetch_all_stations(id_lookup)
        
        # Get temporal and rolling features
        weather_features = self.feature_engineer.process_weather_features(weather_data)
        self.feature_engineer.validate_features(weather_features)
        
        # Merge with street network and segment statistics
        for merged_chunk in self.data_processor.merge_street_weather(street_seg, weather_features):
            # Add historical segment crash statistics to complete feature set
            model_chunk = merged_chunk.merge(
                seg_stats,
                on='segment_id',  
                how='inner',     
                sort=False
            )
            yield model_chunk

# ================= TESTING =================

if __name__ == '__main__':
    """Test the complete data pipeline with sample data"""
    
    pipe = DataPipeline()
    id_lookup = pd.read_csv('../data/id_lookup.csv')
    street_seg = street_encode('../data/street_seg.parquet')
    seg_stats = pd.read_parquet('../data/segment_stats.parquet')
    test_data = pipe.model_input(id_lookup, street_seg, seg_stats)
    
    for chunk in test_data:
        print(f"Processed chunk with {len(chunk)} rows")
        break