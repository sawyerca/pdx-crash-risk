# ==============================================================================
# Standalone Weather Fetch Script
# ==============================================================================
# Purpose: Fetch real-time weather data for all stations and write
#          weather_cache.csv for use by the Render app. Runs as a
#          GitHub Action so requests come from Azure IPs, not Render's
#          shared IP, avoiding Open-Meteo rate limits.
#
# Run from repo root:
#   python app/fetch_weather.py
#
# Input:  data/id_lookup.csv      (relative to repo root / cwd)
# Output: weather_cache.csv       (written to cwd)
# ==============================================================================

import openmeteo_requests
import pandas as pd
from retry_requests import retry
import pytz
import logging
import requests
from urllib3.exceptions import NewConnectionError, MaxRetryError
from datetime import datetime
import gc

# ================= CONFIGURATION =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WEATHER_DTYPES = {
    'temp': 'float16', 'humidity': 'float16', 'rain': 'float16',
    'snow': 'float16', 'snow_depth': 'float16', 'cloud_cover': 'float16',
    'wind_speed': 'float16', 'wind_gusts': 'float16', 'rain_3hr': 'float16'
}

CATEGORICAL_DTYPES = {'location_id': 'int8'}

# ================= UTILITY =================

def optimize_dtypes(df):
    """Optimize DataFrame data types for memory efficiency"""

    df = df.copy()

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

    # Add timeout constants
    CONNECT_TIMEOUT = 8
    READ_TIMEOUT = 20

    def __init__(self):
        """Initialize weather client with API configuration and caching"""

        # Create session with timeout configured
        session = requests.Session()

        # Configure retry with backoff
        retry_session = retry(
            session,
            retries=2,
            backoff_factor=0.2
        )

        # Create Open-Meteo client with retry-enabled session
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = "https://api.open-meteo.com/v1/forecast"

        # Store timeout tuple for requests
        self.timeout = (self.CONNECT_TIMEOUT, self.READ_TIMEOUT)

    def fetch_all_stations(self, id_lookup):
        """Fetch weather data for all weather stations with error handling"""

        logger.info(f"Fetching weather data for {len(id_lookup)} stations...")

        # Track successful and failed station requests
        successful_data = []
        failed_stations = []

        # Process each weather station sequentially
        for idx, station in id_lookup.iterrows():
            try:
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

                # Garbage collection
                gc.collect()

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

        # Clean up intermediate data
        del successful_data
        if failed_stations:
            del fallback_data
        gc.collect()

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
                          "wind_speed_10m", "wind_gusts_10m"],
                "timezone": "America/Los_Angeles",
                "start_hour": start_time.strftime('%Y-%m-%dT%H:00'),
                "end_hour": end_time.strftime('%Y-%m-%dT%H:00'),
                "wind_speed_unit": "mph",
                "temperature_unit": "fahrenheit",
                "precipitation_unit": "inch",
                "timeout": self.timeout
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
                           "cloud_cover", "wind_speed", "wind_gusts"]

            # Map API response variables to standardized column names
            for i, var_name in enumerate(weather_vars):
                hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

            # Create station-specific weather dataframe
            df = pd.DataFrame(hourly_data)
            df['location_id'] = location_id  # Add station identifier

            return df

        # Error handling
        except requests.exceptions.ReadTimeout as e:
            raise RuntimeError(f"Read timeout for station {location_id}: {e}")
        except requests.exceptions.ConnectTimeout as e:
            raise RuntimeError(f"Connection timeout for station {location_id}: {e}")
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Timeout for station {location_id}: {e}")
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
                    'cloud_cover', 'wind_speed', 'wind_gusts']

        # Calculate regional weather averages across all successful stations
        regional_avg = successful_weather.groupby('datetime')[weather_vars].mean().reset_index()

        # Generate station-specific fallback data using vectorized assign
        fallback_dfs = [regional_avg.assign(location_id=failed_id) for failed_id in failed_station_ids]

        return pd.concat(fallback_dfs, ignore_index=True) if fallback_dfs else pd.DataFrame()

# ================= MAIN =================

if __name__ == '__main__':
    id_lookup = pd.read_csv('data/id_lookup.csv')

    weather_client = WeatherClient()
    weather_data = weather_client.fetch_all_stations(id_lookup)

    weather_data.to_csv('weather_cache.csv', index=False)
    logger.info(
        f"Wrote weather_cache.csv — {len(weather_data)} rows, "
        f"{weather_data['datetime'].nunique()} hours, "
        f"{weather_data['location_id'].nunique()} stations"
    )
