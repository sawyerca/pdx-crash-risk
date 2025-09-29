# ==============================================================================
# Dashboard Callback Functions for Traffic Crash Risk Prediction
# ==============================================================================
# Purpose: Orchestrate interactive dashboard functionality including real-time
#          data updates, map visualization, time controls, and user notifications
#          for the Portland crash risk prediction system
# 
# Input Files:
#   - ../models/crash_model.pkl: Trained XGBoost crash prediction model
#   - ../data/street_seg.parquet: Street segment network with weather stations
#   - ../data/id_lookup.csv: Weather station location mappings
#   - ../data/segment_stats.parquet: Historical segment crash statistics
#
# Output:
#   - Interactive web dashboard with real-time crash risk visualization
#   - Lightweight hourly data with on-demand map generation
# ==============================================================================

# ================= IMPORTS =================

from dash import Input, Output
import pandas as pd
import logging
from preprocessing import DataPipeline, street_encode
from predictor import load_model, CrashPredictor
import psutil
from shapely import wkt
from datetime import datetime
import pytz
import numpy as np
import time
from config import get_deck_color, MAP_CONFIG, PERFORMANCE_CONFIG
from bg_updater import BackgroundUpdater


# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# ================= DATA MANAGEMENT CLASS =================

class DataManager:
    """Handles data loading, caching, and basic data operations"""
    
    def __init__(self):
        """Initialize data manager with core datasets"""
        
        logger.info("Loading core datasets...")
        
        # Load trained crash prediction model with calibrator and metadata
        self.model_artifact = load_model('../models/crash_model.pkl')
        
        # Load street network segments with encoded categorical features
        self.street_points = street_encode('../data/street_seg.parquet')
        
        # Load weather station location and identifier mappings
        self.id_lookup = pd.read_csv('../data/id_lookup.csv')

        # Load historical segment-level crash statistics for model features
        self.seg_stats = pd.read_parquet('../data/segment_stats.parquet')
        
        # Initialize data caches
        self.cached_predictions = None  # Temporary: deleted after filtering
        self.cached_sample = None       # Temporary: deleted after hourly data extraction
        self.cached_available_hours = None  # Permanent: lightweight hour list for UI
        self.parsed_geometry = None     # Permanent: segment_id → coordinate arrays
        self.hourly_data = None         # Permanent: lightweight hourly risk data
        
        logger.info("Data manager initialized successfully")
    
    def get_model_artifact(self):
        """Return the loaded model artifact"""

        return self.model_artifact
    
    def get_street_points(self):
        """Return street segment data"""

        return self.street_points
    
    def get_id_lookup(self):
        """Return weather station lookup data"""

        return self.id_lookup
    
    def get_segment_stats(self):
        """Return segment statistics data"""

        return self.seg_stats
    
    def cache_predictions(self, predictions):
        """Cache full prediction dataset (temporary storage)"""

        self.cached_predictions = predictions
    
    def cache_sample(self, sample):
        """Cache filtered sample dataset (temporary storage)"""

        self.cached_sample = sample
    
    def cache_available_hours(self, hours):
        """Cache list of available hours for UI components"""

        self.cached_available_hours = hours
    
    def cache_parsed_geometry(self, geometry_dict):
        """Cache parsed geometry lookup for fast map generation"""

        self.parsed_geometry = geometry_dict
    
    def cache_hourly_data(self, hourly_dict):
        """Cache lightweight hourly risk data for on-demand map generation"""

        self.hourly_data = hourly_dict
    
    def get_cached_predictions(self):
        """Return cached predictions if available"""

        return self.cached_predictions
    
    def get_cached_sample(self):
        """Return cached sample if available"""

        return self.cached_sample
    
    def get_cached_available_hours(self):
        """Return cached list of available hours"""

        return self.cached_available_hours
    
    def get_parsed_geometry(self):
        """Return parsed geometry lookup dict"""

        return self.parsed_geometry
    
    def get_hourly_data(self):
        """Return hourly risk data dict"""

        return self.hourly_data
    
    def clear_cache(self):
        """Clear all cached data to force fresh generation"""

        self.cached_predictions = None
        self.cached_sample = None
        self.cached_available_hours = None
        self.parsed_geometry = None
        self.hourly_data = None

# ================= PREDICTION ENGINE CLASS =================

class PredictionEngine:
    """Handles crash prediction generation and probability processing"""
    
    def __init__(self, data_manager):
        """Initialize prediction engine with data manager reference"""
        
        self.data_manager = data_manager
        
        # Initialize real-time weather data pipeline
        self.pipeline = DataPipeline()
        
        # Initialize crash prediction engine with loaded model
        self.artifact = self.data_manager.get_model_artifact()

        self.predictor = CrashPredictor(self.artifact)
        self.cutoff = self.artifact['cutoff']
        self.filtered_probs_ref = self.artifact['filtered_probs_ref']

        # Build percentile lookup table 
        self.percentile_thresholds = self.build_percentile_lookup()

        logger.info("Prediction engine initialized")
    
    def build_percentile_lookup(self):
        """Build percentile threshold lookup table from reference data"""

        # Calculate the probability value for each percentile (0-100)
        percentiles = np.arange(0, 101)
        thresholds = np.percentile(self.filtered_probs_ref, percentiles)

        return thresholds
    
    def generate_predictions(self):
        """Generate crash predictions for all segments and time periods"""

        # Check cache 
        cached_predictions = self.data_manager.get_cached_predictions()
        if cached_predictions is None:
            logger.info("Generating new predictions...")
            
            # Create chunked data generator 
            data_generator = self.pipeline.model_input(
                self.data_manager.get_id_lookup(), 
                self.data_manager.get_street_points(),
                self.data_manager.get_segment_stats()
            )
            
            # Execute chunked prediction pipeline
            predictions = self.predictor.predict(data_generator)
            
            # Cache results for future use
            self.data_manager.cache_predictions(predictions)
            
            # Log prediction statistics for monitoring
            logger.info(f"Probability range: {predictions['crash_probability'].min():.6f} to {predictions['crash_probability'].max():.6f}")
            logger.info(f"Average probability: {predictions['crash_probability'].mean():.8f}")
            
            return predictions
        
        return cached_predictions
    
    def score_risk(self, sample):
        """Assign risk score based on pre-computed percentile lookup"""
        
        # Fast lookup using searchsorted 
        risk_scores = np.searchsorted(self.percentile_thresholds, sample['crash_probability'], side='right') - 1
        
        # Add risk scores to the sample
        sample['risk_score'] = risk_scores.astype(int)
        
        return sample

    def filter_predictions(self):
        """Filter predictions to focus on segments with elevated crash risk"""

        # Check cache first
        cached_sample = self.data_manager.get_cached_sample()
        
        if cached_sample is None:
            predictions = self.generate_predictions()
            
            # Filter out segments with low risk (for speed)
            logger.info(f"Filtering for predictions with prob > {self.cutoff:6f}")
            sample = predictions[
                predictions['crash_probability'] >= self.cutoff # Filter out bottom portion (based on knee point)
            ].copy()
            logger.info(f"Filtered to {len(sample)} segments")

            logger.info(f"Scoring segments...")
            sample = self.score_risk(sample)

            self.data_manager.cache_sample(sample)
            return sample
        
        return cached_sample

# ================= MAP RENDERER CLASS =================

class MapRenderer:
    """Handles map visualization creation and management"""
    
    def __init__(self, prediction_engine):
        """Initialize map renderer with prediction engine reference"""
        
        self.prediction_engine = prediction_engine
        self.data_manager = prediction_engine.data_manager
        
    def parse_geometry(self, filtered_predictions):
        """Parse WKT geometry strings once and cache coordinate arrays by segment_id"""
        
        logger.info("Parsing geometry strings to coordinate arrays...")
        start_time = time.time()
        
        geometry_dict = {}
        
        # Get unique segments to avoid parsing duplicates across time periods
        unique_segments = filtered_predictions[['segment_id', 'geometry', 'full_name']].drop_duplicates(subset='segment_id')
        
        for _, row in unique_segments.iterrows():
            geom = wkt.loads(row['geometry'])
            if geom.geom_type == 'LineString':
                # Store coordinate array and metadata by segment_id
                coords = [[point[0], point[1]] for point in geom.coords]
                geometry_dict[row['segment_id']] = {
                    'coords': coords,
                    'full_name': row['full_name']
                }
        
        elapsed = time.time() - start_time
        logger.info(f"Parsed {len(geometry_dict)} unique segments in {elapsed:.3f} seconds")
        
        return geometry_dict
    
    def extract_hourly_data(self, filtered_predictions):
        """Extract lightweight hourly risk data indexed by hour"""
        
        logger.info("Extracting hourly risk data...")
        start_time = time.time()
        
        hourly_dict = {}
        available_hours = sorted(filtered_predictions['datetime'].unique())
        
        for i, hour in enumerate(available_hours):
            hour_data = filtered_predictions[filtered_predictions['datetime'] == hour]
            
            # Store only essential data: list of (segment_id, risk_score) tuples
            hourly_dict[i] = hour_data[['segment_id', 'risk_score']].to_dict('records')
        
        elapsed = time.time() - start_time
        logger.info(f"Extracted {len(hourly_dict)} hours of data in {elapsed:.3f} seconds")
        
        return hourly_dict, available_hours
    
    def generate_map(self, hour_index):
        """Generate deck.gl map on-demand by combining cached geometry with hourly data"""
        
        # Retrieve cached data
        geometry_dict = self.data_manager.get_parsed_geometry()
        hourly_dict = self.data_manager.get_hourly_data()
        
        # Validate data availability
        if not geometry_dict or not hourly_dict:
            logger.error("Missing cached data for map generation")
            return {}
        
        if hour_index not in hourly_dict:
            logger.error(f"Hour index {hour_index} not found in hourly data")
            return {}
        
        # Get segments for this hour
        hour_segments = hourly_dict[hour_index]
        
        # Build deck.gl data by combining geometry with risk scores
        deck_data = []
        for segment_info in hour_segments:
            segment_id = segment_info['segment_id']
            risk_score = segment_info['risk_score']
            
            # Lookup pre-parsed geometry
            if segment_id in geometry_dict:
                geom_info = geometry_dict[segment_id]
                color = get_deck_color(risk_score)
                
                deck_data.append({
                    'path': geom_info['coords'],
                    'probability_text': risk_score,
                    'full_name': geom_info['full_name'],
                    'color': color,
                    'width': MAP_CONFIG['path_width']
                })
        
        # Return complete deck.gl map configuration
        return {
            "initialViewState": {
                "latitude": MAP_CONFIG['center']['lat'],
                "longitude": MAP_CONFIG['center']['lon'], 
                "zoom": MAP_CONFIG['initial_zoom'],
                "pitch": 0,
                "bearing": 0
            },
            "controller": True,
            "layers": [{
                "@@type": "PathLayer",
                "id": "crash-risk-layer",
                "data": deck_data,
                "pickable": True,
                "widthScale": 1,
                "widthMinPixels": MAP_CONFIG['min_path_width'],
                "getPath": "@@=path",
                "getColor": "@@=color",
                "getWidth": "@@=width",
                "autoHighlight": True,
                "highlightColor": MAP_CONFIG['highlight_color']
            }],
            "mapStyle": MAP_CONFIG['map_style']
        }
    
    def prepare_data(self):
        """Prepare lightweight data structures for fast on-demand map generation"""
        
        # Log initial system memory state
        ram_start = psutil.virtual_memory()
        logger.info(f"Starting data preparation - RAM: {ram_start.percent:.1f}% used, {ram_start.available / 1024**3:.1f} GB available")
        
        # Get filtered prediction dataset
        filtered_predictions = self.prediction_engine.filter_predictions()
        
        # Parse geometry once and cache
        geometry_dict = self.parse_geometry(filtered_predictions)
        self.data_manager.cache_parsed_geometry(geometry_dict)
        
        # Extract lightweight hourly data
        hourly_dict, available_hours = self.extract_hourly_data(filtered_predictions)
        self.data_manager.cache_hourly_data(hourly_dict)
        self.data_manager.cache_available_hours(available_hours)
        
        logger.info(f"Cached {len(available_hours)} available hours for UI")
        
        # Delete prediction caches to free memory
        logger.info("Data prepared - clearing prediction caches to free memory")
        self.data_manager.cached_predictions = None
        self.data_manager.cached_sample = None
        
        # Force garbage collection to reclaim memory immediately
        import gc
        gc.collect()
        
        # Log final memory state
        ram_final = psutil.virtual_memory()
        logger.info(f"Data preparation complete!")
        logger.info(f"Final RAM usage: {ram_final.percent:.1f}% used, {ram_final.available / 1024**3:.1f} GB available")
        logger.info(f"Memory freed: {(ram_start.percent - ram_final.percent):.1f}% ({(ram_start.used - ram_final.used) / 1024**3:.1f} GB)")

    def get_available_hours(self):
        """Extract sorted list of available prediction hours from cached hours list"""

        cached_hours = self.data_manager.get_cached_available_hours()
        if cached_hours is not None:
            return cached_hours
        return []

# ================= MAIN APPLICATION CLASS =================

class CrashRiskApp:
    """Main application orchestrator coordinating all dashboard components"""
    
    def __init__(self):
        """Initialize the application with all required components and data"""
        
        logger.info("Initializing...")
        
        # Initialize core components in dependency order
        self.data_manager = DataManager()
        self.prediction_engine = PredictionEngine(self.data_manager)
        self.map_renderer = MapRenderer(self.prediction_engine)
        
        # Prepare lightweight data structures for fast map generation
        self.map_renderer.prepare_data()

        # Initialize and start background data refresh system
        self.background_updater = BackgroundUpdater(self)
        self.background_updater.start()
        
        logger.info("Initialized successfully")
    
    def generate_predictions(self):
        """Delegate prediction generation to prediction engine"""

        return self.prediction_engine.generate_predictions()
    
    def filter_predictions(self):
        """Delegate prediction filtering to prediction engine"""

        return self.prediction_engine.filter_predictions()
    
    def generate_map(self, hour_index):
        """Delegate fast on-demand map generation to map renderer"""

        return self.map_renderer.generate_map(hour_index)
    
    def get_available_hours(self):
        """Delegate hour retrieval to map renderer"""

        return self.map_renderer.get_available_hours()
    
    @property
    def cached_predictions(self):
        """Property accessor for cached predictions"""

        return self.data_manager.get_cached_predictions()
    
    @cached_predictions.setter
    def cached_predictions(self, value):
        """Property setter for cached predictions"""

        self.data_manager.cache_predictions(value)
    
    @property
    def cached_sample(self):
        """Property accessor for cached sample"""

        return self.data_manager.get_cached_sample()
    
    @cached_sample.setter
    def cached_sample(self, value):
        """Property setter for cached sample"""

        self.data_manager.cache_sample(value)
    
    @property
    def parsed_geometry(self):
        """Property accessor for parsed geometry"""

        return self.data_manager.get_parsed_geometry()
    
    @parsed_geometry.setter
    def parsed_geometry(self, value):
        """Property setter for parsed geometry"""

        self.data_manager.cache_parsed_geometry(value)
    
    @property
    def hourly_data(self):
        """Property accessor for hourly data"""

        return self.data_manager.get_hourly_data()
    
    @hourly_data.setter
    def hourly_data(self, value):
        """Property setter for hourly data"""

        self.data_manager.cache_hourly_data(value)
    
    def create_hour_marks(self):
        """Generate slider marks for hour selection interface"""
        
        hours = self.get_available_hours()
        if not hours:
            return {}
        
        # Skip the first hour per application logic
        hours = hours[1:]  
        marks = {}
        
        # Create formatted time labels for each available hour
        for i in range(0, len(hours), 1):
            time_str = pd.to_datetime(hours[i]).strftime('%-I %p')
            marks[i] = {
                'label': time_str,
                'style': {
                    'color': '#ffffff', 
                    'fontSize': '11px',
                    'whiteSpace': 'nowrap'  
                }
            }
        
        return marks

# Initialize the main application instance for callback registration
crash_app = CrashRiskApp()

# ================= CALLBACK REGISTRATION =================

def register_callbacks(app):
    """Register all dashboard callbacks for interactive functionality"""
    
    @app.callback(
        [Output('hour-slider', 'marks'),
        Output('hour-slider', 'max'),
        Output('hour-slider', 'value')], 
        [Input('hour-slider', 'id')] 
    )
    def update_slider_marks(_):
        """Configure time slider marks, range, and initial value"""
        
        # Generate formatted time marks for slider
        marks = crash_app.create_hour_marks()
        
        # Calculate maximum slider value accounting for skipped first hour
        available_hours = crash_app.get_available_hours()
        max_value = len(available_hours) - 2 if len(available_hours) > 1 else 0
        
        return marks, max_value, 0
    
    @app.callback(
        Output('crash-heatmap', 'data'),
        [Input('hour-slider', 'value')]
    )
    def update_heatmap(hour_index):
        """Update map visualization based on selected hour using fast on-demand generation"""
        return crash_app.generate_map(hour_index)
   
    @app.callback(
        Output('refresh-notification', 'style'),
        [Input('refresh-check-interval', 'n_intervals'),
        Input('page-load-tracker', 'children')]
    )
    def check_for_refresh_needed(n_intervals, user_load_timestamp):
        """Monitor for updated data and show refresh notification when needed"""
        
        # Parse and store user page load time for comparison
        if user_load_timestamp:
            try:
                crash_app.user_load_time = datetime.fromisoformat(user_load_timestamp.replace('Z', '+00:00'))
            except:
                crash_app.user_load_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        
        # Check background updater status for new data availability
        status_info = crash_app.background_updater.get_status_info()
        
        if (status_info['status'] == 'ready' and 
            status_info['last_update'] and 
            hasattr(crash_app, 'user_load_time')):

            # Show notification if data was updated after user loaded page
            if status_info['last_update'] > crash_app.user_load_time:
                return {
                    'position': 'fixed',
                    'bottom': '150px',
                    'right': '20px',
                    'backgroundColor': '#475569',
                    'color': 'white',
                    'padding': '10px 15px',
                    'borderRadius': '8px',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'zIndex': 2000,
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.3)',
                    'animation': 'fadeIn 0.5s ease-in'
                }
        
        # Hide notification when no new data available
        return {'display': 'none'}
    
    @app.callback(
        Output('page-load-tracker', 'children'),
        [Input('page-load-tracker', 'id')]
    )
    def track_page_load(_):
        """Track user page load time for data freshness comparison"""
        
        from datetime import datetime
        import pytz
        
        # Record current time as user load timestamp
        crash_app.user_load_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        
        return str(crash_app.user_load_time)

    @app.callback(
        Output('selected-datetime-display', 'children'),
        [Input('hour-slider', 'value')]
    )
    def update_selected_datetime(hour_index):
        """Update header display with selected date and time information"""
        
        try:
            # Get available hours and handle edge cases
            hours = crash_app.get_available_hours()
            
            if not hours or hour_index is None:
                return "Loading predictions..."
            
            # Account for skipped first hour in index calculation
            actual_index = hour_index + 1
            
            if actual_index >= len(hours):
                return "Invalid time selection"
            
            # Extract selected hour and format for display
            selected_hour = hours[actual_index]
            selected_dt = pd.to_datetime(selected_hour)
            
            # Create friendly time range and date formatting
            start_time = selected_dt.strftime('%-I %p')
            end_time = (selected_dt + pd.Timedelta(hours=1)).strftime('%-I %p')
            date_part = selected_dt.strftime('%A, %B %d, %Y')
            
            return f"Showing forecasted risks for {date_part}: {start_time} to {end_time}"
            
        except Exception as e:
            logger.error(f"Error updating datetime display: {e}")
            return "Error loading time"