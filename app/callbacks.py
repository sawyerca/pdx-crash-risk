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
#   - Cached prediction data and pre-generated map layers
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
        self.cached_predictions = None  
        self.cached_sample = None       
        
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
        """Cache full prediction dataset"""

        self.cached_predictions = predictions
    
    def cache_sample(self, sample):
        """Cache filtered sample dataset"""

        self.cached_sample = sample
    
    def get_cached_predictions(self):
        """Return cached predictions if available"""

        return self.cached_predictions
    
    def get_cached_sample(self):
        """Return cached sample if available"""

        return self.cached_sample
    
    def clear_cache(self):
        """Clear all cached data to force fresh generation"""

        self.cached_predictions = None
        self.cached_sample = None

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
    
    def build_percentile_lookup(self):
        """Build percentile threshold lookup table from reference data"""

        # Calculate the probability value for each percentile (0-100)
        percentiles = np.arange(0, 101)
        thresholds = np.percentile(self.filtered_probs_ref, percentiles)

        return thresholds

    def score_risk(self, sample, ):
        """Assign risk score based on pre-computed percentile lookup"""
        
        # Fast lookup using searchsorted 
        risk_scores = np.searchsorted(self.percentile_thresholds, sample['crash_probability'], side='right') - 1
        
        # Add risk scores to the sample
        sample['risk_score'] = risk_scores.astype(int)
        
        return sample

    def filter_predictions(self):
        """Filter and score segments with elevated crash risk"""

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
        self.pregenerated_maps = {}
        
    def create_deck_map(self, filtered_predictions):
        """Convert prediction data to deck.gl PathLayer format for visualization"""
        
        deck_data = []
        for _, row in filtered_predictions.iterrows():
            # Convert Well-Known Text geometry to coordinate arrays
            geom = wkt.loads(row['geometry'])
            if geom.geom_type == 'LineString':
                coords = [[point[0], point[1]] for point in geom.coords]
    
                risk = row['risk_score'] 
                color = get_deck_color(risk) 
                
                # Create individual path layer data with styling and metadata
                deck_data.append({
                'path': coords,
                'probability_text': row['risk_score'], 
                'full_name': row['full_name'],
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
    
    def pregenerate_all_maps(self):
        """Pre-generate deck.gl visualization maps for all available hours with memory monitoring"""
        
        # Log initial system memory state
        ram_start = psutil.virtual_memory()
        logger.info(f"Starting map pre-generation - RAM: {ram_start.percent:.1f}% used, {ram_start.available / 1024**3:.1f} GB available")
        
        # Get filtered prediction dataset for map generation
        filtered_predictions = self.prediction_engine.filter_predictions()
        
        # Log memory usage after data loading
        ram_after_sample = psutil.virtual_memory()
        logger.info(f"After loading sample predictions - RAM: {ram_after_sample.percent:.1f}% used")
        
        # Get temporal scope for map generation
        available_hours = self.get_available_hours()
        logger.info(f"Pre-generating {len(available_hours)} maps...")

        # Generate interactive map for each available hour
        for i, hour in enumerate(available_hours):
            logger.info(f"Generating map {i+1}/{len(available_hours)} for hour: {pd.to_datetime(hour).strftime('%I %p')}")
            
            # Filter predictions to specific hour for focused visualization
            hour_predictions = filtered_predictions[
                filtered_predictions['datetime'] == hour
            ].copy()
            
            # Create deck.gl map layer with risk-based styling
            deck_map = self.create_deck_map(hour_predictions)
            self.pregenerated_maps[i] = deck_map
            
            # Periodic memory monitoring during generation process
            if (i + 1) % PERFORMANCE_CONFIG['pregeneration_batch_size'] == 0:
                ram_current = psutil.virtual_memory()
                logger.info(f"Progress: {i+1}/{len(available_hours)} maps - RAM: {ram_current.percent:.1f}% used")
        
        # Log final memory state and generation statistics
        ram_final = psutil.virtual_memory()
        logger.info(f"Pre-generation complete! Generated {len(self.pregenerated_maps)} maps")
        logger.info(f"Final RAM usage: {ram_final.percent:.1f}% used, {ram_final.available / 1024**3:.1f} GB available")

    def get_available_hours(self):
        """Extract sorted list of available prediction hours from cached data"""

        cached_sample = self.prediction_engine.data_manager.get_cached_sample()
        if cached_sample is not None:
            return sorted(cached_sample['datetime'].unique())
        return []
    
    def get_pregenerated_map(self, hour_index):
        """Retrieve pre-generated map for specified hour index"""

        return self.pregenerated_maps.get(hour_index, {})

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
        
        # Pre-generate all visualization maps to minimize user wait times
        self.map_renderer.pregenerate_all_maps()

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
    
    def create_deck_map(self, filtered_predictions):
        """Delegate map creation to map renderer"""

        return self.map_renderer.create_deck_map(filtered_predictions)
    
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
    def pregenerated_maps(self):
        """Property accessor for pregenerated maps"""

        return self.map_renderer.pregenerated_maps
    
    @pregenerated_maps.setter
    def pregenerated_maps(self, value):
        """Property setter for pregenerated maps"""

        self.map_renderer.pregenerated_maps = value
    
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
    
    def filter_by_hour(self, hour_index):
        """Filter cached predictions to specific hour based on slider index"""
        
        cached_sample = self.data_manager.get_cached_sample()
        if cached_sample is None:
            return pd.DataFrame()
        
        hours = self.get_available_hours()
        
        # Adjust index to account for skipped first hour
        actual_index = hour_index + 1
        
        # Validate index bounds
        if actual_index >= len(hours):
            return pd.DataFrame()
        
        # Filter data to selected time period
        selected_hour = hours[actual_index]
        filtered = cached_sample[
            cached_sample['datetime'] == selected_hour
        ].copy()
        
        return filtered

# Initialize the main application instance for callback registration
crash_app = CrashRiskApp()

# ================= CALLBACK REGISTRATION =================

def register_callbacks(app):
    """Register all dashboard callbacks for interactive functionality"""
    
    # Generate and cache the filtered prediction dataset on startup
    crash_app.filter_predictions()
    
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
        """Update map visualization based on selected hour"""
        return crash_app.map_renderer.get_pregenerated_map(hour_index)
   
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