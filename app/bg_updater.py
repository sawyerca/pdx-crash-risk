# ==============================================================================
# Background Update System for Crash Prediction Dashboard
# ==============================================================================
# Purpose: Handle hourly data refresh with lightweight data preparation to ensure
#          fresh predictions without interrupting user experience
# 
# Input Files:
#   - Real-time weather data via preprocessing pipeline
#   - Street segments and crash statistics from data files
#
# Output:
#   - Updated lightweight prediction data for on-demand map generation
# ==============================================================================

# ================= IMPORTS =================

import logging
import threading
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# ================= BACKGROUND UPDATE MANAGER =================

class BackgroundUpdater:
    """Handles scheduled background updates of crash predictions with lightweight data preparation"""
    
    def __init__(self, crash_app):
        """Initialize background updater with crash application instance"""

        # Initialize
        self.crash_app = crash_app
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('America/Los_Angeles'))
        self.update_lock = threading.Lock()
        
        # Status tracking for monitoring and user notifications
        self.last_update_time = None
        self.update_status = "initializing"
        self.error_message = None
        
        # Prepared data staging area for atomic updates (lightweight structures)
        self.prepared_geometry = None
        self.prepared_hourly_data = None
        self.prepared_available_hours = None
    
    def start(self):
        """Start the background scheduler with two-phase update process"""

        try:
            # Schedule preparation phase at XX:55 (5 minutes before hour)
            self.scheduler.add_job(
                self.prepare_update, 
                CronTrigger(minute=55), 
                id='prepare', 
                max_instances=1
            )
            
            # Schedule deployment phase at XX:00 (top of hour)
            self.scheduler.add_job(
                self.deploy_update, 
                CronTrigger(minute=0), 
                id='deploy', 
                max_instances=1
            )
            
            # Start scheduler
            self.scheduler.start()
            logger.info("Background scheduler started")
        
        # Error handling
        except Exception as e:
            logger.error(f"Failed to start background scheduler: {e}")
            self.set_error(str(e))
    
    def prepare_update(self):
        """Generate new predictions and prepare data structures"""

        logger.info("Starting preparation phase - fetching fresh data...")
        self.update_status = "preparing"
        
        try:
            with self.update_lock:
                # Clear existing cached data to force fresh generation
                self.crash_app.cached_predictions = None
                self.crash_app.cached_sample = None
                
                # Generate new prediction data with latest weather
                logger.info("Generating fresh predictions with current weather data")
                new_sample = self.crash_app.filter_predictions()
                
                # Prepare lightweight data structures for on-demand map generation
                logger.info("Preparing lightweight data structures")
                geometry_dict = self.parse_geometry(new_sample)
                hourly_dict, available_hours = self.extract_hourly_data(new_sample)
                
                # Stage prepared data
                self.prepared_geometry = geometry_dict
                self.prepared_hourly_data = hourly_dict
                self.prepared_available_hours = available_hours
                
                logger.info("Preparation phase completed successfully")
                return True
        
        # Error handling with staging cleanup
        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            self.set_error(f"Preparation failed: {e}")
            
            # Clear any partially prepared data to prevent memory leaks
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
            
            return False
    
    def deploy_update(self):
        """Swap in new data"""

        # Validate prepared data availability
        if self.prepared_geometry is None or self.prepared_hourly_data is None:
            logger.warning("No prepared data available for deployment - skipping update")
            
            # Clear staging to prevent accumulation from partial preparations
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
            return
        
        # Additional validation to ensure data quality
        if not self.prepared_geometry or not self.prepared_hourly_data:
            logger.warning("Prepared data is empty or invalid - skipping deployment")
            
            # Clear invalid staging data
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
            return
        
        logger.info("Deploying fresh lightweight data structures...")
        
        try:
            with self.update_lock:
                # Explicitly delete old data structures before replacement
                old_geometry = self.crash_app.parsed_geometry
                old_hourly = self.crash_app.hourly_data
                
                # Atomic data replacement for seamless user experience
                self.crash_app.parsed_geometry = self.prepared_geometry
                self.crash_app.hourly_data = self.prepared_hourly_data
                self.crash_app.data_manager.cache_available_hours(self.prepared_available_hours)
                
                # Force deletion of old data and trigger garbage collection
                del old_geometry, old_hourly
                import gc
                gc.collect()
                
                # Update status tracking
                self.last_update_time = datetime.now(pytz.timezone('America/Los_Angeles'))
                self.update_status = "ready"
                self.error_message = None
                
                logger.info("Deployment completed - users now have access to fresh data")
        
        # Error handling
        except Exception as e:
            logger.error(f"Deployment phase failed: {e}")
            self.set_error(f"Deployment failed: {e}")
        
        finally:
            # Always clear staging area to prevent memory leaks (even on error)
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
    
    def run_full_update(self):
        """Execute complete update cycle for startup or manual refresh"""

        logger.info("Running full update cycle (prepare + deploy)")
        
        # Execute preparation phase
        preparation_success = self.prepare_update()
        
        # Only deploy if preparation succeeded and status is healthy
        if preparation_success and self.update_status != "error":
            self.deploy_update()
        else:
            logger.warning("Skipping deployment due to preparation failure")
            
            # Ensure staging is cleared even when skipping deployment
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
    
    def parse_geometry(self, filtered_predictions):
        """Parse WKT geometry strings once and cache coordinate arrays by segment_id"""
        
        logger.info("Parsing geometry strings to coordinate arrays...")
        start_time = time.time()
        
        from shapely import wkt
        
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
        """Extract hourly risk data indexed by hour"""
        
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
    
    def set_error(self, message):
        """Set error state with message for status reporting"""

        self.update_status = "error"
        self.error_message = message
        logger.error(f"Update system error: {message}")
    
    def get_status_info(self):
        """Get current system status for monitoring and user notifications"""

        return {
            'status': self.update_status,
            'last_update': self.last_update_time,
            'error_message': self.error_message
        }
    
    def stop(self):
        """Gracefully shutdown the background update system"""

        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Background update system stopped")

    def manual_update(self):
        """Trigger manual update for testing or immediate refresh"""
        
        logger.info("Manual update triggered")
        self.run_full_update()