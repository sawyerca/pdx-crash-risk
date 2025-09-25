# ==============================================================================
# Background Update System for Crash Prediction Dashboard
# ==============================================================================
# Purpose: Handle hourly data refresh with two-phase update process to ensure
#          fresh predictions without interrupting user experience
# 
# Input Files:
#   - Real-time weather data via preprocessing pipeline
#   - Street segments and crash statistics from data files
#
# Output:
#   - Updated prediction data and pre-generated map cache
# ==============================================================================

# ================= IMPORTS =================

import logging
import threading
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# ================= BACKGROUND UPDATE MANAGER =================

class BackgroundUpdater:
    """Handles scheduled background updates of crash predictions and visualization maps"""
    
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
        
        # Prepared data staging area for atomic updates
        self.prepared_sample = None
        self.prepared_maps = None
    
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
        """Preparation phase: Generate new predictions and maps"""

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
                
                # Pre-generate all maps
                logger.info("Pre-generating visualization maps for all hours")
                new_maps = self.generate_maps(new_sample)
                
                # Stage prepared data
                self.prepared_sample = new_sample
                self.prepared_maps = new_maps
                
                logger.info("Preparation phase completed successfully")
                return True
        
        # Error handling
        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            self.set_error(f"Preparation failed: {e}")
            return False
    
    def deploy_update(self):
        """Deployment phase: swap in new data"""

        # Validate prepared data availability
        if self.prepared_sample is None or self.prepared_maps is None:
            logger.warning("No prepared data available for deployment - skipping update")
            return
        
        # Additional validation to ensure data quality
        if self.prepared_sample.empty or not self.prepared_maps:
            logger.warning("Prepared data is empty or invalid - skipping deployment")
            return
        
        logger.info("Deploying fresh predictions and maps...")
        
        try:
            with self.update_lock:
                # Atomic data replacement for seamless user experience
                self.crash_app.cached_sample = self.prepared_sample
                self.crash_app.pregenerated_maps = self.prepared_maps
                
                # Update status tracking
                self.last_update_time = datetime.now(pytz.timezone('America/Los_Angeles'))
                self.update_status = "ready"
                self.error_message = None
                
                # Clear staging area to free memory
                self.prepared_sample = None
                self.prepared_maps = None
                
                logger.info("Deployment completed - users now have access to fresh data")
        
        # Error handling
        except Exception as e:
            logger.error(f"Deployment phase failed: {e}")
            self.set_error(f"Deployment failed: {e}")
    
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
    
    def generate_maps(self, sample_data):
        """Generate pre-rendered maps for all available hours"""

        # Initialize vars and get hour range
        maps = {}
        unique_hours = sorted(sample_data['datetime'].unique())
        
        logger.info(f"Generating {len(unique_hours)} pre-rendered maps")
        
        for i, hour in enumerate(unique_hours):
            # Filter data for this specific hour
            hour_data = sample_data[sample_data['datetime'] == hour].copy()
            
            # Generate visualization map using crash app's rendering logic
            maps[i] = self.crash_app.create_deck_map(hour_data)
            
            # Periodic progress logging for long operations
            if (i + 1) % 5 == 0 or (i + 1) == len(unique_hours):
                logger.info(f"Generated {i + 1}/{len(unique_hours)} maps")
        
        return maps
    
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