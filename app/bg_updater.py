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
import gc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from predictor import parse_geometry, extract_hourly_data

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
        
        # Prepared data staging area for updates
        self.prepared_geometry = None
        self.prepared_hourly_data = None
        self.prepared_available_hours = None
        
        # Thread cancellation and result tracking
        self._cancel_preparation = threading.Event()
        self._prep_result = threading.Event()
        self._prep_success = False
    
    def start(self):
        """Start the background scheduler with two-phase update process"""

        try:
            # Schedule preparation phase at XX:50 (10 minutes before hour)
            self.scheduler.add_job(
                self.prepare_update_timeout, 
                CronTrigger(minute=50), 
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

        logger.info("Starting preparation phase: fetching fresh data...")
        self.update_status = "preparing"
        
        try:
            with self.update_lock:
                # Check cancellation before starting
                if self._cancel_preparation.is_set():
                    logger.info("Preparation cancelled before starting")
                    return False
                
                # Clear any stale prepared data from failed cycle
                if self.prepared_geometry is not None:
                    logger.warning("Clearing stale data from incomplete cycle")
                    self.prepared_geometry = None
                    self.prepared_hourly_data = None
                    self.prepared_available_hours = None
                    gc.collect()

                # Clear existing cached data to force fresh generation
                self.crash_app.cached_predictions = None
                self.crash_app.cached_sample = None
                
                # Force cleanup before starting
                gc.collect()
                
                # Generate new prediction data with latest weather
                logger.info("Generating fresh predictions with current weather data")
                new_sample = self.crash_app.filter_predictions()
                
                # Check cancellation after expensive prediction operation
                if self._cancel_preparation.is_set():
                    logger.info("Preparation cancelled after prediction generation")
                    del new_sample
                    gc.collect()
                    return False
                
                # Prepare lightweight data structures for on-demand map generation
                logger.info("Preparing lightweight data structures")
                geometry_dict = parse_geometry(new_sample)
                
                # Check cancellation after geometry parsing
                if self._cancel_preparation.is_set():
                    logger.info("Preparation cancelled after geometry parsing")
                    del new_sample, geometry_dict
                    gc.collect()
                    return False
                
                hourly_dict, available_hours = extract_hourly_data(new_sample)
                
                # Final cancellation check before staging
                if self._cancel_preparation.is_set():
                    logger.info("Preparation cancelled after data extraction")
                    del new_sample, geometry_dict, hourly_dict, available_hours
                    gc.collect()
                    return False
                
                # Clean up sample immediately after extraction
                del new_sample
                gc.collect()
                
                # Stage prepared data
                self.prepared_geometry = geometry_dict
                self.prepared_hourly_data = hourly_dict
                self.prepared_available_hours = available_hours
                
                logger.info("Preparation phase completed successfully")
                return True
        
        except Exception as e:
            logger.error(f"Preparation phase failed: {e}")
            self.set_error(f"Preparation failed: {e}")
            
            # Aggressive cleanup on error
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
            self.crash_app.cached_predictions = None
            self.crash_app.cached_sample = None
            
            # Force garbage collection
            gc.collect()
            
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
        
        logger.info("Deploying fresh data...")
        
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
                gc.collect()
                
                # Update status tracking
                self.last_update_time = datetime.now(pytz.timezone('America/Los_Angeles'))
                self.update_status = "ready"
                self.error_message = None
                
                logger.info("Deployment completed")
        
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
        preparation_success = self.prepare_update_timeout()
        
        # Only deploy if preparation succeeded and status is healthy
        if preparation_success and self.update_status != "error":
            self.deploy_update()
        else:
            logger.warning("Skipping deployment due to preparation failure")
            
            # Ensure staging is cleared even when skipping deployment
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None

    def cleanup_failed_preparation(self):
        """Aggressively clean up all prepared and cached data after failed update"""
        
        logger.info("Cleaning up failed preparation")
        
        with self.update_lock:
            # Clear all staged data
            self.prepared_geometry = None
            self.prepared_hourly_data = None
            self.prepared_available_hours = None
            
            # Clear app caches
            self.crash_app.cached_predictions = None
            self.crash_app.cached_sample = None
            
            # Force garbage collection
            gc.collect()

    def prepare_update_timeout(self):
        """Wrapper that runs prepare_update with 9-minute, 50-second timeout"""
        
        logger.info("Starting preparation")
        
        # Reset cancellation and result flags
        self._cancel_preparation.clear()
        self._prep_result.clear()
        self._prep_success = False
        
        def run_preparation():
            """Inner function to run in thread"""
            try:
                result = self.prepare_update()
                self._prep_success = result
            except Exception as e:
                logger.error(f"Preparation thread exception: {e}")
                self._prep_success = False
            finally:
                self._prep_result.set()
        
        # Create and start preparation thread
        prep_thread = threading.Thread(target=run_preparation)
        prep_thread.daemon = True
        prep_thread.start()
        
        # Wait for completion (9min 50sec timeout leaves 10sec buffer before deploy)
        completed = self._prep_result.wait(timeout=590)
        
        if not completed:
            # Timeout occurred - signal thread to cancel
            logger.error("TIMEOUT: Preparation exceeded 9 minutes")
            self._cancel_preparation.set()
            self.cleanup_failed_preparation()
            self.set_error("Preparation timed out after 9 minutes")
            return False
        
        # Check if preparation succeeded
        if not self._prep_success:
            logger.error("Preparation failed")
            self.cleanup_failed_preparation()
            return False
        
        logger.info("Preparation completed successfully")
        return True
    
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