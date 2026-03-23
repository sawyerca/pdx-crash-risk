# ==============================================================================
# Crash Prediction Engine
# ==============================================================================
# Purpose: Generate crash probabilities for street segments using trained
#           XGBoost model with chunked processing for memory efficiency 
# 
# Input Files:
#   - ../models/crash_model.pkl: Trained crash prediction model
#   - Data generator from preprocessing pipeline
#
# Output:
#   - DataFrame with segment crash probabilities and geometry
# ==============================================================================

# ================= IMPORTS =================

import pickle
import logging
import pandas as pd
from shapely import wkt

# ================= CONFIGURATION =================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

# Optimized data types for memory efficiency
OPT_DTYPES = {
    'segment_id': 'int32', 
    'crash_probability': 'float32',
    'geometry': 'string',
    'full_name': 'string',
    'risk_score': 'int8'
}

# ================= MODEL LOADING =================

def load_model(model_path):
    """Load the trained crash prediction model artifact"""

    # Load model
    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)

    return model_artifact


# ================= PREDICTION ENGINE =================

class CrashPredictor:
    """
    Generates crash probabilities for street segments using model with  
    percentile based risk scores.
    """
    
    def __init__(self, model_artifact):
        """Initialize predictor with pre-loaded model components"""

        # Extract model components from artifact
        self.model_artifact = model_artifact
        self.model = model_artifact['model']
        self.feature_cols = model_artifact['feature_cols']
    
    def predict(self, data_generator):
        """Generate crash probabilities with chunking"""

        # Initialize vars
        all_predictions = []
        chunk_count = 0

        # Process each data chunk from generator
        for chunk_count, merged_chunk in enumerate(data_generator, 1):
            
            # Get unique time periods from this chunk
            unique_times = sorted(merged_chunk['datetime'].unique())
            
            chunk_predictions = []

            # Process each hour individually to manage memory
            for time_stamp in unique_times:
                # Filter data for this specific hour
                hour_data = merged_chunk[merged_chunk['datetime'] == time_stamp].copy()
                
                # Extract features and generate predictions
                X = hour_data[self.feature_cols]
                raw_probs = self.model.predict_proba(X)[:, 1]
                
                # Create output with essential columns
                predictions = pd.DataFrame({
                    'segment_id': hour_data['segment_id'],
                    'geometry': hour_data['geometry'], 
                    'full_name': hour_data['full_name'],
                    'datetime': hour_data['datetime'],
                    'crash_probability': raw_probs
                })

                # Apply all optimized dtypes 
                for col, dtype in OPT_DTYPES.items():
                    if col in predictions.columns:
                        predictions[col] = predictions[col].astype(dtype)
                        
                chunk_predictions.append(predictions)
                
                # Clean up intermediate variables
                del hour_data, X, raw_probs
            
            # Combine predictions for this chunk
            if chunk_predictions:
                chunk_combined = pd.concat(chunk_predictions, ignore_index=True)
                all_predictions.append(chunk_combined)
            
            # Memory cleanup and garbage collection
            del merged_chunk, chunk_predictions
            import gc; gc.collect()
        
        # Combine all chunks into final prediction set
        logger.info("Combining all prediction chunks...")
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        return final_predictions


# ================= GEOMETRY & HOURLY DATA UTILITIES =================

def parse_geometry(filtered_predictions):
    """Parse WKT geometry strings to coordinate arrays indexed by segment_id"""

    logger.info("Parsing geometry strings to coordinate arrays...")

    geometry_dict = {}

    # Get unique segments to avoid parsing duplicates across time periods
    unique_segments = filtered_predictions[['segment_id', 'geometry', 'full_name']].drop_duplicates(subset='segment_id')

    for _, row in unique_segments.iterrows():
        geom = wkt.loads(row['geometry'])
        if geom.geom_type == 'LineString':
            coords = [[point[0], point[1]] for point in geom.coords]
            geometry_dict[row['segment_id']] = {
                'coords': coords,
                'full_name': row['full_name']
            }

    logger.info(f"Parsed {len(geometry_dict)} unique segments")

    return geometry_dict


def extract_hourly_data(filtered_predictions):
    """Extract hourly risk data indexed by position, skipping the first hour for slider alignment"""

    logger.info("Extracting hourly risk data...")

    hourly_dict = {}
    available_hours = sorted(filtered_predictions['datetime'].unique())
    display_hours = available_hours[1:]  # skip first hour for slider alignment

    for i, hour in enumerate(display_hours):
        hour_data = filtered_predictions[filtered_predictions['datetime'] == hour]
        hourly_dict[i] = hour_data[['segment_id', 'risk_score']].to_dict('records')

    return hourly_dict, available_hours

