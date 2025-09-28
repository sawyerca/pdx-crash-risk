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
import numpy as np 

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
    Generates crash probabilities for street segments using model. Uses Box-Cox
    transformed values with empirical CDF to assign a risk score.
    """
    
    def __init__(self, model_artifact):
        """Initialize predictor with pre-loaded model components"""

        # Extract model components from artifact
        self.model_artifact = model_artifact
        self.model = model_artifact['model']
        self.feature_cols = model_artifact['feature_cols']
    
    def predict(self, data_generator):
        """Generate calibrated crash probabilities with chunking"""

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

        # Add memory breakdown
        memory_usage = final_predictions.memory_usage(deep=True)
        total_memory = memory_usage.sum() / (1024 ** 3)
        
        logger.info(f"Final predictions memory usage: {total_memory:.3f} GiB")
        
        return final_predictions

# ================= TESTING =================

if __name__ == '__main__':
    """Test the prediction pipeline with sample data"""

    from preprocessing import street_encode, DataPipeline

    pipe = DataPipeline()
    id_lookup = pd.read_csv('../data/id_lookup.csv')
    street_seg = street_encode('../data/street_seg.parquet')
    seg_stats = pd.read_parquet('../data/segment_stats.parquet')
    merged_data = pipe.model_input(id_lookup, street_seg, seg_stats)
    
    model = load_model('../models/crash_model.pkl')
    predictor = CrashPredictor(model)

    test_predictions = predictor.predict(merged_data)
