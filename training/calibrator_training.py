# ==============================================================================
# Probability Calibrator Training for Crash Prediction Model
# ==============================================================================
# Purpose: Train isotonic regression calibrator with Laplace smoothing to convert
#          model probabilities to real-world crash probabilities with base rate alignment
# 
# Input Files:
#   - ../training/ml_input_data.parquet: Historical ML training dataset
#   - ../data/segment_stats.parquet: Segment-level crash statistics
#   - ../models/crash_model.pkl: Trained XGBoost model
#
# Output:
#   - ../models/probability_calibrator.pkl: Calibrated probability converter
# ==============================================================================

# ================= IMPORTS =================

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import pickle

# ================= BASE RATE CALCULATION =================

def calculate_base_rates():
    """Calculate real-world and training base rates for calibration"""
    
    # Load training data to get actual sampling ratio used in model training
    ml_data = pd.read_parquet('../training/ml_input_data.parquet')
    
    # Calculate training base rate from actual ML dataset
    total_samples = len(ml_data)
    positive_samples = (ml_data['crash_occurred'] == 1).sum()
    training_base_rate = positive_samples / total_samples
    
    # Calculate real-world base rate using crash frequency approach
    crashes = pd.read_csv('../wrangling/crashes.csv', low_memory=False)
    crashes['CRASH_DT'] = pd.to_datetime(crashes['CRASH_DT'])
    
    # Get temporal span of crash data
    start_date = crashes['CRASH_DT'].min()
    end_date = crashes['CRASH_DT'].max()
    years = (end_date - start_date).days / 365.25
    
    # Get total segments in network
    street_seg = pd.read_parquet('../data/street_seg.parquet')
    total_segments = street_seg['segment_id'].nunique()
    
    # Real-world crash rate per segment-hour
    crashes_per_year = len(crashes) / years
    total_segment_hours_per_year = total_segments * 365.25 * 24
    real_base_rate = crashes_per_year / total_segment_hours_per_year

    print("================= RATES =================")
    print(f"Training base rate: {training_base_rate:.6f} ({training_base_rate*100:.4f}%)")
    print(f"Real-world base rate: {real_base_rate:.2e} ({real_base_rate*100:.6f}%)")
    print(f"Base rate ratio (real/training): {real_base_rate/training_base_rate:.2e}")
    
    return real_base_rate, training_base_rate

# ================= SMOOTHED ISOTONIC CALIBRATION =================

def train_smoothed_isotonic_calibrator(model_probs, true_outcomes, n_bins=50, alpha=1.0):
    """Train isotonic calibrator with Laplace smoothing"""
    
    # Create probability bins for aggregation
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Aggregate data into bins with Laplace smoothing
    smoothed_probs = []
    smoothed_rates = []
    bin_counts = []
    
    for i in range(n_bins):
        # Find samples in this probability bin
        in_bin = (model_probs >= bins[i]) & (model_probs < bins[i+1])
        n_samples = in_bin.sum()
        
        if n_samples > 0:
            n_positive = true_outcomes[in_bin].sum()
            # Apply Laplace smoothing: (successes + alpha) / (trials + 2*alpha)
            smoothed_rate = (n_positive + alpha) / (n_samples + 2*alpha)
            
            smoothed_probs.append(bin_centers[i])
            smoothed_rates.append(smoothed_rate)
            bin_counts.append(n_samples)
    
    # Train isotonic regression on smoothed binned data
    isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
    isotonic_calibrator.fit(smoothed_probs, smoothed_rates)
    
    # Test smoothing effect on high probabilities
    test_probs = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
    print("Smoothed calibration results:")
    for p in test_probs:
        if p <= max(model_probs):
            calibrated = isotonic_calibrator.transform([p])[0]
            print(f"  Raw: {p:.3f} -> Calibrated: {calibrated:.4f}")
    
    return isotonic_calibrator

# ================= CALIBRATION VALIDATION =================

def validate_calibration_quality(y_true, y_prob_uncalibrated, y_prob_calibrated):
    """Validate calibration improvement using Brier score and reliability metrics"""
    
    # Calculate Brier score improvement
    brier_uncalibrated = brier_score_loss(y_true, y_prob_uncalibrated)
    brier_calibrated = brier_score_loss(y_true, y_prob_calibrated)
    brier_improvement = brier_uncalibrated - brier_calibrated
    
    # Check monotonicity preservation
    sorted_indices = np.argsort(y_prob_uncalibrated)
    calibrated_sorted = y_prob_calibrated[sorted_indices]
    is_monotonic = np.all(calibrated_sorted[1:] >= calibrated_sorted[:-1])
    
    # Calculate reliability using probability bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    reliability_score = 0.0
    
    for i in range(n_bins):
        bin_mask = (y_prob_calibrated > bin_boundaries[i]) & (y_prob_calibrated <= bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_accuracy = y_true[bin_mask].mean()
            bin_confidence = y_prob_calibrated[bin_mask].mean()
            bin_weight = bin_mask.mean()
            reliability_score += np.abs(bin_confidence - bin_accuracy) * bin_weight
    
    results = {
        'brier_improvement': brier_improvement,
        'brier_improvement_pct': (brier_improvement / brier_uncalibrated) * 100,
        'is_monotonic': is_monotonic,
        'reliability_score': reliability_score
    }
    
    # Log validation results
    print("================= CALIBRATION VALIDATION =================")
    print(f"Brier score improvement: {brier_improvement:.6f} ({results['brier_improvement_pct']:+.2f}%)")
    print(f"Monotonicity preserved: {is_monotonic}")
    print(f"Reliability score: {reliability_score:.4f}")
    
    return results

# ================= MAIN CALIBRATOR TRAINING =================

def train_calibrator():
    """Train isotonic regression calibrator with smoothing"""
    
    # Load historical data and model
    historical_data = pd.read_parquet('../training/ml_input_data.parquet')
    seg_stats = pd.read_parquet('../data/segment_stats.parquet')
    
    # Initialize model
    with open('../models/crash_model.pkl', 'rb') as f:
        model_artifact = pickle.load(f)
    
    model = model_artifact['model']
    feature_cols = model_artifact['feature_cols']
    
    # Add segment statistics to historical data
    historical_data = historical_data.merge(seg_stats, on='segment_id', how='inner')
    
    # Use 2024 data as out-of-time validation set for calibration
    validation_data = historical_data[historical_data['datetime'].dt.year == 2024].copy()
    
    # Generate model predictions on validation set
    X_val = validation_data[feature_cols]
    model_probs = model.predict_proba(X_val)[:, 1]
    true_outcomes = validation_data['crash_occurred'].values

    print("================= INITIAL VALUES =================")
    print(f"Model probability range: {model_probs.min():.4f} - {model_probs.max():.4f}")
    print(f"Validation set positive rate: {true_outcomes.mean():.6f}")
    
    # Train smoothed isotonic regression calibrator
    isotonic_calibrator = train_smoothed_isotonic_calibrator(
        model_probs, 
        true_outcomes, 
        n_bins=50,  # Moderate binning for smoothing
        alpha=1.0   # Standard Laplace smoothing
    )
    
    # Apply calibration and validate quality
    calibrated_probs = isotonic_calibrator.transform(model_probs)
    print(f"Calibrated probability range: {calibrated_probs.min():.4f} - {calibrated_probs.max():.4f}")
    
    validation_results = validate_calibration_quality(true_outcomes, model_probs, calibrated_probs)
    
    # Calculate base rates using corrected approach
    real_base_rate, training_base_rate = calculate_base_rates()
    
    # Create calibrator package with all components
    calibrator_data = {
        'isotonic_calibrator': isotonic_calibrator,
        'real_base_rate': real_base_rate,
        'training_base_rate': training_base_rate,
        'validation_stats': {
            'n_samples': len(validation_data),
            'model_prob_range': (float(model_probs.min()), float(model_probs.max())),
            'calibrated_prob_range': (float(calibrated_probs.min()), float(calibrated_probs.max())),
            'validation_positive_rate': float(true_outcomes.mean())
        },
        'calibration_quality': validation_results,
        'smoothing_params': {
            'n_bins': 50,
            'alpha': 1.0
        }
    }
    
    # Save calibrator
    output_path = '../models/probability_calibrator.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator_data, f)
    
    print(f"Training complete - base rate correction factor: {real_base_rate/training_base_rate:.2e}")
    

# RUN
train_calibrator()
