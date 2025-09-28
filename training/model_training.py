# ==============================================================================
# Traffic Crash Prediction Model Training
# ==============================================================================
# Purpose: Train XGBoost classifier for binary crash prediction with
#          hyperparameter optimization, and model serialization
# 
# Input Files:
#   - ../training/ml_input_data.parquet: Preprocessed crash/weather dataset
#
# Output:
#   - ../models/crash_model.pkl: Trained model with calibrator and metadata
# ==============================================================================

# ================= IMPORTS =================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
   roc_auc_score,
   brier_score_loss,
   average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# ================= LOAD DATA =================

# Load input data and define target columns
data = pd.read_parquet("../training/ml_input_data.parquet")

target_cols = ['crash_occurred', 'svrty_PDO', 'svrty_INJ', 'svrty_FAT']

# ================= DATA PREPARATION =================

# Temporal split:
# Training: 2019-2023
# Test: 2024 
train_df = data[data['datetime'] <= '2023-12-31'].copy()
test_df  = data[data['datetime'].dt.year == 2024].copy()

# Recalculate segment statistics from training data only
train_crashes = train_df[train_df['crash_occurred'] == 1]
segment_stats = train_crashes.groupby('segment_id').size().reset_index(name='count')

# Get all unique segments across all time periods
all_segments = pd.concat([
    train_df['segment_id'],
    test_df['segment_id']
    ]).unique()

# Create complete segment statistics table with zero-fill for unseen segments
segment_stats = pd.DataFrame({'segment_id': all_segments}).merge(
    segment_stats, on='segment_id', how='left'
).fillna(0)

# Engineer segment-level features from historical crash patterns (more can be added)
segment_stats['seg_log_count'] = np.log1p(segment_stats['count'])        # Log-transformed count
segment_stats = segment_stats.drop('count', axis=1)

# Add historical segment features to all datasets
train_df = train_df.merge(segment_stats, on='segment_id', how='left')
test_df  = test_df.merge(segment_stats, on='segment_id', how='left')

# Define feature columns
exclude_cols = target_cols + ['datetime', 'segment_id', 'svrty_NA']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Create feature matrices and target vectors for modeling
X_train, y_train = train_df[feature_cols], train_df['crash_occurred']
X_test, y_test   = test_df[feature_cols], test_df['crash_occurred']

# Calculate class imbalance ratio for XGBoost weighting
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# ================= HYPERPARAMETER TUNING =================

# Define hyperparameter search space for XGBoost
param_distributions = {
    'n_estimators': [500, 700, 1000],                    
    'max_depth': [3, 4, 5, 6],                           
    'learning_rate': [0.01, 0.02, 0.03, 0.05],          
    'subsample': [0.6, 0.7, 0.8, 0.9],                  
    'colsample_bytree': [0.6, 0.7, 0.8],                
    'min_child_weight': [10, 15, 20, 25, 30],           
    'gamma': [0.1, 0.3, 0.5, 1.0, 2.0],                 
    'reg_alpha': [0.1, 0.5, 1.0, 2.0],                  
    'reg_lambda': [2, 5, 10, 15],                        
    'scale_pos_weight': [
        0.2 * imbalance_ratio,  
        0.3 * imbalance_ratio, 
        0.5 * imbalance_ratio,  
        0.7 * imbalance_ratio, 
        1.0 * imbalance_ratio,  
        1.2 * imbalance_ratio, 
        1.5 * imbalance_ratio, 
    ],      
}

# Use time series CV 
cv_strategy = TimeSeriesSplit(n_splits=3)

# Configure randomized search 
random_search = RandomizedSearchCV(
   estimator=XGBClassifier(
       random_state=123,
       n_jobs=-1,
       eval_metric="logloss"
   ),
   param_distributions=param_distributions,
   n_iter=50,                    
   cv=cv_strategy,
   scoring='average_precision',           
   n_jobs=-1,
   random_state=123,
   verbose=1
)

# Execute hyperparameter search
print("================= HYPERPARAMETER SEARCH =================")
random_search.fit(X_train, y_train, verbose=False)

print(f"Best CV AUC: {random_search.best_score_:.4f}")
print(f"Best params: {random_search.best_params_}")

# ================= FINAL MODEL TRAINING =================

# Train final model with best hyperparameters on full training set

best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# ================= FEATURE IMPORTANCE CHECK =================

print("================= FEATURE IMPORTANCE =================")

# Extract and rank feature importance 
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20))

# Specifically examine segment-level features for validation
segment_features = importance_df[importance_df['feature'].str.contains('segment')]
print("\nSegment Feature Importance:")
print(segment_features)

# ================= MODEL EVALUATION =================

print("================= MODEL PERFORMANCE =================")

# Generate final predictions 
y_proba_test = best_model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
test_auc = roc_auc_score(y_test, y_proba_test)            
test_brier = brier_score_loss(y_test, y_proba_test)     
test_avg_precision = average_precision_score(y_test, y_proba_test)  

print(f"Test AUC: {test_auc:.4f}")
print(f"Test Brier Score: {test_brier:.4f}")
print(f"Test Average Precision: {test_avg_precision:.4f}")

# ================= KNEE CUTOFF FOR SCORING =================

def find_knee_cutoff(raw_probs):
    """Use Kneedle algorithm to find knee point"""
    
    # Create curve of percentile vs probability value
    percentiles = np.arange(0, 100, 0.5)  
    prob_values = np.percentile(raw_probs, percentiles)
    
    # Normalize both axes to [0,1] for algorithm
    x_norm = (percentiles - percentiles.min()) / (percentiles.max() - percentiles.min())
    y_norm = (prob_values - prob_values.min()) / (prob_values.max() - prob_values.min())
    
    # Kneedle: find max distance from line connecting start to end
    max_distance = 0
    knee_index = 0
    
    for i in range(1, len(x_norm) - 1):
        # Distance from point to line (start to end)
        x0, y0 = x_norm[0], y_norm[0]    # Start point
        x1, y1 = x_norm[-1], y_norm[-1]  # End point
        x2, y2 = x_norm[i], y_norm[i]    # Current point
        
        # Distance formula
        distance = abs((y1-y0)*x2 - (x1-x0)*y2 + x1*y0 - y1*x0) / np.sqrt((y1-y0)**2 + (x1-x0)**2)
        
        if distance > max_distance:
            max_distance = distance
            knee_index = i
    
    # Convert knee point back to actual probability value
    knee_percentile = percentiles[knee_index]
    knee_probability = np.percentile(raw_probs, knee_percentile)
    
    return knee_probability

# Get raw probabilities for all data (2019-2024) for cutoff calc
all_X = pd.concat([X_train, X_test])
raw_probs_ref = best_model.predict_proba(all_X)[:, 1]

cutoff = find_knee_cutoff(raw_probs_ref)

# Filter to keep only high-risk 
filtered_probs_ref = raw_probs_ref[raw_probs_ref >= cutoff]

# ================= SAVE MODEL =================

# Package model components and metadata 
model_artifact = {
    'model': best_model,
    'best_params': random_search.best_params_,
    'feature_cols': feature_cols,            
    'cutoff': cutoff,  
    'filtered_probs_ref': filtered_probs_ref,
    'metrics': {
        'test_auc': test_auc,
        'test_brier': test_brier,
        'test_avg_precision': test_avg_precision
    }
}

with open('../models/crash_model.pkl', 'wb') as f:
    pickle.dump(model_artifact, f)