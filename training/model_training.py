# ==============================================================================
# Traffic Crash Prediction Model Training
# ==============================================================================
# Purpose: Train XGBoost classifier for binary crash prediction with temporal
#          validation, hyperparameter optimization, and model serialization
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
# Training: all data through Nov 2023
# Validation: Dec 2023 (for hyperparameter tuning and calibration)
# Test: 2024 
train_df = data[data['datetime'] < '2023-12-01'].copy()
val_df   = data[(data['datetime'] >= '2023-12-01') &
                (data['datetime'] < '2024-01-01')].copy()
test_df  = data[data['datetime'].dt.year == 2024].copy()

# Recalculate segment statistics from training data only
train_crashes = train_df[train_df['crash_occurred'] == 1]
segment_stats = train_crashes.groupby('segment_id').size().reset_index(name='count')

# Get all unique segments across all time periods
all_segments = pd.concat([
    train_df['segment_id'],
    val_df['segment_id'],
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
val_df   = val_df.merge(segment_stats, on='segment_id', how='left')
test_df  = test_df.merge(segment_stats, on='segment_id', how='left')

# Define feature columns
exclude_cols = target_cols + ['datetime', 'segment_id', 'svrty_NA']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Create feature matrices and target vectors for modeling
X_train, y_train = train_df[feature_cols], train_df['crash_occurred']
X_val, y_val     = val_df[feature_cols], val_df['crash_occurred']
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
        0.2 * imbalance_ratio,  # Much less crash-eager (~1.8)
        0.3 * imbalance_ratio,  # Less crash-eager (~2.7)  
        0.5 * imbalance_ratio,  # Moderately less (~4.6)
        0.7 * imbalance_ratio,  # Slightly less (~6.4)
        1.0 * imbalance_ratio,  # Natural balance (~9.1)
        1.2 * imbalance_ratio,  # Slightly more (~10.9)
        1.5 * imbalance_ratio,  # More crash-eager (~13.7)
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

# ================= CALIBRATION =================

print("================= CALIBRATION =================")

# Get raw model probabilities on validation set
y_proba_val = best_model.predict_proba(X_val)[:, 1]

# Assess calibration quality using Brier score
brier_before = brier_score_loss(y_val, y_proba_val)
print(f"Validation Brier score before calibration: {brier_before:.4f}")

# Apply isotonic regression for probability calibration
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_proba_val, y_val)

# Evaluate calibration improvement
y_proba_val_cal = calibrator.transform(y_proba_val)
brier_after = brier_score_loss(y_val, y_proba_val_cal)
print(f"Validation Brier score after calibration: {brier_after:.4f}")

# ================= MODEL EVALUATION =================

print("================= MODEL PERFORMANCE =================")

# Generate final predictions 
y_proba_test = best_model.predict_proba(X_test)[:, 1]
y_proba_test_cal = calibrator.transform(y_proba_test)

# Calculate performance metrics
test_auc = roc_auc_score(y_test, y_proba_test_cal)            
test_brier = brier_score_loss(y_test, y_proba_test_cal)     
test_avg_precision = average_precision_score(y_test, y_proba_test_cal)  

print(f"Test AUC: {test_auc:.4f}")
print(f"Test Brier Score: {test_brier:.4f}")
print(f"Test Average Precision: {test_avg_precision:.4f}")

# ================= SAVE MODEL =================

# Package model components and metadata 
model_artifact = {
    'model': best_model,
    'calibrator': calibrator,
    'best_params': random_search.best_params_,
    'feature_cols': feature_cols,              
    'metrics': {
        'test_auc': test_auc,
        'test_brier': test_brier,
        'test_avg_precision': test_avg_precision
    }
}

with open('../models/crash_model.pkl', 'wb') as f:
    pickle.dump(model_artifact, f)