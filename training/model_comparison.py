# ==============================================================================
# Traffic Crash Prediction Model Comparison
# ==============================================================================
# Purpose: Compare ML algorithms using temporal validation to select best
#          crash prediction model and analyze feature importance patterns
# 
# Input Files:
#   - ../training/ml_input_data.parquet: Preprocessed crash/weather dataset
#
# Output:
#   - Console output with model performance comparisons and feature rankings
# ==============================================================================

# ================= IMPORTS =================

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ================= DATA LOADING =================

# Load input data
data = pd.read_parquet("../training/ml_input_data.parquet")

# Define target variables for multi-class prediction capability
target_cols = ['crash_occurred', 'svrty_PDO', 'svrty_INJ', 'svrty_FAT']

# Extract feature columns by excluding targets and metadata
feature_cols = [c for c in data.columns if c not in target_cols + ['datetime', 'svrty_NA']]

# ================= TEMPORAL DATA SPLITTING =================

# Create temporal splits to prevent data leakage in time series context
train_df = data[data['datetime'].dt.year <= 2022]
val_df = data[data['datetime'].dt.year == 2023]
test_df = data[data['datetime'].dt.year == 2024]

# Create feature matrices and target vectors
X_train, y_train = train_df[feature_cols], train_df['crash_occurred']
X_val, y_val = val_df[feature_cols], val_df['crash_occurred']
X_test, y_test = test_df[feature_cols], test_df['crash_occurred']

# ================= SEGMENT STATISTICS CALCULATION =================

# Calculate segment statistics from training data only to prevent leakage
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

# Engineer segment-level features from historical crash patterns
segment_stats['seg_count'] = segment_stats['count']
segment_stats['seg_freq'] = segment_stats['count'] / len(train_crashes)  # Relative frequency
segment_stats['seg_log_count'] = np.log1p(segment_stats['count'])        # Log-transformed count
segment_stats = segment_stats.drop('count', axis=1)

# ================= ADD SEGMENT FEATURES =================

# Add historical segment features to all datasets
train_df = train_df.merge(segment_stats, on='segment_id', how='left')
val_df = val_df.merge(segment_stats, on='segment_id', how='left')
test_df = test_df.merge(segment_stats, on='segment_id', how='left')

# Extract feature columns (exclude targets, metadata, and segment_id)
exclude_cols = target_cols + ['datetime', 'segment_id', 'svrty_NA']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]


# Identify segment-related features for analysis
segment_feature_cols = [c for c in feature_cols if c.startswith('seg_')]

# Create feature matrices and target vectors
X_train, y_train = train_df[feature_cols], train_df['crash_occurred']
X_val, y_val = val_df[feature_cols], val_df['crash_occurred']
X_test, y_test = test_df[feature_cols], test_df['crash_occurred']


# ================= MODEL DEFINITIONS =================

# Define model configurations with appropriate hyperparameters for comparison
models = {
   "LogisticRegression": Pipeline([
       ('scaler', StandardScaler()),  # Required for logistic regression
       ('clf', LogisticRegression(
           max_iter=1000, 
           class_weight='balanced',  # Handle class imbalance
           random_state=42
       ))
   ]),
   
   "RandomForest": RandomForestClassifier(
       n_estimators=200, 
       max_depth=10,
       class_weight='balanced',  # Handle class imbalance
       n_jobs=-1, 
       random_state=42
   ),
   
   "XGBoost": XGBClassifier(
       n_estimators=200, 
       max_depth=5, 
       learning_rate=0.1,
       scale_pos_weight=10,  # Handle severe class imbalance
       n_jobs=-1, 
       random_state=42
   )
}

# ================= CROSS-VALIDATION EVALUATION =================

# Use time series CV to respect temporal structure in crash data
tscv = TimeSeriesSplit(n_splits=5)

# Define scoring metrics appropriate for imbalanced binary classification
scoring = ['roc_auc', 'precision', 'recall']

# Store results for comparison
cv_results = {}

print("================= MODEL COMPARISON RESULTS =================")

# Evaluate each model using cross-validation
for name, model in models.items():
   print(f"\nEvaluating {name}...")
   
   # Perform cross-validation with multiple scoring metrics
   scores = cross_validate(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)
   cv_results[name] = scores
   
   # Display mean performance with standard deviation
   print(f"AUC:       {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")
   print(f"Precision: {scores['test_precision'].mean():.4f} (+/- {scores['test_precision'].std():.4f})")
   print(f"Recall:    {scores['test_recall'].mean():.4f} (+/- {scores['test_recall'].std():.4f})")

# ================= FEATURE IMPORTANCE ANALYSIS =================

print("================= FEATURE IMPORTANCE ANALYSIS =================")

# Train each model on full training set to extract feature importance
for name, model in models.items():
   print(f"\n{name} - Top 10 Features:")
   
   # Fit model on full training data
   model.fit(X_train, y_train)
   
   # Extract feature importance based on model type
   if name == "LogisticRegression":
       # Use absolute values of coefficients for logistic regression
       coefs = np.abs(model.named_steps['clf'].coef_[0])
       importance_df = pd.DataFrame({
           'feature': feature_cols,
           'importance': coefs
       }).sort_values('importance', ascending=False).head(10)
       
   elif name in ["RandomForest", "XGBoost"]:
       # Use built-in feature importance for tree-based models
       importance_df = pd.DataFrame({
           'feature': feature_cols,
           'importance': model.feature_importances_
       }).sort_values('importance', ascending=False).head(10)
   
   # Display ranked feature importance
   for _, row in importance_df.iterrows():
       print(f"  {row['feature']:30s}: {row['importance']:.4f}")
