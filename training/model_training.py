# ==============================================================================
# Traffic Crash Prediction Model Training
# ==============================================================================
# Purpose: Train XGBoost classifier for binary crash prediction with
#          custom objective function, hyperparameter optimization, and 
#          comprehensive evaluation metrics
# 
# Input Files:
#   - ../training/ml_input_data.parquet: Preprocessed input dat
#
# Output:
#   - ../models/crash_model.pkl: Trained model with metadata
#   - Training logs with comprehensive metrics
# ==============================================================================

# ================= IMPORTS =================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    make_scorer
)
from xgboost import XGBClassifier

# ================= COMPREHENSIVE METRICS FUNCTIONS =================

def calculate_ece(y_true, y_proba, n_bins=10):
    """Calculate Expected Calibration Error"""

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find predictions in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        
        if np.sum(in_bin) > 0:
            # Average predicted probability in bin
            avg_confidence = np.mean(y_proba[in_bin])
            # Actual accuracy in bin
            avg_accuracy = np.mean(y_true[in_bin])
            # Weight by proportion of samples in bin
            bin_weight = np.sum(in_bin) / len(y_proba)
            
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
    
    return ece

def calculate_lift_at_k(y_true, y_proba, k_percent):
    """Calculate lift at top K% of predictions"""

    n = len(y_true)
    k_samples = int(n * k_percent / 100)
    
    # Get indices of top K% predictions
    top_k_idx = np.argsort(y_proba)[-k_samples:]
    
    # Calculate positive rate in top K
    positives_in_top_k = np.sum(y_true[top_k_idx])
    positive_rate_top_k = positives_in_top_k / k_samples
    
    # Calculate baseline positive rate
    baseline_positive_rate = np.mean(y_true)
    
    # Lift
    lift = positive_rate_top_k / baseline_positive_rate if baseline_positive_rate > 0 else 0
    
    return lift

def calculate_comprehensive_metrics(y_true, y_proba, label=""):
    """Calculate comprehensive metrics for model evaluation"""

    metrics = {}
    
    # DISCRIMINATION METRICS 
    metrics['auc'] = roc_auc_score(y_true, y_proba)
    metrics['avg_precision'] = average_precision_score(y_true, y_proba)
    
    # CALIBRATION METRICS =
    metrics['brier_score'] = brier_score_loss(y_true, y_proba)
    metrics['log_loss'] = log_loss(y_true, y_proba)
    
    # EXPECTED CALIBRATION ERROR 
    metrics['ece'] = calculate_ece(y_true, y_proba, n_bins=10)
    
    # PRECISION-RECALL AT THRESHOLDS 
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find precision at specific recall levels
    recall_targets = [0.10, 0.25, 0.50, 0.75, 0.90]
    for target_recall in recall_targets:
        idx = np.argmin(np.abs(recall - target_recall))
        metrics[f'precision_at_recall_{int(target_recall*100)}'] = precision[idx]
        metrics[f'threshold_at_recall_{int(target_recall*100)}'] = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    
    # Find recall at specific precision levels
    precision_targets = [0.10, 0.25, 0.50, 0.75, 0.90]
    for target_precision in precision_targets:
        idx = np.argmin(np.abs(precision - target_precision))
        metrics[f'recall_at_precision_{int(target_precision*100)}'] = recall[idx]
    
    #  LIFT ANALYSIS 
    # Calculate lift at top K% of predictions
    percentiles = [1, 5, 10, 20]
    for k in percentiles:
        metrics[f'lift_at_top_{k}pct'] = calculate_lift_at_k(y_true, y_proba, k)
    
    #  CLASSIFICATION METRICS AT OPTIMAL THRESHOLD 
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    metrics['optimal_threshold'] = optimal_threshold
    metrics['f1_at_optimal'] = f1_scores[best_f1_idx]
    metrics['precision_at_optimal'] = precision[best_f1_idx]
    metrics['recall_at_optimal'] = recall[best_f1_idx]
    
    # MCC and Kappa at optimal threshold
    if len(np.unique(y_pred_optimal)) > 1:  # Need both classes for these metrics
        metrics['mcc_at_optimal'] = matthews_corrcoef(y_true, y_pred_optimal)
        metrics['cohen_kappa_at_optimal'] = cohen_kappa_score(y_true, y_pred_optimal)
    else:
        metrics['mcc_at_optimal'] = np.nan
        metrics['cohen_kappa_at_optimal'] = np.nan
    
    # PROBABILITY DISTRIBUTION STATS 
    metrics['prob_mean'] = np.mean(y_proba)
    metrics['prob_std'] = np.std(y_proba)
    metrics['prob_min'] = np.min(y_proba)
    metrics['prob_max'] = np.max(y_proba)
    metrics['prob_median'] = np.median(y_proba)
    
    # Percentage of predictions in different ranges
    metrics['pct_probs_under_01'] = np.mean(y_proba < 0.01) * 100
    metrics['pct_probs_under_05'] = np.mean(y_proba < 0.05) * 100
    metrics['pct_probs_over_50'] = np.mean(y_proba > 0.50) * 100
    
    return metrics

def print_metrics_report(metrics, label=""):
    """Print formatted metrics report"""
    
    print(f"================= {label} METRICS REPORT =================" )
    
    print("\n================= DISCRIMINATION METRICS =================")
    print(f"AUC:                     {metrics['auc']:.4f}")
    print(f"Average Precision:       {metrics['avg_precision']:.4f}")
    
    print("\n================= CALIBRATION METRICS =================")
    print(f"Brier Score:             {metrics['brier_score']:.4f}")
    print(f"Log Loss:                {metrics['log_loss']:.4f}")
    print(f"Expected Calibration Error: {metrics['ece']:.4f}")
    
    print("\n================= PRECISION AT RECALL LEVELS =================---")
    for recall_target in [10, 25, 50, 75, 90]:
        precision_key = f'precision_at_recall_{recall_target}'
        threshold_key = f'threshold_at_recall_{recall_target}'
        print(f"Recall {recall_target}%: Precision={metrics[precision_key]:.4f}, Threshold={metrics[threshold_key]:.6f}")
    
    print("\n================= RECALL AT PRECISION LEVELS =================")
    for precision_target in [10, 25, 50, 75, 90]:
        recall_key = f'recall_at_precision_{precision_target}'
        print(f"Precision {precision_target}%: Recall={metrics[recall_key]:.4f}")
    
    print("\n================= LIFT ANALYSIS =================")
    for k in [1, 5, 10, 20]:
        lift_key = f'lift_at_top_{k}pct'
        print(f"Top {k}% Lift:            {metrics[lift_key]:.2f}x")
    
    print("\n================= OPTIMAL F1 THRESHOLD =================")
    print(f"Optimal Threshold:       {metrics['optimal_threshold']:.6f}")
    print(f"F1 Score:                {metrics['f1_at_optimal']:.4f}")
    print(f"Precision:               {metrics['precision_at_optimal']:.4f}")
    print(f"Recall:                  {metrics['recall_at_optimal']:.4f}")
    if not np.isnan(metrics['mcc_at_optimal']):
        print(f"Matthews Corr Coef:      {metrics['mcc_at_optimal']:.4f}")
        print(f"Cohen's Kappa:           {metrics['cohen_kappa_at_optimal']:.4f}")
    
    print("\n=================--- PROBABILITY DISTRIBUTION =================")
    print(f"Mean:                    {metrics['prob_mean']:.6f}")
    print(f"Std Dev:                 {metrics['prob_std']:.6f}")
    print(f"Min:                     {metrics['prob_min']:.6f}")
    print(f"Max:                     {metrics['prob_max']:.6f}")
    print(f"Median:                  {metrics['prob_median']:.6f}")
    print(f"% Probs < 0.01:          {metrics['pct_probs_under_01']:.2f}%")
    print(f"% Probs < 0.05:          {metrics['pct_probs_under_05']:.2f}%")
    print(f"% Probs > 0.50:          {metrics['pct_probs_over_50']:.2f}%")

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

# Engineer segment-level features from historical crash patterns
segment_stats['seg_log_count'] = np.log1p(segment_stats['count'])
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

# ================= OBJECTIVE AND SCORING FUNCTIONS =================

def confidence_weighted_loss(base_fp_weight, confidence_multiplier):
    """Custom objective function that penalizes confident false positives"""

    def objective(y_true, y_pred):
        # Convert raw predictions to probabilities
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Calculate confidence-weighted false positive penalty
        fp_penalty = base_fp_weight + confidence_multiplier * p
        
        # Compute gradients
        grad = np.where(y_true == 1, p - 1, fp_penalty * p)
        hess = np.where(y_true == 1, p * (1 - p), fp_penalty * p * (1 - p))
        
        return grad, hess
    return objective

def precision_75(y_true, y_proba, **kwargs):
    """Calculate precision when recall is at 75% level"""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find index where recall is closest to 75%
    idx = np.argmin(np.abs(recall - 0.75))
    
    return precision[idx]

# ================= HYPERPARAMETER TUNING =================

# Parameters (experimentally optimized for precision @ 75% recall)
fp_weight = 7.0              
confidence_mult = 2.0       

# Define hyperparameter search space
param_distributions = {
    # Tree structure 
    'n_estimators': [300, 500, 700, 1000],                    
    'max_depth': [3, 4, 5, 6, 7],                           
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07],          
    
    # Sampling 
    'subsample': [0.6, 0.7, 0.8, 0.9],                  
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],                
    
    # Regularization 
    'min_child_weight': [25, 30, 35, 40, 50, 60],      
    'gamma': [0.5, 1.0, 2.0, 3.0, 5.0],             
    'reg_alpha': [1.0, 2.0, 5.0, 10.0, 15.0],             
    'reg_lambda': [10, 15, 20, 30, 50],             
}

# Use time series CV 
cv_strategy = TimeSeriesSplit(n_splits=3)

# Create custom scorer focused on precision at 75% recall
precision_75_score = make_scorer(
    precision_75, 
    needs_proba=True, 
    greater_is_better=True
    )

# Configure randomized search 
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=123,
        n_jobs=-1,
        eval_metric="logloss",
        objective=confidence_weighted_loss(fp_weight, confidence_mult),
    ),
    param_distributions=param_distributions,
    n_iter=40,              
    cv=cv_strategy,
    scoring=precision_75_score,  
    n_jobs=-1,
    random_state=123,
    verbose=2 
)

# Execute hyperparameter search

print("================= STARTING HYPERPARAMETER SEARCH =================")

# Calculate search space size
space_size = np.prod([len(v) for v in param_distributions.values()])
print(f"Search space size: {space_size:,} combinations")

random_search.fit(X_train, y_train, verbose=False)

print("================= HYPERPARAMETER RESULTS =================")
print(f"Best CV Score (Avg Precision): {random_search.best_score_:.4f}")
print(f"\nBest Hyperparameters:")
for param, value in sorted(random_search.best_params_.items()):
    print(f"  {param:20s}: {value}")

# ================= INITIAL MODEL TRAINING =================

print("================= TRAINING FINAL MODEL =================")

# Train final model with best hyperparameters on full training set
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# ================= FEATURE IMPORTANCE ANALYSIS =================

print("================= FEATURE IMPORTANCE ANALYSIS =================")

# Extract and rank feature importance 
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# ================= COMPREHENSIVE MODEL EVALUATION =================

print("================= EVALUATION =================")

# Generate predictions for both train and test sets
y_proba_train = best_model.predict_proba(X_train)[:, 1]
y_proba_test = best_model.predict_proba(X_test)[:, 1]

# Calculate comprehensive metrics for training set
print("Calculating training set metrics...")
train_metrics = calculate_comprehensive_metrics(y_train, y_proba_train, label="Train")
print_metrics_report(train_metrics, label="TRAINING SET")

# Calculate comprehensive metrics for test set
print("Calculating test set metrics...")
test_metrics = calculate_comprehensive_metrics(y_test, y_proba_test, label="Test")
print_metrics_report(test_metrics, label="TEST SET")

# ================= FINAL MODEL TRAINING =================

print("================= RETRAINING ON FULL DATASET =================")

# Combine train and test sets
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test], ignore_index=True)

# Retrain model with best hyperparameters on full dataset
final_model = XGBClassifier(
        random_state=123,
        n_jobs=-1,
        eval_metric="logloss",
        objective=confidence_weighted_loss(fp_weight, confidence_mult),
        **random_search.best_params_
    )
final_model.fit(X_all, y_all)

# ================= KNEE CUTOFF FOR SCORING =================

def find_knee_cutoff(raw_probs):
    """Use Kneedle algorithm to find knee point in probability distribution"""
    
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

print("================= CALCULATING KNEE CUTOFF =================")

# Recalculate cutoff and reference probabilities using ALL data
print("Recalculating cutoff and reference probabilities on full dataset...")
all_probs = final_model.predict_proba(X_all)[:, 1]
cutoff = find_knee_cutoff(all_probs)
filtered_probs_ref = all_probs[all_probs >= cutoff]

print(f"Cutoff probability: {cutoff:.6f}")
print(f"Percentile at cutoff: {(all_probs <= cutoff).mean() * 100:.2f}%")
print(f"Samples above cutoff: {len(filtered_probs_ref):,} ({(all_probs >= cutoff).mean() * 100:.2f}%)")

# ================= SAVE MODEL =================

# Clear the objective from the model to make it picklable
final_model.objective = None

print("================= SAVING MODEL ARTIFACT =================")

# Package model components and metadata 
model_artifact = {
    'model': final_model,
    'best_params': random_search.best_params_,
    'feature_cols': feature_cols,            
    'cutoff': cutoff,  
    'filtered_probs_ref': filtered_probs_ref,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'objective_params': {
        'base_fp_weight': fp_weight,
        'confidence_multiplier': confidence_mult
    }
}

output_path = '../models/crash_model.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(model_artifact, f)


