"""
Machine Learning Model Training Module.

This script:
1. Loads prepared data
2. Splits into train/test sets
3. Trains Random Forest classifier
4. Evaluates model performance
5. Saves trained model as .pkl file

Why Random Forest?
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Industry standard for credit scoring
- Used by banks like JPMC, Wells Fargo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from ml.data_prep import prepare_data_for_training


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Why split?
    - Training set: Used to train the model
    - Test set: Used to evaluate performance on unseen data
    - Prevents overfitting (model memorizing training data)
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing (0.2 = 20%)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    
    Interview Note:
    "I used an 80-20 train-test split, which is standard in industry.
    The test set simulates how the model performs on new applications.
    Random state ensures reproducibility - same split every time."
    """
    print("\n[1/5] Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class distribution in both sets
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Training set class distribution: {y_train.value_counts().to_dict()}")
    print(f"✓ Test set class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    
    Random Forest = Ensemble of Decision Trees
    - Creates multiple decision trees (100 by default)
    - Each tree votes on the prediction
    - Final prediction = Majority vote
    
    Hyperparameters Explained:
    - n_estimators=100: Number of trees (more = better, but slower)
    - max_depth=10: Maximum tree depth (prevents overfitting)
    - min_samples_split=10: Minimum samples to split a node
    - min_samples_leaf=5: Minimum samples in a leaf node
    - class_weight='balanced': Handles imbalanced classes
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        RandomForestClassifier: Trained model
    
    Interview Note:
    "I used Random Forest because:
    1. It handles non-linear patterns in credit data
    2. It's interpretable (can see feature importance)
    3. It's robust to outliers and missing values
    4. It's used by major banks for credit scoring
    
    I tuned hyperparameters to prevent overfitting while
    maintaining good predictive accuracy."
    """
    print("\n[2/5] Training Random Forest model...")
    
    # ============================================
    # MODEL INITIALIZATION
    # ============================================
    """
    Class Weight = 'balanced':
    - Automatically adjusts for imbalanced classes
    - If 80% low risk, 20% high risk, model gives more weight to high risk
    - Prevents model from just predicting "low risk" for everything
    """
    model = RandomForestClassifier(
        n_estimators=100,          # Number of trees
        max_depth=10,              # Max tree depth
        min_samples_split=10,      # Min samples to split
        min_samples_leaf=5,        # Min samples per leaf
        max_features='sqrt',       # Features to consider per split
        class_weight='balanced',   # Handle imbalanced classes
        random_state=42,           # Reproducibility
        n_jobs=-1,                 # Use all CPU cores
        verbose=1                  # Show training progress
    )
    
    # ============================================
    # MODEL TRAINING
    # ============================================
    print("  Training in progress...")
    model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    # ============================================
    # FEATURE IMPORTANCE
    # ============================================
    """
    Feature Importance = How much each feature contributes to predictions.
    
    Higher importance = More influential in decision-making.
    
    Interview Gold: Always analyze feature importance!
    Shows you understand what drives credit risk.
    """
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Metrics Explained:
    - Accuracy: Overall correctness (but can be misleading with imbalanced data)
    - Precision: Of predicted high-risk, how many are actually high-risk?
    - Recall: Of actual high-risk, how many did we catch?
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)
    
    Args:
        model: Trained model
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    
    Returns:
        dict: Performance metrics
    
    Interview Note:
    "For credit risk, I focused on Recall (catching high-risk applicants)
    more than Precision. It's better to reject a few good applicants
    than approve bad ones (which lead to defaults and losses).
    
    However, we need balance - too high recall means rejecting everyone!"
    """
    print("\n[3/5] Evaluating model performance...")
    
    # ============================================
    # PREDICTIONS
    # ============================================
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probability predictions (for ROC-AUC)
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Probability of class 1 (high risk)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # ============================================
    # CALCULATE METRICS
    # ============================================
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    }
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    print("\n  TRAINING SET PERFORMANCE:")
    print(f"    Accuracy:  {metrics['train']['accuracy']:.4f}")
    print(f"    Precision: {metrics['train']['precision']:.4f}")
    print(f"    Recall:    {metrics['train']['recall']:.4f}")
    print(f"    F1 Score:  {metrics['train']['f1']:.4f}")
    print(f"    ROC-AUC:   {metrics['train']['roc_auc']:.4f}")
    
    print("\n  TEST SET PERFORMANCE:")
    print(f"    Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"    Precision: {metrics['test']['precision']:.4f}")
    print(f"    Recall:    {metrics['test']['recall']:.4f}")
    print(f"    F1 Score:  {metrics['test']['f1']:.4f}")
    print(f"    ROC-AUC:   {metrics['test']['roc_auc']:.4f}")
    
    # ============================================
    # CONFUSION MATRIX
    # ============================================
    """
    Confusion Matrix:
    
                    Predicted
                    0 (Low)  1 (High)
    Actual  0 (Low)   TN       FP
            1 (High)  FN       TP
    
    - True Positive (TP): Correctly predicted high risk
    - True Negative (TN): Correctly predicted low risk
    - False Positive (FP): Predicted high risk, actually low (Type I error)
    - False Negative (FN): Predicted low risk, actually high (Type II error)
    
    In credit risk: FN is WORSE than FP!
    (Missing a default is worse than rejecting a good applicant)
    """
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n  CONFUSION MATRIX (Test Set):")
    print(f"    True Negatives:  {cm[0][0]} (Correctly predicted low risk)")
    print(f"    False Positives: {cm[0][1]} (Incorrectly predicted high risk)")
    print(f"    False Negatives: {cm[1][0]} (Missed high-risk - BAD!)")
    print(f"    True Positives:  {cm[1][1]} (Correctly caught high risk)")
    
    # ============================================
    # OVERFITTING CHECK
    # ============================================
    """
    Overfitting = Model memorizes training data, performs poorly on new data
    
    Signs of overfitting:
    - Training accuracy >> Test accuracy
    - Training accuracy > 95% but test accuracy < 80%
    
    Our goal: Similar performance on both sets
    """
    train_test_gap = metrics['train']['accuracy'] - metrics['test']['accuracy']
    print(f"\n  OVERFITTING CHECK:")
    print(f"    Train-Test Accuracy Gap: {train_test_gap:.4f}")
    if train_test_gap < 0.05:
        print(f"    ✓ Model generalizes well (gap < 5%)")
    elif train_test_gap < 0.10:
        print(f"    ⚠ Slight overfitting (gap 5-10%)")
    else:
        print(f"    ✗ Significant overfitting (gap > 10%)")
    
    return metrics


def cross_validate_model(model, X, y):
    """
    Perform k-fold cross-validation.
    
    Cross-Validation = Train/test on multiple splits
    - Divide data into k folds (e.g., 5)
    - Train on 4 folds, test on 1
    - Repeat 5 times, each fold as test once
    - Average results
    
    Why?
    - More robust evaluation
    - Uses all data for both training and testing
    - Reduces variance in performance estimates
    
    Args:
        model: Trained model
        X: All features
        y: All targets
    
    Returns:
        dict: Cross-validation scores
    
    Interview Note:
    "I used 5-fold cross-validation to ensure the model's performance
    wasn't due to a lucky train-test split. This gives a more reliable
    estimate of how the model will perform in production."
    """
    print("\n[4/5] Performing cross-validation...")
    
    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print(f"\n  5-Fold Cross-Validation Results:")
    print(f"    Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"    ROC-AUC:  {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std():.4f})")
    print(f"    Individual fold accuracies: {[f'{s:.4f}' for s in cv_scores]}")
    
    return {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_roc_auc_mean': cv_roc_auc.mean(),
        'cv_roc_auc_std': cv_roc_auc.std()
    }


def save_model(model, feature_names, filename='credit_model.pkl'):
    """
    Save trained model to disk.
    
    Saves as .pkl (pickle) file which includes:
    - Trained model weights
    - Feature names (for prediction)
    - Model hyperparameters
    
    Args:
        model: Trained model
        feature_names: List of feature names
        filename: Output filename
    
    Interview Note:
    "I saved the model as a pickle file which can be loaded later for
    predictions. In production, this model would be deployed to a server
    and called via API endpoints. The .pkl file is version-controlled
    separately from code."
    """
    print("\n[5/5] Saving model to disk...")
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and feature names together
    model_path = os.path.join(model_dir, filename)
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_data, model_path)
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    return model_path


def train_credit_risk_model():
    """
    Master function: Complete ML training pipeline.
    
    Pipeline:
    1. Load and prepare data
    2. Split into train/test
    3. Train Random Forest
    4. Evaluate performance
    5. Cross-validate
    6. Save model
    
    Returns:
        tuple: (model, metrics, model_path)
    """
    print("="*60)
    print("STARTING CREDIT RISK MODEL TRAINING")
    print("="*60)
    
    try:
        # Step 1: Prepare data
        X, y, feature_names = prepare_data_for_training()
        
        # Step 2: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 3: Train model
        model = train_random_forest(X_train, y_train)
        
        # Step 4: Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Step 5: Cross-validate
        cv_metrics = cross_validate_model(model, X, y)
        metrics['cross_validation'] = cv_metrics
        
        # Step 6: Save model
        model_path = save_model(model, feature_names)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Test Accuracy: {metrics['test']['accuracy']:.4f}")
        print(f"✓ Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
        print(f"✓ Ready for integration into Flask API!")
        print("="*60 + "\n")
        
        return model, metrics, model_path
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ============================================
# TEST EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Train the model when script is run directly.
    
    Run this to train the model:
        python ml/train_model.py
    """
    train_credit_risk_model()