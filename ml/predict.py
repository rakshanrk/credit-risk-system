"""
Machine Learning Prediction Module.

This module loads the trained model and makes predictions
on new loan applications.

Used by Flask API to predict credit risk in real-time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from backend.database import get_db_session, close_db_session
from backend.models import Customer, Employment


# ============================================
# LOAD TRAINED MODEL
# ============================================
"""
Load model once when module is imported (not every prediction).

Why load once?
- Loading .pkl file is slow (~100ms)
- Keep model in memory for fast predictions (~1ms)
- Standard practice in production ML systems
"""
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')
MODEL_DATA = None

def load_model():
    """
    Load trained model from disk.
    
    Returns:
        dict: Model data containing model and feature names
    
    Interview Note:
    "I implemented lazy loading - the model loads only when first needed,
    then stays in memory. This optimizes API startup time while ensuring
    fast predictions."
    """
    global MODEL_DATA
    
    if MODEL_DATA is None:
        try:
            print(f"Loading model from {MODEL_PATH}...")
            MODEL_DATA = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded successfully")
            print(f"✓ Training date: {MODEL_DATA.get('training_date', 'Unknown')}")
        except FileNotFoundError:
            print("✗ Model file not found. Please train the model first:")
            print("  python ml/train_model.py")
            raise
    
    return MODEL_DATA


def prepare_features_for_prediction(customer_id, loan_amount, loan_tenure_months, 
                                    interest_rate, loan_purpose):
    """
    Prepare features for a new loan application.
    
    This function:
    1. Fetches customer data from database
    2. Engineers same features as training
    3. Ensures feature order matches training
    
    Args:
        customer_id (int): Customer ID
        loan_amount (float): Requested loan amount
        loan_tenure_months (int): Loan tenure
        interest_rate (float): Interest rate
        loan_purpose (str): Purpose of loan
    
    Returns:
        pd.DataFrame: Features ready for prediction
    
    Interview Note:
    "Feature engineering for prediction must EXACTLY match training.
    Same formulas, same encoding, same order. Any mismatch causes
    incorrect predictions. This is a common production ML bug!"
    """
    session = get_db_session()
    
    try:
        # ============================================
        # FETCH CUSTOMER DATA
        # ============================================
        customer = session.query(Customer).filter_by(customer_id=customer_id).first()
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        
        employment = session.query(Employment).filter_by(customer_id=customer_id).first()
        if not employment:
            raise ValueError(f"Employment data for customer {customer_id} not found")
        
        # ============================================
        # CALCULATE AGE
        # ============================================
        from datetime import datetime
        today = datetime.now().date()
        age = (today - customer.date_of_birth).days / 365.25
        
        # ============================================
        # ENGINEER FEATURES (Same as training!)
        # ============================================
        # Estimated EMI
        estimated_emi = loan_amount / loan_tenure_months
        
        # Debt-to-Income Ratio
        debt_to_income_ratio = (estimated_emi / float(employment.monthly_income)) * 100
        
        # Loan-to-Income Ratio
        loan_to_income_ratio = loan_amount / float(employment.monthly_income)
        
        # High Risk Flag
        high_risk_flag = int(
            (debt_to_income_ratio > 50) or
            (loan_to_income_ratio > 36) or
            (float(employment.monthly_income) < 30000)
        )
        
        # Age Group Encoding
        if age <= 25:
            age_group_encoded = 0
        elif age <= 35:
            age_group_encoded = 1
        elif age <= 45:
            age_group_encoded = 2
        elif age <= 55:
            age_group_encoded = 3
        else:
            age_group_encoded = 4
        
        # Experience Encoding
        exp = float(employment.years_of_experience)
        if exp <= 2:
            experience_encoded = 0
        elif exp <= 5:
            experience_encoded = 1
        elif exp <= 10:
            experience_encoded = 2
        else:
            experience_encoded = 3
        
        # Gender Encoding
        gender_encoded = 1 if customer.gender == 'Male' else 0
        
        # ============================================
        # CREATE FEATURE DICTIONARY
        # ============================================
        features = {
            # Demographic
            'age': age,
            'gender_encoded': gender_encoded,
            'age_group_encoded': age_group_encoded,
            
            # Employment
            'monthly_income': float(employment.monthly_income),
            'years_of_experience': float(employment.years_of_experience),
            'experience_encoded': experience_encoded,
            
            # Loan Details
            'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure_months,
            'interest_rate': interest_rate,
            
            # Engineered Features
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_to_income_ratio': loan_to_income_ratio,
            'high_risk_flag': high_risk_flag,
            'estimated_emi': estimated_emi,
            
            # One-Hot Encoded: Employment Type
            'employment_Self-Employed': 1 if employment.employment_type == 'Self-Employed' else 0,
            'employment_Salaried': 1 if employment.employment_type == 'Salaried' else 0,
            
            # One-Hot Encoded: Loan Purpose (just a few common ones)
            'purpose_Business Expansion': 1 if loan_purpose == 'Business Expansion' else 0,
            'purpose_Debt Consolidation': 1 if loan_purpose == 'Debt Consolidation' else 0,
            'purpose_Education': 1 if loan_purpose == 'Education' else 0,
            'purpose_Home Purchase': 1 if loan_purpose == 'Home Purchase' else 0,
            'purpose_Home Renovation': 1 if loan_purpose == 'Home Renovation' else 0,
            'purpose_Medical Emergency': 1 if loan_purpose == 'Medical Emergency' else 0,
            'purpose_Vehicle Purchase': 1 if loan_purpose == 'Vehicle Purchase' else 0,
            'purpose_Wedding Expenses': 1 if loan_purpose == 'Wedding Expenses' else 0,
            
            # One-Hot Encoded: State (top states)
            'state_Karnataka': 1 if customer.state == 'Karnataka' else 0,
            'state_Maharashtra': 1 if customer.state == 'Maharashtra' else 0,
            'state_Other': 1 if customer.state not in ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Delhi'] else 0,
            'state_Tamil Nadu': 1 if customer.state == 'Tamil Nadu' else 0,
            'state_Telangana': 1 if customer.state == 'Telangana' else 0,
        }
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        return feature_df
        
    finally:
        close_db_session()


def align_features_with_model(feature_df, model_feature_names):
    """
    Ensure features match model's expected feature order.
    
    Why needed?
    - Model expects features in specific order
    - Missing features must be set to 0
    - Extra features must be removed
    
    Args:
        feature_df (pd.DataFrame): Prepared features
        model_feature_names (list): Feature names from training
    
    Returns:
        pd.DataFrame: Aligned features
    
    Interview Note:
    "This is crucial for production ML. Feature alignment prevents
    'silent failures' where model produces garbage predictions
    because features are in wrong order."
    """
    # Create DataFrame with all model features set to 0
    aligned = pd.DataFrame(0, index=[0], columns=model_feature_names)
    
    # Fill in values for features we have
    for col in feature_df.columns:
        if col in aligned.columns:
            aligned[col] = feature_df[col].values[0]
    
    return aligned


def predict_credit_risk(customer_id, loan_amount, loan_tenure_months, 
                        interest_rate, loan_purpose):
    """
    Predict credit risk for a loan application.
    
    This is the main function called by Flask API.
    
    Args:
        customer_id (int): Customer ID
        loan_amount (float): Requested loan amount
        loan_tenure_months (int): Loan tenure in months
        interest_rate (float): Annual interest rate
        loan_purpose (str): Purpose of loan
    
    Returns:
        dict: Prediction results
        {
            'credit_score': 720.50,
            'risk_probability': 0.1234,
            'risk_level': 'Low',
            'recommendation': 'Approve'
        }
    
    Interview Note:
    "The model outputs probability (0-1). I convert this to:
    1. Credit Score (300-850 scale, like FICO)
    2. Risk Level (Low/Medium/High)
    3. Recommendation (Approve/Review/Reject)
    
    This makes predictions actionable for business users."
    """
    # Load model
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Prepare features
    feature_df = prepare_features_for_prediction(
        customer_id, loan_amount, loan_tenure_months, 
        interest_rate, loan_purpose
    )
    
    # Align features with model
    aligned_features = align_features_with_model(feature_df, feature_names)
    
    # ============================================
    # MAKE PREDICTION
    # ============================================
    # Predict probability of high risk (class 1)
    risk_probability = model.predict_proba(aligned_features)[0][1]
    
    # ============================================
    # CONVERT TO CREDIT SCORE (300-850 scale)
    # ============================================
    """
    Credit Score Mapping:
    - Risk Probability 0.0 → Credit Score 850 (Excellent)
    - Risk Probability 0.5 → Credit Score 575 (Average)
    - Risk Probability 1.0 → Credit Score 300 (Poor)
    
    Formula: Credit Score = 850 - (Risk Probability × 550)
    """
    credit_score = 850 - (risk_probability * 550)
    credit_score = max(300, min(850, credit_score))  # Clamp to [300, 850]
    
    # ============================================
    # DETERMINE RISK LEVEL & RECOMMENDATION
    # ============================================
    """
    Risk Levels:
    - Low Risk: Probability < 0.3 (Approve automatically)
    - Medium Risk: Probability 0.3-0.5 (Manual review required)
    - High Risk: Probability > 0.5 (Reject or require collateral)
    """
    if risk_probability < 0.3:
        risk_level = 'Low'
        recommendation = 'Approve'
    elif risk_probability < 0.5:
        risk_level = 'Medium'
        recommendation = 'Manual Review Required'
    else:
        risk_level = 'High'
        recommendation = 'Reject or Require Collateral'
    
    # ============================================
    # BUILD RESPONSE
    # ============================================
    result = {
        'credit_score': round(credit_score, 2),
        'risk_probability': round(risk_probability, 4),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'model_confidence': round(max(risk_probability, 1 - risk_probability), 4),
        'factors': {
            'debt_to_income_ratio': round(float(aligned_features.get('debt_to_income_ratio', [0]).values[0] if 'debt_to_income_ratio' in aligned_features.columns else 0), 2),
            'loan_to_income_ratio': round(float(aligned_features.get('loan_to_income_ratio', [0]).values[0] if 'loan_to_income_ratio' in aligned_features.columns else 0), 2),
            'monthly_income': float(aligned_features.get('monthly_income', [0]).values[0] if 'monthly_income' in aligned_features.columns else 0),
            'loan_amount': loan_amount,
            'high_risk_flag': int(aligned_features.get('high_risk_flag', [0]).values[0] if 'high_risk_flag' in aligned_features.columns else 0)
        }
    }
    
    return result


def batch_predict(applications):
    """
    Predict credit risk for multiple applications at once.
    
    Used for batch processing (e.g., overnight scoring of pending applications).
    
    Args:
        applications (list): List of application dicts
    
    Returns:
        list: List of prediction results
    
    Interview Note:
    "Batch prediction is more efficient than individual predictions.
    In production, we'd score pending applications overnight and
    cache results. This reduces API latency during business hours."
    """
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    results = []
    for app in applications:
        try:
            result = predict_credit_risk(
                app['customer_id'],
                app['loan_amount'],
                app['loan_tenure_months'],
                app['interest_rate'],
                app['loan_purpose']
            )
            results.append(result)
        except Exception as e:
            results.append({
                'error': str(e),
                'customer_id': app.get('customer_id')
            })
    
    return results


# ============================================
# TEST EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Test prediction on a sample customer.
    
    Run this to test predictions:
        python ml/predict.py
    """
    print("="*60)
    print("TESTING CREDIT RISK PREDICTION")
    print("="*60)
    
    # First, get a valid customer ID from database
    try:
        session = get_db_session()
        first_customer = session.query(Customer).first()
        
        if not first_customer:
            print("\n✗ No customers found in database. Please run seed_data.py first.")
            close_db_session()
            exit(1)
        
        customer_id = first_customer.customer_id
        print(f"\n  Using Customer ID: {customer_id} ({first_customer.full_name})")
        close_db_session()
        
        # Test prediction
        result = predict_credit_risk(
            customer_id=customer_id,
            loan_amount=1000000,
            loan_tenure_months=60,
            interest_rate=9.5,
            loan_purpose='Home Purchase'
        )
        
        print("\nPrediction Result:")
        print(f"  Credit Score: {result['credit_score']}")
        print(f"  Risk Probability: {result['risk_probability']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Model Confidence: {result['model_confidence']}")
        print(f"\n  Key Factors:")
        for key, value in result['factors'].items():
            print(f"    {key}: {value}")
        
        print("\n✓ Prediction successful!")
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()