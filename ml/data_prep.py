"""
Machine Learning Data Preparation Module.

This script:
1. Fetches data from the database
2. Engineers features for ML training
3. Prepares training and test datasets
4. Handles missing values and outliers

Why Feature Engineering Matters:
- ML models only understand numbers, not raw text/categories
- Good features = Better predictions
- This is 70% of the ML work in production!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from backend.database import get_db_session, close_db_session
from backend.models import Customer, Application, Employment, Loan
from sqlalchemy import text


def fetch_training_data():
    """
    Fetch data from database and join relevant tables.
    
    Why join tables?
    - ML needs features from multiple sources (customer info, employment, loan history)
    - Database normalization splits data; we need to combine it for analysis
    
    Returns:
        pd.DataFrame: Combined dataset with all features
    
    Interview Note:
    "I used SQL joins to combine data from 4 tables (Customers, Applications,
    Employment, Loans). This creates a unified view of each loan application
    with all relevant features for credit scoring."
    """
    session = get_db_session()
    
    try:
        # ============================================
        # SQL QUERY TO FETCH TRAINING DATA
        # ============================================
        """
        Why SQL instead of ORM?
        - SQL is more efficient for complex joins
        - Easier to optimize with indexes
        - Direct control over what data is fetched
        
        We're joining:
        - Applications (target variable: approved/rejected)
        - Customers (demographics)
        - Employment (income, job stability)
        - Loans (historical performance for approved loans)
        """
        query = text("""
            SELECT 
                -- Customer Demographics
                c.customer_id,
                c.gender,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, c.date_of_birth)) as age,
                c.city,
                c.state,
                
                -- Employment Details
                e.employment_type,
                e.monthly_income,
                e.years_of_experience,
                
                -- Application Details
                a.application_id,
                a.loan_amount,
                a.loan_tenure_months,
                a.interest_rate,
                a.loan_purpose,
                a.application_status,
                a.credit_score,
                a.risk_probability,
                
                -- Loan Performance (if exists)
                l.loan_status,
                l.outstanding_balance,
                l.emi_amount
                
            FROM applications a
            INNER JOIN customers c ON a.customer_id = c.customer_id
            INNER JOIN employment e ON c.customer_id = e.customer_id
            LEFT JOIN loans l ON a.application_id = l.application_id
            
            WHERE a.application_status IN ('Approved', 'Rejected')
            ORDER BY a.application_id
        """)
        
        # Execute query and load into DataFrame
        result = session.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        print(f"✓ Fetched {len(df)} records from database")
        return df
        
    except Exception as e:
        print(f"✗ Error fetching data: {str(e)}")
        raise
        
    finally:
        close_db_session()


def engineer_features(df):
    """
    Create additional features from raw data.
    
    Feature Engineering = Creating new variables that help the model learn better.
    
    Args:
        df (pd.DataFrame): Raw data from database
    
    Returns:
        pd.DataFrame: Data with engineered features
    
    Interview Note:
    "Feature engineering is crucial. I created features like:
    - Debt-to-Income ratio (DTI)
    - Loan-to-Income ratio
    - Age groups (experience buckets)
    These derived features often have more predictive power than raw data."
    """
    print("\n[1/5] Engineering features...")
    
    # ============================================
    # FEATURE 1: DEBT-TO-INCOME RATIO (DTI)
    # ============================================
    """
    DTI = (EMI / Monthly Income) × 100
    
    Why it matters:
    - High DTI = High risk (borrower is over-leveraged)
    - DTI > 50% is typically rejected by banks
    - One of the most important credit risk indicators
    """
    # Calculate EMI (simplified - we could use actual EMI from loans table)
    # EMI ≈ Loan Amount / Tenure (simplified; actual EMI considers interest)
    df['estimated_emi'] = df['loan_amount'] / df['loan_tenure_months']
    df['debt_to_income_ratio'] = (df['estimated_emi'] / df['monthly_income']) * 100
    
    # ============================================
    # FEATURE 2: LOAN-TO-INCOME RATIO
    # ============================================
    """
    How many months' salary is the loan?
    
    Example: If loan = ₹1,200,000 and monthly income = ₹100,000
    Loan-to-Income = 12 (borrower needs 12 months' salary to repay)
    
    Lower is better (easier to repay)
    """
    df['loan_to_income_ratio'] = df['loan_amount'] / df['monthly_income']
    
    # ============================================
    # FEATURE 3: AGE GROUPS (ENCODED IMMEDIATELY)
    # ============================================
    """
    Why group ages?
    - Non-linear relationship between age and default risk
    - 25-35 and 50-60 have different risk profiles
    - Helps model learn age patterns
    
    IMPORTANT: Encode immediately as numeric to avoid pandas categorical issues
    """
    def encode_age_group(age):
        if age <= 25:
            return 0  # Young
        elif age <= 35:
            return 1  # Early_Career
        elif age <= 45:
            return 2  # Mid_Career
        elif age <= 55:
            return 3  # Senior
        else:
            return 4  # Retirement
    
    df['age_group_encoded'] = df['age'].apply(encode_age_group)
    
    # ============================================
    # FEATURE 4: EXPERIENCE BUCKETS (ENCODED IMMEDIATELY)
    # ============================================
    """
    Job stability indicator.
    More experience = More stable income
    """
    def encode_experience(years):
        if years <= 2:
            return 0  # Fresher
        elif years <= 5:
            return 1  # Junior
        elif years <= 10:
            return 2  # Mid_Level
        else:
            return 3  # Senior
    
    df['experience_encoded'] = df['years_of_experience'].apply(encode_experience)
    
    # ============================================
    # FEATURE 5: LOAN AMOUNT CATEGORY (ENCODED)
    # ============================================
    """
    Different risk patterns for different loan sizes.
    Small loans vs Large loans have different default behaviors.
    """
    def encode_loan_amount(amount):
        if amount <= 500000:
            return 0  # Small
        elif amount <= 1500000:
            return 1  # Medium
        elif amount <= 3000000:
            return 2  # Large
        else:
            return 3  # Very_Large
    
    df['loan_amount_category'] = df['loan_amount'].apply(encode_loan_amount)
    
    # ============================================
    # FEATURE 6: TENURE CATEGORY (ENCODED)
    # ============================================
    """
    Short-term vs Long-term loans.
    Longer tenure = Lower EMI but more interest risk
    """
    def encode_tenure(months):
        if months <= 24:
            return 0  # Short_Term
        elif months <= 60:
            return 1  # Medium_Term
        elif months <= 120:
            return 2  # Long_Term
        else:
            return 3  # Very_Long_Term
    
    df['tenure_category'] = df['loan_tenure_months'].apply(encode_tenure)
    
    # ============================================
    # FEATURE 7: HIGH RISK FLAG
    # ============================================
    """
    Simple rule-based flag for obviously risky applications.
    """
    df['high_risk_flag'] = (
        (df['debt_to_income_ratio'] > 50) |
        (df['loan_to_income_ratio'] > 36) |
        (df['monthly_income'] < 30000)
    ).astype(int)
    
    print(f"✓ Created {7} engineered features")
    return df


def encode_categorical_features(df):
    """
    Convert categorical variables to numbers.
    
    CRITICAL FIX: Ensure ALL encoded columns are explicitly numeric (int/float).
    
    Args:
        df (pd.DataFrame): Data with categorical features
    
    Returns:
        pd.DataFrame: Data with encoded features
    
    Interview Note:
    "I used Label Encoding for ordinal categories (like experience levels)
    and One-Hot Encoding for nominal categories (like employment type).
    All encoded features are explicitly cast to numeric types to ensure
    compatibility with scikit-learn."
    """
    print("\n[2/5] Encoding categorical features...")
    
    # ============================================
    # LABEL ENCODING (Ordinal Categories)
    # ============================================
    # Gender: Male=1, Female=0
    df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
    
    # Note: age_group_encoded and experience_encoded already created in engineer_features()
    
    # ============================================
    # ONE-HOT ENCODING (Nominal Categories)
    # ============================================
    # Employment type
    employment_dummies = pd.get_dummies(
        df['employment_type'],
        prefix='employment',
        drop_first=True,
        dtype=int  # CRITICAL: Force integer type
    )
    df = pd.concat([df, employment_dummies], axis=1)
    
    # Loan purpose
    purpose_dummies = pd.get_dummies(
        df['loan_purpose'],
        prefix='purpose',
        drop_first=True,
        dtype=int  # CRITICAL: Force integer type
    )
    df = pd.concat([df, purpose_dummies], axis=1)
    
    # State (Top 5 states only, rest = "Other")
    top_states = df['state'].value_counts().head(5).index
    df['state_grouped'] = df['state'].apply(lambda x: x if x in top_states else 'Other')
    state_dummies = pd.get_dummies(df['state_grouped'], prefix='state', drop_first=True, dtype=int)
    df = pd.concat([df, state_dummies], axis=1)
    
    # ============================================
    # ENSURE ALL NUMERIC FEATURES ARE PROPER TYPES
    # ============================================
    """
    CRITICAL FIX: Convert all numeric columns to explicit numeric types.
    This prevents pandas from treating them as objects.
    """
    numeric_columns = [
        'age', 'monthly_income', 'years_of_experience', 
        'loan_amount', 'loan_tenure_months', 'interest_rate',
        'debt_to_income_ratio', 'loan_to_income_ratio', 
        'estimated_emi', 'high_risk_flag',
        'gender_encoded', 'age_group_encoded', 'experience_encoded',
        'loan_amount_category', 'tenure_category'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✓ Encoded categorical features")
    return df


def create_target_variable(df):
    """
    Create the target variable (what we're predicting).
    
    Target: 1 = Default/High Risk, 0 = Good/Low Risk
    
    Args:
        df (pd.DataFrame): Data with application status
    
    Returns:
        pd.DataFrame: Data with target variable
    
    Interview Note:
    "The target variable is binary (0 or 1) for classification.
    I defined 'high risk' as applications that were rejected OR
    approved but later defaulted. This captures both pre-approval
    risk assessment and post-disbursement performance."
    """
    print("\n[3/5] Creating target variable...")
    
    # ============================================
    # DEFINE "BAD" LOANS (High Risk)
    # ============================================
    """
    We consider a loan "bad" if:
    1. Application was rejected (lender predicted high risk)
    2. Application was approved but loan defaulted
    
    Otherwise, it's a "good" loan.
    """
    df['is_high_risk'] = (
        (df['application_status'] == 'Rejected') |
        (df['loan_status'] == 'Defaulted')
    ).astype(int)
    
    # Check class distribution
    risk_distribution = df['is_high_risk'].value_counts()
    print(f"✓ Target variable created:")
    print(f"  - Low Risk (0): {risk_distribution.get(0, 0)} ({risk_distribution.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  - High Risk (1): {risk_distribution.get(1, 0)} ({risk_distribution.get(1, 0)/len(df)*100:.1f}%)")
    
    return df


def handle_missing_values(df):
    """
    Handle missing/null values.
    
    Strategies:
    1. Fill with median (for numeric features)
    2. Fill with mode (for categorical features)
    3. Drop if >50% missing
    
    Args:
        df (pd.DataFrame): Data with potential missing values
    
    Returns:
        pd.DataFrame: Cleaned data
    
    Interview Note:
    "I used median imputation for numeric features (robust to outliers)
    and mode imputation for categorical features. I also created
    'missing' indicator columns to capture if missingness itself
    is predictive."
    """
    print("\n[4/5] Handling missing values...")
    
    # Check missing values
    missing_count = df.isnull().sum()
    missing_cols = missing_count[missing_count > 0]
    
    if len(missing_cols) > 0:
        print(f"  Missing values found in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            print(f"    - {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  ✓ No missing values found")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print("✓ Missing values handled")
    return df


def select_features_for_training(df):
    """
    Select final features for ML training.
    
    CRITICAL FIX: Explicitly select ALL relevant numeric features.
    
    Args:
        df (pd.DataFrame): Prepared data
    
    Returns:
        tuple: (X, y, feature_names)
    
    Interview Note:
    "I selected features based on domain knowledge of credit risk:
    - Demographics (age, gender)
    - Financial capacity (income, DTI, LTI)
    - Loan characteristics (amount, tenure, purpose)
    - Employment stability (type, experience)
    This ensures the model learns from financially relevant signals."
    """
    print("\n[5/5] Selecting features for training...")
    
    # ============================================
    # EXPLICIT FEATURE LIST
    # ============================================
    """
    CRITICAL: List all features explicitly to ensure they're included.
    This prevents accidental filtering due to data type issues.
    """
    base_features = [
        # Demographic
        'age',
        'gender_encoded',
        'age_group_encoded',
        
        # Financial Capacity (CRITICAL for credit risk!)
        'monthly_income',
        'debt_to_income_ratio',
        'loan_to_income_ratio',
        'estimated_emi',
        
        # Employment
        'years_of_experience',
        'experience_encoded',
        
        # Loan Details
        'loan_amount',
        'loan_tenure_months',
        'interest_rate',
        'loan_amount_category',
        'tenure_category',
        
        # Risk Indicators
        'high_risk_flag'
    ]
    
    # Add one-hot encoded columns
    encoded_cols = [col for col in df.columns if col.startswith(('employment_', 'purpose_', 'state_'))]
    all_features = base_features + encoded_cols
    
    # ============================================
    # FILTER TO AVAILABLE COLUMNS
    # ============================================
    available_features = []
    missing_features = []
    
    for feature in all_features:
        if feature in df.columns:
            # Verify it's actually numeric
            if pd.api.types.is_numeric_dtype(df[feature]):
                available_features.append(feature)
            else:
                print(f"    ⚠ Skipping {feature}: Not numeric (dtype: {df[feature].dtype})")
        else:
            missing_features.append(feature)
    
    if missing_features:
        print(f"  ⚠ Missing features: {missing_features}")
    
    # ============================================
    # VERIFY CRITICAL FEATURES ARE PRESENT
    # ============================================
    critical_features = ['monthly_income', 'debt_to_income_ratio', 'loan_amount']
    missing_critical = [f for f in critical_features if f not in available_features]
    
    if missing_critical:
        print(f"  ❌ ERROR: Critical features missing: {missing_critical}")
        print(f"  This will severely impact model performance!")
    
    print(f"✓ Selected {len(available_features)} features for training:")
    
    # Group features by category for cleaner display
    demographic = [f for f in available_features if f in ['age', 'gender_encoded', 'age_group_encoded']]
    financial = [f for f in available_features if f in ['monthly_income', 'debt_to_income_ratio', 'loan_to_income_ratio', 'estimated_emi']]
    employment = [f for f in available_features if 'employment' in f or 'experience' in f]
    loan_details = [f for f in available_features if 'loan' in f or 'tenure' in f or 'interest' in f]
    location = [f for f in available_features if 'state' in f]
    purpose = [f for f in available_features if 'purpose' in f]
    other = [f for f in available_features if f not in demographic + financial + employment + loan_details + location + purpose]
    
    print(f"\n  Demographic ({len(demographic)}): {demographic}")
    print(f"  Financial ({len(financial)}): {financial}")
    print(f"  Employment ({len(employment)}): {employment}")
    print(f"  Loan Details ({len(loan_details)}): {loan_details}")
    print(f"  Location ({len(location)}): {location}")
    print(f"  Purpose ({len(purpose)}): {purpose}")
    if other:
        print(f"  Other ({len(other)}): {other}")
    
    # ============================================
    # CREATE X AND y
    # ============================================
    X = df[available_features].copy()
    y = df['is_high_risk'].copy()
    
    # Final check: ensure no object dtypes slipped through
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"\n  ❌ ERROR: Found object columns in X: {object_cols}")
        print(f"  Dropping these columns...")
        X = X.drop(columns=object_cols)
        available_features = [f for f in available_features if f not in object_cols]
    
    print(f"\n✓ Final training data shape: X={X.shape}, y={y.shape}")
    
    return X, y, available_features


def prepare_data_for_training():
    """
    Master function: Execute entire data preparation pipeline.
    
    Pipeline Steps:
    1. Fetch data from database
    2. Engineer features
    3. Encode categoricals
    4. Create target variable
    5. Handle missing values
    6. Select features
    
    Returns:
        tuple: (X, y, feature_names)
    
    Usage:
        X, y, features = prepare_data_for_training()
    """
    print("="*60)
    print("STARTING ML DATA PREPARATION")
    print("="*60)
    
    # Execute pipeline
    df = fetch_training_data()
    df = engineer_features(df)
    df = encode_categorical_features(df)
    df = create_target_variable(df)
    df = handle_missing_values(df)
    X, y, feature_names = select_features_for_training(df)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"✓ Total samples: {len(X)}")
    print(f"✓ Total features: {len(feature_names)}")
    print(f"✓ Class balance: {y.value_counts().to_dict()}")
    print("="*60 + "\n")
    
    return X, y, feature_names


# ============================================
# TEST EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Test data preparation pipeline.
    
    Run this to verify data prep works:
        python ml/data_prep.py
    """
    try:
        X, y, features = prepare_data_for_training()
        print("✓ Data preparation successful!")
        print(f"✓ Ready for model training with {X.shape[0]} samples and {X.shape[1]} features")
        
    except Exception as e:
        print(f"✗ Data preparation failed: {str(e)}")
        import traceback
        traceback.print_exc()