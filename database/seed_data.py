"""
FIXED Seed script with CAUSAL default logic.

KEY CHANGES:
1. Defaults are now based on financial risk factors (NOT random)
2. High DTI → High default probability
3. Low income + high loan → High default probability
4. Application approval/rejection based on risk score
5. Loan status based on actual repayment behavior

This creates LEARNABLE patterns for the ML model.
"""

import sys
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_session, close_db_session
from backend.models import (
    Customer, Application, Employment, Loan,
    Collateral, Guarantor, ApprovedLoan, Repayment, NPATracking
)

# ============================================
# CONFIGURATION
# ============================================
NUM_CUSTOMERS = 50
NUM_APPLICATIONS = 75

# ============================================
# HELPER DATA (Same as before)
# ============================================
FIRST_NAMES = [
    "Arjun", "Priya", "Rahul", "Sneha", "Vikram", "Anjali", "Rohan", "Divya",
    "Karthik", "Meera", "Aditya", "Pooja", "Sanjay", "Kavya", "Amit", "Riya",
    "Nikhil", "Shreya", "Rajesh", "Neha", "Manoj", "Swati", "Suresh", "Anita",
    "Vivek", "Deepika", "Akash", "Isha", "Harish", "Tanvi", "Ravi", "Nisha",
    "Ashok", "Pallavi", "Ganesh", "Lakshmi", "Mohan", "Ramya", "Prakash", "Sonia",
    "Ajay", "Kritika", "Varun", "Megha", "Naveen", "Radha", "Sandeep", "Kamala",
    "Sunil", "Geeta"
]

LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Reddy", "Gupta", "Iyer", "Verma",
    "Nair", "Rao", "Joshi", "Mehta", "Desai", "Shah", "Mishra", "Agarwal",
    "Banerjee", "Das", "Chopra", "Malhotra", "Kapoor", "Bhat", "Menon", "Pillai"
]

CITIES = [
    ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bangalore", "Karnataka"),
    ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"),
    ("Pune", "Maharashtra"), ("Ahmedabad", "Gujarat"), ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"), ("Kochi", "Kerala"), ("Chandigarh", "Punjab")
]

COMPANIES = [
    "Infosys", "TCS", "Wipro", "Accenture", "Cognizant", "HCL Technologies",
    "Tech Mahindra", "Capgemini", "L&T Infotech", "Mindtree", "HDFC Bank",
    "ICICI Bank", "Reliance Industries", "Tata Motors", "Asian Paints"
]

JOB_TITLES = [
    "Software Engineer", "Senior Analyst", "Project Manager", "Data Scientist",
    "Business Analyst", "Operations Manager", "Marketing Manager", "HR Manager",
    "Financial Analyst", "Sales Manager", "Product Manager", "System Administrator"
]

LOAN_PURPOSES = [
    "Home Purchase", "Vehicle Purchase", "Business Expansion", "Education",
    "Medical Emergency", "Debt Consolidation", "Home Renovation", "Wedding Expenses"
]

COLLATERAL_TYPES = ["Property", "Vehicle", "Gold", "Securities"]

# ============================================
# UTILITY FUNCTIONS (Same as before)
# ============================================

def random_date(start_year=1970, end_year=2000):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

def random_recent_date(days_ago=365):
    return datetime.now().date() - timedelta(days=random.randint(0, days_ago))

def generate_pan():
    letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
    digits = ''.join(random.choices('0123456789', k=4))
    last_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return f"{letters}{digits}{last_letter}"

def generate_aadhar():
    return ''.join(random.choices('0123456789', k=12))

def generate_phone():
    return f"{random.randint(7, 9)}{random.randint(100000000, 999999999)}"

def calculate_emi(principal, rate_annual, tenure_months):
    rate_monthly = rate_annual / (12 * 100)
    if rate_monthly == 0:
        return principal / tenure_months
    emi = (principal * rate_monthly * (1 + rate_monthly) ** tenure_months) / \
          ((1 + rate_monthly) ** tenure_months - 1)
    return round(emi, 2)

# ============================================
# NEW: CAUSAL RISK CALCULATION
# ============================================

def calculate_risk_score(monthly_income, loan_amount, tenure_months, 
                         employment_type, years_experience):
    """
    Calculate risk score based on REAL financial factors.
    
    Returns:
        tuple: (risk_score 0-100, should_approve bool)
    
    Interview Note:
    "I designed a rules-based credit scoring system that mimics real-world
    underwriting criteria. High DTI ratios, low income, and unstable employment
    increase default risk, which is reflected in the training labels."
    """
    # Calculate DTI (Debt-to-Income Ratio)
    emi = calculate_emi(loan_amount, 10.0, tenure_months)  # Assume 10% rate
    dti = (emi / monthly_income) * 100
    
    # Calculate LTI (Loan-to-Income Ratio)
    lti = loan_amount / monthly_income
    
    # ============================================
    # RISK FACTORS (Additive scoring)
    # ============================================
    risk_score = 0
    
    # Factor 1: DTI Ratio (Most important!)
    if dti > 50:
        risk_score += 40  # Very high risk
    elif dti > 40:
        risk_score += 25  # High risk
    elif dti > 30:
        risk_score += 10  # Medium risk
    else:
        risk_score += 0   # Low risk
    
    # Factor 2: Income Level
    if monthly_income < 30000:
        risk_score += 20  # Low income = high risk
    elif monthly_income < 50000:
        risk_score += 10
    elif monthly_income > 100000:
        risk_score -= 10  # High income = lower risk
    
    # Factor 3: Loan Size relative to income
    if lti > 36:
        risk_score += 15  # Loan is too large
    elif lti > 24:
        risk_score += 8
    
    # Factor 4: Employment Stability
    if employment_type == 'Self-Employed':
        risk_score += 10  # Less stable income
    elif employment_type == 'Business':
        risk_score += 5
    
    # Factor 5: Experience (Job stability indicator)
    if years_experience < 2:
        risk_score += 10  # New job = risky
    elif years_experience < 5:
        risk_score += 5
    elif years_experience > 10:
        risk_score -= 5  # Stable career
    
    # Factor 6: Loan Tenure
    if tenure_months > 120:
        risk_score += 5  # Long tenure = more uncertainty
    
    # Clamp to 0-100
    risk_score = max(0, min(100, risk_score))
    
    # ============================================
    # APPROVAL LOGIC
    # ============================================
    """
    Banks typically reject if:
    - Risk score > 60
    - DTI > 50%
    - Income < 25000
    """
    should_approve = (
        risk_score < 60 and
        dti < 50 and
        monthly_income >= 25000
    )
    
    return risk_score, should_approve


def calculate_default_probability(risk_score):
    """
    Convert risk score to default probability.
    
    Risk Score → Default Probability mapping:
    - 0-20: 5-10% chance
    - 20-40: 10-25% chance
    - 40-60: 25-50% chance
    - 60-80: 50-75% chance
    - 80-100: 75-95% chance
    """
    if risk_score < 20:
        return random.uniform(0.05, 0.10)
    elif risk_score < 40:
        return random.uniform(0.10, 0.25)
    elif risk_score < 60:
        return random.uniform(0.25, 0.50)
    elif risk_score < 80:
        return random.uniform(0.50, 0.75)
    else:
        return random.uniform(0.75, 0.95)


# ============================================
# DATA GENERATION FUNCTIONS
# ============================================

def create_customers(session, count=50):
    print(f"\n[1/9] Creating {count} customers...")
    customers = []
    
    for i in range(count):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        city, state = random.choice(CITIES)
        
        customer = Customer(
            full_name=f"{first_name} {last_name}",
            date_of_birth=random_date(),
            gender=random.choice(['Male', 'Female']),
            email=f"{first_name.lower()}.{last_name.lower()}{i}@email.com",
            phone=generate_phone(),
            address=f"{random.randint(1, 999)}, {random.choice(['MG Road', 'Park Street', 'Main Road', 'Gandhi Nagar'])}",
            city=city,
            state=state,
            pincode=f"{random.randint(100000, 999999)}",
            pan_number=generate_pan(),
            aadhar_number=generate_aadhar()
        )
        customers.append(customer)
    
    session.bulk_save_objects(customers)
    session.commit()
    print(f"✓ Created {count} customers")
    return session.query(Customer).all()


def create_employment(session, customers):
    print(f"\n[2/9] Creating employment records...")
    employment_records = []
    
    for customer in customers:
        # Realistic income distribution
        income_tier = random.choices(
            ['low', 'medium', 'high'],
            weights=[0.3, 0.5, 0.2]
        )[0]
        
        if income_tier == 'low':
            monthly_income = Decimal(random.randint(25000, 50000))
        elif income_tier == 'medium':
            monthly_income = Decimal(random.randint(50000, 100000))
        else:
            monthly_income = Decimal(random.randint(100000, 250000))
        
        employment = Employment(
            customer_id=customer.customer_id,
            employer_name=random.choice(COMPANIES),
            job_title=random.choice(JOB_TITLES),
            employment_type=random.choice(['Salaried', 'Self-Employed', 'Business']),
            monthly_income=monthly_income,
            years_of_experience=Decimal(random.uniform(1.0, 25.0)).quantize(Decimal('0.1')),
            employer_phone=generate_phone(),
            employment_start_date=random_recent_date(days_ago=3650)
        )
        employment_records.append(employment)
    
    session.bulk_save_objects(employment_records)
    session.commit()
    print(f"✓ Created {len(employment_records)} employment records")


def create_applications(session, customers, num_total=75):
    """
    FIXED: Applications now use CAUSAL risk scoring.
    """
    print(f"\n[3/9] Creating {num_total} applications with CAUSAL risk logic...")
    applications = []
    approved_count = 0
    rejected_count = 0
    pending_count = 0
    
    for i in range(num_total):
        customer = random.choice(customers)
        employment = session.query(Employment).filter_by(customer_id=customer.customer_id).first()
        
        # Generate loan details
        loan_amount = Decimal(random.choice([200000, 500000, 1000000, 1500000, 2000000, 3000000]))
        tenure_months = random.choice([12, 24, 36, 48, 60, 84, 120])
        interest_rate = Decimal(random.uniform(8.5, 14.5)).quantize(Decimal('0.01'))
        
        # ============================================
        # CAUSAL RISK CALCULATION
        # ============================================
        risk_score, should_approve = calculate_risk_score(
            float(employment.monthly_income),
            float(loan_amount),
            tenure_months,
            employment.employment_type,
            float(employment.years_of_experience)
        )
        
        default_probability = calculate_default_probability(risk_score)
        
        # Convert risk score to credit score (inverse relationship)
        credit_score = Decimal(850 - (risk_score * 5.5))  # 0 risk → 850, 100 risk → 300
        
        # Determine status
        if should_approve:
            if approved_count < 50:
                status = 'Approved'
                approved_count += 1
            else:
                status = 'Pending'
                pending_count += 1
        else:
            if rejected_count < 15:
                status = 'Rejected'
                rejected_count += 1
            else:
                status = 'Pending'
                pending_count += 1
        
        application = Application(
            customer_id=customer.customer_id,
            loan_amount=loan_amount,
            loan_purpose=random.choice(LOAN_PURPOSES),
            loan_tenure_months=tenure_months,
            interest_rate=interest_rate,
            application_date=random_recent_date(days_ago=180),
            application_status=status,
            credit_score=credit_score,
            risk_probability=Decimal(str(default_probability)).quantize(Decimal('0.0001')),
            remarks=f"Risk Score: {risk_score}/100, DTI calculated"
        )
        applications.append(application)
    
    session.bulk_save_objects(applications)
    session.commit()
    print(f"✓ Created {num_total} applications (Approved: {approved_count}, Rejected: {rejected_count}, Pending: {pending_count})")
    return session.query(Application).filter_by(application_status='Approved').all()


def create_approved_loans(session, approved_applications):
    print(f"\n[4/9] Creating approved loan records...")
    approved_loans = []
    
    for app in approved_applications:
        approved_loan = ApprovedLoan(
            application_id=app.application_id,
            approved_by=random.choice(['System', 'John Doe', 'Jane Smith', 'Risk Committee']),
            approval_date=app.application_date + timedelta(days=random.randint(3, 15)),
            approved_amount=app.loan_amount,
            approved_tenure_months=app.loan_tenure_months,
            final_interest_rate=app.interest_rate,
            conditions="Standard terms and conditions apply"
        )
        approved_loans.append(approved_loan)
    
    session.bulk_save_objects(approved_loans)
    session.commit()
    print(f"✓ Created {len(approved_loans)} approved loan records")


def create_loans(session, approved_applications):
    """
    FIXED: Loan defaults now based on risk probability (NOT random).
    """
    print(f"\n[5/9] Creating disbursed loans with CAUSAL default logic...")
    loans = []
    
    for app in approved_applications:
        disbursed_amount = app.loan_amount * Decimal('0.99')
        emi = calculate_emi(float(app.loan_amount), float(app.interest_rate), app.loan_tenure_months)
        
        # ============================================
        # CAUSAL DEFAULT LOGIC
        # ============================================
        """
        High risk_probability → Higher chance of default
        """
        default_threshold = random.random()
        
        if float(app.risk_probability) > default_threshold:
            # This loan WILL default
            loan_status = 'Defaulted'
            outstanding = app.loan_amount * Decimal(random.uniform(0.5, 0.9))
        elif random.random() < 0.2:
            # 20% chance of early closure (low risk loans)
            loan_status = 'Closed'
            outstanding = Decimal('0.00')
        else:
            # Active loan
            loan_status = 'Active'
            months_passed = random.randint(3, app.loan_tenure_months - 1)
            outstanding = app.loan_amount - (Decimal(emi) * Decimal(months_passed) * Decimal('0.7'))
            outstanding = max(outstanding, Decimal('0.00'))
        
        disbursement_date = app.application_date + timedelta(days=random.randint(5, 20))
        
        loan = Loan(
            application_id=app.application_id,
            customer_id=app.customer_id,
            loan_amount=app.loan_amount,
            disbursed_amount=disbursed_amount,
            interest_rate=app.interest_rate,
            tenure_months=app.loan_tenure_months,
            emi_amount=Decimal(emi),
            disbursement_date=disbursement_date,
            loan_status=loan_status,
            outstanding_balance=outstanding
        )
        loans.append(loan)
    
    session.bulk_save_objects(loans)
    session.commit()
    
    defaulted = len([l for l in loans if l.loan_status == 'Defaulted'])
    print(f"✓ Created {len(loans)} loans ({defaulted} defaulted based on risk probability)")
    return session.query(Loan).all()


def create_collateral(session, loans):
    print(f"\n[6/9] Creating collateral records...")
    collateral_records = []
    
    loans_with_collateral = random.sample(loans, int(len(loans) * 0.6))
    
    for loan in loans_with_collateral:
        collateral_type = random.choice(COLLATERAL_TYPES)
        collateral_value = loan.loan_amount * Decimal(random.uniform(1.2, 1.5))
        
        collateral = Collateral(
            loan_id=loan.loan_id,
            collateral_type=collateral_type,
            collateral_value=collateral_value,
            valuation_date=loan.disbursement_date - timedelta(days=random.randint(1, 30)),
            description=f"{collateral_type} valued at ₹{collateral_value:,.2f}"
        )
        collateral_records.append(collateral)
    
    session.bulk_save_objects(collateral_records)
    session.commit()
    print(f"✓ Created {len(collateral_records)} collateral records")


def create_guarantors(session, loans):
    print(f"\n[7/9] Creating guarantor records...")
    guarantors = []
    
    loans_with_guarantors = random.sample(loans, int(len(loans) * 0.4))
    
    for loan in loans_with_guarantors:
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        
        guarantor = Guarantor(
            loan_id=loan.loan_id,
            guarantor_name=f"{first_name} {last_name}",
            guarantor_relationship=random.choice(['Father', 'Mother', 'Spouse', 'Sibling', 'Friend']),
            guarantor_phone=generate_phone(),
            guarantor_email=f"{first_name.lower()}.{last_name.lower()}@email.com",
            guarantor_address=f"{random.randint(1, 999)}, {random.choice(['MG Road', 'Park Street'])}",
            guarantor_pan=generate_pan(),
            guarantor_income=Decimal(random.randint(40000, 150000))
        )
        guarantors.append(guarantor)
    
    session.bulk_save_objects(guarantors)
    session.commit()
    print(f"✓ Created {len(guarantors)} guarantor records")


def create_repayments(session, loans):
    print(f"\n[8/9] Creating repayment records...")
    repayments = []
    
    for loan in loans:
        if loan.loan_status == 'Defaulted':
            continue
        
        if loan.loan_status == 'Closed':
            num_repayments = loan.tenure_months
        else:
            num_repayments = random.randint(3, min(12, loan.tenure_months))
        
        for i in range(num_repayments):
            emi_due_date = loan.disbursement_date + timedelta(days=30 * (i + 1))
            payment_date = emi_due_date + timedelta(days=random.randint(-5, 10))
            
            interest_component = float(loan.emi_amount) * 0.4
            principal_component = float(loan.emi_amount) - interest_component
            
            is_late = random.random() < 0.1
            late_fee = Decimal(random.randint(100, 500)) if is_late else Decimal('0.00')
            
            repayment = Repayment(
                loan_id=loan.loan_id,
                payment_date=payment_date,
                emi_due_date=emi_due_date,
                amount_paid=loan.emi_amount + late_fee,
                principal_paid=Decimal(principal_component),
                interest_paid=Decimal(interest_component),
                payment_method=random.choice(['Bank Transfer', 'UPI', 'Card', 'Cheque']),
                payment_status='Success',
                late_fee=late_fee
            )
            repayments.append(repayment)
    
    session.bulk_save_objects(repayments)
    session.commit()
    print(f"✓ Created {len(repayments)} repayment records")


def create_npa_tracking(session, loans):
    print(f"\n[9/9] Creating NPA tracking records...")
    npa_records = []
    
    defaulted_loans = [loan for loan in loans if loan.loan_status == 'Defaulted']
    
    for loan in defaulted_loans:
        overdue_days = random.randint(90, 365)
        
        if overdue_days < 180:
            classification = 'Sub-Standard'
        elif overdue_days < 365:
            classification = 'Doubtful'
        else:
            classification = 'Loss'
        
        last_repayment = session.query(Repayment).filter_by(loan_id=loan.loan_id).order_by(Repayment.payment_date.desc()).first()
        last_payment_date = last_repayment.payment_date if last_repayment else None
        
        npa = NPATracking(
            loan_id=loan.loan_id,
            overdue_days=overdue_days,
            overdue_amount=loan.outstanding_balance,
            npa_classification=classification,
            last_payment_date=last_payment_date,
            recovery_action="Legal notice sent" if overdue_days > 180 else "Follow-up call scheduled",
            added_to_npa_date=(datetime.now().date() - timedelta(days=overdue_days - 90))
        )
        npa_records.append(npa)
    
    session.bulk_save_objects(npa_records)
    session.commit()
    print(f"✓ Created {len(npa_records)} NPA tracking records")


# ============================================
# MAIN EXECUTION
# ============================================

def seed_database():
    print("=" * 60)
    print("STARTING DATABASE SEEDING (WITH CAUSAL LOGIC)")
    print("=" * 60)
    
    session = get_db_session()
    
    try:
        existing_customers = session.query(Customer).count()
        if existing_customers > 0:
            print(f"\n⚠ Database already contains {existing_customers} customers.")
            response = input("Do you want to clear existing data and reseed? (yes/no): ")
            if response.lower() != 'yes':
                print("Seeding cancelled.")
                return
            
            print("\nClearing existing data...")
            session.query(NPATracking).delete()
            session.query(Repayment).delete()
            session.query(Guarantor).delete()
            session.query(Collateral).delete()
            session.query(Loan).delete()
            session.query(ApprovedLoan).delete()
            session.query(Application).delete()
            session.query(Employment).delete()
            session.query(Customer).delete()
            session.commit()
            print("✓ Existing data cleared")
        
        customers = create_customers(session, NUM_CUSTOMERS)
        create_employment(session, customers)
        approved_apps = create_applications(session, customers, NUM_APPLICATIONS)
        create_approved_loans(session, approved_apps)
        loans = create_loans(session, approved_apps)
        create_collateral(session, loans)
        create_guarantors(session, loans)
        create_repayments(session, loans)
        create_npa_tracking(session, loans)
        
        print("\n" + "=" * 60)
        print("DATABASE SEEDING COMPLETED (WITH CAUSAL LOGIC)!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Customers: {session.query(Customer).count()}")
        print(f"  - Applications: {session.query(Application).count()}")
        print(f"  - Loans: {session.query(Loan).count()}")
        print(f"  - Defaulted Loans: {session.query(Loan).filter_by(loan_status='Defaulted').count()}")
        print(f"  - NPA Tracking: {session.query(NPATracking).count()}")
        print("\n✓ Data now has CAUSAL patterns (defaults based on DTI/income/risk)!")
        
    except Exception as e:
        session.rollback()
        print(f"\n✗ Error during seeding: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        close_db_session()


if __name__ == "__main__":
    seed_database()