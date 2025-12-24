"""
Seed script to populate the database with realistic dummy data.

This script generates:
- 50 Customers
- 75 Applications (Approved: 50, Rejected: 15, Pending: 10)
- 50 Employment records
- 50 Loans (for approved applications)
- 30 Collateral records
- 20 Guarantors
- 50 Approved_Loans records
- 200+ Repayments (multiple per loan)
- 10 NPA_Tracking records (defaulted loans)

Why we need dummy data:
- Test API endpoints without manual data entry
- Train ML model with diverse examples
- Demonstrate portfolio analytics in dashboard
"""

import sys
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_session, close_db_session
from backend.models import (
    Customer, Application, Employment, Loan,
    Collateral, Guarantor, ApprovedLoan, Repayment, NPATracking
)

# CONFIGURATION
NUM_CUSTOMERS = 50
NUM_APPLICATIONS = 75  # More applications than customers (some apply multiple times)
NUM_APPROVED = 50
NUM_REJECTED = 15
NUM_PENDING = 10

# HELPER DATA
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

# UTILITY FUNCTIONS

def random_date(start_year=1970, end_year=2000):
    """Generate random date of birth (age 24-54)"""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def random_recent_date(days_ago=365):
    """Generate random date within last N days"""
    return datetime.now().date() - timedelta(days=random.randint(0, days_ago))


def generate_pan():
    """Generate fake PAN number (format: ABCDE1234F)"""
    letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
    digits = ''.join(random.choices('0123456789', k=4))
    last_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return f"{letters}{digits}{last_letter}"


def generate_aadhar():
    """Generate fake Aadhar number (12 digits)"""
    return ''.join(random.choices('0123456789', k=12))


def generate_phone():
    """Generate fake phone number"""
    return f"{random.randint(7, 9)}{random.randint(100000000, 999999999)}"


def calculate_emi(principal, rate_annual, tenure_months):
    """
    Calculate EMI using standard formula.
    EMI = [P × r × (1 + r)^n] / [(1 + r)^n - 1]
    
    Interview Note: This is the industry-standard EMI calculation formula.
    """
    rate_monthly = rate_annual / (12 * 100)  # Convert annual % to monthly decimal
    if rate_monthly == 0:
        return principal / tenure_months
    emi = (principal * rate_monthly * (1 + rate_monthly) ** tenure_months) / \
          ((1 + rate_monthly) ** tenure_months - 1)
    return round(emi, 2)


# DATA GENERATION FUNCTIONS

def create_customers(session, count=50):
    """Generate customer records"""
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
    print(f"Created {count} customers")
    return session.query(Customer).all()


def create_employment(session, customers):
    """Generate employment records"""
    print(f"\n[2/9] Creating employment records...")
    employment_records = []
    
    for customer in customers:
        employment = Employment(
            customer_id=customer.customer_id,
            employer_name=random.choice(COMPANIES),
            job_title=random.choice(JOB_TITLES),
            employment_type=random.choice(['Salaried', 'Self-Employed', 'Business']),
            monthly_income=Decimal(random.randint(30000, 200000)),
            years_of_experience=Decimal(random.uniform(1.0, 25.0)).quantize(Decimal('0.1')),
            employer_phone=generate_phone(),
            employment_start_date=random_recent_date(days_ago=3650)  # Up to 10 years ago
        )
        employment_records.append(employment)
    
    session.bulk_save_objects(employment_records)
    session.commit()
    print(f"Created {len(employment_records)} employment records")


def create_applications(session, customers, num_total=75):
    """Generate loan applications with varied statuses"""
    print(f"\n[3/9] Creating {num_total} applications...")
    applications = []
    
    # Ensure we have enough approved applications
    statuses = ['Approved'] * NUM_APPROVED + ['Rejected'] * NUM_REJECTED + ['Pending'] * NUM_PENDING
    random.shuffle(statuses)
    
    for i in range(num_total):
        customer = random.choice(customers)
        loan_amount = Decimal(random.choice([200000, 500000, 1000000, 1500000, 2000000, 3000000]))
        tenure_months = random.choice([12, 24, 36, 48, 60, 84, 120])
        interest_rate = Decimal(random.uniform(8.5, 14.5)).quantize(Decimal('0.01'))
        
        # Generate credit score and risk probability
        # Lower risk = higher credit score
        risk_prob = Decimal(random.uniform(0.05, 0.45)).quantize(Decimal('0.0001'))
        credit_score = Decimal(850 - (risk_prob * 1000)).quantize(Decimal('0.01'))
        
        application = Application(
            customer_id=customer.customer_id,
            loan_amount=loan_amount,
            loan_purpose=random.choice(LOAN_PURPOSES),
            loan_tenure_months=tenure_months,
            interest_rate=interest_rate,
            application_date=random_recent_date(days_ago=180),
            application_status=statuses[i],
            credit_score=credit_score,
            risk_probability=risk_prob,
            remarks=f"Application processed based on credit score: {credit_score}"
        )
        applications.append(application)
    
    session.bulk_save_objects(applications)
    session.commit()
    print(f"Created {num_total} applications (Approved: {NUM_APPROVED}, Rejected: {NUM_REJECTED}, Pending: {NUM_PENDING})")
    return session.query(Application).filter_by(application_status='Approved').all()


def create_approved_loans(session, approved_applications):
    """Create approved_loans records for approved applications"""
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
    print(f"Created {len(approved_loans)} approved loan records")


def create_loans(session, approved_applications):
    """Create disbursed loans"""
    print(f"\n[5/9] Creating disbursed loans...")
    loans = []
    
    for app in approved_applications:
        disbursed_amount = app.loan_amount * Decimal('0.99')  # 1% processing fee deducted
        emi = calculate_emi(float(app.loan_amount), float(app.interest_rate), app.loan_tenure_months)
        
        # Some loans are active, some closed, some defaulted
        status_weights = [0.7, 0.2, 0.1]  # 70% active, 20% closed, 10% defaulted
        loan_status = random.choices(['Active', 'Closed', 'Defaulted'], weights=status_weights)[0]
        
        disbursement_date = app.application_date + timedelta(days=random.randint(5, 20))
        
        # Calculate outstanding balance based on status
        if loan_status == 'Closed':
            outstanding = Decimal('0.00')
        elif loan_status == 'Defaulted':
            outstanding = app.loan_amount * Decimal(random.uniform(0.5, 0.9))
        else:  # Active
            months_passed = min(random.randint(3, app.loan_tenure_months - 1), app.loan_tenure_months)
            outstanding = app.loan_amount - (Decimal(emi) * Decimal(months_passed) * Decimal('0.7'))
            outstanding = max(outstanding, Decimal('0.00'))
        
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
    print(f"Created {len(loans)} disbursed loans")
    return session.query(Loan).all()


def create_collateral(session, loans):
    """Create collateral records for ~60% of loans"""
    print(f"\n[6/9] Creating collateral records...")
    collateral_records = []
    
    # Only 60% of loans have collateral
    loans_with_collateral = random.sample(loans, int(len(loans) * 0.6))
    
    for loan in loans_with_collateral:
        collateral_type = random.choice(COLLATERAL_TYPES)
        
        # Collateral value is typically 1.2-1.5x loan amount
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
    """Create guarantor records for ~40% of loans"""
    print(f"\n[7/9] Creating guarantor records...")
    guarantors = []
    
    # Only 40% of loans have guarantors
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
    """Create repayment records for active and closed loans"""
    print(f"\n[8/9] Creating repayment records...")
    repayments = []
    
    for loan in loans:
        if loan.loan_status == 'Defaulted':
            continue  # Defaulted loans have no recent repayments
        
        # Number of repayments based on loan status
        if loan.loan_status == 'Closed':
            num_repayments = loan.tenure_months
        else:  # Active
            num_repayments = random.randint(3, min(12, loan.tenure_months))
        
        for i in range(num_repayments):
            emi_due_date = loan.disbursement_date + timedelta(days=30 * (i + 1))
            payment_date = emi_due_date + timedelta(days=random.randint(-5, 10))
            
            # Split EMI into principal and interest (simplified)
            interest_component = float(loan.emi_amount) * 0.4  # ~40% interest
            principal_component = float(loan.emi_amount) - interest_component
            
            # 10% chance of late payment (with late fee)
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
    print(f"Created {len(repayments)} repayment records")


def create_npa_tracking(session, loans):
    """Create NPA tracking records for defaulted loans"""
    print(f"\n[9/9] Creating NPA tracking records...")
    npa_records = []
    
    defaulted_loans = [loan for loan in loans if loan.loan_status == 'Defaulted']
    
    for loan in defaulted_loans:
        overdue_days = random.randint(90, 365)
        
        # NPA classification based on overdue days
        if overdue_days < 180:
            classification = 'Sub-Standard'
        elif overdue_days < 365:
            classification = 'Doubtful'
        else:
            classification = 'Loss'
        
        # Get last repayment date if exists
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


# MAIN EXECUTION

def seed_database():
    """Main function to seed all tables"""
    print("=" * 60)
    print("STARTING DATABASE SEEDING")
    print("=" * 60)
    
    session = get_db_session()
    
    try:
        # Check if data already exists
        existing_customers = session.query(Customer).count()
        if existing_customers > 0:
            print(f"\nDatabase already contains {existing_customers} customers.")
            response = input("Do you want to clear existing data and reseed? (yes/no): ")
            if response.lower() != 'yes':
                print("Seeding cancelled.")
                return
            
            # Clear all tables (in reverse order due to foreign keys)
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
        
        # Create data in proper order
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
        print("DATABASE SEEDING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Customers: {session.query(Customer).count()}")
        print(f"  - Applications: {session.query(Application).count()}")
        print(f"  - Employment Records: {session.query(Employment).count()}")
        print(f"  - Loans: {session.query(Loan).count()}")
        print(f"  - Collateral: {session.query(Collateral).count()}")
        print(f"  - Guarantors: {session.query(Guarantor).count()}")
        print(f"  - Approved Loans: {session.query(ApprovedLoan).count()}")
        print(f"  - Repayments: {session.query(Repayment).count()}")
        print(f"  - NPA Tracking: {session.query(NPATracking).count()}")
        print("\nReady for API testing and ML training!")
        
    except Exception as e:
        session.rollback()
        print(f"\n✗ Error during seeding: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        close_db_session()


if __name__ == "__main__":
    seed_database()