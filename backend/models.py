"""
SQLAlchemy ORM Models for all 9 database tables.

These classes represent database tables as Python objects.
Each class maps to one table, and each attribute maps to a column.

Why ORM (Object-Relational Mapping)?
- Write Python code instead of SQL queries
- Type safety and IDE autocomplete
- Automatic relationship handling
- Easier to maintain and test
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, String, Date, DateTime, Numeric, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base

# MODEL 1: CUSTOMER
class Customer(Base):
    """
    Represents a customer in the system.
    
    Relationships:
    - One customer can have many applications
    - One customer can have many loans
    - One customer can have many employment records
    """
    __tablename__ = 'customers'
    
    customer_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10), CheckConstraint("gender IN ('Male', 'Female', 'Other')"))
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(15), nullable=False)
    address = Column(Text, nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=False)
    pincode = Column(String(10), nullable=False)
    pan_number = Column(String(10), unique=True, nullable=False)
    aadhar_number = Column(String(12), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships (enables easy navigation between related tables)
    applications = relationship("Application", back_populates="customer", cascade="all, delete-orphan")
    loans = relationship("Loan", back_populates="customer", cascade="all, delete-orphan")
    employment_records = relationship("Employment", back_populates="customer", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Customer(id={self.customer_id}, name='{self.full_name}', email='{self.email}')>"


# MODEL 2: APPLICATION
class Application(Base):
    """
    Represents a loan application submitted by a customer.
    
    Relationships:
    - Belongs to one customer
    - Can have one approved_loan record
    - Can have one loan record (if approved and disbursed)
    """
    __tablename__ = 'applications'
    
    application_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False)
    loan_amount = Column(Numeric(15, 2), CheckConstraint('loan_amount > 0'), nullable=False)
    loan_purpose = Column(String(100), nullable=False)
    loan_tenure_months = Column(Integer, CheckConstraint('loan_tenure_months > 0'), nullable=False)
    interest_rate = Column(Numeric(5, 2), nullable=False)
    application_date = Column(Date, default=func.current_date())
    application_status = Column(
        String(20), 
        CheckConstraint("application_status IN ('Pending', 'Approved', 'Rejected')"),
        default='Pending'
    )
    credit_score = Column(Numeric(5, 2))  # Populated by ML model
    risk_probability = Column(Numeric(5, 4))  # Probability of default (0-1)
    remarks = Column(Text)
    
    # Relationships
    customer = relationship("Customer", back_populates="applications")
    approved_loan = relationship("ApprovedLoan", back_populates="application", uselist=False, cascade="all, delete-orphan")
    loan = relationship("Loan", back_populates="application", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Application(id={self.application_id}, customer_id={self.customer_id}, status='{self.application_status}')>"


# MODEL 3: EMPLOYMENT
class Employment(Base):
    """
    Stores employment details for income verification.
    
    Relationships:
    - Belongs to one customer
    """
    __tablename__ = 'employment'
    
    employment_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False)
    employer_name = Column(String(100), nullable=False)
    job_title = Column(String(100), nullable=False)
    employment_type = Column(
        String(20),
        CheckConstraint("employment_type IN ('Salaried', 'Self-Employed', 'Business')")
    )
    monthly_income = Column(Numeric(12, 2), CheckConstraint('monthly_income > 0'), nullable=False)
    years_of_experience = Column(Numeric(4, 1), nullable=False)
    employer_phone = Column(String(15))
    employment_start_date = Column(Date, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="employment_records")
    
    def __repr__(self):
        return f"<Employment(id={self.employment_id}, customer_id={self.customer_id}, employer='{self.employer_name}')>"


# MODEL 4: LOAN
class Loan(Base):
    """
    Represents an approved and disbursed loan.
    
    Relationships:
    - Belongs to one customer
    - Belongs to one application
    - Can have multiple repayments
    - Can have multiple collateral records
    - Can have multiple guarantors
    - Can have one NPA tracking record
    """
    __tablename__ = 'loans'
    
    loan_id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(Integer, ForeignKey('applications.application_id', ondelete='CASCADE'), unique=True, nullable=False)
    customer_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False)
    loan_amount = Column(Numeric(15, 2), nullable=False)
    disbursed_amount = Column(Numeric(15, 2), nullable=False)
    interest_rate = Column(Numeric(5, 2), nullable=False)
    tenure_months = Column(Integer, nullable=False)
    emi_amount = Column(Numeric(12, 2), nullable=False)
    disbursement_date = Column(Date, nullable=False)
    loan_status = Column(
        String(20),
        CheckConstraint("loan_status IN ('Active', 'Closed', 'Defaulted')"),
        default='Active'
    )
    outstanding_balance = Column(Numeric(15, 2), nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="loans")
    application = relationship("Application", back_populates="loan")
    repayments = relationship("Repayment", back_populates="loan", cascade="all, delete-orphan")
    collaterals = relationship("Collateral", back_populates="loan", cascade="all, delete-orphan")
    guarantors = relationship("Guarantor", back_populates="loan", cascade="all, delete-orphan")
    npa_tracking = relationship("NPATracking", back_populates="loan", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Loan(id={self.loan_id}, customer_id={self.customer_id}, status='{self.loan_status}')>"


# MODEL 5: COLLATERAL
class Collateral(Base):
    """
    Stores collateral/security details for secured loans.
    
    Relationships:
    - Belongs to one loan
    """
    __tablename__ = 'collateral'
    
    collateral_id = Column(Integer, primary_key=True, autoincrement=True)
    loan_id = Column(Integer, ForeignKey('loans.loan_id', ondelete='CASCADE'), nullable=False)
    collateral_type = Column(
        String(50),
        CheckConstraint("collateral_type IN ('Property', 'Vehicle', 'Gold', 'Securities', 'Other')"),
        nullable=False
    )
    collateral_value = Column(Numeric(15, 2), CheckConstraint('collateral_value > 0'), nullable=False)
    valuation_date = Column(Date, nullable=False)
    description = Column(Text)
    
    # Relationships
    loan = relationship("Loan", back_populates="collaterals")
    
    def __repr__(self):
        return f"<Collateral(id={self.collateral_id}, loan_id={self.loan_id}, type='{self.collateral_type}')>"


# MODEL 6: GUARANTOR
class Guarantor(Base):
    """
    Stores guarantor information for loans.
    
    Relationships:
    - Belongs to one loan
    """
    __tablename__ = 'guarantors'
    
    guarantor_id = Column(Integer, primary_key=True, autoincrement=True)
    loan_id = Column(Integer, ForeignKey('loans.loan_id', ondelete='CASCADE'), nullable=False)
    guarantor_name = Column(String(100), nullable=False)
    guarantor_relationship = Column(String(50), nullable=False)
    guarantor_phone = Column(String(15), nullable=False)
    guarantor_email = Column(String(100))
    guarantor_address = Column(Text, nullable=False)
    guarantor_pan = Column(String(10), nullable=False)
    guarantor_income = Column(Numeric(12, 2), nullable=False)
    
    # Relationships
    loan = relationship("Loan", back_populates="guarantors")
    
    def __repr__(self):
        return f"<Guarantor(id={self.guarantor_id}, loan_id={self.loan_id}, name='{self.guarantor_name}')>"


# MODEL 7: APPROVED_LOAN
class ApprovedLoan(Base):
    """
    Tracks final approval decisions for applications.
    
    Relationships:
    - Belongs to one application
    """
    __tablename__ = 'approved_loans'
    
    approval_id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(Integer, ForeignKey('applications.application_id', ondelete='CASCADE'), unique=True, nullable=False)
    approved_by = Column(String(100), nullable=False)
    approval_date = Column(Date, default=func.current_date())
    approved_amount = Column(Numeric(15, 2), nullable=False)
    approved_tenure_months = Column(Integer, nullable=False)
    final_interest_rate = Column(Numeric(5, 2), nullable=False)
    conditions = Column(Text)
    
    # Relationships
    application = relationship("Application", back_populates="approved_loan")
    
    def __repr__(self):
        return f"<ApprovedLoan(id={self.approval_id}, application_id={self.application_id})>"


# MODEL 8: REPAYMENT
class Repayment(Base):
    """
    Tracks EMI payments made by customers.
    
    Relationships:
    - Belongs to one loan
    """
    __tablename__ = 'repayments'
    
    repayment_id = Column(Integer, primary_key=True, autoincrement=True)
    loan_id = Column(Integer, ForeignKey('loans.loan_id', ondelete='CASCADE'), nullable=False)
    payment_date = Column(Date, nullable=False)
    emi_due_date = Column(Date, nullable=False)
    amount_paid = Column(Numeric(12, 2), CheckConstraint('amount_paid > 0'), nullable=False)
    principal_paid = Column(Numeric(12, 2), nullable=False)
    interest_paid = Column(Numeric(12, 2), nullable=False)
    payment_method = Column(
        String(20),
        CheckConstraint("payment_method IN ('Bank Transfer', 'Cheque', 'Cash', 'UPI', 'Card')")
    )
    payment_status = Column(
        String(20),
        CheckConstraint("payment_status IN ('Success', 'Failed', 'Pending')"),
        default='Success'
    )
    late_fee = Column(Numeric(10, 2), default=0)
    
    # Relationships
    loan = relationship("Loan", back_populates="repayments")
    
    def __repr__(self):
        return f"<Repayment(id={self.repayment_id}, loan_id={self.loan_id}, amount={self.amount_paid})>"


# MODEL 9: NPA_TRACKING
class NPATracking(Base):
    """
    Tracks Non-Performing Assets (loans overdue > 90 days).
    
    Relationships:
    - Belongs to one loan
    """
    __tablename__ = 'npa_tracking'
    
    npa_id = Column(Integer, primary_key=True, autoincrement=True)
    loan_id = Column(Integer, ForeignKey('loans.loan_id', ondelete='CASCADE'), nullable=False)
    overdue_days = Column(Integer, CheckConstraint('overdue_days >= 90'), nullable=False)
    overdue_amount = Column(Numeric(15, 2), nullable=False)
    npa_classification = Column(
        String(20),
        CheckConstraint("npa_classification IN ('Sub-Standard', 'Doubtful', 'Loss')")
    )
    last_payment_date = Column(Date)
    recovery_action = Column(Text)
    added_to_npa_date = Column(Date, default=func.current_date())
    
    # Relationships
    loan = relationship("Loan", back_populates="npa_tracking")
    
    def __repr__(self):
        return f"<NPATracking(id={self.npa_id}, loan_id={self.loan_id}, classification='{self.npa_classification}')>"