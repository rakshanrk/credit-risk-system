-- ============================================
-- CREDIT RISK ASSESSMENT DATABASE SCHEMA
-- 9 Tables: Customers, Applications, Employment, 
--           Loans, Collateral, Guarantors, 
--           Approved_Loans, Repayments, NPA_Tracking
-- ============================================

-- Drop tables if they exist (for clean re-runs)
DROP TABLE IF EXISTS npa_tracking CASCADE;
DROP TABLE IF EXISTS repayments CASCADE;
DROP TABLE IF EXISTS approved_loans CASCADE;
DROP TABLE IF EXISTS guarantors CASCADE;
DROP TABLE IF EXISTS collateral CASCADE;
DROP TABLE IF EXISTS loans CASCADE;
DROP TABLE IF EXISTS employment CASCADE;
DROP TABLE IF EXISTS applications CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ============================================
-- TABLE 1: CUSTOMERS
-- Stores basic customer information
-- ============================================
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(15) NOT NULL,
    address TEXT NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    pincode VARCHAR(10) NOT NULL,
    pan_number VARCHAR(10) UNIQUE NOT NULL,
    aadhar_number VARCHAR(12) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TABLE 2: APPLICATIONS
-- Tracks loan applications submitted by customers
-- ============================================
CREATE TABLE applications (
    application_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    loan_amount DECIMAL(15, 2) NOT NULL CHECK (loan_amount > 0),
    loan_purpose VARCHAR(100) NOT NULL,
    loan_tenure_months INT NOT NULL CHECK (loan_tenure_months > 0),
    interest_rate DECIMAL(5, 2) NOT NULL,
    application_date DATE DEFAULT CURRENT_DATE,
    application_status VARCHAR(20) DEFAULT 'Pending' 
        CHECK (application_status IN ('Pending', 'Approved', 'Rejected')),
    credit_score DECIMAL(5, 2), -- ML model will populate this
    risk_probability DECIMAL(5, 4), -- Probability of default (0-1)
    remarks TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ============================================
-- TABLE 3: EMPLOYMENT
-- Stores employment details for income verification
-- ============================================
CREATE TABLE employment (
    employment_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    employer_name VARCHAR(100) NOT NULL,
    job_title VARCHAR(100) NOT NULL,
    employment_type VARCHAR(20) CHECK (employment_type IN ('Salaried', 'Self-Employed', 'Business')),
    monthly_income DECIMAL(12, 2) NOT NULL CHECK (monthly_income > 0),
    years_of_experience DECIMAL(4, 1) NOT NULL,
    employer_phone VARCHAR(15),
    employment_start_date DATE NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ============================================
-- TABLE 4: LOANS
-- Stores details of all loans (approved & disbursed)
-- ============================================
CREATE TABLE loans (
    loan_id SERIAL PRIMARY KEY,
    application_id INT NOT NULL UNIQUE,
    customer_id INT NOT NULL,
    loan_amount DECIMAL(15, 2) NOT NULL,
    disbursed_amount DECIMAL(15, 2) NOT NULL,
    interest_rate DECIMAL(5, 2) NOT NULL,
    tenure_months INT NOT NULL,
    emi_amount DECIMAL(12, 2) NOT NULL,
    disbursement_date DATE NOT NULL,
    loan_status VARCHAR(20) DEFAULT 'Active' 
        CHECK (loan_status IN ('Active', 'Closed', 'Defaulted')),
    outstanding_balance DECIMAL(15, 2) NOT NULL,
    FOREIGN KEY (application_id) REFERENCES applications(application_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- ============================================
-- TABLE 5: COLLATERAL
-- Stores collateral/security details for secured loans
-- ============================================
CREATE TABLE collateral (
    collateral_id SERIAL PRIMARY KEY,
    loan_id INT NOT NULL,
    collateral_type VARCHAR(50) NOT NULL 
        CHECK (collateral_type IN ('Property', 'Vehicle', 'Gold', 'Securities', 'Other')),
    collateral_value DECIMAL(15, 2) NOT NULL CHECK (collateral_value > 0),
    valuation_date DATE NOT NULL,
    description TEXT,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

-- TABLE 6: GUARANTORS
-- Stores guarantor information for loans
-- ============================================
CREATE TABLE guarantors (
    guarantor_id SERIAL PRIMARY KEY,
    loan_id INT NOT NULL,
    guarantor_name VARCHAR(100) NOT NULL,
    guarantor_relationship VARCHAR(50) NOT NULL,
    guarantor_phone VARCHAR(15) NOT NULL,
    guarantor_email VARCHAR(100),
    guarantor_address TEXT NOT NULL,
    guarantor_pan VARCHAR(10) NOT NULL,
    guarantor_income DECIMAL(12, 2) NOT NULL,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

-- ============================================
-- TABLE 7: APPROVED_LOANS
-- Tracks final approval decisions
-- ============================================
CREATE TABLE approved_loans (
    approval_id SERIAL PRIMARY KEY,
    application_id INT NOT NULL UNIQUE,
    approved_by VARCHAR(100) NOT NULL, -- Officer/System name
    approval_date DATE DEFAULT CURRENT_DATE,
    approved_amount DECIMAL(15, 2) NOT NULL,
    approved_tenure_months INT NOT NULL,
    final_interest_rate DECIMAL(5, 2) NOT NULL,
    conditions TEXT, -- Any special conditions
    FOREIGN KEY (application_id) REFERENCES applications(application_id) ON DELETE CASCADE
);

-- ============================================
-- TABLE 8: REPAYMENTS
-- Tracks EMI payments made by customers
-- ============================================
CREATE TABLE repayments (
    repayment_id SERIAL PRIMARY KEY,
    loan_id INT NOT NULL,
    payment_date DATE NOT NULL,
    emi_due_date DATE NOT NULL,
    amount_paid DECIMAL(12, 2) NOT NULL CHECK (amount_paid > 0),
    principal_paid DECIMAL(12, 2) NOT NULL,
    interest_paid DECIMAL(12, 2) NOT NULL,
    payment_method VARCHAR(20) CHECK (payment_method IN ('Bank Transfer', 'Cheque', 'Cash', 'UPI', 'Card')),
    payment_status VARCHAR(20) DEFAULT 'Success' 
        CHECK (payment_status IN ('Success', 'Failed', 'Pending')),
    late_fee DECIMAL(10, 2) DEFAULT 0,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

-- TABLE 9: NPA_TRACKING
-- Tracks Non-Performing Assets (loans overdue > 90 days)
CREATE TABLE npa_tracking (
    npa_id SERIAL PRIMARY KEY,
    loan_id INT NOT NULL,
    overdue_days INT NOT NULL CHECK (overdue_days >= 90),
    overdue_amount DECIMAL(15, 2) NOT NULL,
    npa_classification VARCHAR(20) CHECK (npa_classification IN ('Sub-Standard', 'Doubtful', 'Loss')),
    last_payment_date DATE,
    recovery_action TEXT,
    added_to_npa_date DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

-- ============================================
-- INDEXES for Performance Optimization
-- (Interviewers love this!)
-- ============================================
CREATE INDEX idx_applications_customer ON applications(customer_id);
CREATE INDEX idx_applications_status ON applications(application_status);
CREATE INDEX idx_loans_customer ON loans(customer_id);
CREATE INDEX idx_loans_status ON loans(loan_status);
CREATE INDEX idx_repayments_loan ON repayments(loan_id);
CREATE INDEX idx_repayments_date ON repayments(payment_date);
CREATE INDEX idx_npa_loan ON npa_tracking(loan_id);

-- ============================================
-- SUCCESS MESSAGE
-- ============================================
DO $$
BEGIN
    RAISE NOTICE 'Database schema created successfully!';
    RAISE NOTICE 'Total tables: 9';
    RAISE NOTICE 'Indexes created: 7';
END $$;