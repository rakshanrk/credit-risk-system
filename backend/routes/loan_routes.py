"""
Loan Management API Endpoints.

Handles:
- Loan application submission
- Application status retrieval
- Loan approval/rejection (will integrate ML model in Phase 3)

This is where we'll integrate the ML credit scoring model later.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from decimal import Decimal
from backend.database import get_db_session, close_db_session
from backend.models import Application, Customer, Loan, ApprovedLoan
from backend.utils.calculations import calculate_emi

# ============================================
# BLUEPRINT DEFINITION
# ============================================
loan_bp = Blueprint('loan', __name__)


# ============================================
# ENDPOINT 1: SUBMIT LOAN APPLICATION
# ============================================
@loan_bp.route('/apply', methods=['POST'])
def submit_application():
    """
    Submit a new loan application.
    
    This endpoint creates an Application record.
    In Phase 3, we'll add ML model prediction here to calculate
    credit_score and risk_probability.
    
    Request Body (JSON):
    {
        "customer_id": 51,
        "loan_amount": 500000,
        "loan_purpose": "Home Purchase",
        "loan_tenure_months": 60,
        "interest_rate": 9.5
    }
    
    Response (201):
    {
        "message": "Loan application submitted successfully",
        "application_id": 76,
        "status": "Pending",
        "credit_score": 720.50,
        "risk_probability": 0.1234
    }
    
    Interview Note:
    - This is where ML integration happens (Phase 3)
    - Risk probability determines approval/rejection
    - Credit score is calculated by Random Forest model
    """
    session = get_db_session()
    
    try:
        data = request.get_json()
        
        # ============================================
        # INPUT VALIDATION
        # ============================================
        required_fields = [
            'customer_id', 'loan_amount', 'loan_purpose',
            'loan_tenure_months', 'interest_rate'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Verify customer exists
        customer = session.query(Customer).filter_by(
            customer_id=data['customer_id']
        ).first()
        
        if not customer:
            return jsonify({
                "error": "Customer not found"
            }), 404
        
        # Validate loan amount
        if data['loan_amount'] <= 0:
            return jsonify({
                "error": "Loan amount must be greater than 0"
            }), 400
        
        # Validate tenure
        if data['loan_tenure_months'] <= 0:
            return jsonify({
                "error": "Loan tenure must be greater than 0"
            }), 400
        
        # ============================================
        # ML MODEL PREDICTION
        # ============================================
        """
        Use trained ML model to predict credit risk.
        
        The model returns:
        - credit_score (300-850)
        - risk_probability (0-1)
        - recommendation (Approve/Review/Reject)
        """
        try:
            from ml.predict import predict_credit_risk
            
            prediction = predict_credit_risk(
                customer_id=data['customer_id'],
                loan_amount=data['loan_amount'],
                loan_tenure_months=data['loan_tenure_months'],
                interest_rate=data['interest_rate'],
                loan_purpose=data['loan_purpose']
            )
            
            credit_score = Decimal(str(prediction['credit_score']))
            risk_probability = Decimal(str(prediction['risk_probability']))
            
            # Determine initial status based on ML recommendation
            if prediction['recommendation'] == 'Approve':
                initial_status = 'Approved'
            elif prediction['recommendation'] == 'Reject or Require Collateral':
                initial_status = 'Rejected'
            else:
                initial_status = 'Pending'  # Manual review
            
        except Exception as e:
            # Fallback if model fails (shouldn't happen in production)
            print(f"ML prediction failed: {str(e)}")
            credit_score = Decimal('650.00')
            risk_probability = Decimal('0.3000')
            initial_status = 'Pending'
        
        # ============================================
        # CREATE APPLICATION RECORD
        # ============================================
        new_application = Application(
            customer_id=data['customer_id'],
            loan_amount=Decimal(str(data['loan_amount'])),
            loan_purpose=data['loan_purpose'],
            loan_tenure_months=data['loan_tenure_months'],
            interest_rate=Decimal(str(data['interest_rate'])),
            application_status=initial_status,
            credit_score=credit_score,
            risk_probability=risk_probability,
            remarks=f"Credit score: {credit_score}, Risk: {risk_probability}"
        )
        
        session.add(new_application)
        session.commit()
        
        return jsonify({
            "message": "Loan application submitted successfully",
            "application_id": new_application.application_id,
            "status": new_application.application_status,
            "credit_score": float(credit_score),
            "risk_probability": float(risk_probability),
            "recommendation": "Low risk" if risk_probability < 0.2 else "Medium risk" if risk_probability < 0.4 else "High risk"
        }), 201
        
    except Exception as e:
        session.rollback()
        return jsonify({
            "error": f"Failed to submit application: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 2: GET ALL APPLICATIONS
# ============================================
@loan_bp.route('/applications', methods=['GET'])
def get_applications():
    """
    Retrieve all loan applications with filters.
    
    Query Parameters:
        ?status=Approved - Filter by status
        ?customer_id=51 - Filter by customer
        ?limit=20 - Pagination limit
    
    Response (200):
    {
        "applications": [...],
        "total": 75
    }
    """
    session = get_db_session()
    
    try:
        # Build query with optional filters
        query = session.query(Application)
        
        # Filter by status if provided
        status = request.args.get('status')
        if status:
            query = query.filter_by(application_status=status)
        
        # Filter by customer if provided
        customer_id = request.args.get('customer_id', type=int)
        if customer_id:
            query = query.filter_by(customer_id=customer_id)
        
        # Pagination
        limit = request.args.get('limit', default=50, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        applications = query.limit(limit).offset(offset).all()
        total = query.count()
        
        # Convert to JSON
        app_list = []
        for app in applications:
            app_list.append({
                "application_id": app.application_id,
                "customer_id": app.customer_id,
                "customer_name": app.customer.full_name,
                "loan_amount": float(app.loan_amount),
                "loan_purpose": app.loan_purpose,
                "tenure_months": app.loan_tenure_months,
                "interest_rate": float(app.interest_rate),
                "status": app.application_status,
                "credit_score": float(app.credit_score) if app.credit_score else None,
                "risk_probability": float(app.risk_probability) if app.risk_probability else None,
                "application_date": app.application_date.strftime('%Y-%m-%d')
            })
        
        return jsonify({
            "applications": app_list,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch applications: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 3: GET APPLICATION BY ID
# ============================================
@loan_bp.route('/applications/<int:application_id>', methods=['GET'])
def get_application_by_id(application_id):
    """
    Get detailed information about a specific application.
    
    Response (200):
    {
        "application_id": 76,
        "customer": {...},
        "loan_details": {...},
        "status": "Approved"
    }
    """
    session = get_db_session()
    
    try:
        application = session.query(Application).filter_by(
            application_id=application_id
        ).first()
        
        if not application:
            return jsonify({
                "error": "Application not found"
            }), 404
        
        # Get customer details
        customer = application.customer
        
        app_data = {
            "application_id": application.application_id,
            "customer": {
                "customer_id": customer.customer_id,
                "full_name": customer.full_name,
                "email": customer.email,
                "phone": customer.phone
            },
            "loan_details": {
                "loan_amount": float(application.loan_amount),
                "loan_purpose": application.loan_purpose,
                "tenure_months": application.loan_tenure_months,
                "interest_rate": float(application.interest_rate)
            },
            "status": application.application_status,
            "credit_score": float(application.credit_score) if application.credit_score else None,
            "risk_probability": float(application.risk_probability) if application.risk_probability else None,
            "application_date": application.application_date.strftime('%Y-%m-%d'),
            "remarks": application.remarks
        }
        
        return jsonify(app_data), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch application: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 4: GET ALL LOANS
# ============================================
@loan_bp.route('/loans', methods=['GET'])
def get_loans():
    """
    Retrieve all disbursed loans.
    
    Query Parameters:
        ?status=Active - Filter by loan status
        ?customer_id=51 - Filter by customer
    
    Response (200):
    {
        "loans": [...],
        "total": 50
    }
    """
    session = get_db_session()
    
    try:
        query = session.query(Loan)
        
        # Filter by status
        status = request.args.get('status')
        if status:
            query = query.filter_by(loan_status=status)
        
        # Filter by customer
        customer_id = request.args.get('customer_id', type=int)
        if customer_id:
            query = query.filter_by(customer_id=customer_id)
        
        # Pagination
        limit = request.args.get('limit', default=50, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        loans = query.limit(limit).offset(offset).all()
        total = query.count()
        
        # Convert to JSON
        loan_list = []
        for loan in loans:
            loan_list.append({
                "loan_id": loan.loan_id,
                "customer_id": loan.customer_id,
                "customer_name": loan.customer.full_name,
                "loan_amount": float(loan.loan_amount),
                "disbursed_amount": float(loan.disbursed_amount),
                "interest_rate": float(loan.interest_rate),
                "tenure_months": loan.tenure_months,
                "emi_amount": float(loan.emi_amount),
                "outstanding_balance": float(loan.outstanding_balance),
                "loan_status": loan.loan_status,
                "disbursement_date": loan.disbursement_date.strftime('%Y-%m-%d')
            })
        
        return jsonify({
            "loans": loan_list,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch loans: {str(e)}"
        }), 500
        
    finally:
        close_db_session()