import numpy as np

# Initial salary values
salary_user = 125000
salary_partner = 150000
salary_growth = 0.03  # 3% raise per year

# Retirement contribution (pre-tax)
retirement_contribution = 0.15  # 15% of salary

# Tax parameters (approximate federal tax brackets for married filing jointly, 2024)
standard_deduction = 29200  # Standard deduction for married filing jointly
tax_brackets = [(0.10, 23000), (0.12, 94300), (0.22, 201050), (0.24, 383900)]  # Simplified brackets

# Initial loan balances
private_loan_user = 58000  # User's private loans (6% interest)
gov_loan_user = 30000  # User's government loans (4% interest)

private_loan_partner = 60000  # Partner's private loans (7% interest)
gov_loan_partner = 60000  # Partner's government loans (4% interest)

# Loan payments per month
private_payment_user = 1500
gov_payment_user = 1000
private_payment_partner = 1600
gov_payment_partner = 900

# Monthly expenses
rent = 2000  # Rent & utilities
rent_increase = 100  # Rent increase per year

groceries = 500  # Groceries & gas
groceries_increase = 0.02  # 2% increase per year

fixed_expenses = 500  # Fixed subscriptions, car insurance, etc.

# Simulation for 10 years
years = 10
net_income_per_year = []

# Function to calculate taxes owed
def calculate_taxes(income):
    taxable_income = max(income - standard_deduction, 0)  # Apply standard deduction
    taxes_owed = 0
    for rate, bracket in tax_brackets:
        if taxable_income > bracket:
            taxes_owed += bracket * rate
            taxable_income -= bracket
        else:
            taxes_owed += taxable_income * rate
            break
    return taxes_owed

# Loan repayment tracking
user_private_paid = False
partner_private_paid = False

for year in range(years):
    # Salary growth
    salary_user *= (1 + salary_growth)
    salary_partner *= (1 + salary_growth)
    total_salary = salary_user + salary_partner

    # Pre-tax retirement contributions
    retirement_user = salary_user * retirement_contribution
    retirement_partner = salary_partner * retirement_contribution
    taxable_income = total_salary - (retirement_user + retirement_partner)

    # Calculate taxes
    taxes = calculate_taxes(taxable_income)

    # Loan payments (pay extra to government loans after private loans are paid)
    total_private_payment = 0
    total_gov_payment = 0

    if not user_private_paid:
        if private_loan_user > 0:
            total_private_payment += private_payment_user * 12
            private_loan_user *= (1 + 0.06 / 12) ** 12  # Apply interest
            private_loan_user -= private_payment_user * 12
        if private_loan_user <= 0:
            user_private_paid = True

    if not partner_private_paid:
        if private_loan_partner > 0:
            total_private_payment += private_payment_partner * 12
            private_loan_partner *= (1 + 0.07 / 12) ** 12  # Apply interest
            private_loan_partner -= private_payment_partner * 12
        if private_loan_partner <= 0:
            partner_private_paid = True

    # After private loans are paid, extra payment goes to government loans
    if user_private_paid:
        gov_payment_user += private_payment_user

    if partner_private_paid:
        gov_payment_partner += private_payment_partner

    if gov_loan_user > 0:
        total_gov_payment += gov_payment_user * 12
        gov_loan_user *= (1 + 0.04 / 12) ** 12  # Apply interest
        gov_loan_user -= gov_payment_user * 12

    if gov_loan_partner > 0:
        total_gov_payment += gov_payment_partner * 12
        gov_loan_partner *= (1 + 0.04 / 12) ** 12  # Apply interest
        gov_loan_partner -= gov_payment_partner * 12

    # Adjust expenses
    rent += rent_increase  # Increase rent yearly
    groceries *= (1 + groceries_increase)  # Increase groceries yearly

    # Calculate net income
    total_expenses = (rent * 12) + (groceries * 12) + (fixed_expenses * 12)
    total_loan_payments = total_private_payment + total_gov_payment

    net_income = taxable_income - taxes - total_expenses - total_loan_payments
    net_income_per_year.append(net_income)

net_income_per_year
