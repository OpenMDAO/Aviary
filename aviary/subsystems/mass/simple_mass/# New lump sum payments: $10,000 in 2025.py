# New lump sum payments: $10,000 in 2025, $4,000 in 2026 and 2027
lump_sum_payments = {0: 10000, 12: 4000, 24: 4000}  # Months from now when lump sums occur

# Reset loan balances
private_loan_balance = 58000  
gov_loan_balance = 28760.60  

# Reset months counter
months = 0

private_interest_rate = 0.05 / 12  # Monthly interest rate for private loans (5% annually)
gov_interest_rate = 0.04 / 12      # Monthly interest rate for government loans (4% annually)

# Payment schedule
initial_months = (2028 - 2025) * 12  # Number of months until January 2028
private_payment_initial = 360
gov_payment_initial = 0  # Government loans in forbearance

private_payment_after = 1500
gov_payment_after = 1000

# Payments until January 2028 with updated lump sum payments
for month in range(initial_months):
    if private_loan_balance > 0:
        interest = private_loan_balance * private_interest_rate
        private_loan_balance += interest - private_payment_initial
        private_loan_balance = max(private_loan_balance, 0)  # Ensure no negative balance

    # Government loan accrues interest, but lump sum payments occur at specified months
    if gov_loan_balance > 0:
        interest = gov_loan_balance * gov_interest_rate
        gov_loan_balance += interest

        if month in lump_sum_payments:
            gov_loan_balance -= lump_sum_payments[month]
            gov_loan_balance = max(gov_loan_balance, 0)

    months += 1

# Payments starting January 2028
while private_loan_balance > 0 or gov_loan_balance > 0:
    if private_loan_balance > 0:
        interest = private_loan_balance * private_interest_rate
        private_loan_balance += interest - private_payment_after
        private_loan_balance = max(private_loan_balance, 0)

    if gov_loan_balance > 0:
        interest = gov_loan_balance * gov_interest_rate
        gov_loan_balance += interest - gov_payment_after
        gov_loan_balance = max(gov_loan_balance, 0)

    months += 1

    # If private loans are paid off, redirect full payment to government loans
    if private_loan_balance == 0 and gov_loan_balance > 0:
        gov_payment_after += private_payment_after
        private_payment_after = 0

    # If government loans are paid off, redirect full payment to private loans
    if gov_loan_balance == 0 and private_loan_balance > 0:
        private_payment_after += gov_payment_after
        gov_payment_after = 0

months
