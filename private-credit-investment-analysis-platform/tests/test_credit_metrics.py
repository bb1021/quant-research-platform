from src.credit_metrics import CreditAssumptions, calculate_credit_metrics


def test_credit_metrics_core_ratios():
    assumptions = CreditAssumptions(revenue=100, ebitda=25, ebit=20, cash=10, total_debt=90, interest_expense=10, enterprise_value=300, free_cash_flow=15)
    metrics = calculate_credit_metrics(assumptions)

    assert metrics["EV / EBITDA"] == 12
    assert metrics["Debt / EBITDA"] == 3.6
    assert metrics["Net Debt / EBITDA"] == 3.2
    assert metrics["EBITDA / Interest"] == 2.5
    assert metrics["EBIT / Interest"] == 2.0
    assert metrics["FCF Conversion"] == 0.6


def test_credit_metrics_handles_zero_interest():
    assumptions = CreditAssumptions(interest_expense=0)
    metrics = calculate_credit_metrics(assumptions)

    assert metrics["EBITDA / Interest"] != metrics["EBITDA / Interest"]
