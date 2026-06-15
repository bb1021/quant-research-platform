from __future__ import annotations

import pandas as pd

from .credit_metrics import CreditAssumptions
from .utils import clamp, safe_divide


DEFAULT_SCENARIOS = {
    "Base": {
        "revenue_growth": 0.08,
        "ebitda_margin": 0.27,
        "interest_rate": 0.095,
        "capex_pct_revenue": 0.045,
        "working_capital_impact": 8.0,
        "cost_savings": 6.0,
        "debt_repayment": 18.0,
    },
    "Downside": {
        "revenue_growth": -0.04,
        "ebitda_margin": 0.21,
        "interest_rate": 0.105,
        "capex_pct_revenue": 0.050,
        "working_capital_impact": 13.0,
        "cost_savings": 2.0,
        "debt_repayment": 0.0,
    },
    "Severe Downside": {
        "revenue_growth": -0.14,
        "ebitda_margin": 0.16,
        "interest_rate": 0.115,
        "capex_pct_revenue": 0.055,
        "working_capital_impact": 20.0,
        "cost_savings": 0.0,
        "debt_repayment": -22.0,
    },
}


def calculate_scenario(assumptions: CreditAssumptions, case_name: str, case: dict[str, float]) -> dict[str, float | str]:
    revenue = assumptions.revenue * (1 + case["revenue_growth"])
    ebitda = revenue * case["ebitda_margin"]
    capex = revenue * case["capex_pct_revenue"]
    interest = assumptions.total_debt * case["interest_rate"]
    free_cash_flow = ebitda - interest - capex - case["working_capital_impact"] + case["cost_savings"]
    debt = assumptions.total_debt - case["debt_repayment"] if case["debt_repayment"] >= 0 else assumptions.total_debt + abs(case["debt_repayment"])
    cash = assumptions.cash + max(free_cash_flow, 0.0) - max(-free_cash_flow, 0.0)
    liquidity = max(cash, 0.0) + assumptions.revolver_availability
    net_debt = debt - max(cash, 0.0)
    net_leverage = safe_divide(net_debt, ebitda)
    interest_coverage = safe_divide(ebitda, interest)
    resilience = downside_resilience_score(free_cash_flow, liquidity, net_leverage, interest_coverage)
    return {
        "Case": case_name,
        "Revenue": revenue,
        "EBITDA": ebitda,
        "Free Cash Flow": free_cash_flow,
        "Net Debt": net_debt,
        "Net Leverage": net_leverage,
        "Interest Coverage": interest_coverage,
        "Liquidity": liquidity,
        "Downside Resilience Score": resilience,
    }


def downside_resilience_score(fcf: float, liquidity: float, net_leverage: float, coverage: float) -> float:
    fcf_score = 100 if fcf >= 0 else clamp(55 + fcf * 2.0)
    liquidity_score = clamp(liquidity / 100.0 * 100)
    leverage_score = clamp(100 - max(net_leverage - 4.0, 0) / 2.5 * 100)
    coverage_score = clamp((coverage - 1.0) / 2.0 * 100)
    return round(0.30 * fcf_score + 0.25 * liquidity_score + 0.25 * leverage_score + 0.20 * coverage_score, 1)


def run_scenarios(
    assumptions: CreditAssumptions,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    scenario_set = scenarios or DEFAULT_SCENARIOS
    rows = [calculate_scenario(assumptions, name, case) for name, case in scenario_set.items()]
    return pd.DataFrame(rows)
