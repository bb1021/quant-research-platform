from __future__ import annotations

import pandas as pd

from .acquisition_screening import AcquisitionAssumptions
from .utils import clamp, safe_divide


DEFAULT_SCENARIOS = {
    "Downside": {
        "revenue_growth": 0.06,
        "ebitda_margin": 0.15,
        "capex_pct_revenue": 0.07,
        "working_capital_pct_revenue": 0.04,
        "interest_rate": 0.085,
        "synergy_savings": 4.0,
        "integration_costs": 26.0,
        "debt_repayment": -12.0,
    },
    "Base": {
        "revenue_growth": 0.18,
        "ebitda_margin": 0.23,
        "capex_pct_revenue": 0.055,
        "working_capital_pct_revenue": 0.025,
        "interest_rate": 0.065,
        "synergy_savings": 14.0,
        "integration_costs": 18.0,
        "debt_repayment": 10.0,
    },
    "Upside": {
        "revenue_growth": 0.30,
        "ebitda_margin": 0.29,
        "capex_pct_revenue": 0.045,
        "working_capital_pct_revenue": 0.015,
        "interest_rate": 0.055,
        "synergy_savings": 26.0,
        "integration_costs": 13.0,
        "debt_repayment": 28.0,
    },
}


def calculate_case(assumptions: AcquisitionAssumptions, case_name: str, case: dict[str, float]) -> dict[str, float | str]:
    revenue = assumptions.revenue * (1 + case["revenue_growth"])
    ebitda = revenue * case["ebitda_margin"]
    capex = revenue * case["capex_pct_revenue"]
    working_capital = revenue * case["working_capital_pct_revenue"]
    interest = assumptions.debt * case["interest_rate"]
    fcf = ebitda - capex - working_capital - interest + case["synergy_savings"] - case["integration_costs"]
    cash = assumptions.cash + max(fcf, 0.0) - max(-fcf, 0.0)
    debt = max(0.0, assumptions.debt - case["debt_repayment"]) if case["debt_repayment"] >= 0 else assumptions.debt + abs(case["debt_repayment"])
    net_debt = debt - max(cash, 0.0)
    net_leverage = safe_divide(net_debt, ebitda, default=0.0)
    liquidity = max(cash, 0.0)
    burden = safe_divide(case["integration_costs"], ebitda, default=0.0)
    break_even_margin = safe_divide(capex + working_capital + interest + case["integration_costs"] - case["synergy_savings"], revenue, default=0.0)
    resilience = downside_resilience_score(fcf, liquidity, net_leverage, burden)

    return {
        "Case": case_name,
        "Revenue": revenue,
        "EBITDA": ebitda,
        "FCF": fcf,
        "Net Debt": net_debt,
        "Net Leverage": net_leverage,
        "Liquidity": liquidity,
        "Integration Cost Burden": burden,
        "Break-even EBITDA Margin": break_even_margin,
        "Downside Resilience Score": resilience,
    }


def downside_resilience_score(fcf: float, liquidity: float, leverage: float, integration_burden: float) -> float:
    fcf_score = 100 if fcf > 0 else clamp(60 + fcf * 2)
    liquidity_score = clamp(liquidity / 50 * 100)
    leverage_score = clamp(100 - max(leverage - 2.0, 0) / 3.5 * 100)
    burden_score = clamp(100 - integration_burden / 0.40 * 100)
    return round(0.35 * fcf_score + 0.25 * liquidity_score + 0.25 * leverage_score + 0.15 * burden_score, 1)


def run_scenarios(
    assumptions: AcquisitionAssumptions,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    scenario_set = scenarios or DEFAULT_SCENARIOS
    rows = [calculate_case(assumptions, name, case) for name, case in scenario_set.items()]
    return pd.DataFrame(rows)


def scenario_interpretation(scenarios: pd.DataFrame) -> str:
    if scenarios.empty:
        return "No scenario output is available."
    downside = scenarios.loc[scenarios["Case"] == "Downside"]
    base = scenarios.loc[scenarios["Case"] == "Base"]
    if downside.empty or base.empty:
        return "Scenario set should include Base and Downside cases for board review."
    downside_score = float(downside["Downside Resilience Score"].iloc[0])
    base_leverage = float(base["Net Leverage"].iloc[0])
    if downside_score >= 70 and base_leverage <= 2.5:
        return "The target shows resilient downside liquidity and manageable base-case leverage."
    if downside_score >= 50:
        return "The target is financeable, but downside resilience depends on integration cost control and liquidity discipline."
    return "The downside case is fragile and should trigger deeper financing, cost, and cash conversion diligence."
