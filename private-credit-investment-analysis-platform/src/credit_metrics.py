from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .utils import safe_divide


@dataclass
class CreditAssumptions:
    company_name: str = "Apex Data Platforms"
    sector: str = "B2B vertical software"
    revenue: float = 265.0
    ebitda: float = 72.0
    ebit: float = 58.0
    free_cash_flow: float = 38.0
    cash: float = 32.0
    total_debt: float = 310.0
    interest_expense: float = 29.0
    capex: float = 11.0
    working_capital_impact: float = 8.0
    revolver_availability: float = 55.0
    enterprise_value: float = 790.0
    existing_debt_maturity: float = 85.0
    selected_leverage_multiple: float = 4.75

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def calculate_credit_metrics(assumptions: CreditAssumptions) -> dict[str, float]:
    net_debt = assumptions.total_debt - assumptions.cash
    liquidity = assumptions.cash + assumptions.revolver_availability
    monthly_cash_burn = max(-assumptions.free_cash_flow / 12.0, 0.0)
    liquidity_runway = safe_divide(liquidity, monthly_cash_burn, default=36.0 if assumptions.free_cash_flow >= 0 else 0.0)
    debt_capacity = assumptions.ebitda * assumptions.selected_leverage_multiple
    return {
        "Revenue": assumptions.revenue,
        "EBITDA": assumptions.ebitda,
        "EBIT": assumptions.ebit,
        "Free Cash Flow": assumptions.free_cash_flow,
        "Cash": assumptions.cash,
        "Total Debt": assumptions.total_debt,
        "Net Debt": net_debt,
        "Liquidity": liquidity,
        "EV / EBITDA": safe_divide(assumptions.enterprise_value, assumptions.ebitda),
        "Debt / EBITDA": safe_divide(assumptions.total_debt, assumptions.ebitda),
        "Net Debt / EBITDA": safe_divide(net_debt, assumptions.ebitda),
        "EBITDA / Interest": safe_divide(assumptions.ebitda, assumptions.interest_expense),
        "EBIT / Interest": safe_divide(assumptions.ebit, assumptions.interest_expense),
        "FCF Conversion": safe_divide(assumptions.free_cash_flow, assumptions.ebitda),
        "FCF / Debt": safe_divide(assumptions.free_cash_flow, assumptions.total_debt),
        "Debt / Enterprise Value": safe_divide(assumptions.total_debt, assumptions.enterprise_value),
        "Liquidity Runway Months": liquidity_runway,
        "Debt Capacity": debt_capacity,
        "Incremental Debt Capacity": debt_capacity - assumptions.total_debt,
        "Refinancing Need": assumptions.existing_debt_maturity,
    }


def metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(metrics.items(), columns=["Metric", "Value"])


def credit_strengths(assumptions: CreditAssumptions, metrics: dict[str, float]) -> list[str]:
    strengths: list[str] = []
    if metrics["EBITDA / Interest"] >= 2.25:
        strengths.append("Interest coverage is above a conservative direct lending threshold.")
    if metrics["FCF Conversion"] >= 0.40:
        strengths.append("Free cash flow conversion supports deleveraging capacity.")
    if metrics["Liquidity"] >= assumptions.existing_debt_maturity * 0.75:
        strengths.append("Available liquidity covers a meaningful portion of near-term maturities.")
    if metrics["Net Debt / EBITDA"] <= 4.0:
        strengths.append("Net leverage is manageable relative to recurring EBITDA.")
    if not strengths:
        strengths.append("Credit strengths require further validation through diligence.")
    return strengths


def credit_risks(assumptions: CreditAssumptions, metrics: dict[str, float]) -> list[str]:
    risks: list[str] = []
    if metrics["Debt / EBITDA"] > 5.0:
        risks.append("Gross leverage is elevated for a private credit hold.")
    if metrics["EBITDA / Interest"] < 2.0:
        risks.append("Interest coverage is thin and sensitive to rate or earnings pressure.")
    if metrics["FCF Conversion"] < 0.25:
        risks.append("Low free cash flow conversion limits debt paydown capacity.")
    if assumptions.existing_debt_maturity > metrics["Liquidity"]:
        risks.append("Near-term maturity requirement exceeds available cash and revolver liquidity.")
    if metrics["Debt / Enterprise Value"] > 0.55:
        risks.append("Debt quantum represents a high share of enterprise value.")
    if not risks:
        risks.append("No critical credit risk identified from headline metrics, subject to diligence.")
    return risks
