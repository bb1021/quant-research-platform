from __future__ import annotations

import pandas as pd

from .acquisition_screening import AcquisitionAssumptions, accounting_alignment_risk, net_debt
from .utils import risk_band, safe_divide


def analyse_financial_due_diligence(
    assumptions: AcquisitionAssumptions,
    capex_pct_revenue: float = 0.055,
    working_capital_pct_revenue: float = 0.025,
    cash_conversion: float = 0.72,
) -> dict[str, object]:
    revenue = assumptions.revenue
    ebitda = assumptions.ebitda
    capex = revenue * capex_pct_revenue
    working_capital_impact = revenue * working_capital_pct_revenue
    free_cash_flow = ebitda * cash_conversion - capex - working_capital_impact
    monthly_cash_burn = max(-free_cash_flow / 12, 0.0)
    liquidity_runway = safe_divide(assumptions.cash, monthly_cash_burn, default=36.0 if free_cash_flow >= 0 else 0.0)
    leverage = safe_divide(net_debt(assumptions), ebitda, default=0.0)
    reporting_complexity = min(100.0, assumptions.jurisdictions * 8 + assumptions.employees / 30)

    metrics = {
        "Revenue": revenue,
        "Revenue Growth": assumptions.revenue_growth,
        "EBITDA": ebitda,
        "EBITDA Margin": assumptions.ebitda_margin,
        "Capex": capex,
        "Working Capital Impact": working_capital_impact,
        "Free Cash Flow": free_cash_flow,
        "Cash": assumptions.cash,
        "Debt": assumptions.debt,
        "Net Debt": net_debt(assumptions),
        "Net Leverage": leverage,
        "Liquidity Runway Months": liquidity_runway,
        "Reporting Complexity Score": reporting_complexity,
        "Accounting Alignment Risk": accounting_alignment_risk(assumptions.accounting_framework),
    }

    quality_flags = quality_of_earnings_flags(assumptions, metrics)
    interpretation = interpret_diligence(assumptions, metrics, quality_flags)
    return {
        "metrics": metrics,
        "quality_flags": quality_flags,
        "interpretation": interpretation,
        "metric_table": pd.DataFrame(metrics.items(), columns=["Metric", "Value"]),
    }


def quality_of_earnings_flags(assumptions: AcquisitionAssumptions, metrics: dict[str, float | str]) -> list[str]:
    flags: list[str] = []
    if assumptions.ebitda_margin < 0.12:
        flags.append("Low EBITDA margin increases sensitivity to revenue softness and cost dis-synergies.")
    if float(metrics["Free Cash Flow"]) < 0:
        flags.append("Free cash flow is negative after capex and working-capital impact.")
    if assumptions.revenue_growth > 0.30 and assumptions.ebitda_margin < 0.18:
        flags.append("Growth-margin combination may indicate heavy acquisition spend or undercapitalised operations.")
    if float(metrics["Net Leverage"]) > 3.0:
        flags.append("Leverage requires covenant, refinancing, and interest sensitivity review.")
    if assumptions.jurisdictions > 6:
        flags.append("Reporting perimeter is complex, with elevated statutory and tax diligence needs.")
    if metrics["Accounting Alignment Risk"] != "Low":
        flags.append("Accounting policies may require conversion or harmonisation before consolidation.")
    if not flags:
        flags.append("Quality of earnings profile appears workable, subject to revenue and adjustment validation.")
    return flags


def interpret_diligence(
    assumptions: AcquisitionAssumptions,
    metrics: dict[str, float | str],
    quality_flags: list[str],
) -> str:
    fcf = float(metrics["Free Cash Flow"])
    leverage = float(metrics["Net Leverage"])
    liquidity = float(metrics["Liquidity Runway Months"])
    fcf_view = "cash-generative" if fcf >= 0 else "cash-consuming"
    leverage_risk = risk_band(leverage, 1.5, 3.0).lower()
    liquidity_view = "adequate" if liquidity >= 18 else "constrained"
    return (
        f"{assumptions.company_name} appears {fcf_view} on current assumptions with net leverage of "
        f"{leverage:.1f}x and {liquidity:.0f} months of liquidity runway. Leverage risk is {leverage_risk}, "
        f"and liquidity appears {liquidity_view}. The main diligence focus should be: {quality_flags[0]}"
    )


def diligence_summary_table(diligence: dict[str, object]) -> pd.DataFrame:
    metrics = diligence["metrics"]
    rows = [
        ("Revenue", metrics["Revenue"]),
        ("EBITDA", metrics["EBITDA"]),
        ("EBITDA Margin", metrics["EBITDA Margin"]),
        ("Free Cash Flow", metrics["Free Cash Flow"]),
        ("Net Debt", metrics["Net Debt"]),
        ("Net Leverage", metrics["Net Leverage"]),
        ("Liquidity Runway Months", metrics["Liquidity Runway Months"]),
        ("Reporting Complexity Score", metrics["Reporting Complexity Score"]),
        ("Accounting Alignment Risk", metrics["Accounting Alignment Risk"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
