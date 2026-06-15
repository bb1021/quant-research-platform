from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .utils import clamp, safe_divide, score_band


@dataclass
class AcquisitionAssumptions:
    company_name: str = "Northstar Apps"
    sector: str = "Mobile subscription software"
    revenue: float = 180.0
    revenue_growth: float = 0.22
    ebitda: float = 42.0
    ebitda_margin: float = 0.233
    cash: float = 28.0
    debt: float = 95.0
    purchase_price: float = 640.0
    employees: int = 420
    jurisdictions: int = 7
    accounting_framework: str = "IFRS"
    strategic_rationale: str = "Expands subscription revenue, adds product depth, and creates cross-sell potential across a scaled consumer technology platform."

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def enterprise_value(assumptions: AcquisitionAssumptions) -> float:
    return assumptions.purchase_price + assumptions.debt - assumptions.cash


def net_debt(assumptions: AcquisitionAssumptions) -> float:
    return assumptions.debt - assumptions.cash


def accounting_alignment_risk(framework: str) -> str:
    text = (framework or "").strip().lower()
    if text == "ifrs":
        return "Low"
    if "mixed" in text or "unknown" in text:
        return "High"
    return "Medium"


def score_acquisition(assumptions: AcquisitionAssumptions) -> dict[str, float | str]:
    ev = enterprise_value(assumptions)
    margin = assumptions.ebitda_margin if assumptions.ebitda_margin else safe_divide(assumptions.ebitda, assumptions.revenue, 0.0)
    ev_revenue = safe_divide(ev, assumptions.revenue)
    ev_ebitda = safe_divide(ev, assumptions.ebitda)
    leverage = safe_divide(net_debt(assumptions), assumptions.ebitda, default=0.0)
    accounting_risk = accounting_alignment_risk(assumptions.accounting_framework)

    growth_score = clamp(assumptions.revenue_growth / 0.30 * 100)
    margin_score = clamp(margin / 0.30 * 100)
    leverage_score = clamp(100 - max(leverage - 1.5, 0) / 4.0 * 100)
    valuation_score = clamp(100 - max(ev_ebitda - 10.0, 0) / 15.0 * 100) if np.isfinite(ev_ebitda) else 35
    complexity_penalty = min(max(assumptions.jurisdictions - 3, 0) * 3, 18)
    accounting_penalty = {"Low": 0, "Medium": 8, "High": 18}.get(accounting_risk, 10)
    rationale_score = 88 if len((assumptions.strategic_rationale or "").strip()) > 80 else 62

    score = (
        0.22 * growth_score
        + 0.20 * margin_score
        + 0.18 * leverage_score
        + 0.18 * valuation_score
        + 0.12 * rationale_score
        + 0.10 * clamp(100 - complexity_penalty - accounting_penalty)
    )

    return {
        "score": round(clamp(score), 1),
        "score_band": score_band(score),
        "enterprise_value": ev,
        "ev_revenue": ev_revenue,
        "ev_ebitda": ev_ebitda,
        "net_debt": net_debt(assumptions),
        "net_leverage": leverage,
        "ebitda_margin": margin,
        "accounting_alignment_risk": accounting_risk,
        "integration_complexity_score": round(clamp(complexity_penalty + accounting_penalty + assumptions.jurisdictions * 2), 1),
    }


def risk_flags(assumptions: AcquisitionAssumptions, metrics: dict[str, float | str]) -> list[str]:
    flags: list[str] = []
    if assumptions.revenue_growth < 0.08:
        flags.append("Revenue growth is below high-growth platform thresholds.")
    if float(metrics["ebitda_margin"]) < 0.15:
        flags.append("EBITDA margin leaves limited room for integration disruption.")
    if float(metrics["net_leverage"]) > 3.0:
        flags.append("Net leverage is elevated and may constrain post-close flexibility.")
    if float(metrics["ev_ebitda"]) > 18.0:
        flags.append("Entry multiple requires synergy confidence and durable growth.")
    if assumptions.jurisdictions >= 8:
        flags.append("Multi-jurisdiction footprint increases tax, payroll, and reporting complexity.")
    if metrics["accounting_alignment_risk"] == "High":
        flags.append("Accounting framework is mixed or unknown, requiring early reporting alignment work.")
    if not flags:
        flags.append("No single critical red flag, diligence should focus on validation rather than remediation.")
    return flags


def diligence_priorities(assumptions: AcquisitionAssumptions, metrics: dict[str, float | str]) -> list[str]:
    priorities = [
        "Validate revenue quality, cohort retention, pricing power, refunds, and deferred revenue treatment.",
        "Reconcile management EBITDA to reported accounts and identify non-recurring adjustments.",
        "Map cash, debt, restricted cash, and working-capital seasonality before signing.",
    ]
    if float(metrics["net_leverage"]) > 2.5:
        priorities.append("Stress test debt capacity, covenants, interest sensitivity, and liquidity runway.")
    if assumptions.jurisdictions > 5:
        priorities.append("Review local statutory reporting, tax registrations, payroll entities, and transfer-pricing exposure.")
    if metrics["accounting_alignment_risk"] != "Low":
        priorities.append("Assess IFRS conversion requirements, revenue-recognition policies, and consolidation readiness.")
    return priorities


def strategic_fit_summary(assumptions: AcquisitionAssumptions, metrics: dict[str, float | str]) -> str:
    return (
        f"{assumptions.company_name} screens as '{metrics['score_band']}' with an attractiveness score of "
        f"{metrics['score']}/100. The target combines {assumptions.revenue_growth:.1%} revenue growth, "
        f"{metrics['ebitda_margin']:.1%} EBITDA margin, and {metrics['net_leverage']:.1f}x net leverage. "
        "Strategic fit depends on confirming revenue durability, integration feasibility, and reporting readiness."
    )


def screen_acquisition(assumptions: AcquisitionAssumptions) -> dict[str, object]:
    metrics = score_acquisition(assumptions)
    return {
        "assumptions": assumptions.to_dict(),
        "metrics": metrics,
        "risk_flags": risk_flags(assumptions, metrics),
        "diligence_priorities": diligence_priorities(assumptions, metrics),
        "strategic_fit_summary": strategic_fit_summary(assumptions, metrics),
    }


def screening_table(result: dict[str, object]) -> pd.DataFrame:
    metrics = result["metrics"]
    rows = [
        ("Attractiveness score", metrics["score"]),
        ("Enterprise value", metrics["enterprise_value"]),
        ("EV / Revenue", metrics["ev_revenue"]),
        ("EV / EBITDA", metrics["ev_ebitda"]),
        ("Net debt", metrics["net_debt"]),
        ("Net leverage", metrics["net_leverage"]),
        ("Accounting alignment risk", metrics["accounting_alignment_risk"]),
        ("Integration complexity score", metrics["integration_complexity_score"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
