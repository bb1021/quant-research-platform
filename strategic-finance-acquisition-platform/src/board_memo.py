from __future__ import annotations

import os

import pandas as pd
import requests

from .acquisition_screening import AcquisitionAssumptions
from .utils import money, multiple, pct


def _metric(metrics: dict[str, object], key: str, default: float = float("nan")) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_board_memo(
    assumptions: AcquisitionAssumptions,
    screening: dict[str, object],
    diligence: dict[str, object],
    scenarios: pd.DataFrame,
    integration_plan: pd.DataFrame,
    use_llm: bool = False,
) -> str:
    memo = deterministic_board_memo(assumptions, screening, diligence, scenarios, integration_plan)
    if use_llm and os.getenv("OPENAI_API_KEY"):
        return enhance_with_llm(memo)
    return memo


def deterministic_board_memo(
    assumptions: AcquisitionAssumptions,
    screening: dict[str, object],
    diligence: dict[str, object],
    scenarios: pd.DataFrame,
    integration_plan: pd.DataFrame,
) -> str:
    metrics = screening["metrics"]
    diligence_metrics = diligence["metrics"]
    flags = screening["risk_flags"]
    priorities = screening["diligence_priorities"]
    quality_flags = diligence["quality_flags"]
    top_integration = integration_plan.head(4)
    scenario_lines = _scenario_lines(scenarios)
    management_questions = _management_questions(assumptions, flags, quality_flags)
    recommendation = _recommendation(metrics, diligence_metrics, scenarios)

    return f"""# Board Memo: {assumptions.company_name}

## Executive Summary
{assumptions.company_name} screens as **{metrics["score_band"]}** with an acquisition attractiveness score of **{metrics["score"]}/100**. The target generates {money(assumptions.revenue)} of revenue, {money(assumptions.ebitda)} of EBITDA, and an EBITDA margin of {pct(_metric(metrics, "ebitda_margin"))}. Net leverage is {multiple(_metric(metrics, "net_leverage"))}, and the current reporting readiness profile is shaped by {assumptions.jurisdictions} jurisdictions and {assumptions.accounting_framework} accounting.

## Strategic Rationale
{assumptions.strategic_rationale}

The strategic question is whether the acquisition strengthens the group platform while remaining governable within Bending Spoons-style finance discipline: clear unit economics, rapid reporting cadence, reliable cash visibility, and decisive post-close execution.

## Financial Overview
- Revenue: {money(assumptions.revenue)}
- Revenue growth: {pct(assumptions.revenue_growth)}
- EBITDA: {money(assumptions.ebitda)}
- EBITDA margin: {pct(_metric(metrics, "ebitda_margin"))}
- Free cash flow estimate: {money(_metric(diligence_metrics, "Free Cash Flow"))}
- Net debt: {money(_metric(metrics, "net_debt"))}
- Liquidity runway: {_metric(diligence_metrics, "Liquidity Runway Months"):.0f} months

## Valuation Snapshot
- Enterprise value: {money(_metric(metrics, "enterprise_value"))}
- EV / Revenue: {multiple(_metric(metrics, "ev_revenue"))}
- EV / EBITDA: {multiple(_metric(metrics, "ev_ebitda"))}
- Purchase price: {money(assumptions.purchase_price)}

The valuation is attractive only if diligence confirms revenue durability, cash conversion, and achievable synergy delivery. Multiple discipline should be linked to integration complexity rather than headline growth alone.

## Key Diligence Findings
{_bullet_list(flags + priorities[:2])}

## Accounting and Reporting Considerations
- Accounting alignment risk: {metrics["accounting_alignment_risk"]}
- Reporting complexity score: {_metric(diligence_metrics, "Reporting Complexity Score"):.1f}/100
- Primary quality of earnings point: {quality_flags[0]}

Finance should validate revenue recognition, deferred revenue, refunds, management adjustments, statutory reporting obligations, and consolidation mapping before final approval.

## Integration Risks
{_integration_bullets(top_integration)}

## Base / Downside / Upside Case Analysis
{scenario_lines}

{_scenario_summary(scenarios)}

## 100-Day Integration Plan
{_integration_bullets(integration_plan.head(8), include_timeline=True)}

## Key Questions for Management
{_bullet_list(management_questions)}

## Recommendation
{recommendation}
"""


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _integration_bullets(plan: pd.DataFrame, include_timeline: bool = False) -> str:
    rows = []
    for _, row in plan.iterrows():
        prefix = f"{row['Workstream']} ({row['Timeline']})" if include_timeline else row["Workstream"]
        rows.append(f"- {prefix}: {row['Priority']} priority, {row['Risk Level']} risk. {row['Rationale']}")
    return "\n".join(rows)


def _scenario_lines(scenarios: pd.DataFrame) -> str:
    lines = []
    for _, row in scenarios.iterrows():
        lines.append(
            f"- {row['Case']}: revenue {money(row['Revenue'])}, EBITDA {money(row['EBITDA'])}, "
            f"FCF {money(row['FCF'])}, net leverage {multiple(row['Net Leverage'])}, "
            f"resilience score {row['Downside Resilience Score']:.1f}/100."
        )
    return "\n".join(lines)


def _scenario_summary(scenarios: pd.DataFrame) -> str:
    if scenarios.empty:
        return "Scenario analysis is unavailable."
    downside = scenarios.loc[scenarios["Case"] == "Downside"]
    if downside.empty:
        return "Board review should include an explicit downside case before approval."
    score = float(downside["Downside Resilience Score"].iloc[0])
    if score >= 70:
        return "The downside case appears resilient enough for continued diligence."
    if score >= 50:
        return "The downside case is workable, but the deal should be gated by liquidity and integration-cost controls."
    return "The downside case is fragile and should be a gating item before signing."


def _management_questions(
    assumptions: AcquisitionAssumptions,
    risk_flags: list[str],
    quality_flags: list[str],
) -> list[str]:
    return [
        "Which revenue cohorts, geographies, and products drive the current growth rate?",
        "What adjustments bridge reported EBITDA to management EBITDA, and which are recurring?",
        "How much cash is unrestricted, and what intra-month liquidity volatility should be expected?",
        "Which finance systems and ledgers are business-critical during the first 100 days?",
        f"How mature is statutory reporting across the {assumptions.jurisdictions} operating jurisdictions?",
        f"What evidence addresses this diligence concern: {risk_flags[0]}",
        f"What evidence addresses this quality-of-earnings concern: {quality_flags[0]}",
    ]


def _recommendation(metrics: dict[str, object], diligence_metrics: dict[str, object], scenarios: pd.DataFrame) -> str:
    score = _metric(metrics, "score")
    leverage = _metric(metrics, "net_leverage")
    runway = _metric(diligence_metrics, "Liquidity Runway Months")
    downside = scenarios.loc[scenarios["Case"] == "Downside"]
    resilience = float(downside["Downside Resilience Score"].iloc[0]) if not downside.empty else 0.0

    if score >= 70 and leverage <= 2.5 and runway >= 18 and resilience >= 55:
        return "Proceed to confirmatory diligence with focus on revenue quality, cash conversion, and integration execution."
    if score >= 50 and resilience >= 45:
        return "Proceed selectively, but make signing conditional on finance-system readiness, liquidity validation, and synergy evidence."
    return "Do not proceed without deeper diligence and revised economics; current assumptions do not yet clear the finance-control threshold."


def enhance_with_llm(memo: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return memo

    prompt = (
        "Improve the following M&A board memo for a strategic finance graduate programme audience. "
        "Keep the facts, avoid trading language, and preserve the markdown section structure.\n\n"
        f"{memo}"
    )
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise strategic finance and M&A integration adviser."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() or memo
    except Exception:
        return memo
