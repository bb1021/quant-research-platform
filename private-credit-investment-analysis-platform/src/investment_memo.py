from __future__ import annotations

import os

import pandas as pd
import requests

from .credit_metrics import CreditAssumptions
from .utils import money, multiple, pct


def generate_investment_memo(
    assumptions: CreditAssumptions,
    metrics: dict[str, float],
    strengths: list[str],
    risks: list[str],
    scenarios: pd.DataFrame,
    covenant_analysis: dict[str, object],
    recommendation: dict[str, object],
    use_llm: bool = False,
) -> str:
    memo = deterministic_memo(assumptions, metrics, strengths, risks, scenarios, covenant_analysis, recommendation)
    if use_llm and os.getenv("OPENAI_API_KEY"):
        return enhance_with_llm(memo)
    return memo


def deterministic_memo(
    assumptions: CreditAssumptions,
    metrics: dict[str, float],
    strengths: list[str],
    risks: list[str],
    scenarios: pd.DataFrame,
    covenant_analysis: dict[str, object],
    recommendation: dict[str, object],
) -> str:
    return f"""# Private Credit Investment Memo: {assumptions.company_name}

## Executive Summary
{assumptions.company_name} screens as **{recommendation["category"]}** with a credit score of **{recommendation["score"]}/100**. The business generates {money(assumptions.revenue)} of revenue, {money(assumptions.ebitda)} of EBITDA, {multiple(metrics["Net Debt / EBITDA"])} net leverage, and {multiple(metrics["EBITDA / Interest"])} EBITDA interest coverage.

## Company Overview
The company operates in {assumptions.sector}. The preliminary credit view focuses on recurring earnings quality, cash conversion, capital intensity, liquidity coverage, refinancing needs, and downside debt service capacity.

## Transaction Overview
- Enterprise value: {money(assumptions.enterprise_value)}
- EV / EBITDA: {multiple(metrics["EV / EBITDA"])}
- Total debt: {money(assumptions.total_debt)}
- Net debt: {money(metrics["Net Debt"])}
- Existing maturity amount: {money(assumptions.existing_debt_maturity)}

## Capital Structure
- Cash: {money(assumptions.cash)}
- Revolver availability: {money(assumptions.revolver_availability)}
- Liquidity: {money(metrics["Liquidity"])}
- Debt / enterprise value: {pct(metrics["Debt / Enterprise Value"])}
- Debt capacity at selected multiple: {money(metrics["Debt Capacity"])}

## Key Credit Strengths
{_bullets(strengths)}

## Key Credit Risks
{_bullets(risks)}

## Financial Analysis
- EBITDA margin: {pct(assumptions.ebitda / assumptions.revenue if assumptions.revenue else 0)}
- FCF conversion: {pct(metrics["FCF Conversion"])}
- FCF / debt: {pct(metrics["FCF / Debt"])}
- EBIT / interest: {multiple(metrics["EBIT / Interest"])}
- Liquidity runway: {metrics["Liquidity Runway Months"]:.0f} months

## Downside Case Analysis
{_scenario_lines(scenarios)}

## Covenant Headroom
- First breach: {covenant_analysis["first_breach"]}
- Key risk driver: {covenant_analysis["key_driver"]}

{_covenant_lines(covenant_analysis["table"])}

## Liquidity and Refinancing Risk
Liquidity of {money(metrics["Liquidity"])} should be compared against {money(assumptions.existing_debt_maturity)} of near-term maturities. The refinancing workstream should focus on maturity wall timing, revolver conditions, excess cash leakage, and sponsor support.

## Diligence Questions
- What percentage of EBITDA is recurring, contracted, or usage-linked?
- How resilient is gross retention under customer budget pressure?
- Which costs are fixed versus variable in a revenue decline?
- What drives working capital volatility by quarter?
- Are capex requirements maintenance-like or growth-dependent?
- What is the maturity profile by instrument and lender group?
- Which covenant definitions differ from management reporting metrics?

## Investment Recommendation
Recommendation: **{recommendation["category"]}**.

{_bullets(recommendation["rationale"])}
"""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _scenario_lines(scenarios: pd.DataFrame) -> str:
    lines = []
    for _, row in scenarios.iterrows():
        lines.append(
            f"- {row['Case']}: revenue {money(row['Revenue'])}, EBITDA {money(row['EBITDA'])}, "
            f"FCF {money(row['Free Cash Flow'])}, net leverage {multiple(row['Net Leverage'])}, "
            f"interest coverage {multiple(row['Interest Coverage'])}, liquidity {money(row['Liquidity'])}, "
            f"resilience {row['Downside Resilience Score']:.1f}/100."
        )
    return "\n".join(lines)


def _covenant_lines(covenant_table: pd.DataFrame) -> str:
    lines = []
    for _, row in covenant_table.iterrows():
        lines.append(
            f"- {row['Case']}: {row['Status']}, net leverage headroom {multiple(row['Net Leverage Headroom'])}, "
            f"coverage headroom {multiple(row['Interest Coverage Headroom'])}, liquidity headroom {money(row['Liquidity Headroom'])}."
        )
    return "\n".join(lines)


def enhance_with_llm(memo: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return memo
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise private credit investment committee memo writer."},
                    {"role": "user", "content": f"Improve this private credit memo while preserving facts and markdown sections:\n\n{memo}"},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip() or memo
    except Exception:
        return memo
