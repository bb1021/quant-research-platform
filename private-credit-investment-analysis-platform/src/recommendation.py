from __future__ import annotations

import pandas as pd


def recommend_credit(metrics: dict[str, float], scenarios: pd.DataFrame, covenant_analysis: dict[str, object]) -> dict[str, object]:
    net_leverage = float(metrics["Net Debt / EBITDA"])
    coverage = float(metrics["EBITDA / Interest"])
    fcf_conversion = float(metrics["FCF Conversion"])
    liquidity = float(metrics["Liquidity"])
    severe = scenarios.loc[scenarios["Case"] == "Severe Downside"]
    severe_score = float(severe["Downside Resilience Score"].iloc[0]) if not severe.empty else 0.0
    covenant_pass = str(covenant_analysis["first_breach"]) == "None"

    score = 100.0
    score -= max(net_leverage - 3.5, 0) * 12
    score -= max(2.5 - coverage, 0) * 16
    score -= max(0.35 - fcf_conversion, 0) * 80
    score -= 15 if liquidity < 50 else 0
    score -= max(60 - severe_score, 0) * 0.45
    score -= 18 if not covenant_pass else 0
    score = max(0.0, min(100.0, score))

    if score >= 75:
        category = "Attractive"
    elif score >= 55:
        category = "Watchlist"
    elif score >= 35:
        category = "High risk"
    else:
        category = "Avoid"

    rationale = [
        f"Net leverage is {net_leverage:.1f}x.",
        f"EBITDA interest coverage is {coverage:.1f}x.",
        f"FCF conversion is {fcf_conversion:.1%}.",
        f"Severe downside resilience score is {severe_score:.1f}/100.",
        f"Covenant result: {'no breach' if covenant_pass else 'breach in ' + str(covenant_analysis['first_breach'])}.",
    ]
    return {"score": round(score, 1), "category": category, "rationale": rationale}
