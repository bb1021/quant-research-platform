from __future__ import annotations

import pandas as pd


def analyse_covenants(
    scenarios: pd.DataFrame,
    max_net_leverage: float = 5.25,
    min_interest_coverage: float = 1.75,
    min_liquidity: float = 35.0,
) -> dict[str, object]:
    rows = []
    first_breach = "None"
    key_driver = "No breach under current cases."
    for _, row in scenarios.iterrows():
        leverage_headroom = max_net_leverage - float(row["Net Leverage"])
        coverage_headroom = float(row["Interest Coverage"]) - min_interest_coverage
        liquidity_headroom = float(row["Liquidity"]) - min_liquidity
        status = "Pass" if leverage_headroom >= 0 and coverage_headroom >= 0 and liquidity_headroom >= 0 else "Fail"
        if status == "Fail" and first_breach == "None":
            first_breach = row["Case"]
            drivers = {
                "Net leverage": leverage_headroom,
                "Interest coverage": coverage_headroom,
                "Liquidity": liquidity_headroom,
            }
            key_driver = min(drivers, key=drivers.get)
        rows.append(
            {
                "Case": row["Case"],
                "Net Leverage": row["Net Leverage"],
                "Net Leverage Headroom": leverage_headroom,
                "Interest Coverage": row["Interest Coverage"],
                "Interest Coverage Headroom": coverage_headroom,
                "Liquidity": row["Liquidity"],
                "Liquidity Headroom": liquidity_headroom,
                "Status": status,
            }
        )
    table = pd.DataFrame(rows)
    return {"table": table, "first_breach": first_breach, "key_driver": key_driver}
