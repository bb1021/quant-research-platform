import numpy as np
import pandas as pd

from src.factors import calculate_factors


def _prices():
    dates = pd.bdate_range("2023-01-02", periods=320)
    rows = []
    for ticker, slope in {"AAA": 0.003, "BBB": 0.001, "CCC": -0.0005}.items():
        price = 100.0
        for date in dates:
            price *= 1 + slope
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000,
                    "returns": slope,
                }
            )
    return pd.DataFrame(rows)


def test_calculate_factors_outputs_rankings():
    factors = calculate_factors(_prices())
    latest = factors.sort_values("date").groupby("ticker").tail(1)

    assert {"mom_12m", "vol_60d", "mean_reversion_5d", "ma_crossover", "relative_strength_rank"}.issubset(factors.columns)
    best = latest.sort_values("mom_12m", ascending=False).iloc[0]["ticker"]
    assert best == "AAA"
    assert np.isfinite(latest["mom_3m"]).all()

