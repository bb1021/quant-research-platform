from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import TRADING_DAYS


FACTOR_DIRECTIONS: dict[str, str] = {
    "mom_12m": "higher",
    "mom_6m": "higher",
    "mom_3m": "higher",
    "vol_60d": "lower",
    "vol_20d": "lower",
    "mean_reversion_5d": "higher",
    "ma_crossover": "higher",
    "relative_strength_score": "higher",
}

FACTOR_COLUMNS = list(FACTOR_DIRECTIONS)


def _rank_by_date(df: pd.DataFrame, column: str, higher_is_better: bool = True) -> pd.Series:
    ascending = not higher_is_better
    return df.groupby("date")[column].rank(method="min", ascending=ascending)


def _score_by_date(df: pd.DataFrame, column: str, higher_is_better: bool = True) -> pd.Series:
    ascending = higher_is_better
    return df.groupby("date")[column].rank(method="average", pct=True, ascending=ascending)


def calculate_factors(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        columns = ["date", "ticker", *FACTOR_COLUMNS, *[f"{c}_rank" for c in FACTOR_COLUMNS]]
        return pd.DataFrame(columns=columns)

    required = {"date", "ticker", "adj_close", "returns"}
    missing = required.difference(price_data.columns)
    if missing:
        raise ValueError(f"price_data missing required columns: {sorted(missing)}")

    df = price_data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce").fillna(0.0)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = df.groupby("ticker", group_keys=False)
    factors = df[["date", "ticker"]].copy()
    factors["mom_12m"] = grouped["adj_close"].pct_change(TRADING_DAYS)
    factors["mom_6m"] = grouped["adj_close"].pct_change(126)
    factors["mom_3m"] = grouped["adj_close"].pct_change(63)
    factors["vol_60d"] = grouped["returns"].rolling(60).std().reset_index(level=0, drop=True) * np.sqrt(TRADING_DAYS)
    factors["vol_20d"] = grouped["returns"].rolling(20).std().reset_index(level=0, drop=True) * np.sqrt(TRADING_DAYS)
    recent_return_5d = grouped["adj_close"].pct_change(5)
    factors["mean_reversion_5d"] = -recent_return_5d

    ma_fast = grouped["adj_close"].rolling(50).mean().reset_index(level=0, drop=True)
    ma_slow = grouped["adj_close"].rolling(200).mean().reset_index(level=0, drop=True)
    factors["ma_crossover"] = (ma_fast / ma_slow) - 1.0
    factors["ma_signal"] = (ma_fast > ma_slow).astype(float)

    factors["relative_strength_rank"] = factors.groupby("date")["mom_12m"].rank(method="min", ascending=False)
    factors["relative_strength_score"] = factors.groupby("date")["mom_12m"].rank(method="average", pct=True)

    for col, direction in FACTOR_DIRECTIONS.items():
        higher = direction == "higher"
        factors[f"{col}_rank"] = _rank_by_date(factors, col, higher_is_better=higher)
        factors[f"{col}_score"] = _score_by_date(factors, col, higher_is_better=higher)

    factors.replace([np.inf, -np.inf], np.nan, inplace=True)
    return factors.sort_values(["date", "ticker"]).reset_index(drop=True)

