from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Normalize ticker input while preserving order."""
    if isinstance(tickers, str):
        raw = tickers.replace("\n", ",").replace(";", ",").split(",")
    else:
        raw = list(tickers)

    seen: set[str] = set()
    clean: list[str] = []
    for item in raw:
        ticker = str(item).strip().upper()
        ticker = " ".join(ticker.split())
        if ticker and ticker not in seen:
            seen.add(ticker)
            clean.append(ticker)
    return clean


def coerce_date(value: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def today_timestamp() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def clean_returns(series: pd.Series) -> pd.Series:
    returns = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return returns.fillna(0.0)


def latest_rows(df: pd.DataFrame, group_col: str = "ticker", date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values([group_col, date_col])
    return ordered.groupby(group_col, as_index=False).tail(1).reset_index(drop=True)

