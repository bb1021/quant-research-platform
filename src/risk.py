from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import drawdown_series, rolling_sharpe, rolling_volatility


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(1.0 - confidence))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    var = value_at_risk(clean, confidence)
    tail = clean[clean <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def risk_summary(returns: pd.Series, equity_curve: pd.Series | None = None) -> dict[str, float]:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    equity = equity_curve if equity_curve is not None else (1.0 + clean).cumprod()
    dd = drawdown_series(equity)
    return {
        "Daily VaR 95%": value_at_risk(clean, 0.95),
        "Daily CVaR 95%": conditional_value_at_risk(clean, 0.95),
        "Worst Day": float(clean.min()) if not clean.empty else np.nan,
        "Best Day": float(clean.max()) if not clean.empty else np.nan,
        "Current Drawdown": float(dd.iloc[-1]) if not dd.empty else np.nan,
    }


__all__ = [
    "conditional_value_at_risk",
    "drawdown_series",
    "risk_summary",
    "rolling_sharpe",
    "rolling_volatility",
    "value_at_risk",
]

