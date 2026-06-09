from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_index(items: list[str] | pd.Index) -> pd.Index:
    return pd.Index([str(x).upper() for x in items])


def _normalize_weights(weights: pd.Series) -> pd.Series:
    weights = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)
    total = weights.sum()
    if total <= 0:
        return _with_warning(equal_weights(list(weights.index)), "Weight normalisation failed; using equal weights.")
    return weights / total


def _with_warning(weights: pd.Series, warning: str | None = None) -> pd.Series:
    if warning:
        weights.attrs["warning"] = warning
    return weights


def equal_weights(tickers: list[str] | pd.Index) -> pd.Series:
    index = _as_index(tickers)
    if len(index) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(index), index=index, dtype=float)


def volatility_weights(volatility: pd.Series | dict[str, float]) -> pd.Series:
    vol = pd.Series(volatility, dtype=float)
    if vol.empty:
        return pd.Series(dtype=float)
    vol.index = _as_index(vol.index)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    median = vol[vol > 0].median()
    if not np.isfinite(median) or median <= 0:
        return _with_warning(equal_weights(vol.index), "Volatility inputs were unavailable; using equal weights.")
    fill_value = median if np.isfinite(median) and median > 0 else 1.0
    inv_vol = 1.0 / vol.fillna(fill_value).clip(lower=1e-8)
    return _normalize_weights(inv_vol)


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 1.0,
) -> pd.Series:
    tickers = _as_index(expected_returns.index)
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    try:
        mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        cov = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy(dtype=float)
        cov = cov + np.eye(len(tickers)) * 1e-8

        def objective(w: np.ndarray) -> float:
            variance = float(w.T @ cov @ w)
            ret = float(w @ mu)
            return risk_aversion * variance - ret

        x0 = np.repeat(1.0 / len(tickers), len(tickers))
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(tickers),
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return _with_warning(equal_weights(tickers), "Mean-variance optimisation failed; using equal weights.")
        return _normalize_weights(pd.Series(result.x, index=tickers))
    except Exception:
        return _with_warning(equal_weights(tickers), "Mean-variance optimisation failed; using equal weights.")


def risk_parity_weights(covariance: pd.DataFrame) -> pd.Series:
    tickers = _as_index(covariance.index)
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    try:
        cov = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy(dtype=float)
        cov = cov + np.eye(len(tickers)) * 1e-8

        def risk_contributions(w: np.ndarray) -> np.ndarray:
            portfolio_vol = np.sqrt(max(float(w.T @ cov @ w), 1e-12))
            marginal = cov @ w / portfolio_vol
            return w * marginal

        def objective(w: np.ndarray) -> float:
            rc = risk_contributions(w)
            target = rc.mean()
            return float(((rc - target) ** 2).sum())

        x0 = np.repeat(1.0 / len(tickers), len(tickers))
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(tickers),
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            diagonal_vol = np.sqrt(np.diag(cov))
            fallback = volatility_weights(pd.Series(diagonal_vol, index=tickers))
            return _with_warning(fallback, "Risk parity optimisation failed; using inverse-volatility fallback.")
        return _normalize_weights(pd.Series(result.x, index=tickers))
    except Exception:
        return _with_warning(equal_weights(tickers), "Risk parity optimisation failed; using equal weights.")


def build_portfolio_weights(
    selected_tickers: list[str],
    method: str = "equal",
    volatility: pd.Series | None = None,
    trailing_returns: pd.DataFrame | None = None,
) -> pd.Series:
    selected = [str(t).upper() for t in selected_tickers]
    method_key = method.lower().strip().replace("-", "_")
    if not selected:
        return pd.Series(dtype=float)
    if method_key in {"equal", "equal_weight", "equal_weighted"}:
        return equal_weights(selected)
    if method_key in {"volatility", "volatility_weighted", "inverse_vol"}:
        if volatility is None or volatility.empty:
            return _with_warning(equal_weights(selected), "Volatility inputs were unavailable; using equal weights.")
        return volatility_weights(volatility.reindex(selected))
    if trailing_returns is None or trailing_returns.empty:
        if method_key in {"mean_variance", "mean_variance_optimization", "mvo", "risk_parity", "risk_parity_approx"}:
            return _with_warning(equal_weights(selected), "Trailing returns were unavailable; using equal weights.")
        return equal_weights(selected)

    trailing = trailing_returns.reindex(columns=selected).dropna(how="all")
    if len(trailing) < 20:
        if method_key in {"mean_variance", "mean_variance_optimization", "mvo", "risk_parity", "risk_parity_approx"}:
            return _with_warning(equal_weights(selected), "Insufficient trailing returns for optimisation; using equal weights.")
        return equal_weights(selected)
    expected = trailing.mean().fillna(0.0) * 252
    cov = trailing.cov().fillna(0.0) * 252
    if method_key in {"mean_variance", "mean_variance_optimization", "mvo"}:
        return mean_variance_weights(expected, cov)
    if method_key in {"risk_parity", "risk_parity_approx"}:
        return risk_parity_weights(cov)
    return equal_weights(selected)
