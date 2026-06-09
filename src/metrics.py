from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import TRADING_DAYS


def _returns(returns: pd.Series | list[float] | np.ndarray | None) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return series


def equity_from_returns(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    clean = _returns(returns)
    if clean.empty:
        return pd.Series(dtype=float)
    return initial_value * (1.0 + clean).cumprod()


def calculate_cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    total_return = float((1.0 + clean).prod())
    years = len(clean) / periods_per_year
    if years <= 0 or total_return <= 0:
        return np.nan
    return total_return ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    clean = _returns(returns)
    if len(clean) < 2:
        return np.nan
    excess = clean - risk_free_rate / periods_per_year
    vol = excess.std(ddof=1)
    if vol <= 0 or not np.isfinite(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    excess = clean - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std <= 0 or not np.isfinite(downside_std):
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    equity = pd.Series(equity_curve, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    dd = drawdown_series(equity_curve)
    if dd.empty:
        return np.nan
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    equity = equity_from_returns(clean)
    mdd = abs(max_drawdown(equity))
    cagr = calculate_cagr(clean, periods_per_year)
    if mdd <= 0 or not np.isfinite(mdd):
        return np.nan
    return float(cagr / mdd)


def beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> tuple[float, float]:
    joined = pd.concat(
        [_returns(portfolio_returns).rename("portfolio"), _returns(benchmark_returns).rename("benchmark")],
        axis=1,
    ).dropna()
    if len(joined) < 2:
        return np.nan, np.nan
    bench_var = joined["benchmark"].var(ddof=1)
    if bench_var <= 0 or not np.isfinite(bench_var):
        return np.nan, np.nan
    cov = joined["portfolio"].cov(joined["benchmark"])
    beta = cov / bench_var
    daily_rf = risk_free_rate / periods_per_year
    alpha_daily = (joined["portfolio"].mean() - daily_rf) - beta * (joined["benchmark"].mean() - daily_rf)
    alpha = alpha_daily * periods_per_year
    return float(beta), float(alpha)


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    joined = pd.concat(
        [_returns(portfolio_returns).rename("portfolio"), _returns(benchmark_returns).rename("benchmark")],
        axis=1,
    ).dropna()
    if len(joined) < 2:
        return np.nan
    active = joined["portfolio"] - joined["benchmark"]
    tracking_error = active.std(ddof=1)
    if tracking_error <= 0 or not np.isfinite(tracking_error):
        return np.nan
    return float(active.mean() / tracking_error * np.sqrt(periods_per_year))


def win_rate(returns: pd.Series) -> float:
    clean = _returns(returns)
    if clean.empty:
        return np.nan
    return float((clean > 0).mean())


def rolling_volatility(returns: pd.Series, window: int = 63, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    return _returns(returns).rolling(window).std() * np.sqrt(periods_per_year)


def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = TRADING_DAYS) -> pd.Series:
    clean = _returns(returns)
    rolling_mean = clean.rolling(window).mean()
    rolling_std = clean.rolling(window).std()
    return rolling_mean / rolling_std.replace(0.0, np.nan) * np.sqrt(periods_per_year)


def calculate_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    clean = _returns(portfolio_returns)
    equity = equity_from_returns(clean)
    metrics = {
        "CAGR": calculate_cagr(clean),
        "Annualized Volatility": annualized_volatility(clean),
        "Sharpe Ratio": sharpe_ratio(clean, risk_free_rate=risk_free_rate),
        "Sortino Ratio": sortino_ratio(clean, risk_free_rate=risk_free_rate),
        "Max Drawdown": max_drawdown(equity),
        "Calmar Ratio": calmar_ratio(clean),
        "Win Rate": win_rate(clean),
        "Average Turnover": float(_returns(turnover).mean()) if turnover is not None and len(_returns(turnover)) else np.nan,
    }
    if benchmark_returns is not None:
        beta, alpha = beta_alpha(clean, benchmark_returns, risk_free_rate=risk_free_rate)
        metrics["Beta vs Benchmark"] = beta
        metrics["Alpha vs Benchmark"] = alpha
        metrics["Information Ratio"] = information_ratio(clean, benchmark_returns)
    else:
        metrics["Beta vs Benchmark"] = np.nan
        metrics["Alpha vs Benchmark"] = np.nan
        metrics["Information Ratio"] = np.nan
    return metrics

