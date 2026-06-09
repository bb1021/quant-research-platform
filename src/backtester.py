from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .factors import FACTOR_DIRECTIONS
from .metrics import calculate_performance_metrics, drawdown_series
from .portfolio import build_portfolio_weights
from .utils import coerce_date


@dataclass
class StrategyConfig:
    selected_factor: str = "mom_12m"
    top_n: int = 10
    rebalance_frequency: str = "M"
    transaction_cost: float = 0.001
    weighting_method: str = "equal"
    benchmark_ticker: str = "SPY"
    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 1.0
    ascending: bool | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "StrategyConfig":
        cfg = dict(config or {})
        aliases = {
            "factor": "selected_factor",
            "rebalance": "rebalance_frequency",
            "benchmark": "benchmark_ticker",
            "start": "start_date",
            "end": "end_date",
            "weighting": "weighting_method",
            "starting_capital": "initial_capital",
        }
        for old, new in aliases.items():
            if old in cfg and new not in cfg:
                cfg[new] = cfg.pop(old)
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in fields})


def _first_trading_day_each_month(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.empty:
        return index
    series = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(series.groupby(index.to_period("M")).min().to_list())


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    freq = str(frequency or "M").upper()
    if freq in {"D", "DAILY"}:
        return index
    if freq in {"W", "WEEKLY"}:
        series = pd.Series(index=index, data=index)
        return pd.DatetimeIndex(series.groupby(index.to_period("W")).min().to_list())
    return _first_trading_day_each_month(index)


def _prepare_inputs(price_data: pd.DataFrame, factor_data: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    if price_data is None or price_data.empty:
        raise ValueError("price_data is empty")
    if factor_data is None or factor_data.empty:
        raise ValueError("factor_data is empty")
    if cfg.selected_factor not in factor_data.columns:
        raise ValueError(f"selected factor '{cfg.selected_factor}' not found in factor_data")

    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["returns"] = pd.to_numeric(prices["returns"], errors="coerce").fillna(0.0)

    factors = factor_data.copy()
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    factors["ticker"] = factors["ticker"].astype(str).str.upper()

    start = coerce_date(cfg.start_date) or max(prices["date"].min(), factors["date"].min())
    end = coerce_date(cfg.end_date) or min(prices["date"].max(), factors["date"].max())
    prices = prices.loc[prices["date"].between(start, end)]
    factors = factors.loc[factors["date"].between(start, end)]
    if prices.empty or factors.empty:
        raise ValueError("No overlapping price and factor data for requested period")

    returns = prices.pivot(index="date", columns="ticker", values="returns").sort_index().fillna(0.0)
    factor_pivot = factors.pivot(index="date", columns="ticker", values=cfg.selected_factor).sort_index()
    common_dates = returns.index.intersection(factor_pivot.index)
    returns = returns.loc[common_dates]
    factor_pivot = factor_pivot.loc[common_dates]
    return returns, factor_pivot, common_dates


def _select_top(
    signal_row: pd.Series,
    top_n: int,
    benchmark_ticker: str,
    selected_factor: str,
    ascending: bool | None,
) -> list[str]:
    signal = signal_row.dropna().copy()
    benchmark = benchmark_ticker.upper()
    if benchmark in signal.index:
        signal = signal.drop(index=benchmark)
    if signal.empty:
        return []
    if ascending is None:
        direction = FACTOR_DIRECTIONS.get(selected_factor, "higher")
        ascending = direction == "lower"
    selected = signal.sort_values(ascending=ascending).head(max(int(top_n), 1)).index.tolist()
    return [str(t).upper() for t in selected]


def run_backtest(
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    strategy_config: dict[str, Any] | StrategyConfig,
) -> dict[str, Any]:
    cfg = strategy_config if isinstance(strategy_config, StrategyConfig) else StrategyConfig.from_dict(strategy_config)
    returns, factor_pivot, dates = _prepare_inputs(price_data, factor_data, cfg)
    signal_pivot = factor_pivot.shift(1)
    rebalances = set(_rebalance_dates(dates, cfg.rebalance_frequency))
    benchmark = str(cfg.benchmark_ticker or "").upper()
    initial_capital = float(cfg.initial_capital or 1.0)
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        initial_capital = 1.0

    portfolio_returns: list[float] = []
    portfolio_values: list[float] = []
    turnover_values: list[float] = []
    weights_records: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    warning_records: list[str] = []
    seen_warnings: set[str] = set()
    current_weights = pd.Series(0.0, index=returns.columns, dtype=float)
    portfolio_value = initial_capital

    for current_date in dates:
        day_turnover = 0.0
        day_cost = 0.0
        if current_date in rebalances:
            selected = _select_top(
                signal_pivot.loc[current_date],
                cfg.top_n,
                benchmark,
                cfg.selected_factor,
                cfg.ascending,
            )
            if selected:
                signal_date = dates[dates.get_loc(current_date) - 1] if dates.get_loc(current_date) > 0 else current_date
                trailing_returns = returns.loc[:signal_date, selected].tail(252)
                volatility = None
                if "vol_60d" in factor_data.columns:
                    vol_row = (
                        factor_data.assign(date=pd.to_datetime(factor_data["date"]).dt.normalize())
                        .loc[lambda x: x["date"] <= signal_date]
                        .sort_values("date")
                        .groupby("ticker")
                        .tail(1)
                        .set_index("ticker")["vol_60d"]
                    )
                    volatility = vol_row.reindex(selected)
                target_raw = build_portfolio_weights(
                    selected,
                    method=cfg.weighting_method,
                    volatility=volatility,
                    trailing_returns=trailing_returns,
                )
                warning = target_raw.attrs.get("warning")
                if warning and warning not in seen_warnings:
                    seen_warnings.add(warning)
                    warning_records.append(f"{current_date.date()}: {warning}")
                target = target_raw.reindex(returns.columns, fill_value=0.0)
            else:
                target = pd.Series(0.0, index=returns.columns, dtype=float)
                warning = "No valid securities selected from shifted factor signals; portfolio held cash."
                if warning not in seen_warnings:
                    seen_warnings.add(warning)
                    warning_records.append(f"{current_date.date()}: {warning}")

            trade = target - current_weights.reindex(returns.columns, fill_value=0.0)
            day_turnover = float(trade.abs().sum())
            day_cost = day_turnover * float(cfg.transaction_cost)
            for ticker, trade_weight in trade[trade.abs() > 1e-12].items():
                trade_records.append(
                    {
                        "date": current_date,
                        "ticker": ticker,
                        "previous_weight": float(current_weights.get(ticker, 0.0)),
                        "target_weight": float(target.get(ticker, 0.0)),
                        "trade_weight": float(trade_weight),
                        "transaction_cost": float(abs(trade_weight) * cfg.transaction_cost),
                    }
                )
            current_weights = target

        day_return_vector = returns.loc[current_date].reindex(current_weights.index).fillna(0.0)
        gross_return = float((current_weights * day_return_vector).sum())
        net_return = gross_return - day_cost
        portfolio_value *= 1.0 + net_return
        portfolio_returns.append(net_return)
        portfolio_values.append(portfolio_value)
        turnover_values.append(day_turnover)

        if current_weights.sum() > 0:
            drifted = current_weights * (1.0 + day_return_vector)
            denom = drifted.sum()
            current_weights = drifted / denom if denom > 0 else current_weights
        for ticker, weight in current_weights[current_weights.abs() > 1e-10].items():
            weights_records.append({"date": current_date, "ticker": ticker, "weight": float(weight)})

    result = pd.DataFrame(
        {
            "date": dates,
            "portfolio_returns": portfolio_returns,
            "portfolio_value": portfolio_values,
            "turnover": turnover_values,
        }
    )
    if benchmark in returns.columns:
        result["benchmark_returns"] = returns[benchmark].to_numpy()
        result["benchmark_value"] = initial_capital * (1.0 + result["benchmark_returns"]).cumprod()
    else:
        result["benchmark_returns"] = np.nan
        result["benchmark_value"] = np.nan
        if benchmark:
            warning = f"Benchmark ticker {benchmark} was unavailable; benchmark-relative metrics are not shown."
            if warning not in seen_warnings:
                warning_records.append(warning)
    result["drawdown"] = drawdown_series(result["portfolio_value"]).reindex(result.index).to_numpy()

    weights = pd.DataFrame(weights_records, columns=["date", "ticker", "weight"])
    trades = pd.DataFrame(
        trade_records,
        columns=["date", "ticker", "previous_weight", "target_weight", "trade_weight", "transaction_cost"],
    )
    metrics = calculate_performance_metrics(
        result["portfolio_returns"],
        result["benchmark_returns"] if result["benchmark_returns"].notna().any() else None,
        result["turnover"],
    )
    return {
        "equity_curve": result,
        "weights": weights,
        "trades": trades,
        "metrics": metrics,
        "config": cfg.__dict__,
        "warnings": warning_records,
    }
