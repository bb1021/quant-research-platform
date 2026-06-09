import pandas as pd
import pytest

from src.backtester import run_backtest


def _price_frame():
    dates = pd.bdate_range("2024-01-01", periods=80)
    rows = []
    for ticker, daily_return in {"AAA": 0.002, "BBB": 0.001, "SPY": 0.0015}.items():
        price = 100.0
        for date in dates:
            price *= 1 + daily_return
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
                    "returns": daily_return,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_output_shape():
    prices = _price_frame()
    factors = prices[["date", "ticker"]].copy()
    factors["mom_12m"] = factors["ticker"].map({"AAA": 2.0, "BBB": 1.0, "SPY": 0.0})

    result = run_backtest(prices, factors, {"selected_factor": "mom_12m", "top_n": 1, "benchmark_ticker": "SPY"})

    equity = result["equity_curve"]
    assert {"date", "portfolio_returns", "portfolio_value", "benchmark_returns", "drawdown"}.issubset(equity.columns)
    assert len(equity) == prices["date"].nunique()
    assert not result["weights"].empty


def test_backtest_supports_initial_capital_and_missing_benchmark():
    prices = _price_frame()
    factors = prices[["date", "ticker"]].copy()
    factors["mom_12m"] = factors["ticker"].map({"AAA": 2.0, "BBB": 1.0, "SPY": 0.0})

    result = run_backtest(
        prices,
        factors,
        {
            "selected_factor": "mom_12m",
            "top_n": 1,
            "benchmark_ticker": "QQQ",
            "initial_capital": 250000,
        },
    )

    equity = result["equity_curve"]
    assert equity["portfolio_value"].iloc[0] > 0
    assert equity["portfolio_value"].iloc[-1] > 250000
    assert equity["benchmark_value"].isna().all()
    assert result["warnings"]


def test_backtest_shifts_signals_to_avoid_same_day_lookahead():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_rows = []
    returns = {
        "AAA": [0.0, 1.0, 0.0, 0.0],
        "BBB": [0.0, 0.0, 0.0, 0.0],
        "SPY": [0.0, 0.0, 0.0, 0.0],
    }
    for ticker, vals in returns.items():
        price = 100.0
        for date, ret in zip(dates, vals):
            price *= 1 + ret
            price_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000,
                    "returns": ret,
                }
            )
    prices = pd.DataFrame(price_rows)
    factors = prices[["date", "ticker"]].copy()
    factors["signal"] = 0.0
    factors.loc[(factors["date"] == dates[0]) & (factors["ticker"] == "BBB"), "signal"] = 10.0
    factors.loc[(factors["date"] == dates[1]) & (factors["ticker"] == "AAA"), "signal"] = 10.0

    result = run_backtest(
        prices,
        factors,
        {
            "selected_factor": "signal",
            "top_n": 1,
            "rebalance_frequency": "D",
            "benchmark_ticker": "SPY",
            "transaction_cost": 0.0,
        },
    )

    equity = result["equity_curve"]
    day_two = equity.loc[equity["date"] == dates[1], "portfolio_returns"].iloc[0]
    assert day_two == pytest.approx(0.0)
