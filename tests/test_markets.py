import pandas as pd

from src.factors import calculate_factors
from src.markets import deterministic_market_brief, generate_trade_ideas, market_snapshot


def _price_frame():
    dates = pd.bdate_range("2023-01-02", periods=320)
    rows = []
    for ticker, daily_return in {"AAA": 0.0025, "BBB": -0.0015, "SPY": 0.001}.items():
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


def _market_frame():
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows = []
    for ticker, daily_return in {"^GSPC": 0.001, "^VIX": -0.002}.items():
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


def test_trade_ideas_include_risk_reward_fields():
    prices = _price_frame()
    factors = calculate_factors(prices)

    ideas = generate_trade_ideas(prices, factors, benchmark_ticker="SPY", max_ideas=4)

    assert not ideas.empty
    assert {"ticker", "direction", "target", "stop_loss", "risk_reward", "volatility_context"}.issubset(ideas.columns)
    assert ideas["risk_reward"].notna().all()


def test_market_brief_fallback_uses_computed_snapshot():
    prices = _price_frame()
    factors = calculate_factors(prices)
    market_data = _market_frame()
    snapshot = market_snapshot(market_data)

    brief = deterministic_market_brief(prices, factors, market_data, benchmark_ticker="SPY")

    assert not snapshot.empty
    assert "Market Overview" in brief
    assert "Potential Trade Themes" in brief
    assert "Desk-Style Summary" in brief
