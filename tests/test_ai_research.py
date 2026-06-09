import pandas as pd

from src.ai_research import generate_research_report
from src.factors import calculate_factors


def _price_frame():
    dates = pd.bdate_range("2023-01-02", periods=320)
    rows = []
    price = 100.0
    for date in dates:
        price *= 1.001
        rows.append(
            {
                "date": date,
                "ticker": "AAA",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adj_close": price,
                "volume": 1000,
                "returns": 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_ai_report_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prices = _price_frame()
    factors = calculate_factors(prices)

    report = generate_research_report("AAA", prices, factors, use_llm=True)

    assert "Executive Summary" in report
    assert "Price Performance Overview" in report
    assert "Benchmark-Relative Analysis" in report
    assert "Final Analyst-Style Conclusion" in report
