import pandas as pd
import pytest

from src import data_loader


def _synthetic_prices(tickers, start, end):
    dates = pd.bdate_range(start, end)
    rows = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            price = 100 + i * 10
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000,
                    "returns": 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["returns"] = frame.groupby("ticker")["adj_close"].pct_change().fillna(0.0)
    return frame[data_loader.PRICE_COLUMNS]


def test_get_price_data_formats_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "DB_PATH", tmp_path / "market_data.duckdb")
    monkeypatch.setattr(data_loader, "_download_price_data", _synthetic_prices)

    first = data_loader.get_price_data(["aapl", "msft"], start="2024-01-01", end="2024-01-10", refresh=True)

    assert list(first.columns) == data_loader.PRICE_COLUMNS
    assert set(first["ticker"]) == {"AAPL", "MSFT"}
    assert first.groupby("ticker")["returns"].first().eq(0.0).all()
    assert first["returns"].iloc[1] == pytest.approx(0.0)

    def fail_download(*args, **kwargs):
        raise AssertionError("cache should be used")

    monkeypatch.setattr(data_loader, "_download_price_data", fail_download)
    second = data_loader.get_price_data("AAPL,MSFT", start="2024-01-01", end="2024-01-10", refresh=False)
    assert len(second) == len(first)


def test_return_calculation_from_formatted_yfinance_frame():
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "Open": [100, 110, 121],
            "High": [100, 110, 121],
            "Low": [100, 110, 121],
            "Close": [100, 110, 121],
            "Adj Close": [100, 110, 121],
            "Volume": [1, 1, 1],
        }
    ).set_index("Date")
    formatted = data_loader._format_yfinance_frame(raw, "ABC")
    assert formatted["returns"].tolist() == pytest.approx([0.0, 0.10, 0.10])


def test_formats_yfinance_multiindex_price_ticker_frame():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"],
         ["ABC"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[100, 101, 99, 100, 100, 1000], [110, 111, 109, 110, 110, 1200]],
        index=dates,
        columns=columns,
    )

    formatted = data_loader._format_yfinance_frame(raw, "ABC")

    assert list(formatted.columns) == data_loader.PRICE_COLUMNS
    assert formatted["ticker"].unique().tolist() == ["ABC"]
    assert formatted["returns"].tolist() == pytest.approx([0.0, 0.10])
