from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only without installed deps
    yf = None

from .utils import coerce_date, ensure_directory, normalize_tickers, today_timestamp


DB_PATH = Path(os.getenv("MARKET_DATA_DB", "data/market_data.duckdb"))

PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "returns",
]


def _connect() -> duckdb.DuckDBPyConnection:
    ensure_directory(DB_PATH.parent)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            date DATE,
            ticker VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume DOUBLE,
            returns DOUBLE
        )
        """
    )
    return con


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _query_cached(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not tickers:
        return _empty_price_frame()
    con = _connect()
    placeholders = ", ".join(["?"] * len(tickers))
    params: list[object] = [start.date(), end.date(), *tickers]
    query = f"""
        SELECT {", ".join(PRICE_COLUMNS)}
        FROM prices
        WHERE date >= ? AND date <= ? AND ticker IN ({placeholders})
        ORDER BY date, ticker
    """
    try:
        cached = con.execute(query, params).fetchdf()
    finally:
        con.close()
    return _coerce_price_schema(cached)


def _coerce_price_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_price_frame()
    out = df.copy()
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[PRICE_COLUMNS]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    numeric_cols = [c for c in PRICE_COLUMNS if c not in {"date", "ticker"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["returns"] = out["returns"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _cache_coverage(cached: pd.DataFrame, tickers: list[str]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
    coverage: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}
    for ticker in tickers:
        ticker_df = cached.loc[cached["ticker"] == ticker]
        if ticker_df.empty:
            coverage[ticker] = None
        else:
            coverage[ticker] = (ticker_df["date"].min(), ticker_df["date"].max())
    return coverage


def _stale_tickers(
    cached: pd.DataFrame,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    coverage = _cache_coverage(cached, tickers)
    start_tolerance = start + pd.Timedelta(days=10)
    end_tolerance = end - pd.Timedelta(days=7)
    stale: list[str] = []
    for ticker, bounds in coverage.items():
        if bounds is None:
            stale.append(ticker)
            continue
        min_date, max_date = bounds
        if min_date > start_tolerance or max_date < end_tolerance:
            stale.append(ticker)
    return stale


def _download_one_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run pip install -r requirements.txt.")

    download_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=download_end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _format_yfinance_frame(raw, ticker)


def _download_price_data(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for ticker in tickers:
        try:
            frame = _download_one_ticker(ticker, start, end)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            errors[ticker] = str(exc)

    if not frames:
        if errors:
            message = "; ".join(f"{ticker}: {err}" for ticker, err in errors.items())
            raise ValueError(f"No market data downloaded. Provider errors: {message}")
        return _empty_price_frame()
    return _coerce_price_schema(pd.concat(frames, ignore_index=True))


def _format_yfinance_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_price_frame()

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        ticker_upper = ticker.upper()
        selected = False
        for level in range(frame.columns.nlevels):
            level_values = pd.Index(frame.columns.get_level_values(level).astype(str).str.upper())
            if ticker_upper in set(level_values):
                original_label = next(
                    label
                    for label in frame.columns.get_level_values(level).unique()
                    if str(label).upper() == ticker_upper
                )
                frame = frame.xs(original_label, axis=1, level=level)
                selected = True
                break
        if isinstance(frame.columns, pd.MultiIndex):
            if selected and frame.columns.nlevels == 1:
                frame.columns = frame.columns.get_level_values(0)
            else:
                non_ticker_levels = [
                    level
                    for level in range(frame.columns.nlevels)
                    if ticker_upper not in set(pd.Index(frame.columns.get_level_values(level).astype(str).str.upper()))
                ]
                level = non_ticker_levels[0] if non_ticker_levels else -1
                frame.columns = frame.columns.get_level_values(level)

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename)
    frame = frame.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_col: "date"})
    frame["ticker"] = ticker.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in frame.columns:
            frame[col] = np.nan
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.sort_values("date").drop_duplicates(["date", "ticker"], keep="last")
    price_cols = ["open", "high", "low", "close", "adj_close"]
    frame[price_cols] = frame[price_cols].ffill().bfill()
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.dropna(subset=["date", "ticker", "adj_close"])
    frame["returns"] = frame.groupby("ticker")["adj_close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _coerce_price_schema(frame[PRICE_COLUMNS])


def _write_cache(price_data: pd.DataFrame, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> None:
    if price_data.empty:
        return
    clean = _coerce_price_schema(price_data)
    con = _connect()
    placeholders = ", ".join(["?"] * len(tickers))
    delete_params: list[object] = [start.date(), end.date(), *tickers]
    try:
        con.execute(
            f"""
            DELETE FROM prices
            WHERE date >= ? AND date <= ? AND ticker IN ({placeholders})
            """,
            delete_params,
        )
        con.register("incoming_prices", clean[PRICE_COLUMNS])
        con.execute(f"INSERT INTO prices SELECT {', '.join(PRICE_COLUMNS)} FROM incoming_prices")
    finally:
        con.close()


def get_price_data(
    tickers: str | list[str],
    start: str | pd.Timestamp = "2015-01-01",
    end: str | pd.Timestamp | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Download or load daily OHLCV data with local DuckDB caching.

    Returns columns: date, ticker, open, high, low, close, adj_close, volume, returns.
    """
    normalized = normalize_tickers(tickers)
    if not normalized:
        return _empty_price_frame()

    start_ts = coerce_date(start) or pd.Timestamp("2015-01-01")
    end_ts = coerce_date(end) or today_timestamp()
    if end_ts < start_ts:
        raise ValueError("end date must be on or after start date")

    cached = _query_cached(normalized, start_ts, end_ts) if not refresh else _empty_price_frame()
    stale = normalized if refresh else _stale_tickers(cached, normalized, start_ts, end_ts)

    if stale:
        downloaded = _download_price_data(stale, start_ts, end_ts)
        if not downloaded.empty:
            _write_cache(downloaded, stale, start_ts, end_ts)
        cached = _query_cached(normalized, start_ts, end_ts)

    if cached.empty:
        return _empty_price_frame()

    cached = cached.loc[cached["date"].between(start_ts, end_ts)].copy()
    ordered = cached.sort_values(["ticker", "date"])
    returns = ordered.groupby("ticker")["adj_close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cached.loc[returns.index, "returns"] = returns
    return _coerce_price_schema(cached)
