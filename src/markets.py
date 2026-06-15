from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from .data_loader import get_price_data
from .utils import TRADING_DAYS, latest_rows


@dataclass(frozen=True)
class MarketInstrument:
    name: str
    ticker: str
    category: str


# Yahoo Finance proxies used for broad markets instruments:
# ^TNX is the CBOE US 10-year Treasury yield index, DX-Y.NYB is the US Dollar Index,
# GC=F and CL=F are liquid front-month gold and WTI crude futures proxies.
MARKET_INSTRUMENTS: tuple[MarketInstrument, ...] = (
    MarketInstrument("S&P 500", "^GSPC", "Equity Index"),
    MarketInstrument("Nasdaq", "^IXIC", "Equity Index"),
    MarketInstrument("FTSE 100", "^FTSE", "Equity Index"),
    MarketInstrument("US Dollar Index", "DX-Y.NYB", "FX"),
    MarketInstrument("US 10Y Yield", "^TNX", "Rates"),
    MarketInstrument("Gold", "GC=F", "Commodities"),
    MarketInstrument("WTI Crude", "CL=F", "Commodities"),
    MarketInstrument("VIX", "^VIX", "Volatility"),
)

MARKET_LABELS = {item.ticker: item.name for item in MARKET_INSTRUMENTS}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def load_market_dashboard_data(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start).normalize() if start is not None else pd.Timestamp.today().normalize() - pd.Timedelta(days=240)
    end_ts = pd.to_datetime(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    return get_price_data([item.ticker for item in MARKET_INSTRUMENTS], start=start_ts, end=end_ts, refresh=refresh)


def market_snapshot(market_data: pd.DataFrame) -> pd.DataFrame:
    if market_data is None or market_data.empty:
        return pd.DataFrame(columns=["instrument", "ticker", "category", "date", "level", "daily_move", "trend_1m"])

    df = market_data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "date"])
    latest = latest_rows(df, group_col="ticker")
    latest["instrument"] = latest["ticker"].map(MARKET_LABELS).fillna(latest["ticker"])
    latest["category"] = latest["ticker"].map({item.ticker: item.category for item in MARKET_INSTRUMENTS}).fillna("Market")

    month_returns = (
        df.groupby("ticker")["adj_close"]
        .apply(lambda s: s.iloc[-1] / s.iloc[-22] - 1.0 if len(s.dropna()) >= 22 and s.iloc[-22] else np.nan)
        .rename("trend_1m")
        .reset_index()
    )
    latest = latest.merge(month_returns, on="ticker", how="left")
    latest = latest.rename(columns={"adj_close": "level", "returns": "daily_move"})
    return latest[["instrument", "ticker", "category", "date", "level", "daily_move", "trend_1m"]].sort_values("instrument")


def normalised_market_trends(market_data: pd.DataFrame) -> pd.DataFrame:
    if market_data is None or market_data.empty:
        return pd.DataFrame(columns=["date", "instrument", "normalised_level"])
    df = market_data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"])
    df["instrument"] = df["ticker"].map(MARKET_LABELS).fillna(df["ticker"])
    base = df.groupby("ticker")["adj_close"].transform("first").replace(0, np.nan)
    df["normalised_level"] = df["adj_close"] / base * 100.0
    return df[["date", "instrument", "normalised_level"]].dropna()


def interpret_market_conditions(snapshot: pd.DataFrame) -> str:
    if snapshot is None or snapshot.empty:
        return "Market dashboard data is not loaded yet."

    moves = snapshot.dropna(subset=["daily_move"]).copy()
    if moves.empty:
        return "Market direction is mixed with insufficient daily move data."

    equity_names = {"S&P 500", "Nasdaq", "FTSE 100"}
    equity_move = moves.loc[moves["instrument"].isin(equity_names), "daily_move"].mean()
    vix_move = moves.loc[moves["instrument"] == "VIX", "daily_move"].mean()
    rates_move = moves.loc[moves["instrument"] == "US 10Y Yield", "daily_move"].mean()

    if equity_move > 0.004 and (pd.isna(vix_move) or vix_move < 0):
        regime = "risk-on"
    elif equity_move < -0.004 or (pd.notna(vix_move) and vix_move > 0.05):
        regime = "risk-off"
    else:
        regime = "balanced"

    rates_text = "rates are stable"
    if pd.notna(rates_move) and abs(rates_move) > 0.015:
        rates_text = "rates are moving materially higher" if rates_move > 0 else "rates are moving lower"
    return f"Current cross-asset tone is {regime}; equity indices average {equity_move:.2%} on the day and {rates_text}."


def _ticker_history(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return price_data.loc[price_data["ticker"].astype(str).str.upper() == ticker.upper()].sort_values("date").copy()


def _period_return(price_data: pd.DataFrame, ticker: str, periods: int) -> float:
    history = _ticker_history(price_data, ticker)
    if len(history) <= periods:
        return np.nan
    prior = _finite(history["adj_close"].iloc[-periods])
    latest = _finite(history["adj_close"].iloc[-1])
    if prior <= 0:
        return np.nan
    return latest / prior - 1.0


def generate_trade_ideas(
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    max_ideas: int = 8,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "direction",
        "signal_strength",
        "horizon",
        "rationale",
        "supporting_factors",
        "main_risk",
        "entry_price",
        "target",
        "stop_loss",
        "expected_upside",
        "expected_downside",
        "risk_reward",
        "volatility_context",
    ]
    if price_data is None or price_data.empty or factor_data is None or factor_data.empty:
        return pd.DataFrame(columns=columns)

    latest_factors = latest_rows(factor_data, group_col="ticker").copy()
    latest_prices = latest_rows(price_data, group_col="ticker")[["ticker", "adj_close"]].rename(columns={"adj_close": "entry_price"})
    latest = latest_factors.merge(latest_prices, on="ticker", how="left")
    benchmark_return = _period_return(price_data, benchmark_ticker, 63)
    rows: list[dict[str, Any]] = []

    for _, row in latest.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker or ticker == benchmark_ticker.upper():
            continue
        entry = _finite(row.get("entry_price"))
        if not np.isfinite(entry) or entry <= 0:
            continue

        mom_12m = _finite(row.get("mom_12m"))
        mom_3m = _finite(row.get("mom_3m"))
        vol_20d = _finite(row.get("vol_20d"))
        vol_60d = _finite(row.get("vol_60d"))
        mean_reversion = _finite(row.get("mean_reversion_5d"), 0.0)
        ma_crossover = _finite(row.get("ma_crossover"))
        rel_3m = _period_return(price_data, ticker, 63) - benchmark_return if np.isfinite(benchmark_return) else np.nan

        score = 0.0
        score += np.clip(mom_12m / 0.35, -1.0, 1.0) * 30 if np.isfinite(mom_12m) else 0
        score += np.clip(mom_3m / 0.15, -1.0, 1.0) * 22 if np.isfinite(mom_3m) else 0
        score += np.clip(ma_crossover / 0.08, -1.0, 1.0) * 18 if np.isfinite(ma_crossover) else 0
        score += np.clip(rel_3m / 0.12, -1.0, 1.0) * 18 if np.isfinite(rel_3m) else 0
        score += np.clip(mean_reversion / 0.05, -1.0, 1.0) * 12 if np.isfinite(mean_reversion) else 0

        if score >= 25:
            direction = "Long"
            horizon = "Medium-term"
        elif score <= -25:
            direction = "Short"
            horizon = "Short-term"
        else:
            direction = "Neutral"
            horizon = "Watchlist"

        signal_strength = min(100.0, abs(score))
        annual_vol = vol_20d if np.isfinite(vol_20d) else vol_60d
        if not np.isfinite(annual_vol) or annual_vol <= 0:
            annual_vol = 0.25
        horizon_vol = annual_vol / np.sqrt(TRADING_DAYS) * np.sqrt(20)
        # Indicative target and stop bands are scaled from recent realised volatility.
        band = float(np.clip(horizon_vol * 1.2, 0.03, 0.18))
        stop_band = band * 0.55

        if direction == "Short":
            target = entry * (1.0 - band)
            stop = entry * (1.0 + stop_band)
            upside = (entry - target) / entry
            downside = (stop - entry) / entry
        elif direction == "Long":
            target = entry * (1.0 + band)
            stop = entry * (1.0 - stop_band)
            upside = (target - entry) / entry
            downside = (entry - stop) / entry
        else:
            target = entry * (1.0 + band * 0.35)
            stop = entry * (1.0 - band * 0.35)
            upside = (target - entry) / entry
            downside = (entry - stop) / entry

        supporting = [
            f"12m momentum {mom_12m:.1%}" if np.isfinite(mom_12m) else "12m momentum n/a",
            f"3m momentum {mom_3m:.1%}" if np.isfinite(mom_3m) else "3m momentum n/a",
            f"benchmark-relative 3m {rel_3m:.1%}" if np.isfinite(rel_3m) else "benchmark-relative n/a",
            f"20d realised vol {annual_vol:.1%}",
        ]
        rationale = (
            "Positive trend, benchmark-relative strength, and constructive moving-average profile."
            if direction == "Long"
            else "Weak trend and negative benchmark-relative performance warrant caution."
            if direction == "Short"
            else "Signals are mixed, so the name is better monitored than expressed."
        )
        main_risk = (
            "Momentum reversal or broader risk-off tape."
            if direction == "Long"
            else "Short-covering rally or positive earnings surprise."
            if direction == "Short"
            else "A cleaner signal may emerge only after the next catalyst."
        )
        rows.append(
            {
                "ticker": ticker,
                "direction": direction,
                "signal_strength": signal_strength,
                "horizon": horizon,
                "rationale": rationale,
                "supporting_factors": "; ".join(supporting),
                "main_risk": main_risk,
                "entry_price": entry,
                "target": target,
                "stop_loss": stop,
                "expected_upside": upside,
                "expected_downside": downside,
                "risk_reward": upside / downside if downside > 0 else np.nan,
                "volatility_context": f"{annual_vol:.1%} annualised realised volatility",
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    ideas = pd.DataFrame(rows)
    return ideas.sort_values("signal_strength", ascending=False).head(max_ideas).reset_index(drop=True)


def deterministic_market_brief(
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark_ticker: str = "SPY",
) -> str:
    snapshot = market_snapshot(market_data)
    conditions = interpret_market_conditions(snapshot)
    movers = snapshot.sort_values("daily_move", ascending=False).dropna(subset=["daily_move"]) if not snapshot.empty else pd.DataFrame()
    top_mover = movers.iloc[0] if not movers.empty else None
    weak_mover = movers.iloc[-1] if not movers.empty else None
    ideas = generate_trade_ideas(price_data, factor_data, benchmark_ticker=benchmark_ticker, max_ideas=3)

    theme = "factor dispersion"
    if not ideas.empty:
        long_count = int((ideas["direction"] == "Long").sum())
        short_count = int((ideas["direction"] == "Short").sum())
        theme = "selective upside leadership" if long_count > short_count else "defensive risk management"

    key_movers = (
        f"{top_mover['instrument']} is the strongest dashboard instrument at {top_mover['daily_move']:.2%}, "
        f"while {weak_mover['instrument']} is weakest at {weak_mover['daily_move']:.2%}."
        if top_mover is not None and weak_mover is not None
        else "Key mover data is unavailable."
    )
    idea_lines = "\n".join(
        f"- {row.ticker}: {row.direction}, strength {row.signal_strength:.0f}/100, {row.rationale}"
        for row in ideas.itertuples()
    ) or "- No trade ideas yet. Load a universe with enough factor history."

    return f"""# AI Market Brief

## Market Overview
{conditions}

## Key Movers
{key_movers}

## Risk Sentiment
Risk sentiment is best described as {theme}. Cross-asset dashboard instruments and the selected universe should be reviewed together before sizing any idea.

## Potential Trade Themes
{idea_lines}

## Portfolio and Risk Implications
Monitor concentration in high-volatility names, compare new exposures against {benchmark_ticker.upper()}, and keep stop levels aligned with realised volatility rather than arbitrary price points.

## Desk-Style Summary
The current tape supports a selective, risk-budget-aware research process: focus on names with confirmed momentum and benchmark-relative strength, keep downside scenarios visible, and avoid treating any single factor as a standalone investment conclusion.
"""


def _llm_market_brief(template: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise global markets desk research assistant."},
                    {"role": "user", "content": f"Refine this market brief without inventing facts:\n\n{template}"},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def generate_market_brief(
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    use_llm: bool = False,
) -> str:
    template = deterministic_market_brief(price_data, factor_data, market_data, benchmark_ticker=benchmark_ticker)
    if use_llm:
        llm = _llm_market_brief(template)
        if llm:
            return llm
    return template
