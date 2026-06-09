from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from .metrics import drawdown_series


load_dotenv()


@dataclass
class ResearchContext:
    ticker: str
    as_of: str
    latest_close: float
    returns_1m: float
    returns_3m: float
    returns_6m: float
    returns_12m: float
    realized_vol_60d: float
    max_drawdown: float
    momentum_rank: float | None
    factor_profile: dict[str, float]
    strategy_status: str
    benchmark_ticker: str | None
    beta_vs_benchmark: float
    alpha_vs_benchmark: float
    information_ratio: float


def _period_return(history: pd.DataFrame, periods: int) -> float:
    if len(history) <= periods:
        return np.nan
    latest = history["adj_close"].iloc[-1]
    prior = history["adj_close"].iloc[-periods]
    if prior <= 0:
        return np.nan
    return float(latest / prior - 1.0)


def _fmt_pct(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.1%}"


def _fmt_num(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:,.2f}"


def build_research_context(
    ticker: str,
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    backtest_result: dict[str, Any] | None = None,
) -> ResearchContext:
    ticker = ticker.upper()
    history = price_data.loc[price_data["ticker"].astype(str).str.upper() == ticker].copy()
    if history.empty:
        raise ValueError(f"No price data available for {ticker}")
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history.sort_values("date")
    returns = pd.to_numeric(history["returns"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()

    factor_profile: dict[str, float] = {}
    momentum_rank = None
    if factor_data is not None and not factor_data.empty:
        f = factor_data.loc[factor_data["ticker"].astype(str).str.upper() == ticker].copy()
        if not f.empty:
            f["date"] = pd.to_datetime(f["date"]).dt.normalize()
            latest_factors = f.sort_values("date").iloc[-1]
            for col in ["mom_12m", "mom_6m", "mom_3m", "vol_60d", "vol_20d", "mean_reversion_5d", "ma_crossover"]:
                if col in latest_factors.index:
                    factor_profile[col] = float(latest_factors[col]) if pd.notna(latest_factors[col]) else np.nan
            rank_col = "relative_strength_rank"
            if rank_col in latest_factors.index and pd.notna(latest_factors[rank_col]):
                momentum_rank = float(latest_factors[rank_col])

    strategy_status = "Not evaluated in the latest backtest."
    benchmark_ticker = None
    beta_vs_benchmark = np.nan
    alpha_vs_benchmark = np.nan
    information_ratio_value = np.nan
    if backtest_result and isinstance(backtest_result.get("weights"), pd.DataFrame):
        config = backtest_result.get("config", {})
        benchmark_ticker = str(config.get("benchmark_ticker", "") or "").upper() or None
        metrics = backtest_result.get("metrics", {})
        beta_vs_benchmark = float(metrics.get("Beta vs Benchmark", np.nan))
        alpha_vs_benchmark = float(metrics.get("Alpha vs Benchmark", np.nan))
        information_ratio_value = float(metrics.get("Information Ratio", np.nan))
        weights = backtest_result["weights"]
        if not weights.empty:
            latest_date = pd.to_datetime(weights["date"]).max()
            latest_weights = weights.loc[pd.to_datetime(weights["date"]) == latest_date]
            row = latest_weights.loc[latest_weights["ticker"].astype(str).str.upper() == ticker]
            if row.empty:
                strategy_status = "Excluded from the latest strategy portfolio."
            else:
                strategy_status = f"Included in the latest strategy portfolio at {float(row['weight'].iloc[0]):.1%} weight."

    return ResearchContext(
        ticker=ticker,
        as_of=str(history["date"].iloc[-1].date()),
        latest_close=float(history["adj_close"].iloc[-1]),
        returns_1m=_period_return(history, 21),
        returns_3m=_period_return(history, 63),
        returns_6m=_period_return(history, 126),
        returns_12m=_period_return(history, 252),
        realized_vol_60d=float(returns.tail(60).std() * np.sqrt(252)) if len(returns.tail(60)) > 1 else np.nan,
        max_drawdown=float(drawdown_series(equity).min()),
        momentum_rank=momentum_rank,
        factor_profile=factor_profile,
        strategy_status=strategy_status,
        benchmark_ticker=benchmark_ticker,
        beta_vs_benchmark=beta_vs_benchmark,
        alpha_vs_benchmark=alpha_vs_benchmark,
        information_ratio=information_ratio_value,
    )


def deterministic_report(context: ResearchContext) -> str:
    f = context.factor_profile
    momentum_text = (
        f"ranked #{int(context.momentum_rank)} on 12-month relative strength"
        if context.momentum_rank is not None and np.isfinite(context.momentum_rank)
        else "has insufficient relative strength history for a stable cross-sectional rank"
    )
    bull_points = []
    bear_points = []
    if f.get("mom_12m", np.nan) > 0:
        bull_points.append("Positive 12-month momentum indicates persistent buyer sponsorship.")
    else:
        bear_points.append("Weak or unavailable 12-month momentum reduces trend confirmation.")
    if f.get("ma_crossover", np.nan) > 0:
        bull_points.append("The medium-term moving average profile remains constructive.")
    else:
        bear_points.append("The moving average profile is not yet supportive.")
    if context.realized_vol_60d < 0.35:
        bull_points.append("Realized volatility is not elevated relative to typical single-name equity risk.")
    else:
        bear_points.append("Elevated realized volatility raises sizing and drawdown risk.")

    if not bull_points:
        bull_points.append("Upside case depends on improving momentum, lower volatility, and stronger factor breadth.")
    if not bear_points:
        bear_points.append("The main bear case is factor mean reversion after a strong run.")
    benchmark_text = (
        f"Versus {context.benchmark_ticker}, the latest backtest estimates beta at {_fmt_num(context.beta_vs_benchmark)}, "
        f"annualized alpha at {_fmt_pct(context.alpha_vs_benchmark)}, and information ratio at {_fmt_num(context.information_ratio)}."
        if context.benchmark_ticker
        else "Benchmark-relative statistics are unavailable because no benchmark backtest was supplied."
    )

    return f"""# Equity Research Report: {context.ticker}

As of: {context.as_of}

## Executive Summary
{context.ticker} last traded at {_fmt_num(context.latest_close)} on an adjusted close basis. The stock {momentum_text}. The quantitative profile shows 1-month return of {_fmt_pct(context.returns_1m)}, 3-month return of {_fmt_pct(context.returns_3m)}, and 12-month return of {_fmt_pct(context.returns_12m)}. {context.strategy_status}

## Price Performance Overview
- 1-month return: {_fmt_pct(context.returns_1m)}
- 3-month return: {_fmt_pct(context.returns_3m)}
- 6-month return: {_fmt_pct(context.returns_6m)}
- 12-month return: {_fmt_pct(context.returns_12m)}
- Latest adjusted close: {_fmt_num(context.latest_close)}

## Factor Profile
- 12-month momentum: {_fmt_pct(f.get("mom_12m", np.nan))}
- 6-month momentum: {_fmt_pct(f.get("mom_6m", np.nan))}
- 3-month momentum: {_fmt_pct(f.get("mom_3m", np.nan))}
- 60-day realized volatility: {_fmt_pct(f.get("vol_60d", context.realized_vol_60d))}
- 20-day realized volatility: {_fmt_pct(f.get("vol_20d", np.nan))}
- Mean reversion signal: {_fmt_pct(f.get("mean_reversion_5d", np.nan))}
- Moving average crossover: {_fmt_pct(f.get("ma_crossover", np.nan))}

## Volatility and Drawdown Analysis
The 60-day realized volatility is {_fmt_pct(context.realized_vol_60d)} and the maximum drawdown over the available sample is {_fmt_pct(context.max_drawdown)}. Position sizing should reflect both standalone volatility and benchmark-relative contribution. Fundamental data is represented as a placeholder in this MVP, so valuation and earnings quality require external validation before investment use.

## Benchmark-Relative Analysis
{benchmark_text}

## Strategy Inclusion Commentary
{context.strategy_status}

## Bull Case
{" ".join(bull_points)}

## Bear Case
{" ".join(bear_points)}

## Key Risks
- Factor crowding may reverse recent momentum signals.
- Company-specific earnings, guidance, or balance sheet shocks are not captured by price-only factors.
- Liquidity, transaction costs, and tax assumptions are simplified.
- Backtest results are sensitive to universe selection, rebalance timing, and benchmark availability.

## Final Analyst-Style Conclusion
{context.ticker} is best viewed through a disciplined, factor-aware process rather than a standalone narrative. The current quantitative evidence supports inclusion only when its momentum, volatility, and ranking profile remain competitive versus the selected universe. This report is research support, not investment advice.
"""


def _llm_report(context: ResearchContext, template: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = f"{base_url}/chat/completions"
    prompt = (
        "Rewrite the following deterministic equity research report in an institutional style. "
        "Use only the provided facts. Keep the same section headings and do not invent fundamentals.\n\n"
        f"{template}"
    )
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a cautious institutional quant equity research assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def generate_research_report(
    ticker: str,
    price_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    backtest_result: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> str:
    context = build_research_context(ticker, price_data, factor_data, backtest_result)
    template = deterministic_report(context)
    if use_llm:
        llm = _llm_report(context, template)
        if llm:
            return llm
    return template
