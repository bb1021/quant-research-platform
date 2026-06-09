from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_directory


def save_report(report_text: str, ticker: str, output_dir: str | Path = "reports", extension: str = "md") -> Path:
    directory = ensure_directory(output_dir)
    safe_ticker = "".join(ch for ch in ticker.upper() if ch.isalnum() or ch in {"-", "_"})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = extension.lower().lstrip(".")
    path = directory / f"{safe_ticker}_research_report_{timestamp}.{ext}"
    path.write_text(report_text, encoding="utf-8")
    return path


def markdown_to_text(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            clean = clean.lstrip("#").strip()
        if clean.startswith("- "):
            clean = clean[2:].strip()
        lines.append(clean)
    return "\n".join(lines)


def metrics_to_markdown(metrics: dict[str, float] | pd.DataFrame, title: str = "Performance Metrics") -> str:
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        frame = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    if frame.empty:
        return f"## {title}\n\nNo metrics available.\n"
    rows = ["| Metric | Value |", "|---|---|"]
    for _, row in frame.iterrows():
        rows.append(f"| {row['Metric']} | {row['Value']} |")
    return f"## {title}\n\n" + "\n".join(rows) + "\n"


def backtest_summary_markdown(backtest_result: dict[str, Any]) -> str:
    config = backtest_result.get("config", {})
    metrics = backtest_result.get("metrics", {})
    weights = backtest_result.get("weights", pd.DataFrame())
    trades = backtest_result.get("trades", pd.DataFrame())
    equity = backtest_result.get("equity_curve", pd.DataFrame())
    warnings = backtest_result.get("warnings", [])

    lines = [
        "# Backtest Summary",
        "",
        "## Configuration",
        f"- Factor: {config.get('selected_factor', 'n/a')}",
        f"- Top N: {config.get('top_n', 'n/a')}",
        f"- Weighting: {config.get('weighting_method', 'n/a')}",
        f"- Rebalance frequency: {config.get('rebalance_frequency', 'n/a')}",
        f"- Transaction cost: {config.get('transaction_cost', 'n/a')}",
        f"- Benchmark: {config.get('benchmark_ticker', 'n/a')}",
        f"- Initial capital: {config.get('initial_capital', 'n/a')}",
        "",
        metrics_to_markdown(metrics).strip(),
        "",
    ]

    if isinstance(equity, pd.DataFrame) and not equity.empty:
        start = equity["date"].min()
        end = equity["date"].max()
        final_value = equity["portfolio_value"].iloc[-1]
        lines.extend(
            [
                "## Equity Curve",
                f"- Start: {start.date() if hasattr(start, 'date') else start}",
                f"- End: {end.date() if hasattr(end, 'date') else end}",
                f"- Final portfolio value: {final_value:,.2f}",
                "",
            ]
        )

    if isinstance(weights, pd.DataFrame) and not weights.empty:
        latest_date = pd.to_datetime(weights["date"]).max()
        latest = weights.loc[pd.to_datetime(weights["date"]) == latest_date].sort_values("weight", ascending=False)
        lines.append("## Latest Holdings")
        for _, row in latest.head(25).iterrows():
            lines.append(f"- {row['ticker']}: {row['weight']:.2%}")
        lines.append("")

    if isinstance(trades, pd.DataFrame) and not trades.empty:
        lines.extend(["## Recent Rebalances"])
        for _, row in trades.tail(20).iterrows():
            date = pd.to_datetime(row["date"]).date()
            lines.append(
                f"- {date} {row['ticker']}: {row['previous_weight']:.2%} to {row['target_weight']:.2%}"
            )
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
