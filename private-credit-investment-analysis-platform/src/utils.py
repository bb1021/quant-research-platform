from __future__ import annotations

from pathlib import Path

import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator is None or denominator == 0 or not np.isfinite(denominator):
        return default
    return numerator / denominator


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if value is None or not np.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def money(value: float, unit: str = "m") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if unit == "m":
        return f"{sign}${value:,.1f}m"
    return f"{sign}${value:,.0f}"


def pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"


def multiple(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}x"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def markdown_to_text(markdown: str) -> str:
    text = markdown
    for old, new in {"# ": "", "## ": "", "**": "", "- ": " - ", "|": " "}.items():
        text = text.replace(old, new)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
