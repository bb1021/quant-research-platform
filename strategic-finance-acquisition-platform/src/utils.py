from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    if denominator is None or denominator == 0 or not np.isfinite(denominator):
        return default
    return numerator / denominator


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if value is None or not np.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def normalise_weights(weights: Iterable[float]) -> list[float]:
    values = [max(float(weight), 0.0) for weight in weights]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [value / total for value in values]


def money(value: float, unit: str = "m") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    prefix = "-" if value < 0 else ""
    value = abs(value)
    if unit == "m":
        return f"{prefix}€{value:,.1f}m"
    return f"{prefix}€{value:,.0f}"


def pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1%}"


def multiple(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.1f}x"


def score_band(score: float) -> str:
    if score >= 75:
        return "High conviction"
    if score >= 55:
        return "Proceed selectively"
    if score >= 40:
        return "Requires focused diligence"
    return "High risk"


def risk_band(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Low"
    if value <= high:
        return "Medium"
    return "High"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def markdown_to_text(markdown: str) -> str:
    replacements = {
        "## ": "",
        "# ": "",
        "**": "",
        "- ": " - ",
        "|": " ",
    }
    text = markdown
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
