from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .utils import ensure_directory, markdown_to_text


def save_report(report_text: str, company_name: str, output_dir: str | Path = "reports", extension: str = "md") -> Path:
    directory = ensure_directory(output_dir)
    safe_name = "".join(ch for ch in company_name.lower().replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = extension.lstrip(".")
    path = directory / f"{safe_name}_investment_memo_{timestamp}.{suffix}"
    path.write_text(markdown_to_text(report_text) if suffix == "txt" else report_text, encoding="utf-8")
    return path


__all__ = ["save_report", "markdown_to_text"]
