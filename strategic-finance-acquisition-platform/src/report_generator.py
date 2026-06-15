from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .utils import ensure_directory, markdown_to_text


def save_report(report_text: str, company_name: str, output_dir: str | Path = "reports", extension: str = "md") -> Path:
    directory = ensure_directory(output_dir)
    safe_name = "".join(ch for ch in company_name.lower().replace(" ", "_") if ch.isalnum() or ch in {"_", "-"})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = extension.lstrip(".")
    path = directory / f"{safe_name}_board_memo_{timestamp}.{suffix}"
    if suffix == "txt":
        path.write_text(markdown_to_text(report_text), encoding="utf-8")
    else:
        path.write_text(report_text, encoding="utf-8")
    return path


def dataframe_to_markdown(frame: pd.DataFrame, title: str) -> str:
    if frame is None or frame.empty:
        return f"## {title}\n\nNo data available.\n"
    return f"## {title}\n\n{frame.to_markdown(index=False)}\n"


__all__ = ["save_report", "dataframe_to_markdown", "markdown_to_text"]
