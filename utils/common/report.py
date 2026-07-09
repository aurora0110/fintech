from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def make_result_dir(base_dir: str | Path, task_name: str, timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / f"{task_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_df(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, **kwargs)
    return out_path


def write_markdown_report(path: str | Path, content: str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path

