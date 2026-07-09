from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def check_min_sample_count(sample_count: int, minimum: int) -> bool:
    return int(sample_count) >= int(minimum)


def ensure_positive_equity(final_equity: float) -> None:
    if final_equity <= 0:
        raise ValueError(f"final equity must be positive, got {final_equity}")


def ensure_file_exists(path: str | Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(str(path))

