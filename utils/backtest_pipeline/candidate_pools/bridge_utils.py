from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd

from utils.strategy_feature_cache import StrategyFeatureCache


def iter_data_files(data_dir: str) -> Iterable[Path]:
    return sorted(Path(data_dir).glob("*.txt"))


def scan_with_check(
    data_dir: str,
    scan_fn: Callable[[str, StrategyFeatureCache], list],
    strategy_family: str,
    candidate_pool: str,
    signal_type: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_data_files(data_dir):
        cache = StrategyFeatureCache(str(path))
        try:
            result = scan_fn(str(path), cache)
        except Exception:
            continue
        if not result or result[0] != 1:
            continue
        code = path.stem
        raw = cache.raw_df()
        if raw is None or raw.empty:
            continue
        signal_date = pd.Timestamp(raw["date"].iloc[-1])
        entry_date = signal_date
        stop_loss_price = None
        close_price = None
        base_score = 0.0
        note = ""

        if strategy_family in {"b2", "b3", "brick"}:
            stop_loss_price = float(result[1])
            close_price = float(result[2])
            base_score = float(result[3])
            note = str(result[4])
        elif strategy_family == "pin":
            base_score = 1.0
            note = str(result[3]) if len(result) >= 4 else str(result[1])

        rows.append(
            {
                "code": code,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "strategy_family": strategy_family,
                "candidate_pool": candidate_pool,
                "signal_type": signal_type,
                "base_score": base_score,
                "stop_loss_price": stop_loss_price,
                "close_price": close_price,
                "note": note,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "code",
                "signal_date",
                "entry_date",
                "strategy_family",
                "candidate_pool",
                "signal_type",
                "base_score",
                "stop_loss_price",
                "close_price",
                "note",
            ]
        )
    return pd.DataFrame(rows)
