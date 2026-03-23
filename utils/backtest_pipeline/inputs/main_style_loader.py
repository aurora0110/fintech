from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from core.data_loader import load_price_directory
from utils.backtest_pipeline.types import DataConfig


def _apply_exclude_ranges(
    stock_data: Dict[str, pd.DataFrame],
    exclude_ranges: List[tuple[str, str]],
) -> Dict[str, pd.DataFrame]:
    if not exclude_ranges:
        return stock_data

    out: Dict[str, pd.DataFrame] = {}
    for code, df in stock_data.items():
        filtered = df.copy()
        for start_str, end_str in exclude_ranges:
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            filtered = filtered[(filtered.index < start) | (filtered.index > end)]
        if len(filtered) >= 30:
            out[code] = filtered
    return out


def load_market_data(config: DataConfig) -> Tuple[Dict[str, pd.DataFrame], List[pd.Timestamp]]:
    """参考 main.py / core.data_loader 的输入流程，做统一数据入口。"""
    data_dir = Path(config.data_dir)
    stock_data, all_dates = load_price_directory(str(data_dir))
    stock_data = _apply_exclude_ranges(stock_data, config.exclude_ranges)
    valid_dates = sorted({dt for df in stock_data.values() for dt in df.index})
    return stock_data, valid_dates
