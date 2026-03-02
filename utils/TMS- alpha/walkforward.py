from __future__ import annotations

from typing import List, Tuple
import pandas as pd


def generate_walkforward_splits(
    dates: pd.Series,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    ds = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique()))
    if len(ds) == 0:
        return []
    years = sorted({d.year for d in ds})
    if len(years) < (train_years + test_years):
        # Fallback split: 70/30 by date.
        sorted_dates = pd.Series(ds.sort_values())
        cut = int(len(sorted_dates) * 0.7)
        if cut <= 0 or cut >= len(sorted_dates):
            return []
        return [
            (
                sorted_dates.iloc[0],
                sorted_dates.iloc[cut - 1],
                sorted_dates.iloc[cut],
                sorted_dates.iloc[-1],
            )
        ]

    splits = []
    start_idx = 0
    while True:
        tr_start_y = years[start_idx]
        tr_end_y = tr_start_y + train_years - 1
        te_start_y = tr_end_y + 1
        te_end_y = te_start_y + test_years - 1
        if te_end_y > years[-1]:
            break
        tr_dates = ds[(ds.year >= tr_start_y) & (ds.year <= tr_end_y)]
        te_dates = ds[(ds.year >= te_start_y) & (ds.year <= te_end_y)]
        if not tr_dates.empty and not te_dates.empty:
            splits.append((tr_dates.min(), tr_dates.max(), te_dates.min(), te_dates.max()))
        start_idx += step_years
        if start_idx >= len(years):
            break
    return splits
