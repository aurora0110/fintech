from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from config import FactorConfig, RunConfig


def normalize_factors(df: pd.DataFrame, cfg: RunConfig, factor_list: List[FactorConfig]) -> pd.DataFrame:
    out = df.copy()
    for f in factor_list:
        if f.name not in out.columns:
            continue
        score_col = f"{f.name}_score"
        if f.kind == "continuous":
            if cfg.normalize_method == "rank":
                out[score_col] = out.groupby(cfg.date_col)[f.name].rank(pct=True, method="average")
                out[score_col] = out[score_col] * 2.0 - 1.0  # [0,1] -> [-1,1]
            else:
                g = out.groupby(cfg.date_col)[f.name]
                mu = g.transform("mean")
                sigma = g.transform("std").replace(0, np.nan)
                out[score_col] = ((out[f.name] - mu) / sigma).fillna(0.0)

            if f.direction == "lower_better":
                out[score_col] = -out[score_col]
        else:
            out[score_col] = 0.0
    return out

