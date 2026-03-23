from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
B1_BASE_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
B1_TEMPLATE_SIGNAL_DIR = ROOT / "results" / "b1_txt_template_signal_v2_full_20260322_201735"


@lru_cache(maxsize=8)
def load_candidate_frame(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)
    for col in ["signal_date", "entry_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_b1_base_candidates() -> pd.DataFrame:
    return load_candidate_frame(str(B1_BASE_SIGNAL_DIR / "candidate_enriched.csv")).copy()


def load_b1_template_candidates() -> pd.DataFrame:
    return load_candidate_frame(str(B1_TEMPLATE_SIGNAL_DIR / "candidate_enriched.csv")).copy()
