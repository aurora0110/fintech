from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_exit_and_param_opt import (  # type: ignore
    add_base_features,
    load_one_csv,
    DATA_DIR,
    EXCLUDE_START,
    EXCLUDE_END,
)
from utils.tmp.run_b2_type14_param_search_cached import (  # type: ignore
    select_type1,
    select_type4,
)
from utils.tmp.run_b2_type14_split_account_opt_20260314 import (  # type: ignore
    simulate_portfolio,
    AccountConfig,
)
from utils.tmp.run_distribution_point_zone_calibration_20260314 import (  # type: ignore
    add_distribution_point_zone_labels,
)


RESULT_DIR = ROOT / "results/b2_distribution_point_zone_exit_ab_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TYPE1_PARAMS = {
    "ret1_min": 0.04,
    "upper_shadow_body_ratio": 0.40,
    "j_max": 90.0,
    "type1_near_ratio": 1.02,
    "type1_j_rank20_max": 0.10,
}
TYPE4_PARAMS = {
    "ret1_min": 0.03,
    "upper_shadow_body_ratio": 0.80,
    "j_max": 100.0,
    "type4_touch_ratio": 1.01,
}

ACCOUNT_CFG = AccountConfig(
    name="pos5_new3_b100_cap20_equal",
    max_positions=5,
    daily_new_limit=3,
    daily_budget_frac=1.0,
    position_cap_frac=0.2,
    allocation_mode="equal",
)

TYPE1_CANDIDATES_PATH = ROOT / "results/b2_type14_param_search_cached_20260313/type1_candidates.csv"
TYPE4_CANDIDATES_PATH = ROOT / "results/b2_type14_param_search_cached_20260313/type4_candidates.csv"


@dataclass(frozen=True)
class ExitVariant:
    name: str
    max_hold_days: int
    take_profit: float | None = None
    use_distribution_exit: bool = False


def load_all_data() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob("*.txt"))
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        raw = load_one_csv(path)
        if raw is None:
            continue
        x = add_distribution_point_zone_labels(add_base_features(raw))
        x = x[(x["date"] < EXCLUDE_START) | (x["date"] > EXCLUDE_END)].reset_index(drop=True)
        if len(x) >= 150:
            dfs[path.stem] = x
        if idx % 500 == 0:
            print(f"数据加载进度: {idx}/{total}")
    return dfs


def load_candidates(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["signal_date", "entry_date"])


def _distribution_sell_signal(row: pd.Series) -> bool:
    return bool(row.get("point_any", False)) or bool(row.get("zone_end", False))


def _exit_trade(x: pd.DataFrame, signal_idx: int, variant: ExitVariant):
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return len(x) - 1, float(x.iloc[-1]["close"]), "insufficient_data"
    entry_open = float(x.at[entry_idx, "open"])
    max_exit_idx = min(entry_idx + variant.max_hold_days, len(x) - 1)
    for i in range(entry_idx, max_exit_idx + 1):
        row = x.iloc[i]
        if variant.take_profit is not None:
            tp_price = entry_open * (1.0 + variant.take_profit)
            if float(row["high"]) >= tp_price:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, "open"]), f"tp_{variant.take_profit:.2f}"
        if variant.use_distribution_exit and _distribution_sell_signal(row):
            next_idx = min(i + 1, len(x) - 1)
            return next_idx, float(x.at[next_idx, "open"]), "point_or_zone_end_exit"
    return max_exit_idx, float(x.iloc[max_exit_idx]["close"]), f"hold_{variant.max_hold_days}_close"


def build_trade_table(signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], variant: ExitVariant, tag: str) -> pd.DataFrame:
    rows: List[dict] = []
    for rec in signals.itertuples(index=False):
        x = dfs[rec.code]
        exit_idx, exit_price, reason = _exit_trade(x, int(rec.signal_idx), variant)
        entry_open = float(rec.entry_open)
        ret = exit_price / entry_open - 1.0
        path = x.iloc[int(rec.entry_idx): exit_idx + 1].copy()
        rows.append(
            {
                "tag": tag,
                "variant": variant.name,
                "code": rec.code,
                "signal_idx": int(rec.signal_idx),
                "signal_date": rec.signal_date,
                "entry_idx": int(rec.entry_idx),
                "entry_date": rec.entry_date,
                "exit_idx": int(exit_idx),
                "exit_date": x.at[exit_idx, "date"],
                "entry_open": entry_open,
                "exit_price": exit_price,
                "return": ret,
                "reason": reason,
                "sort_score": float(rec.sort_score),
                "max_favorable": float(path["high"].max() / entry_open - 1.0),
                "max_adverse": float(path["low"].min() / entry_open - 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "sample_count": 0,
            "success_rate": np.nan,
            "avg_return": np.nan,
            "avg_max_favorable": np.nan,
            "avg_max_adverse": np.nan,
        }
    return {
        "sample_count": int(len(trades)),
        "success_rate": float((trades["return"] > 0).mean()),
        "avg_return": float(trades["return"].mean()),
        "avg_max_favorable": float(trades["max_favorable"].mean()),
        "avg_max_adverse": float(trades["max_adverse"].mean()),
    }


def run_type(tag: str, signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], variants: List[ExitVariant]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        trades = build_trade_table(signals, dfs, variant, tag)
        trade_summary = summarize_trades(trades)
        account_summary = simulate_portfolio(trades, dfs, ACCOUNT_CFG)
        rows.append({"tag": tag, "variant": variant.name, **trade_summary, **account_summary})
    return pd.DataFrame(rows)


def main():
    dfs = load_all_data()
    type1_candidates = load_candidates(TYPE1_CANDIDATES_PATH)
    type4_candidates = load_candidates(TYPE4_CANDIDATES_PATH)
    type1_signals = select_type1(type1_candidates, TYPE1_PARAMS)
    type4_signals = select_type4(type4_candidates, TYPE4_PARAMS)

    type1_variants = [
        ExitVariant("baseline_tp10_hold30", 30, take_profit=0.10, use_distribution_exit=False),
        ExitVariant("distribution_only_hold30", 30, take_profit=None, use_distribution_exit=True),
        ExitVariant("tp10_plus_distribution_hold30", 30, take_profit=0.10, use_distribution_exit=True),
    ]
    type4_variants = [
        ExitVariant("baseline_hold20", 20, take_profit=None, use_distribution_exit=False),
        ExitVariant("distribution_only_hold20", 20, take_profit=None, use_distribution_exit=True),
    ]

    type1_df = run_type("type1", type1_signals, dfs, type1_variants)
    type4_df = run_type("type4", type4_signals, dfs, type4_variants)
    comparison = pd.concat([type1_df, type4_df], ignore_index=True)
    comparison.to_csv(RESULT_DIR / "comparison.csv", index=False)

    distribution_counts = []
    for tag, sigs in [("type1", type1_signals), ("type4", type4_signals)]:
        subset = []
        for rec in sigs.itertuples(index=False):
            x = dfs[rec.code]
            start = int(rec.entry_idx)
            end = min(start + 40, len(x) - 1)
            path = x.iloc[start:end + 1]
            subset.append(
                {
                    "tag": tag,
                    "code": rec.code,
                    "point_any_count": int(path["point_any"].sum()),
                    "zone_end_count": int(path["zone_end"].sum()),
                    "zone_any_count": int(path["zone_any"].sum()),
                }
            )
        distribution_counts.extend(subset)
    pd.DataFrame(distribution_counts).to_csv(RESULT_DIR / "distribution_tag_counts.csv", index=False)

    summary = {
        "result_dir": str(RESULT_DIR),
        "type1_signal_count": int(len(type1_signals)),
        "type4_signal_count": int(len(type4_signals)),
        "comparison_rows": int(len(comparison)),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
