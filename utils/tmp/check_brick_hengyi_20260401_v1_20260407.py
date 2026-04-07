from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter
from utils import brickfilter_case_rank_lgbm_top20 as case_mod
from utils import technical_indicators as ti
from utils.brick_optimize import brickfilter_case_first_v1_20260326 as case_first
from utils.brick_optimize import brickfilter_case_recall_v1_20260327 as case_recall
from utils.brick_optimize import run_brick_case_rank_daily_stream_v2_20260328 as phase0
from utils.brick_optimize import run_brick_case_rank_model_search_v1_20260327 as rank_model


FILE_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260402/SZ#000703.txt")
TARGET_DATE = pd.Timestamp("2026-04-01")
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260402")


def check_brick_filter() -> dict:
    raw = ti._load_price_data(str(FILE_PATH))
    cols = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    df = raw.rename(columns=cols)[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = "SZ#000703"
    df = df[df["date"] <= TARGET_DATE].copy().reset_index(drop=True)
    feat = brick_filter.add_features(df)
    latest = feat.iloc[-1]
    mask_a = bool(latest["pattern_a"]) and float(latest["rebound_ratio"]) >= 0.8
    mask_b = bool(latest["pattern_b"]) and float(latest["rebound_ratio"]) >= 1.0
    legacy_ok = bool(latest["signal_base"]) and float(latest["ret1"]) <= 0.08 and (mask_a or mask_b) and float(latest["trend_line"]) > float(latest["long_line"])
    perfect_ok = legacy_ok and (-0.03 <= float(latest["ret1"]) <= 0.11)
    return {
        "selected": bool(perfect_ok),
        "pattern_a": bool(latest["pattern_a"]),
        "pattern_b": bool(latest["pattern_b"]),
        "signal_base": bool(latest["signal_base"]),
        "ret1": float(latest["ret1"]),
        "rebound_ratio": float(latest["rebound_ratio"]) if pd.notna(latest["rebound_ratio"]) else None,
        "trend_gt_long": bool(float(latest["trend_line"]) > float(latest["long_line"])),
        "pullback_shrinking": bool(latest["pullback_shrinking"]) if pd.notna(latest["pullback_shrinking"]) else None,
        "signal_vs_ma5": float(latest["signal_vs_ma5"]) if pd.notna(latest["signal_vs_ma5"]) else None,
        "not_sideways": bool(latest["not_sideways"]) if pd.notna(latest["not_sideways"]) else None,
        "close_location": float(latest["close_location"]) if pd.notna(latest["close_location"]) else None,
        "upper_shadow_pct": float(latest["upper_shadow_pct"]) if pd.notna(latest["upper_shadow_pct"]) else None,
    }


def check_case_rank() -> dict:
    rec = case_first._record_for_date(str(FILE_PATH), TARGET_DATE.strftime("%Y-%m-%d"), required_lens=case_recall.CASE_SEQ_LENS)
    if rec is None:
        return {"candidate": False, "selected": False}
    cand = pd.DataFrame([rec])
    enriched = case_recall.enrich_candidates_for_date(TARGET_DATE, cand, DATA_DIR)
    if enriched.empty:
        return {"candidate": True, "selected": False, "enriched": False}
    enriched = rank_model._prepare_features(enriched)
    bundle = case_mod._load_runtime_bundle()
    enriched["model_score"] = phase0._score_with_model(bundle["model"], enriched)
    row = enriched.iloc[0]
    return {
        "candidate": True,
        "selected": bool(float(row["model_score"]) >= float(bundle["threshold"])),
        "model_score": float(row["model_score"]),
        "threshold": float(bundle["threshold"]),
        "same_type_case_sim_score": float(row["same_type_case_sim_score"]) if pd.notna(row["same_type_case_sim_score"]) else None,
        "perfect_case_sim_score": float(row["perfect_case_sim_score"]) if pd.notna(row["perfect_case_sim_score"]) else None,
        "perfect_case_quality_score": float(row["perfect_case_quality_score"]) if pd.notna(row["perfect_case_quality_score"]) else None,
        "recall_score": float(row["recall_score"]) if pd.notna(row["recall_score"]) else None,
        "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else None,
        "rebound_ratio": float(row["rebound_ratio"]) if pd.notna(row["rebound_ratio"]) else None,
        "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else None,
    }


def main() -> None:
    payload = {
        "code": "000703",
        "stock_name": "恒逸石化",
        "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
        "brick_filter": check_brick_filter(),
        "brickfilter_case_rank_lgbm_top20": check_case_rank(),
        "case_rank_runtime": case_mod.debug_summary(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
