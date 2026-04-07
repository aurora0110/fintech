from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "20260402"
PERFECT_B3_DIR = ROOT / "data" / "完美图" / "B3"
NAME_CODE_MAP = ROOT / "results" / "b1_name_code_map_cache_20260315.json"
OUTPUT_ROOT = ROOT / "results"
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
MAX_WORKERS = max(1, min((mp.cpu_count() or 4), 10))
HORIZONS = [1, 3, 5, 10]


def load_module(path: Path, module_name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


technical = load_module(ROOT / "utils" / "technical_indicators.py", "b3_brick_analysis_technical")
b3filter = load_module(ROOT / "utils" / "B3filter.py", "b3_brick_analysis_b3")
brick_filter = load_module(ROOT / "utils" / "brick_filter.py", "b3_brick_analysis_brick")


@dataclass
class AnalysisConfig:
    output_dir: Path
    data_dir: Path
    max_workers: int


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_one_daily(path: Path) -> pd.DataFrame:
    df = technical._load_price_data(str(path))
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(
        columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    )[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = path.stem
    return out.sort_values("date").reset_index(drop=True)


def _allowed_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["date"] < EXCLUDE_START) | (df["date"] > EXCLUDE_END)].copy()


def _signal_forward_return(df: pd.DataFrame, idx: int, horizon: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + horizon
    if entry_idx >= len(df) or exit_idx >= len(df):
        return None
    entry_price = float(df.iloc[entry_idx]["open"])
    exit_price = float(df.iloc[exit_idx]["close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return None
    return float(exit_price / entry_price - 1.0)


def _analyze_one_file(file_path_str: str) -> list[dict[str, Any]]:
    path = Path(file_path_str)
    raw = _load_one_daily(path)
    raw = _allowed_dates(raw)
    if raw.empty or len(raw) < 130:
        return []

    b3x = b3filter.add_features(raw)
    brickx = brick_filter.add_features(raw)
    x = b3x.merge(
        brickx[["date", "brick_red", "brick_green", "prev_green_streak"]],
        on="date",
        how="left",
        suffixes=("", "_brick"),
    )
    if x.empty:
        return []

    x["brick_red"] = x["brick_red"].fillna(False).astype(bool)
    x["brick_green"] = x["brick_green"].fillna(False).astype(bool)
    x["prev_green_streak"] = pd.to_numeric(x["prev_green_streak"], errors="coerce").fillna(0.0)

    prev_red = x["brick_red"].shift(1).fillna(False).astype(bool)
    prev_green_streak_before_first_red = x["prev_green_streak"].shift(1).fillna(0.0)
    x["second_red_after_green_ge2"] = x["brick_red"] & prev_red & (prev_green_streak_before_first_red >= 2)
    x["second_red_after_green_ge3"] = x["brick_red"] & prev_red & (prev_green_streak_before_first_red >= 3)
    x["second_red_after_green_ge4"] = x["brick_red"] & prev_red & (prev_green_streak_before_first_red >= 4)

    rows: list[dict[str, Any]] = []
    for idx, row in x.iterrows():
        if not bool(row.get("b3_signal", False)):
            continue
        item: dict[str, Any] = {
            "code": path.stem,
            "signal_date": pd.Timestamp(row["date"]),
            "second_red_after_green_ge2": bool(row["second_red_after_green_ge2"]),
            "second_red_after_green_ge3": bool(row["second_red_after_green_ge3"]),
            "second_red_after_green_ge4": bool(row["second_red_after_green_ge4"]),
            "prev_green_streak_before_first_red": float(prev_green_streak_before_first_red.iloc[idx]) if idx < len(prev_green_streak_before_first_red) else float("nan"),
            "brick_red_today": bool(row["brick_red"]),
            "brick_red_prev": bool(prev_red.iloc[idx]),
            "trend_line": float(row.get("trend_line", np.nan)),
            "long_line": float(row.get("long_line", np.nan)),
            "J": float(row.get("J", np.nan)),
            "ret1": float(row.get("ret1", np.nan)),
            "amplitude": float(row.get("amplitude", np.nan)),
            "vol_vs_prev": float(row.get("vol_vs_prev", np.nan)),
        }
        for h in HORIZONS:
            item[f"ret_h{h}"] = _signal_forward_return(x, idx, h)
        rows.append(item)
    return rows


def _pool_map(func, payloads: list[Any], max_workers: int) -> list[Any]:
    if not payloads:
        return []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(func, payloads, chunksize=1)


def _summarize_condition(df: pd.DataFrame, cond_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flag_value, g in df.groupby(cond_col, dropna=False):
        item = {
            "condition": cond_col,
            "flag": bool(flag_value),
            "signal_count": int(len(g)),
        }
        for h in HORIZONS:
            s = pd.to_numeric(g[f"ret_h{h}"], errors="coerce").dropna()
            item[f"avg_ret_h{h}"] = float(s.mean()) if not s.empty else float("nan")
            item[f"win_rate_h{h}"] = float((s > 0).mean()) if not s.empty else float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def _load_perfect_b3_name_map() -> dict[str, str]:
    obj = json.loads(NAME_CODE_MAP.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for image_path in PERFECT_B3_DIR.glob("*.png"):
        name = image_path.stem
        code = obj.get(name)
        if code:
            out[name] = code
    return out


def run(cfg: AnalysisConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(cfg.output_dir, "scanning_files", data_dir=str(cfg.data_dir), max_workers=cfg.max_workers)

    file_paths = sorted(cfg.data_dir.glob("*.txt"))
    payloads = [str(p) for p in file_paths]
    parts = _pool_map(_analyze_one_file, payloads, cfg.max_workers)
    rows = [row for part in parts for row in part]
    signal_df = pd.DataFrame(rows)
    if signal_df.empty:
        raise RuntimeError("未找到任何 B3 信号")
    signal_df["signal_date"] = pd.to_datetime(signal_df["signal_date"])
    signal_df = signal_df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    signal_df.to_csv(cfg.output_dir / "b3_signal_detail.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for cond_col in ["second_red_after_green_ge2", "second_red_after_green_ge3", "second_red_after_green_ge4"]:
        summaries.append(_summarize_condition(signal_df, cond_col))
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(cfg.output_dir / "condition_return_summary.csv", index=False, encoding="utf-8-sig")

    perfect_map = _load_perfect_b3_name_map()
    perfect_codes = set(perfect_map.values())
    perfect_signal_df = signal_df[signal_df["code"].isin(perfect_codes)].copy()
    perfect_signal_df.to_csv(cfg.output_dir / "perfect_b3_named_stock_signals.csv", index=False, encoding="utf-8-sig")

    perfect_stock_summary = []
    for name, code in sorted(perfect_map.items()):
        g = perfect_signal_df[perfect_signal_df["code"] == code].copy()
        perfect_stock_summary.append(
            {
                "stock_name": name,
                "code": code,
                "signal_count": int(len(g)),
                "ge2_count": int(g["second_red_after_green_ge2"].sum()) if not g.empty else 0,
                "ge3_count": int(g["second_red_after_green_ge3"].sum()) if not g.empty else 0,
                "ge4_count": int(g["second_red_after_green_ge4"].sum()) if not g.empty else 0,
                "avg_ret_h3": float(pd.to_numeric(g["ret_h3"], errors="coerce").mean()) if not g.empty else float("nan"),
                "avg_ret_h5": float(pd.to_numeric(g["ret_h5"], errors="coerce").mean()) if not g.empty else float("nan"),
            }
        )
    perfect_stock_summary_df = pd.DataFrame(perfect_stock_summary)
    perfect_stock_summary_df.to_csv(cfg.output_dir / "perfect_b3_named_stock_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "data_dir": str(cfg.data_dir),
            "exclude_window": [str(EXCLUDE_START.date()), str(EXCLUDE_END.date())],
            "signal_definition": "current_B3filter",
            "structure_definition": {
                "ge2": "B3 signal on second red brick, and the first red brick is preceded by at least 2 consecutive green bricks",
                "ge3": "same, but green streak >= 3",
                "ge4": "same, but green streak >= 4",
            },
            "return_definition": "next_day_open_buy_then_horizon_close_sell",
            "horizons": HORIZONS,
        },
        "total_b3_signals": int(len(signal_df)),
        "signal_date_count": int(signal_df["signal_date"].nunique()),
        "perfect_b3_named_stock_count": int(len(perfect_map)),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(cfg.output_dir, "finished", total_b3_signals=int(len(signal_df)))


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"b3_second_red_brick_analysis_v1_{ts}"
    cfg = AnalysisConfig(output_dir=output_dir, data_dir=DATA_DIR, max_workers=MAX_WORKERS)
    run(cfg)


if __name__ == "__main__":
    main()
