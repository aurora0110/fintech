from __future__ import annotations

import json
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
OUTPUT_ROOT = ROOT / "results"
BACKTEST_START = pd.Timestamp("2024-04-02")
BACKTEST_END = pd.Timestamp("2026-04-02")
MAX_WORKERS = max(1, min((mp.cpu_count() or 4), 10))
HORIZONS = [1, 3, 5, 10]


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


b3filter = load_module(ROOT / "utils" / "B3filter.py", "b3_volcmp_b3")
technical = load_module(ROOT / "utils" / "technical_indicators.py", "b3_volcmp_technical")


@dataclass
class AnalysisConfig:
    output_dir: Path
    data_dir: Path
    max_workers: int


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_one_daily(path: Path) -> pd.DataFrame:
    df = technical._load_price_data(str(path))
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(
        columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    )[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = path.stem
    out = out[(out["date"] >= BACKTEST_START) & (out["date"] <= BACKTEST_END)].copy()
    return out.sort_values("date").reset_index(drop=True)


def signal_forward_return(df: pd.DataFrame, idx: int, horizon: int) -> float | None:
    entry_idx = idx + 1
    exit_idx = idx + horizon
    if entry_idx >= len(df) or exit_idx >= len(df):
        return None
    entry_price = float(df.iloc[entry_idx]["open"])
    exit_price = float(df.iloc[exit_idx]["close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return None
    return float(exit_price / entry_price - 1.0)


def analyze_one_file(file_path_str: str) -> list[dict[str, Any]]:
    path = Path(file_path_str)
    raw = load_one_daily(path)
    if raw.empty or len(raw) < 130:
        return []

    x = b3filter.add_features(raw)
    if x.empty:
        return []
    x = x.copy()
    base_signal = (
        x["prev_b2_any"].fillna(False)
        & x["bull_close"].fillna(False)
        & (pd.to_numeric(x["ret1"], errors="coerce") < b3filter.B3_RET1_MAX)
        & (pd.to_numeric(x["amplitude"], errors="coerce") < b3filter.B3_AMPLITUDE_MAX)
    )
    vol_vs_prev = pd.to_numeric(x["vol_vs_prev"], errors="coerce")
    x["signal_no_volume_req"] = base_signal
    x["signal_volume_expand"] = base_signal & (vol_vs_prev > 1.0)
    x["signal_volume_shrink"] = base_signal & (vol_vs_prev < 1.0)

    rows: list[dict[str, Any]] = []
    for idx, row in x.iterrows():
        for signal_name, volume_group in [
            ("signal_no_volume_req", "no_requirement"),
            ("signal_volume_expand", "volume_expand"),
            ("signal_volume_shrink", "volume_shrink"),
        ]:
            if not bool(row.get(signal_name, False)):
                continue
            item: dict[str, Any] = {
                "code": path.stem,
                "signal_date": pd.Timestamp(row["date"]),
                "volume_group": volume_group,
                "vol_vs_prev": float(row.get("vol_vs_prev", np.nan)),
                "ret1": float(row.get("ret1", np.nan)),
                "amplitude": float(row.get("amplitude", np.nan)),
                "b3_score": float(row.get("b3_score", np.nan)),
            }
            for h in HORIZONS:
                item[f"ret_h{h}"] = signal_forward_return(x, idx, h)
            rows.append(item)
    return rows


def pool_map(func, payloads: list[str], max_workers: int) -> list[Any]:
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        return pool.map(func, payloads, chunksize=1)


def summarize_group(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, g in df.groupby("volume_group", dropna=False):
        item = {"volume_group": group_name, "signal_count": int(len(g))}
        for h in HORIZONS:
            s = pd.to_numeric(g[f"ret_h{h}"], errors="coerce").dropna()
            item[f"avg_ret_h{h}"] = float(s.mean()) if not s.empty else float("nan")
            item[f"win_rate_h{h}"] = float((s > 0).mean()) if not s.empty else float("nan")
            item[f"median_ret_h{h}"] = float(s.median()) if not s.empty else float("nan")
        rows.append(item)
    order = {"no_requirement": 0, "volume_expand": 1, "volume_shrink": 2}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sort_key"] = out["volume_group"].map(order).fillna(999)
        out = out.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def run(cfg: AnalysisConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    update_progress(cfg.output_dir, "scanning_files", data_dir=str(cfg.data_dir), max_workers=cfg.max_workers)

    payloads = [str(p) for p in sorted(cfg.data_dir.glob("*.txt"))]
    parts = pool_map(analyze_one_file, payloads, cfg.max_workers)
    rows = [row for part in parts for row in part]
    signal_df = pd.DataFrame(rows)
    if signal_df.empty:
        raise RuntimeError("最近两年窗口内未找到任何 B3 基础信号")
    signal_df["signal_date"] = pd.to_datetime(signal_df["signal_date"])
    signal_df = signal_df.sort_values(["signal_date", "code", "volume_group"]).reset_index(drop=True)
    signal_df.to_csv(cfg.output_dir / "b3_volume_signal_detail.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize_group(signal_df)
    summary_df.to_csv(cfg.output_dir / "b3_volume_condition_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "data_dir": str(cfg.data_dir),
            "backtest_window": [str(BACKTEST_START.date()), str(BACKTEST_END.date())],
            "signal_definition": "current_B3filter_base_signal_without_or_with_volume_rule",
            "volume_groups": {
                "no_requirement": "base B3 structure, no signal-day volume rule",
                "volume_expand": "base B3 structure with signal-day volume > previous day",
                "volume_shrink": "base B3 structure with signal-day volume < previous day",
            },
            "return_definition": "next_day_open_buy_then_horizon_close_sell",
            "horizons": HORIZONS,
        },
        "total_rows": int(len(signal_df)),
        "signal_date_count": int(signal_df["signal_date"].nunique()),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(cfg.output_dir, "finished", total_rows=int(len(signal_df)))


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = AnalysisConfig(
        output_dir=OUTPUT_ROOT / f"b3_volume_condition_compare_v1_{ts}",
        data_dir=DATA_DIR,
        max_workers=MAX_WORKERS,
    )
    run(cfg)


if __name__ == "__main__":
    main()
