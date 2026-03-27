from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
DAILY_DIR = ROOT / "data" / "20260324"
SIGNAL_SOURCE = RESULT_ROOT / "brick_minute_execution_compare_v1_full_day5_parallel_20260325_r4" / "selected_signals.csv"
CHAMPION_RESULT_DIR = RESULT_ROOT / "brick_hybrid_local_search_minoc_full_20260326_r2"
BASE_SCRIPT_PATH = ROOT / "utils" / "tmp" / "run_brick_intraday_minute_compare_v1_20260325.py"
HYBRID_SCRIPT_PATH = ROOT / "utils" / "tmp" / "run_brick_hybrid_local_search_v1_20260325.py"
CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RAW_CASE_DIR = ROOT / "data" / "20260312"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import compute_metrics
from core.data_loader import _read_txt

INITIAL_CAPITAL = 1_000_000.0
MAX_POSITIONS = 10
DAILY_NEW_LIMIT = 10
DAILY_BUDGET_FRAC = 1.0
POSITION_CAP_FRAC = 0.10
ALLOCATION_MODE = "equal"
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.001
STAMP_DUTY_RATE = 0.001
MIN_LOT = 100
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 8))
CONTINUATION_LOOKAHEAD = 3
CONTINUATION_EXTRA_RET = 0.02


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT_PATH, "brick_strong_tail_compare_base_v1")
hybrid = load_module(HYBRID_SCRIPT_PATH, "brick_strong_tail_compare_hybrid_v1")


@dataclass(frozen=True)
class AccountConfig:
    initial_capital: float = INITIAL_CAPITAL
    max_positions: int = MAX_POSITIONS
    daily_new_limit: int = DAILY_NEW_LIMIT
    daily_budget_frac: float = DAILY_BUDGET_FRAC
    position_cap_frac: float = POSITION_CAP_FRAC
    allocation_mode: str = ALLOCATION_MODE
    commission_rate: float = COMMISSION_RATE
    slippage_rate: float = SLIPPAGE_RATE
    stamp_duty_rate: float = STAMP_DUTY_RATE
    min_lot: int = MIN_LOT


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def load_champion_trades(file_limit: int) -> pd.DataFrame:
    summary = json.loads((CHAMPION_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    strategy_key = str(summary["strategy_best"])
    trades = pd.read_csv(
        CHAMPION_RESULT_DIR / "hybrid_local_trades.csv",
        parse_dates=["signal_date", "entry_date", "exit_date"],
    )
    trades = trades[trades["strategy_key"] == strategy_key].copy()
    if trades.empty:
        raise RuntimeError(f"未找到当前冠军交易: {strategy_key}")
    trades = trades.sort_values(["signal_date", "code", "signal_idx"]).reset_index(drop=True)
    if file_limit > 0:
        keep_codes = sorted(trades["code"].astype(str).unique())[:file_limit]
        trades = trades[trades["code"].astype(str).isin(keep_codes)].copy()
    return trades.reset_index(drop=True)


def load_signal_scores(file_limit: int) -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_SOURCE, parse_dates=["signal_date", "entry_date"])
    if file_limit > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    df["sort_score"] = pd.to_numeric(df.get("base_score", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).copy()
    df["daily_rank"] = df.groupby("signal_date").cumcount() + 1
    df["daily_count"] = df.groupby("signal_date")["code"].transform("count")
    df["top30_flag"] = df["daily_rank"] <= np.maximum(1, np.ceil(df["daily_count"] * 0.3))
    return df[
        ["code", "signal_idx", "signal_date", "entry_date", "sort_score", "base_score", "daily_rank", "daily_count", "top30_flag"]
    ].drop_duplicates(["code", "signal_idx", "signal_date", "entry_date"])


def attach_signal_meta(trades: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    out = trades.merge(score_df, on=["code", "signal_idx", "signal_date", "entry_date"], how="left")
    out["sort_score"] = pd.to_numeric(out["sort_score"], errors="coerce").fillna(0.0)
    out["daily_rank"] = pd.to_numeric(out["daily_rank"], errors="coerce")
    out["daily_count"] = pd.to_numeric(out["daily_count"], errors="coerce")
    out["top30_flag"] = out["top30_flag"].fillna(False).astype(bool)
    return out


def _build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    feat = base.brick_base.build_feature_df(raw.copy())
    feat = base.brick_ranking.add_long_line(feat)
    full_range = (feat["high"] - feat["low"]).replace(0, np.nan)
    feat["close_position"] = (feat["close"] - feat["low"]) / full_range
    feat["upper_shadow_ratio"] = (feat["high"] - np.maximum(feat["open"], feat["close"])) / full_range
    feat["trend_spread_clip"] = ((feat["trend_line"] - feat["long_line"]) / feat["close"]).clip(lower=-1.0, upper=1.0)
    feat["trend_slope_5"] = feat["trend_line"] / feat["trend_line"].shift(5) - 1.0
    feat["dist_20d_high"] = feat["close"] / feat["high"].rolling(20).max() - 1.0
    return feat


def _load_feature_subset_for_code(code: str) -> tuple[str, pd.DataFrame | None]:
    path = DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return code, None
    raw = _read_txt(str(path))
    if raw is None or raw.empty:
        return code, None
    daily_df = raw[(raw["date"] < base.EXCLUDE_START) | (raw["date"] > base.EXCLUDE_END)].copy()
    daily_df = daily_df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "open", "high", "low", "close"])
    if daily_df.empty:
        return code, None
    feat = _build_feature_frame(daily_df)
    keep = feat[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "trend_line",
            "long_line",
            "brick_green",
            "close_position",
            "upper_shadow_ratio",
            "trend_spread_clip",
            "trend_slope_5",
            "dist_20d_high",
            "signal_vs_ma5",
            "rebound_ratio",
        ]
    ].copy()
    keep["brick_green"] = keep["brick_green"].fillna(False).astype(bool)
    return code, keep


def build_feature_registry(codes: list[str], max_workers: int, progress_cb: Any | None = None) -> dict[str, pd.DataFrame]:
    codes = sorted(set(codes))
    out: dict[str, pd.DataFrame] = {}
    total = len(codes)
    if total == 0:
        return out
    if max_workers <= 1:
        for idx, code in enumerate(codes, start=1):
            _, feat = _load_feature_subset_for_code(code)
            if feat is not None and not feat.empty:
                out[code] = feat
            if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
                progress_cb(idx, total)
        return out
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_feature_subset_for_code, code): code for code in codes}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                code, feat = future.result()
                if feat is not None and not feat.empty:
                    out[code] = feat
                if progress_cb is not None and (completed == 1 or completed % 100 == 0 or completed == total):
                    progress_cb(completed, total)
    except Exception:
        out = {}
        for idx, code in enumerate(codes, start=1):
            _, feat = _load_feature_subset_for_code(code)
            if feat is not None and not feat.empty:
                out[code] = feat
            if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
                progress_cb(idx, total, fallback="serial")
    return out


def infer_exit_session(exit_reason: str) -> str:
    reason = str(exit_reason)
    if "next_open" in reason:
        return "open"
    if "same_day_hybrid" in reason or "same_day" in reason:
        return "intraday"
    if "close" in reason:
        return "close"
    return "close"


def previous_trading_day(code: str, ref_date: pd.Timestamp, feature_registry: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    feat = feature_registry.get(code)
    if feat is None or feat.empty:
        return None
    dates = feat.loc[feat["date"] < ref_date, "date"]
    if dates.empty:
        return None
    return pd.Timestamp(dates.iloc[-1])


def get_feature_row(code: str, day: pd.Timestamp, feature_registry: dict[str, pd.DataFrame]) -> pd.Series | None:
    feat = feature_registry.get(code)
    if feat is None or feat.empty:
        return None
    m = feat.loc[feat["date"] == day]
    if m.empty:
        return None
    return m.iloc[-1]


def parse_case_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    for path in sorted(CASE_DIR.glob("*.png")):
        m = pat.match(path.name)
        if not m:
            continue
        rows.append({"stock_name": m.group(1), "signal_date": pd.to_datetime(m.group(2), format="%Y%m%d"), "case_file": str(path)})
    return pd.DataFrame(rows)


def build_name_code_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    valid_codes = {p.stem for p in DAILY_DIR.glob("*.txt")}
    for path in RAW_CASE_DIR.glob("*.txt"):
        try:
            first_line = path.read_text(encoding="gbk", errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit() and path.stem in valid_codes:
            mapping[parts[1]] = path.stem
    return mapping


def load_case_daily(code: str) -> pd.DataFrame | None:
    path = DAILY_DIR / f"{code}.txt"
    if not path.exists():
        return None
    raw = _read_txt(str(path))
    if raw is None or raw.empty:
        return None
    raw = raw[(raw["date"] < base.EXCLUDE_START) | (raw["date"] > base.EXCLUDE_END)].copy()
    raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "open", "high", "low", "close"])
    if raw.empty:
        return None
    return raw


def load_min5_fast(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["date_text", "time", "open", "high", "low", "close", "volume", "amount"],
            encoding="gbk",
            engine="c",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=["date_text", "time", "open", "high", "low", "close", "volume", "amount"],
            encoding="utf-8",
            engine="c",
        )
    if df.empty:
        return pd.DataFrame(columns=["datetime", "date", "time", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date_text"], format="%Y/%m/%d", errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce").round().astype("Int64").astype(str).str.replace("<NA>", "", regex=False).str.zfill(4)
    df["datetime"] = pd.to_datetime(
        df["date_text"].astype(str) + " " + df["time"],
        format="%Y/%m/%d %H%M",
        errors="coerce",
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "date", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "date", "time", "open", "high", "low", "close", "volume"]]


def load_case_min5(code: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "202603245min" / f"{code}.txt"
    if not path.exists():
        return None
    df = load_min5_fast(path)
    return None if df.empty else df


def analyze_perfect_cases(result_dir: Path, progress_cb: Any | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = parse_case_files()
    name_code = build_name_code_map()
    if cases.empty:
        return pd.DataFrame(), {"usable_case_count": 0}

    rows: list[dict[str, Any]] = []
    total_cases = len(cases)
    for idx, case in enumerate(cases.itertuples(index=False), start=1):
        code = name_code.get(str(case.stock_name))
        if not code:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total_cases):
                progress_cb(idx, total_cases, usable_cases=len(rows), skipped_reason="missing_code")
            continue
        daily = load_case_daily(code)
        if daily is None or daily.empty:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total_cases):
                progress_cb(idx, total_cases, usable_cases=len(rows), skipped_reason="missing_daily")
            continue
        feat = _build_feature_frame(daily)
        match_idx = feat.index[feat["date"] == pd.Timestamp(case.signal_date)].tolist()
        if not match_idx:
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total_cases):
                progress_cb(idx, total_cases, usable_cases=len(rows), skipped_reason="missing_signal_date")
            continue
        signal_idx = int(match_idx[-1])
        entry_idx = signal_idx + 1
        if entry_idx >= len(feat):
            if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total_cases):
                progress_cb(idx, total_cases, usable_cases=len(rows), skipped_reason="missing_entry_date")
            continue
        min5 = load_case_min5(code)
        sim = hybrid.simulate_one_trade(
            code=code,
            signal_date=pd.Timestamp(case.signal_date),
            entry_date=pd.Timestamp(feat.at[entry_idx, "date"]),
            signal_idx=signal_idx,
            daily_df=feat[["date", "open", "high", "low", "close", "volume"]],
            min5_df=min5,
            tp_pct=0.055,
            stop_multiplier=1.0,
            stop_base="min_oc",
        )
        if sim.trade is None:
            continue
        trade = sim.trade
        exit_date = pd.Timestamp(trade["exit_date"])
        later = feat.loc[feat["date"] > exit_date].head(CONTINUATION_LOOKAHEAD)
        future_max = float(later["high"].max()) if not later.empty else np.nan
        continue_up = bool(np.isfinite(future_max) and future_max / float(trade["exit_price"]) - 1.0 >= CONTINUATION_EXTRA_RET)
        sig_row = feat.iloc[signal_idx]
        rows.append(
            {
                "stock_name": case.stock_name,
                "code": code,
                "signal_date": pd.Timestamp(case.signal_date),
                "strategy_exit_reason": str(trade["exit_reason"]),
                "hit_tp": str(trade["exit_reason"]).startswith("tp_"),
                "continue_up_after_tp": bool(str(trade["exit_reason"]).startswith("tp_") and continue_up),
                "close_position": float(sig_row["close_position"]),
                "upper_shadow_ratio": float(sig_row["upper_shadow_ratio"]),
                "trend_spread_clip": float(sig_row["trend_spread_clip"]),
                "trend_slope_5": float(sig_row["trend_slope_5"]) if pd.notna(sig_row["trend_slope_5"]) else np.nan,
                "dist_20d_high": float(sig_row["dist_20d_high"]) if pd.notna(sig_row["dist_20d_high"]) else np.nan,
                "signal_vs_ma5": float(sig_row["signal_vs_ma5"]) if pd.notna(sig_row["signal_vs_ma5"]) else np.nan,
                "rebound_ratio": float(sig_row["rebound_ratio"]) if pd.notna(sig_row["rebound_ratio"]) else np.nan,
            }
        )
        if progress_cb is not None and (idx == 1 or idx % 10 == 0 or idx == total_cases):
            progress_cb(idx, total_cases, usable_cases=len(rows))

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        return case_df, {"usable_case_count": 0}

    case_df.to_csv(result_dir / "perfect_case_continuation_analysis.csv", index=False, encoding="utf-8-sig")
    tp_df = case_df[case_df["hit_tp"]].copy()
    pos = tp_df[tp_df["continue_up_after_tp"]].copy()
    neg = tp_df[~tp_df["continue_up_after_tp"]].copy()
    stats = {
        "usable_case_count": int(len(case_df)),
        "tp_case_count": int(len(tp_df)),
        "continue_up_case_count": int(len(pos)),
        "non_continue_case_count": int(len(neg)),
    }
    return case_df, stats


def derive_case_rule(case_df: pd.DataFrame) -> dict[str, Any]:
    tp_df = case_df[case_df["hit_tp"]].copy()
    pos = tp_df[tp_df["continue_up_after_tp"]].copy()
    neg = tp_df[~tp_df["continue_up_after_tp"]].copy()
    candidate_cols = [
        "close_position",
        "upper_shadow_ratio",
        "trend_spread_clip",
        "trend_slope_5",
        "dist_20d_high",
        "signal_vs_ma5",
        "rebound_ratio",
    ]
    if len(pos) < 3 or len(neg) < 3:
        return {
            "type": "fallback",
            "features": [
                {"column": "close_position", "direction": ">=", "threshold": 0.80},
                {"column": "upper_shadow_ratio", "direction": "<=", "threshold": 0.20},
                {"column": "trend_spread_clip", "direction": ">=", "threshold": 0.045},
                {"column": "trend_slope_5", "direction": ">=", "threshold": 0.005},
            ],
            "min_pass_count": 3,
            "source": {"positive_count": int(len(pos)), "negative_count": int(len(neg))},
        }

    feature_rules: list[dict[str, Any]] = []
    scored: list[tuple[float, dict[str, Any]]] = []
    for col in candidate_cols:
        pos_med = float(pos[col].median())
        neg_med = float(neg[col].median())
        if not np.isfinite(pos_med) or not np.isfinite(neg_med):
            continue
        direction = ">=" if pos_med >= neg_med else "<="
        threshold = (pos_med + neg_med) / 2.0
        separation = abs(pos_med - neg_med)
        scored.append(
            (
                separation,
                {
                    "column": col,
                    "direction": direction,
                    "threshold": float(threshold),
                    "positive_median": pos_med,
                    "negative_median": neg_med,
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    feature_rules = [rule for _, rule in scored[:4]]
    return {
        "type": "derived_from_perfect_cases",
        "features": feature_rules,
        "min_pass_count": max(2, min(3, len(feature_rules))),
        "source": {"positive_count": int(len(pos)), "negative_count": int(len(neg))},
    }


def is_rule_strong(row: pd.Series, trigger_row: pd.Series | None) -> bool:
    if trigger_row is None:
        return False
    conditions = [
        bool(pd.notna(trigger_row["close"]) and pd.notna(trigger_row["trend_line"]) and float(trigger_row["close"]) > float(trigger_row["trend_line"])),
        bool(not bool(trigger_row["brick_green"])),
        bool(pd.notna(trigger_row["close_position"]) and float(trigger_row["close_position"]) >= 0.75),
        bool(pd.notna(trigger_row["upper_shadow_ratio"]) and float(trigger_row["upper_shadow_ratio"]) <= 0.25),
        bool(row.get("top30_flag", False)),
    ]
    return sum(conditions) >= 4


def is_case_strong(signal_row: pd.Series | None, case_rule: dict[str, Any]) -> bool:
    if signal_row is None:
        return False
    pass_count = 0
    for rule in case_rule.get("features", []):
        col = str(rule["column"])
        val = signal_row.get(col)
        if not pd.notna(val):
            continue
        if rule["direction"] == ">=" and float(val) >= float(rule["threshold"]):
            pass_count += 1
        elif rule["direction"] == "<=" and float(val) <= float(rule["threshold"]):
            pass_count += 1
    return pass_count >= int(case_rule.get("min_pass_count", 1))


def find_green_exit(code: str, partial_exit_date: pd.Timestamp, partial_session: str, feature_registry: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, float, str]:
    feat = feature_registry.get(code)
    if feat is None or feat.empty:
        raise RuntimeError(f"缺少 {code} 的特征表，无法搜索绿砖退出")
    if partial_session == "open":
        mask = feat["date"] >= partial_exit_date
    else:
        mask = feat["date"] > partial_exit_date
    future = feat[mask].copy()
    if future.empty:
        last = feat.iloc[-1]
        return pd.Timestamp(last["date"]), float(last["close"]), "end_of_data_close_after_partial"
    green = future[future["brick_green"]]
    if not green.empty:
        row = green.iloc[0]
        return pd.Timestamp(row["date"]), float(row["close"]), "green_close_after_partial"
    last = future.iloc[-1]
    return pd.Timestamp(last["date"]), float(last["close"]), "end_of_window_close_after_partial"


def build_strategy_events(trades: pd.DataFrame, feature_registry: dict[str, pd.DataFrame], case_rule: dict[str, Any]) -> dict[str, pd.DataFrame]:
    labels = ["champion_full_exit", "partial_all", "partial_rule_strong", "partial_case_strong"]
    rows_map: dict[str, list[dict[str, Any]]] = {k: [] for k in labels}

    for row in trades.itertuples(index=False):
        position_id = f"{row.code}|{int(row.signal_idx)}|{pd.Timestamp(row.entry_date).date()}"
        base_payload = {
            "position_id": position_id,
            "code": row.code,
            "signal_idx": int(row.signal_idx),
            "signal_date": pd.Timestamp(row.signal_date),
            "entry_date": pd.Timestamp(row.entry_date),
            "entry_price": float(row.entry_price),
            "sort_score": float(row.sort_score),
            "strategy_key": str(row.strategy_key),
        }
        full_exit_event = {
            **base_payload,
            "leg_name": "full_exit",
            "exit_date": pd.Timestamp(row.exit_date),
            "exit_session": infer_exit_session(str(row.exit_reason)),
            "exit_price": float(row.exit_price),
            "exit_ratio": 1.0,
            "exit_reason": str(row.exit_reason),
        }
        rows_map["champion_full_exit"].append({"strategy_label": "champion_full_exit", **full_exit_event})

        reason = str(row.exit_reason)
        if not reason.startswith("tp_"):
            for label in ["partial_all", "partial_rule_strong", "partial_case_strong"]:
                rows_map[label].append({"strategy_label": label, **full_exit_event})
            continue

        partial_session = infer_exit_session(reason)
        partial_date = pd.Timestamp(row.exit_date)
        partial_price = float(row.exit_price)
        signal_feature = get_feature_row(str(row.code), pd.Timestamp(row.signal_date), feature_registry)
        trigger_date = previous_trading_day(str(row.code), partial_date, feature_registry) if partial_session == "open" else partial_date
        trigger_feature = get_feature_row(str(row.code), trigger_date, feature_registry) if trigger_date is not None else None
        rule_strong = is_rule_strong(pd.Series(row._asdict()), trigger_feature)
        case_strong = is_case_strong(signal_feature, case_rule)

        green_date, green_price, green_reason = find_green_exit(
            code=str(row.code),
            partial_exit_date=partial_date,
            partial_session=partial_session,
            feature_registry=feature_registry,
        )

        partial_80 = {
            **base_payload,
            "leg_name": "tp_partial_80",
            "exit_date": partial_date,
            "exit_session": partial_session,
            "exit_price": partial_price,
            "exit_ratio": 0.8,
            "exit_reason": f"{reason}_80pct",
        }
        tail_20 = {
            **base_payload,
            "leg_name": "green_tail_20",
            "exit_date": green_date,
            "exit_session": "close",
            "exit_price": green_price,
            "exit_ratio": 0.2,
            "exit_reason": green_reason,
        }
        rows_map["partial_all"].extend(
            [
                {"strategy_label": "partial_all", **partial_80},
                {"strategy_label": "partial_all", **tail_20},
            ]
        )
        if rule_strong:
            rows_map["partial_rule_strong"].extend(
                [
                    {"strategy_label": "partial_rule_strong", **partial_80},
                    {"strategy_label": "partial_rule_strong", **tail_20},
                ]
            )
        else:
            rows_map["partial_rule_strong"].append({"strategy_label": "partial_rule_strong", **full_exit_event})
        if case_strong:
            rows_map["partial_case_strong"].extend(
                [
                    {"strategy_label": "partial_case_strong", **partial_80},
                    {"strategy_label": "partial_case_strong", **tail_20},
                ]
            )
        else:
            rows_map["partial_case_strong"].append({"strategy_label": "partial_case_strong", **full_exit_event})

    out: dict[str, pd.DataFrame] = {}
    for label, rows in rows_map.items():
        out[label] = pd.DataFrame(rows).sort_values(["entry_date", "code", "signal_idx", "exit_date", "leg_name"]).reset_index(drop=True)
    return out


def _fast_load_close_series(path: Path) -> pd.Series | None:
    df = _read_txt(str(path))
    if df is None or df.empty:
        return None
    df = df[(df["date"] < base.EXCLUDE_START) | (df["date"] > base.EXCLUDE_END)].copy()
    if df.empty:
        return None
    return df[["date", "close"]].dropna(subset=["date", "close"]).set_index("date")["close"].astype(float)


def build_close_map(codes: list[str], progress_cb: Any | None = None) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    relevant: dict[str, pd.Series] = {}
    all_dates: set[pd.Timestamp] = set()
    unique_codes = sorted(set(codes))
    total = len(unique_codes)
    for idx, code in enumerate(unique_codes, start=1):
        path = DAILY_DIR / f"{code}.txt"
        if not path.exists():
            continue
        s = _fast_load_close_series(path)
        if s is None or s.empty:
            continue
        relevant[code] = s
        all_dates.update(s.index.tolist())
        if progress_cb is not None and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_cb(idx, total)
    market_dates = pd.DatetimeIndex(sorted(all_dates))
    close_map: dict[str, pd.Series] = {}
    for code, s in relevant.items():
        close_map[code] = s.reindex(market_dates).ffill()
    return market_dates, close_map


def _compute_calmar(annual_return: float, max_drawdown_abs: float) -> float:
    if not np.isfinite(annual_return):
        return float("nan")
    if not np.isfinite(max_drawdown_abs) or max_drawdown_abs <= 0:
        return float("nan")
    return float(annual_return / max_drawdown_abs)


def _max_losing_streak(returns: list[float]) -> int:
    streak = 0
    max_streak = 0
    for r in returns:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def simulate_real_account_event_based(
    events: pd.DataFrame,
    close_map: dict[str, pd.Series],
    market_dates: pd.DatetimeIndex,
    config: AccountConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if events.empty:
        raise RuntimeError("事件表为空，无法回测")

    plans = events[
        ["position_id", "code", "signal_idx", "signal_date", "entry_date", "entry_price", "sort_score", "strategy_label", "strategy_key"]
    ].drop_duplicates(["position_id"]).copy()
    plans = plans.sort_values(["entry_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    entries_by_date = {
        d: g.sort_values(["sort_score", "code"], ascending=[False, True]).to_dict("records")
        for d, g in plans.groupby("entry_date")
    }
    exits_by_date_session: dict[tuple[pd.Timestamp, str], list[dict[str, Any]]] = {}
    for rec in events.sort_values(["exit_date", "exit_session", "code"]).to_dict("records"):
        key = (pd.Timestamp(rec["exit_date"]), str(rec["exit_session"]))
        exits_by_date_session.setdefault(key, []).append(rec)

    cash = float(config.initial_capital)
    positions: dict[str, dict[str, Any]] = {}
    executed_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    def process_exit_event(current_date: pd.Timestamp, session: str) -> None:
        nonlocal cash
        for ev in exits_by_date_session.get((current_date, session), []):
            position_id = str(ev["position_id"])
            if position_id not in positions:
                continue
            pos = positions[position_id]
            total_shares = int(pos["shares_total"])
            remaining_shares = int(pos["shares_remaining"])
            if remaining_shares <= 0:
                positions.pop(position_id, None)
                continue

            if str(ev["leg_name"]) == "tp_partial_80":
                target = int((total_shares * 0.8) // config.min_lot) * config.min_lot
                if target <= 0 or target >= total_shares:
                    shares_to_sell = remaining_shares
                else:
                    shares_to_sell = min(target, remaining_shares)
            else:
                shares_to_sell = remaining_shares

            raw_exit_price = float(ev["exit_price"])
            exit_price = raw_exit_price * (1.0 - config.slippage_rate)
            gross_cash = shares_to_sell * exit_price
            fee = gross_cash * config.commission_rate
            tax = gross_cash * config.stamp_duty_rate
            cash += gross_cash - fee - tax

            entry_fee_alloc = pos["entry_fee_total"] * (shares_to_sell / total_shares)
            pnl = (exit_price - pos["entry_price_exec"]) * shares_to_sell - entry_fee_alloc - fee - tax
            cost_base = pos["entry_price_exec"] * shares_to_sell + entry_fee_alloc
            realized_return = pnl / cost_base if cost_base > 0 else float("nan")
            pos["shares_remaining"] = remaining_shares - shares_to_sell

            executed_rows.append(
                {
                    "strategy_label": pos["strategy_label"],
                    "strategy_key": pos["strategy_key"],
                    "position_id": position_id,
                    "code": pos["code"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "exit_session": session,
                    "leg_name": ev["leg_name"],
                    "entry_price_raw": pos["entry_price_raw"],
                    "entry_price_exec": pos["entry_price_exec"],
                    "exit_price_raw": raw_exit_price,
                    "exit_price_exec": exit_price,
                    "shares_sold": shares_to_sell,
                    "shares_remaining_after": pos["shares_remaining"],
                    "gross_entry_cost_alloc": pos["entry_price_exec"] * shares_to_sell,
                    "entry_fee_alloc": entry_fee_alloc,
                    "exit_fee_tax": fee + tax,
                    "pnl": pnl,
                    "return_pct_net": realized_return,
                    "exit_reason": ev["exit_reason"],
                    "sort_score": pos["sort_score"],
                }
            )
            if pos["shares_remaining"] <= 0:
                positions.pop(position_id, None)

    for current_date in market_dates:
        process_exit_event(current_date, "open")

        equity_before_entry = cash
        for _, pos in positions.items():
            mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price_exec"]))
            equity_before_entry += pos["shares_remaining"] * mark_price

        entry_candidates = entries_by_date.get(current_date, [])
        available_slots = max(config.max_positions - len(positions), 0)
        if entry_candidates and available_slots > 0:
            to_add: list[dict[str, Any]] = []
            for rec in entry_candidates:
                position_id = str(rec["position_id"])
                if position_id in positions:
                    continue
                to_add.append(rec)
                if len(to_add) >= min(available_slots, config.daily_new_limit):
                    break
            if to_add:
                investable = min(cash, equity_before_entry * config.daily_budget_frac)
                if investable > 0:
                    weights = np.full(len(to_add), 1.0 / len(to_add), dtype=float)
                    per_pos_cap = equity_before_entry * config.position_cap_frac
                    for rec, weight in zip(to_add, weights):
                        raw_entry_price = float(rec["entry_price"])
                        entry_price = raw_entry_price * (1.0 + config.slippage_rate)
                        alloc = min(investable * float(weight), per_pos_cap, cash)
                        if alloc <= 0 or entry_price <= 0:
                            continue
                        shares = int(alloc / entry_price / config.min_lot) * config.min_lot
                        if shares <= 0:
                            continue
                        gross_cost = shares * entry_price
                        fee = gross_cost * config.commission_rate
                        total_cost = gross_cost + fee
                        if total_cost > cash:
                            continue
                        cash -= total_cost
                        positions[str(rec["position_id"])] = {
                            "code": str(rec["code"]),
                            "signal_date": pd.Timestamp(rec["signal_date"]),
                            "entry_date": current_date,
                            "shares_total": shares,
                            "shares_remaining": shares,
                            "entry_price_raw": raw_entry_price,
                            "entry_price_exec": entry_price,
                            "entry_fee_total": fee,
                            "sort_score": float(rec["sort_score"]),
                            "strategy_label": str(rec["strategy_label"]),
                            "strategy_key": str(rec["strategy_key"]),
                        }

        process_exit_event(current_date, "intraday")
        process_exit_event(current_date, "close")

        equity = cash
        for _, pos in positions.items():
            mark_price = float(close_map[pos["code"]].get(current_date, pos["entry_price_exec"]))
            equity += pos["shares_remaining"] * mark_price
        equity_rows.append({"date": current_date, "equity": equity, "cash": cash, "position_count": len(positions)})

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_curve = pd.Series(equity_df["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(equity_df["date"]), dtype=float)
    metrics = compute_metrics(equity_curve)
    max_drawdown_abs = float(metrics["max_drawdown"])
    annual_return = float(metrics["annual_return"])
    sharpe = float(metrics["sharpe"])
    calmar = _compute_calmar(annual_return, max_drawdown_abs)
    executed_df = pd.DataFrame(executed_rows).sort_values(["exit_date", "entry_date", "code", "leg_name"]).reset_index(drop=True)
    if executed_df.empty:
        raise RuntimeError("账户回放没有成交")

    position_summary = (
        executed_df.groupby(["strategy_label", "position_id"], as_index=False)
        .agg(
            code=("code", "first"),
            signal_date=("signal_date", "first"),
            entry_date=("entry_date", "first"),
            exit_date=("exit_date", "max"),
            total_pnl=("pnl", "sum"),
            gross_entry_cost=("gross_entry_cost_alloc", "sum"),
            entry_fee=("entry_fee_alloc", "sum"),
            strategy_key=("strategy_key", "first"),
        )
    )
    position_summary["trade_return_net"] = position_summary["total_pnl"] / (position_summary["gross_entry_cost"] + position_summary["entry_fee"])
    wins = position_summary.loc[position_summary["trade_return_net"] > 0, "trade_return_net"]
    losses = position_summary.loc[position_summary["trade_return_net"] < 0, "trade_return_net"]
    avg_trade_return = float(position_summary["trade_return_net"].mean())
    success_rate = float((position_summary["trade_return_net"] > 0).mean())
    hold_return = float(equity_df.iloc[-1]["equity"] / config.initial_capital - 1.0)
    summary = {
        "final_multiple": float(metrics["final_multiple"]),
        "annual_return": annual_return,
        "holding_return": hold_return,
        "max_drawdown": -max_drawdown_abs,
        "sharpe": sharpe,
        "calmar": calmar,
        "trade_count": int(len(position_summary)),
        "success_rate": success_rate,
        "avg_trade_return": avg_trade_return,
        "avg_win_return": float(wins.mean()) if not wins.empty else float("nan"),
        "avg_loss_return": float(losses.mean()) if not losses.empty else float("nan"),
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if not losses.empty else float("inf"),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else float("inf"),
        "max_losing_streak": int(_max_losing_streak(position_summary["trade_return_net"].tolist())),
        "equity_days": int(metrics["days"]),
        "final_equity": float(equity_df.iloc[-1]["equity"]),
    }
    return equity_df, executed_df, summary


def run_chain(result_dir: Path, file_limit: int, max_workers: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(result_dir, "loading_champion_trades", file_limit=file_limit, max_workers=max_workers)
    trades = load_champion_trades(file_limit=file_limit)
    score_df = load_signal_scores(file_limit=file_limit)
    trades = attach_signal_meta(trades, score_df)
    trades.to_csv(result_dir / "champion_source_trades.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "champion_trades_ready", trade_rows=int(len(trades)))

    codes = sorted(trades["code"].astype(str).unique())

    def feature_progress(done: int, total: int, **extra: Any) -> None:
        update_progress(result_dir, "loading_features", done_codes=done, total_codes=total, **extra)

    feature_registry = build_feature_registry(codes, max_workers=max_workers, progress_cb=feature_progress)
    pd.DataFrame([{"code": code, "has_feature": bool(code in feature_registry)} for code in codes]).to_csv(
        result_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig"
    )
    update_progress(result_dir, "features_ready", feature_codes=int(len(feature_registry)))

    case_df, case_stats = analyze_perfect_cases(
        result_dir,
        progress_cb=lambda done, total, **extra: update_progress(
            result_dir,
            "analyzing_perfect_cases",
            done_cases=done,
            total_cases=total,
            **extra,
        ),
    )
    case_rule = derive_case_rule(case_df) if not case_df.empty else {"type": "fallback", "features": [], "min_pass_count": 1}
    (result_dir / "case_rule.json").write_text(json.dumps(case_rule, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "case_rule_ready", **case_stats)

    event_map = build_strategy_events(trades, feature_registry, case_rule)
    for label, df in event_map.items():
        df.to_csv(result_dir / f"{label}_events.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "events_ready", strategy_count=len(event_map))

    market_dates, close_map = build_close_map(
        codes,
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=done, total_codes=total),
    )
    update_progress(result_dir, "close_map_ready", market_days=int(len(market_dates)), close_codes=int(len(close_map)))

    config = AccountConfig()
    summary_rows: list[dict[str, Any]] = []
    compare_targets = {}
    for label, events in event_map.items():
        equity_df, exec_df, summary = simulate_real_account_event_based(events, close_map, market_dates, config)
        equity_df.to_csv(result_dir / f"{label}_equity.csv", index=False, encoding="utf-8-sig")
        exec_df.to_csv(result_dir / f"{label}_executed_trades.csv", index=False, encoding="utf-8-sig")
        summary_rows.append({"label": label, **summary})
        compare_targets[label] = summary
        update_progress(result_dir, f"{label}_ready", trade_count=int(summary["trade_count"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(result_dir / "real_account_summary.csv", index=False, encoding="utf-8-sig")

    base_summary = compare_targets["champion_full_exit"]
    comparisons = {}
    for label in ["partial_all", "partial_rule_strong", "partial_case_strong"]:
        s = compare_targets[label]
        comparisons[f"champion_vs_{label}"] = {
            "annual_return_diff": float(s["annual_return"] - base_summary["annual_return"]),
            "holding_return_diff": float(s["holding_return"] - base_summary["holding_return"]),
            "avg_trade_return_diff": float(s["avg_trade_return"] - base_summary["avg_trade_return"]),
            "success_rate_diff": float(s["success_rate"] - base_summary["success_rate"]),
            "max_drawdown_diff": float(s["max_drawdown"] - base_summary["max_drawdown"]),
            "sharpe_diff": float(s["sharpe"] - base_summary["sharpe"]),
            "calmar_diff": float(s["calmar"] - base_summary["calmar"]),
            "payoff_ratio_diff": float(s["payoff_ratio"] - base_summary["payoff_ratio"]),
            "profit_factor_diff": float(s["profit_factor"] - base_summary["profit_factor"]),
        }

    summary = {
        "assumptions": {
            "initial_capital": config.initial_capital,
            "max_positions": config.max_positions,
            "daily_new_limit": config.daily_new_limit,
            "daily_budget_frac": config.daily_budget_frac,
            "position_cap_frac": config.position_cap_frac,
            "allocation_mode": config.allocation_mode,
            "commission_rate": config.commission_rate,
            "slippage_rate": config.slippage_rate,
            "stamp_duty_rate": config.stamp_duty_rate,
            "min_lot": config.min_lot,
            "signal_pool": "brick.formal_best",
            "champion_strategy": "formal_best + 当日止损次日止盈 + min(open,close)止损 + 5.5%止盈",
            "partial_all": "所有盈利单先卖80%，剩余20%等第一次绿砖当日收盘卖出",
            "partial_rule_strong": "仅规则强单留20%尾仓，其余盈利单仍按冠军策略全仓止盈",
            "partial_case_strong": "仅案例共性强单留20%尾仓，其余盈利单仍按冠军策略全仓止盈",
            "tail_assumption": "尾仓20%不再受3天持有上限限制，若直到样本结束都未转绿，则最后一天收盘卖出",
            "case_continuation_definition": f'止盈后{CONTINUATION_LOOKAHEAD}个交易日内相对原止盈卖价再上涨至少{CONTINUATION_EXTRA_RET:.0%}',
        },
        "case_rule": case_rule,
        "case_stats": case_stats,
        "summary_rows": summary_rows,
        "comparison": comparisons,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_limit = int(args.file_limit) if args.mode == "smoke" else 0
    result_dir = Path(args.output_dir)
    try:
        run_chain(result_dir=result_dir, file_limit=file_limit, max_workers=int(args.max_workers))
    except Exception as exc:
        write_error(result_dir, exc)
        raise


if __name__ == "__main__":
    main()
