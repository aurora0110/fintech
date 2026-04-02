from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
SIMILARITY_PATH = ROOT / "utils" / "tmp" / "similarity_filter_research.py"
PERFECT_CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RISK_CASE_DIR = ROOT / "data" / "完美图" / "出货图"

CASE_TYPE_NAME = {
    1: "N型低位首次启动",
    2: "横盘后突然启动",
    3: "连续上涨后中继绿砖再转红",
    4: "其他",
}

# 用户已明确给出的典型案例，优先作为人工主类型锚点。
MANUAL_CASE_TYPE: dict[str, int] = {
    "华盛锂电|2025-10-28": 1,
    "万邦达|2026-03-24": 1,
    "仁信新材|2026-03-24": 1,
    "华峰化学|2025-12-15": 1,
    "大金重工|2026-01-15": 1,
    "江航装备|2026-02-03": 1,
    "联特科技|2026-03-02": 1,
    "航发动力|2026-02-12": 2,
    "广立微|2025-07-24": 2,
    "华银电力|2026-03-24": 2,
    "华润江中|2026-03-09": 2,
    "华纳药厂|2025-05-28": 2,
    "幸福蓝海|2025-07-21": 2,
    "微芯生物|2025-05-28": 2,
    "三峡水利|2026-03-24": 2,
    "福能股份|2026-03-24": 2,
    "海光信息|2025-08-20": 3,
    "航发动力|2026-01-27": 3,
}

CASE_TYPE_SCORE = {
    1: 1.00,
    2: 0.92,
    3: 0.82,
    4: 0.45,
}

_MODULE_CACHE: dict[str, Any] = {}
_RISK_PROFILE_CACHE: dict[str, dict[str, float]] = {}


def _load_module(path: Path, module_name: str):
    cache_key = f"{module_name}:{path}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


sim = _load_module(SIMILARITY_PATH, "brick_case_semantics_similarity")


def _compute_signal_vs_ma5_proxy(x: pd.DataFrame) -> pd.Series:
    vol_prev_ma5 = pd.to_numeric(x["volume"], errors="coerce").shift(1).rolling(5).mean()
    return pd.to_numeric(x["volume"], errors="coerce") / vol_prev_ma5.replace(0, np.nan)


def code_key(value: Any) -> str:
    text = str(value)
    m = re.search(r"(\d{6})", text)
    return m.group(1) if m else text


def build_name_code_map(data_dir: str | Path) -> dict[str, str]:
    data_path = Path(data_dir)
    if data_path.name == "normal":
        date_root = data_path.parent
    else:
        date_root = data_path
    mapping: dict[str, str] = {}
    for path in sorted(date_root.glob("*.txt")):
        try:
            parts = path.read_text(encoding="gbk", errors="ignore").splitlines()[0].strip().split()
        except Exception:
            continue
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[1]] = path.stem
    return mapping


def parse_case_images(case_dir: Path, skip_counter_examples: bool = True) -> pd.DataFrame:
    pat = re.compile(r"(.+?)(\d{8})(?:-[0-9a-f]+)?\.png$")
    rows: list[dict[str, Any]] = []
    for path in sorted(case_dir.glob("*.png")):
        if skip_counter_examples and ("反例" in path.name or path.stem == "案例图"):
            continue
        m = pat.match(path.name)
        if not m:
            continue
        stock_name, ds = m.groups()
        signal_date = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
        if pd.isna(signal_date):
            continue
        rows.append(
            {
                "stock_name": stock_name,
                "signal_date": signal_date,
                "case_file": str(path),
                "case_key": f"{stock_name}|{signal_date.strftime('%Y-%m-%d')}",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["stock_name", "signal_date"]).reset_index(drop=True)


def _manual_case_type(stock_name: str, signal_date: pd.Timestamp) -> int | None:
    return MANUAL_CASE_TYPE.get(f"{stock_name}|{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}")


def infer_case_type_from_values(
    *,
    stock_name: str | None = None,
    signal_date: pd.Timestamp | None = None,
    prev_green_streak: float | int | None = None,
    rebound_ratio: float | None = None,
    signal_ret: float | None = None,
    upper_shadow_pct: float | None = None,
    body_ratio: float | None = None,
    close_to_trend: float | None = None,
    close_to_long: float | None = None,
) -> tuple[int, str]:
    if stock_name is not None and signal_date is not None:
        manual = _manual_case_type(stock_name, pd.Timestamp(signal_date))
        if manual is not None:
            return manual, "manual"

    pgs = float(prev_green_streak or 0.0)
    rebound = float(rebound_ratio) if rebound_ratio is not None and np.isfinite(rebound_ratio) else float("nan")
    ret = float(signal_ret) if signal_ret is not None and np.isfinite(signal_ret) else float("nan")
    upper = float(upper_shadow_pct) if upper_shadow_pct is not None and np.isfinite(upper_shadow_pct) else float("nan")
    body = float(body_ratio) if body_ratio is not None and np.isfinite(body_ratio) else float("nan")
    ctt = float(close_to_trend) if close_to_trend is not None and np.isfinite(close_to_trend) else float("nan")
    ctl = float(close_to_long) if close_to_long is not None and np.isfinite(close_to_long) else float("nan")

    if pgs >= 3 and (np.isnan(upper) or upper <= 0.35):
        return 1, "heuristic"
    if pgs in (1.0, 2.0) and (np.isnan(rebound) or rebound <= 2.6) and (np.isnan(upper) or upper <= 0.35):
        return 2, "heuristic"
    if pgs <= 1.0 and (not np.isnan(ret) and ret > 0) and (
        (not np.isnan(rebound) and rebound >= 3.0)
        or (not np.isnan(ctl) and ctl >= 0.10)
        or (not np.isnan(ctt) and ctt >= 0.08)
    ):
        return 3, "heuristic"
    if pgs in (1.0, 2.0) and (not np.isnan(body) and body >= 0.72) and (not np.isnan(ret) and ret > 0):
        return 2, "heuristic"
    return 4, "fallback"


def build_case_type_features(
    *,
    stock_name: str | None = None,
    signal_date: pd.Timestamp | None = None,
    prev_green_streak: float | int | None = None,
    prev_red_streak: float | int | None = None,
    rebound_ratio: float | None = None,
    signal_ret: float | None = None,
    upper_shadow_pct: float | None = None,
    body_ratio: float | None = None,
    close_to_trend: float | None = None,
    close_to_long: float | None = None,
) -> dict[str, Any]:
    case_type, case_source = infer_case_type_from_values(
        stock_name=stock_name,
        signal_date=signal_date,
        prev_green_streak=prev_green_streak,
        rebound_ratio=rebound_ratio,
        signal_ret=signal_ret,
        upper_shadow_pct=upper_shadow_pct,
        body_ratio=body_ratio,
        close_to_trend=close_to_trend,
        close_to_long=close_to_long,
    )
    prev_red = float(prev_red_streak or 0.0)
    early_red = bool(case_type in (1, 2) and prev_red <= 1.0)
    return {
        "brick_case_type": int(case_type),
        "brick_case_type_name": CASE_TYPE_NAME[int(case_type)],
        "brick_case_type_score": float(CASE_TYPE_SCORE[int(case_type)]),
        "brick_case_type_source": case_source,
        "early_red_stage_flag": early_red,
        "early_red_stage_flag_num": float(early_red),
    }


def load_case_day_features(case_dir: Path, data_dir: Path) -> pd.DataFrame:
    case_df = parse_case_images(case_dir, skip_counter_examples=(case_dir == PERFECT_CASE_DIR))
    if case_df.empty:
        return case_df
    name_map = build_name_code_map(data_dir)
    rows: list[dict[str, Any]] = []
    for row in case_df.itertuples(index=False):
        code = name_map.get(row.stock_name)
        if not code:
            continue
        file_path = data_dir / f"{code}.txt"
        if not file_path.exists():
            continue
        raw = sim.load_stock_data(str(file_path))
        if raw is None or raw.empty:
            continue
        feat = sim.compute_relaxed_brick_features(raw).reset_index(drop=True)
        if feat.empty:
            continue
        feat["signal_vs_ma5_proxy"] = _compute_signal_vs_ma5_proxy(feat)
        match = feat[feat["date"] == pd.Timestamp(row.signal_date)]
        if match.empty:
            continue
        r = match.iloc[-1]
        rec = {
            "stock_name": row.stock_name,
            "signal_date": pd.Timestamp(row.signal_date),
            "code": code,
            "code_key": code_key(code),
            "case_file": row.case_file,
            "signal_ret": float(r.get("signal_ret", np.nan)),
            "prev_green_streak": float(r.get("prev_green_streak", np.nan)),
            "prev_red_streak": float(r.get("prev_red_streak", np.nan)),
            "rebound_ratio": float(r.get("rebound_ratio", np.nan)) if pd.notna(r.get("rebound_ratio", np.nan)) else np.nan,
            "body_ratio": float(r.get("body_ratio", np.nan)),
            "upper_shadow_pct": float(r.get("upper_shadow_pct", np.nan)),
            "lower_shadow_pct": float(r.get("lower_shadow_pct", np.nan)),
            "close_to_trend": float(r.get("close_to_trend", np.nan)),
            "close_to_long": float(r.get("close_to_long", np.nan)),
            "trend_spread": float(r.get("trend_spread", np.nan)),
            "brick_red_len": float(r.get("brick_red_len", np.nan)),
            "brick_green_len_prev": float(r.get("brick_green_len", np.nan)),
            "pattern_a_relaxed": bool(r.get("pattern_a_relaxed", False)),
            "pattern_b_relaxed": bool(r.get("pattern_b_relaxed", False)),
            "signal_relaxed": bool(r.get("signal_relaxed", False)),
        }
        rec.update(
            build_case_type_features(
                stock_name=row.stock_name,
                signal_date=pd.Timestamp(row.signal_date),
                prev_green_streak=rec["prev_green_streak"],
                prev_red_streak=rec["prev_red_streak"],
                rebound_ratio=rec["rebound_ratio"],
                signal_ret=rec["signal_ret"],
                upper_shadow_pct=rec["upper_shadow_pct"],
                body_ratio=rec["body_ratio"],
                close_to_trend=rec["close_to_trend"],
                close_to_long=rec["close_to_long"],
            )
        )
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "stock_name"]).reset_index(drop=True)


def build_risk_profile(data_dir: Path) -> dict[str, float]:
    cache_key = str(data_dir.resolve())
    if cache_key in _RISK_PROFILE_CACHE:
        return _RISK_PROFILE_CACHE[cache_key]
    risk_df = load_case_day_features(RISK_CASE_DIR, data_dir)
    if risk_df.empty:
        profile = {
            "signal_ret_threshold": -0.035,
            "upper_shadow_threshold": 0.18,
            "lower_shadow_threshold": 0.25,
            "close_to_long_threshold": 0.18,
            "body_ratio_threshold": 0.56,
        }
    else:
        profile = {
            "signal_ret_threshold": float(risk_df["signal_ret"].quantile(0.75)),
            "upper_shadow_threshold": float(risk_df["upper_shadow_pct"].quantile(0.50)),
            "lower_shadow_threshold": float(risk_df["lower_shadow_pct"].quantile(0.75)),
            "close_to_long_threshold": float(risk_df["close_to_long"].quantile(0.50)),
            "body_ratio_threshold": float(risk_df["body_ratio"].quantile(0.25)),
        }
    _RISK_PROFILE_CACHE[cache_key] = profile
    return profile


def distribution_event_score_from_row(row: pd.Series | dict[str, Any], profile: dict[str, float]) -> float:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    signal_ret = float(series.get("signal_ret", np.nan)) if pd.notna(series.get("signal_ret", np.nan)) else np.nan
    upper = float(series.get("upper_shadow_pct", np.nan)) if pd.notna(series.get("upper_shadow_pct", np.nan)) else np.nan
    lower = float(series.get("lower_shadow_pct", np.nan)) if pd.notna(series.get("lower_shadow_pct", np.nan)) else np.nan
    ctl = float(series.get("close_to_long", np.nan)) if pd.notna(series.get("close_to_long", np.nan)) else np.nan
    body = float(series.get("body_ratio", np.nan)) if pd.notna(series.get("body_ratio", np.nan)) else np.nan

    score = 0.0
    if not np.isnan(signal_ret) and signal_ret <= profile["signal_ret_threshold"]:
        score += 0.35
    if not np.isnan(upper) and upper >= profile["upper_shadow_threshold"]:
        score += 0.20
    if not np.isnan(lower) and lower >= profile["lower_shadow_threshold"]:
        score += 0.15
    if not np.isnan(ctl) and ctl >= profile["close_to_long_threshold"]:
        score += 0.15
    if not np.isnan(body) and body <= profile["body_ratio_threshold"]:
        score += 0.15
    return float(min(1.0, score))


def compute_recent_distribution_risk_features(
    feature_df: pd.DataFrame,
    signal_idx: int,
    profile: dict[str, float],
) -> dict[str, float]:
    if feature_df.empty:
        return {
            "risk_distribution_recent_20": 0.0,
            "risk_distribution_recent_30": 0.0,
            "risk_distribution_recent_60": 0.0,
        }
    history = feature_df.iloc[: signal_idx + 1].copy()
    if history.empty:
        return {
            "risk_distribution_recent_20": 0.0,
            "risk_distribution_recent_30": 0.0,
            "risk_distribution_recent_60": 0.0,
        }
    scores = history.apply(lambda row: distribution_event_score_from_row(row, profile), axis=1)
    out: dict[str, float] = {}
    for window in (20, 30, 60):
        subset = scores.tail(window)
        out[f"risk_distribution_recent_{window}"] = float(subset.max()) if not subset.empty else 0.0
    return out


def enrich_case_type_and_risk_from_values(
    *,
    stock_name: str | None = None,
    signal_date: pd.Timestamp | None = None,
    prev_green_streak: float | int | None = None,
    prev_red_streak: float | int | None = None,
    rebound_ratio: float | None = None,
    signal_ret: float | None = None,
    upper_shadow_pct: float | None = None,
    body_ratio: float | None = None,
    close_to_trend: float | None = None,
    close_to_long: float | None = None,
    feature_df: pd.DataFrame | None = None,
    signal_idx: int | None = None,
    risk_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    rec = build_case_type_features(
        stock_name=stock_name,
        signal_date=signal_date,
        prev_green_streak=prev_green_streak,
        prev_red_streak=prev_red_streak,
        rebound_ratio=rebound_ratio,
        signal_ret=signal_ret,
        upper_shadow_pct=upper_shadow_pct,
        body_ratio=body_ratio,
        close_to_trend=close_to_trend,
        close_to_long=close_to_long,
    )
    if feature_df is not None and signal_idx is not None and risk_profile is not None:
        rec.update(compute_recent_distribution_risk_features(feature_df, int(signal_idx), risk_profile))
    else:
        rec.update(
            {
                "risk_distribution_recent_20": 0.0,
                "risk_distribution_recent_30": 0.0,
                "risk_distribution_recent_60": 0.0,
            }
        )
    return rec
