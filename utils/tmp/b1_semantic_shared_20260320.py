from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
V1_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
V2_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v2_20260320.py"
RISK_SCRIPT = ROOT / "utils" / "market_risk_tags.py"
EXTRA_FACTOR_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_extra_factor_ab.py"

SEQ_LEN = 21
BUY_DELAY_DAYS = 1
MAX_HOLD_DAYS = 60
NEGATIVE_MAX_DRAWDOWN_30 = -0.10


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = load_module(V1_SCRIPT, "b1_sem_base")
case_mod = load_module(V2_SCRIPT, "b1_sem_case")
risk_mod = load_module(RISK_SCRIPT, "b1_sem_risk")
extra_mod = load_module(EXTRA_FACTOR_SCRIPT, "b1_sem_extra")


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, default: float = np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def _bars_since(flag: pd.Series, max_lookback: int = 120) -> pd.Series:
    out = np.full(len(flag), np.nan, dtype=float)
    last_true = -10_000
    arr = flag.fillna(False).to_numpy(dtype=bool)
    for i in range(len(arr)):
        if arr[i]:
            last_true = i
        dist = i - last_true
        if dist <= max_lookback:
            out[i] = float(dist)
    return pd.Series(out, index=flag.index)


def _first_pullback_after_cross(cross_up: pd.Series, pullback_any: pd.Series, max_lookback: int = 30) -> pd.Series:
    out = np.zeros(len(cross_up), dtype=bool)
    cross_idx = cross_up[cross_up.fillna(False)].index.to_list()
    pull_arr = pullback_any.fillna(False).to_numpy(dtype=bool)
    for idx in cross_idx:
        left = int(idx) + 1
        right = min(len(pull_arr), left + max_lookback)
        for j in range(left, right):
            if pull_arr[j]:
                out[j] = True
                break
    return pd.Series(out, index=cross_up.index)


def add_semantic_buy_features(df: pd.DataFrame) -> pd.DataFrame:
    x = base_mod.compute_b1_features(df).copy()

    x["prev_low"] = x["low"].shift(1)
    x["prev_close"] = x["close"].shift(1)
    x["prev_trend"] = x["trend_line"].shift(1)
    x["prev_long"] = x["long_line"].shift(1)

    x["near_trend_pullback"] = (
        (x["low"] <= x["trend_line"] * 1.02)
        & (x["close"] >= x["trend_line"] * 0.96)
    ) | (
        (x["prev_low"] <= x["prev_trend"] * 1.02)
        & (x["prev_close"] >= x["prev_trend"] * 0.96)
    )
    x["near_long_pullback"] = (
        (x["low"] <= x["long_line"] * 1.02)
        & (x["close"] >= x["long_line"] * 0.96)
    ) | (
        (x["prev_low"] <= x["prev_long"] * 1.02)
        & (x["prev_close"] >= x["prev_long"] * 0.96)
    )
    x["pullback_any"] = x["near_trend_pullback"] | x["near_long_pullback"]

    x["low_level_context"] = x["long_line"] > x["trend_line"]
    x["recent_low_level_context_20"] = x["low_level_context"].shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
    x["cross_up_event"] = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
    x["bars_since_cross_up"] = _bars_since(x["cross_up_event"], max_lookback=60)
    x["first_pullback_after_cross"] = _first_pullback_after_cross(x["cross_up_event"], x["pullback_any"], max_lookback=30)

    x["half_volume"] = (x["vol_vs_prev"] <= 0.55) | (x["signal_vs_ma5"] <= 0.75)
    x["semi_shrink"] = (x["vol_vs_prev"] <= 0.75) | (x["signal_vs_ma5"] <= 0.90)

    x["ret20"] = x["close"].pct_change(20)
    x["ret30"] = x["close"].pct_change(30)
    x["ret_std_5"] = x["close"].pct_change().rolling(5, min_periods=3).std()
    x["ret_std_10"] = x["close"].pct_change().rolling(10, min_periods=5).std()
    x["ret_std_20"] = x["close"].pct_change().rolling(20, min_periods=10).std()
    x["ret_skew_10"] = x["close"].pct_change().rolling(10, min_periods=6).skew()
    x["ret_skew_20"] = x["close"].pct_change().rolling(20, min_periods=10).skew()
    x["ret_kurt_20"] = x["close"].pct_change().rolling(20, min_periods=10).kurt()
    x["price_skew_20"] = x["close"].rolling(20, min_periods=10).skew()
    x["up_count_5"] = (x["close"].pct_change() > 0).rolling(5, min_periods=1).sum()
    x["down_count_5"] = (x["close"].pct_change() < 0).rolling(5, min_periods=1).sum()
    x["up_count_10"] = (x["close"].pct_change() > 0).rolling(10, min_periods=1).sum()
    x["down_count_10"] = (x["close"].pct_change() < 0).rolling(10, min_periods=1).sum()
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - x["close"].shift(1)).abs(),
            (x["low"] - x["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr14_pct"] = _safe_div(tr.rolling(14, min_periods=5).mean(), x["close"])
    roll20_high = x["high"].rolling(20, min_periods=10).max()
    roll20_low = x["low"].rolling(20, min_periods=10).min()
    roll60_high = x["high"].rolling(60, min_periods=20).max()
    roll60_low = x["low"].rolling(60, min_periods=20).min()
    x["dist_20d_high"] = _safe_div(x["close"], roll20_high) - 1.0
    x["dist_20d_low"] = _safe_div(x["close"], roll20_low) - 1.0
    x["dist_60d_high"] = _safe_div(x["close"], roll60_high) - 1.0
    x["dist_60d_low"] = _safe_div(x["close"], roll60_low) - 1.0
    x["range_pos_20"] = _safe_div(x["close"] - roll20_low, roll20_high - roll20_low)
    x["range_pos_60"] = _safe_div(x["close"] - roll60_low, roll60_high - roll60_low)
    vol_mean20 = x["volume"].rolling(20, min_periods=5).mean()
    vol_std20 = x["volume"].rolling(20, min_periods=5).std()
    x["volume_z20"] = _safe_div(x["volume"] - vol_mean20, vol_std20)
    x["volume_rank20"] = x["volume"].rolling(20, min_periods=5).rank(pct=True)
    x["ma_bull_alignment"] = ((x["ma5"] > x["ma10"]) & (x["ma10"] > x["ma20"])).astype(int)
    x["ma_bear_alignment"] = ((x["ma5"] < x["ma10"]) & (x["ma10"] < x["ma20"])).astype(int)
    diff = x["close"].diff()
    up = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    rs = _safe_div(up.rolling(14, min_periods=5).mean(), down.rolling(14, min_periods=5).mean())
    x["rsi14"] = 100 - 100 / (1 + rs)

    extra_df = pd.DataFrame(
        {
            "OPEN": x["open"],
            "HIGH": x["high"],
            "LOW": x["low"],
            "CLOSE": x["close"],
            "VOLUME": x["volume"],
            "J": x["J"],
            "trend_ok": x["trend_line"] > x["long_line"],
            "bullish_filter": True,
        }
    )
    extra_df = extra_mod.add_extra_features(extra_df)
    x["double_bull"] = extra_df["double_bull"].astype(bool)
    x["double_bull_exist_60"] = extra_df["double_bull_exist_60"].astype(bool)
    x["above_double_bull_close"] = extra_df["above_double_bull_close"].fillna(False).astype(bool)
    x["above_double_bull_high"] = extra_df["above_double_bull_high"].fillna(False).astype(bool)
    x["key_k_close"] = pd.to_numeric(extra_df["key_k_close"], errors="coerce")
    x["key_k_support"] = extra_df["key_k_support"].fillna(False).astype(bool)

    risk_base = x[["open", "high", "low", "close", "volume", "trend_line", "long_line"]].copy()
    risk_df = risk_mod.add_risk_features(risk_base, precomputed_base=risk_base)
    risk_cols = [
        "recent_heavy_bear_top_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
        "recent_stair_bear_20",
        "risk_fast_rise_10d_40",
        "risk_segment_rise_slope_10_006",
        "risk_distribution_any_20",
    ]
    for col in risk_cols:
        x[col] = risk_df[col].astype(bool)
    x["no_distribution_risk"] = ~x["risk_distribution_any_20"].fillna(False).astype(bool)

    x["semantic_base"] = (x["J"] < 13) & x["pullback_any"] & x["no_distribution_risk"]
    x["semantic_uptrend_pullback"] = x["semantic_base"] & (x["trend_line"] > x["long_line"])
    x["semantic_low_cross_pullback"] = (
        (x["J"] < 13)
        & x["first_pullback_after_cross"]
        & x["recent_low_level_context_20"]
        & x["no_distribution_risk"]
    )
    x["semantic_confirmed"] = x["semantic_base"] & (
        x["half_volume"] | x["double_bull_exist_60"] | x["key_k_support"]
    )
    x["semantic_strict"] = (
        (x["semantic_uptrend_pullback"] | x["semantic_low_cross_pullback"])
        & (x["semi_shrink"] | x["double_bull_exist_60"] | x["key_k_support"])
    )
    x["semantic_trend_focus"] = (
        (x["J"] < 13)
        & x["near_trend_pullback"]
        & x["no_distribution_risk"]
        & (x["trend_slope_5"] > -0.02)
        & (x["long_slope_5"] > -0.02)
    )
    x["semantic_long_focus"] = (
        (x["J"] < 13)
        & x["near_long_pullback"]
        & x["no_distribution_risk"]
    )
    x["semantic_candidate"] = x["semantic_base"]

    x["buy_semantic_score"] = (
        x["semantic_base"].astype(int)
        + x["semantic_uptrend_pullback"].astype(int)
        + x["semantic_low_cross_pullback"].astype(int) * 1.5
        + x["half_volume"].astype(int)
        + x["double_bull_exist_60"].astype(int)
        + x["key_k_support"].astype(int)
        + x["above_double_bull_close"].astype(int) * 0.5
        + x["near_trend_pullback"].astype(int) * 0.5
        + x["near_long_pullback"].astype(int) * 0.5
    ).astype(float)
    return x


def extract_sequence(window_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return base_mod.extract_sequence(window_df)


def future_metrics(x: pd.DataFrame, idx: int) -> Dict[str, Any]:
    return base_mod.future_metrics(x, idx)


def build_semantic_candidates_for_one_stock(file_path: str) -> List[Dict[str, Any]]:
    df = base_mod.load_stock_data(file_path)
    if df is None or df.empty:
        return []
    x = add_semantic_buy_features(df)
    code = str(x["code"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for idx in range(max(base_mod.MIN_BARS, SEQ_LEN - 1), len(x) - (BUY_DELAY_DAYS + 1)):
        row = x.iloc[idx]
        if not bool(row["semantic_candidate"]):
            continue
        metrics = future_metrics(x, idx)
        if not metrics:
            continue
        seq_window = x.iloc[idx - SEQ_LEN + 1 : idx + 1]
        if len(seq_window) != SEQ_LEN:
            continue
        seq_map = extract_sequence(seq_window)
        rows.append(
            {
                "code": code,
                "signal_date": pd.Timestamp(row["date"]),
                "signal_idx": int(idx),
                "entry_date": metrics["entry_date"],
                "entry_price": metrics["entry_price"],
                "stop_loss_price": metrics["stop_loss_price"],
                "ret_3d": metrics["ret_3d"],
                "ret_5d": metrics["ret_5d"],
                "ret_10d": metrics["ret_10d"],
                "ret_20d": metrics["ret_20d"],
                "ret_30d": metrics["ret_30d"],
                "up_3d": metrics["up_3d"],
                "up_5d": metrics["up_5d"],
                "up_10d": metrics["up_10d"],
                "up_20d": metrics["up_20d"],
                "up_30d": metrics["up_30d"],
                "min_close_ret_30": metrics["min_close_ret_30"],
                "max_high_ret_30": metrics["max_high_ret_30"],
                "negative_30d": metrics["negative_30d"],
                "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else 0.0,
                "ret3": float(row["ret3"]) if pd.notna(row["ret3"]) else 0.0,
                "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else 0.0,
                "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else 0.0,
                "ret20": float(row["ret20"]) if pd.notna(row["ret20"]) else 0.0,
                "ret30": float(row["ret30"]) if pd.notna(row["ret30"]) else 0.0,
                "signal_ret": float(row["signal_ret"]) if pd.notna(row["signal_ret"]) else 0.0,
                "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else 0.0,
                "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else 0.0,
                "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else 0.0,
                "signal_vs_ma5": float(row["signal_vs_ma5"]) if pd.notna(row["signal_vs_ma5"]) else 0.0,
                "vol_vs_prev": float(row["vol_vs_prev"]) if pd.notna(row["vol_vs_prev"]) else 0.0,
                "ret_std_5": float(row["ret_std_5"]) if pd.notna(row["ret_std_5"]) else 0.0,
                "ret_std_10": float(row["ret_std_10"]) if pd.notna(row["ret_std_10"]) else 0.0,
                "ret_std_20": float(row["ret_std_20"]) if pd.notna(row["ret_std_20"]) else 0.0,
                "ret_skew_10": float(row["ret_skew_10"]) if pd.notna(row["ret_skew_10"]) else 0.0,
                "ret_skew_20": float(row["ret_skew_20"]) if pd.notna(row["ret_skew_20"]) else 0.0,
                "ret_kurt_20": float(row["ret_kurt_20"]) if pd.notna(row["ret_kurt_20"]) else 0.0,
                "price_skew_20": float(row["price_skew_20"]) if pd.notna(row["price_skew_20"]) else 0.0,
                "atr14_pct": float(row["atr14_pct"]) if pd.notna(row["atr14_pct"]) else 0.0,
                "dist_20d_high": float(row["dist_20d_high"]) if pd.notna(row["dist_20d_high"]) else 0.0,
                "dist_20d_low": float(row["dist_20d_low"]) if pd.notna(row["dist_20d_low"]) else 0.0,
                "dist_60d_high": float(row["dist_60d_high"]) if pd.notna(row["dist_60d_high"]) else 0.0,
                "dist_60d_low": float(row["dist_60d_low"]) if pd.notna(row["dist_60d_low"]) else 0.0,
                "range_pos_20": float(row["range_pos_20"]) if pd.notna(row["range_pos_20"]) else 0.0,
                "range_pos_60": float(row["range_pos_60"]) if pd.notna(row["range_pos_60"]) else 0.0,
                "volume_z20": float(row["volume_z20"]) if pd.notna(row["volume_z20"]) else 0.0,
                "volume_rank20": float(row["volume_rank20"]) if pd.notna(row["volume_rank20"]) else 0.0,
                "up_count_5": float(row["up_count_5"]) if pd.notna(row["up_count_5"]) else 0.0,
                "down_count_5": float(row["down_count_5"]) if pd.notna(row["down_count_5"]) else 0.0,
                "up_count_10": float(row["up_count_10"]) if pd.notna(row["up_count_10"]) else 0.0,
                "down_count_10": float(row["down_count_10"]) if pd.notna(row["down_count_10"]) else 0.0,
                "ma_bull_alignment": bool(row["ma_bull_alignment"]),
                "ma_bear_alignment": bool(row["ma_bear_alignment"]),
                "rsi14": float(row["rsi14"]) if pd.notna(row["rsi14"]) else 0.0,
                "body_ratio": float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0,
                "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else 0.0,
                "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else 0.0,
                "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else 0.0,
                "ma5_slope_5": float(row["ma5_slope_5"]) if pd.notna(row["ma5_slope_5"]) else 0.0,
                "ma10_slope_5": float(row["ma10_slope_5"]) if pd.notna(row["ma10_slope_5"]) else 0.0,
                "ma20_slope_5": float(row["ma20_slope_5"]) if pd.notna(row["ma20_slope_5"]) else 0.0,
                "trend_slope_5": float(row["trend_slope_5"]) if pd.notna(row["trend_slope_5"]) else 0.0,
                "long_slope_5": float(row["long_slope_5"]) if pd.notna(row["long_slope_5"]) else 0.0,
                "near_trend_pullback": bool(row["near_trend_pullback"]),
                "near_long_pullback": bool(row["near_long_pullback"]),
                "pullback_any": bool(row["pullback_any"]),
                "recent_low_level_context_20": bool(row["recent_low_level_context_20"]),
                "cross_up_event": bool(row["cross_up_event"]),
                "bars_since_cross_up": float(row["bars_since_cross_up"]) if pd.notna(row["bars_since_cross_up"]) else np.nan,
                "first_pullback_after_cross": bool(row["first_pullback_after_cross"]),
                "half_volume": bool(row["half_volume"]),
                "semi_shrink": bool(row["semi_shrink"]),
                "double_bull_exist_60": bool(row["double_bull_exist_60"]),
                "above_double_bull_close": bool(row["above_double_bull_close"]),
                "above_double_bull_high": bool(row["above_double_bull_high"]),
                "key_k_support": bool(row["key_k_support"]),
                "recent_heavy_bear_top_20": bool(row["recent_heavy_bear_top_20"]),
                "recent_failed_breakout_20": bool(row["recent_failed_breakout_20"]),
                "top_distribution_20": bool(row["top_distribution_20"]),
                "recent_stair_bear_20": bool(row["recent_stair_bear_20"]),
                "risk_distribution_any_20": bool(row["risk_distribution_any_20"]),
                "semantic_base": bool(row["semantic_base"]),
                "semantic_candidate": bool(row["semantic_candidate"]),
                "semantic_uptrend_pullback": bool(row["semantic_uptrend_pullback"]),
                "semantic_low_cross_pullback": bool(row["semantic_low_cross_pullback"]),
                "semantic_confirmed": bool(row["semantic_confirmed"]),
                "semantic_strict": bool(row["semantic_strict"]),
                "semantic_trend_focus": bool(row["semantic_trend_focus"]),
                "semantic_long_focus": bool(row["semantic_long_focus"]),
                "buy_semantic_score": float(row["buy_semantic_score"]),
                "seq_map": seq_map,
            }
        )
    return rows


def enrich_case_with_semantics(row: pd.Series, mapping: Dict[str, str]) -> Dict[str, Any]:
    out = case_mod.enrich_case(row, mapping)
    if out.get("status") != "ok":
        return out
    code = str(out["code"])
    cache = getattr(enrich_case_with_semantics, "_feature_cache", None)
    if cache is None:
        cache = {}
        setattr(enrich_case_with_semantics, "_feature_cache", cache)
    feat = cache.get(code)
    if feat is None:
        path = base_mod.DATA_DIR / f"{code}.txt"
        df = base_mod.load_stock_data(str(path))
        if df is None or df.empty:
            out["status"] = "load_failed"
            return out
        feat = add_semantic_buy_features(df)
        cache[code] = feat
    ds = pd.to_datetime(feat["date"]).dt.strftime("%Y%m%d")
    idxs = np.flatnonzero(ds.to_numpy() == str(row["buy_date"]))
    if len(idxs) == 0:
        out["status"] = "buy_date_missing"
        return out
    idx = int(idxs[-1])
    if idx < SEQ_LEN - 1:
        out["status"] = "bars_insufficient"
        return out
    feat_row = feat.iloc[idx]
    out.update(
        {
            "near_trend_pullback": bool(feat_row["near_trend_pullback"]),
            "near_long_pullback": bool(feat_row["near_long_pullback"]),
            "pullback_any": bool(feat_row["pullback_any"]),
            "recent_low_level_context_20": bool(feat_row["recent_low_level_context_20"]),
            "first_pullback_after_cross": bool(feat_row["first_pullback_after_cross"]),
            "half_volume": bool(feat_row["half_volume"]),
            "semi_shrink": bool(feat_row["semi_shrink"]),
            "double_bull_exist_60": bool(feat_row["double_bull_exist_60"]),
            "above_double_bull_close": bool(feat_row["above_double_bull_close"]),
            "above_double_bull_high": bool(feat_row["above_double_bull_high"]),
            "key_k_support": bool(feat_row["key_k_support"]),
            "recent_heavy_bear_top_20": bool(feat_row["recent_heavy_bear_top_20"]),
            "recent_failed_breakout_20": bool(feat_row["recent_failed_breakout_20"]),
            "top_distribution_20": bool(feat_row["top_distribution_20"]),
            "recent_stair_bear_20": bool(feat_row["recent_stair_bear_20"]),
            "risk_distribution_any_20": bool(feat_row["risk_distribution_any_20"]),
            "semantic_base": bool(feat_row["semantic_base"]),
            "semantic_uptrend_pullback": bool(feat_row["semantic_uptrend_pullback"]),
            "semantic_low_cross_pullback": bool(feat_row["semantic_low_cross_pullback"]),
            "semantic_confirmed": bool(feat_row["semantic_confirmed"]),
            "semantic_strict": bool(feat_row["semantic_strict"]),
            "semantic_trend_focus": bool(feat_row["semantic_trend_focus"]),
            "semantic_long_focus": bool(feat_row["semantic_long_focus"]),
            "buy_semantic_score": float(feat_row["buy_semantic_score"]),
            "ret20": float(feat_row["ret20"]) if pd.notna(feat_row["ret20"]) else 0.0,
            "ret30": float(feat_row["ret30"]) if pd.notna(feat_row["ret30"]) else 0.0,
            "ret_std_5": float(feat_row["ret_std_5"]) if pd.notna(feat_row["ret_std_5"]) else 0.0,
            "ret_std_10": float(feat_row["ret_std_10"]) if pd.notna(feat_row["ret_std_10"]) else 0.0,
            "ret_std_20": float(feat_row["ret_std_20"]) if pd.notna(feat_row["ret_std_20"]) else 0.0,
            "ret_skew_10": float(feat_row["ret_skew_10"]) if pd.notna(feat_row["ret_skew_10"]) else 0.0,
            "ret_skew_20": float(feat_row["ret_skew_20"]) if pd.notna(feat_row["ret_skew_20"]) else 0.0,
            "ret_kurt_20": float(feat_row["ret_kurt_20"]) if pd.notna(feat_row["ret_kurt_20"]) else 0.0,
            "price_skew_20": float(feat_row["price_skew_20"]) if pd.notna(feat_row["price_skew_20"]) else 0.0,
            "atr14_pct": float(feat_row["atr14_pct"]) if pd.notna(feat_row["atr14_pct"]) else 0.0,
            "dist_20d_high": float(feat_row["dist_20d_high"]) if pd.notna(feat_row["dist_20d_high"]) else 0.0,
            "dist_20d_low": float(feat_row["dist_20d_low"]) if pd.notna(feat_row["dist_20d_low"]) else 0.0,
            "dist_60d_high": float(feat_row["dist_60d_high"]) if pd.notna(feat_row["dist_60d_high"]) else 0.0,
            "dist_60d_low": float(feat_row["dist_60d_low"]) if pd.notna(feat_row["dist_60d_low"]) else 0.0,
            "range_pos_20": float(feat_row["range_pos_20"]) if pd.notna(feat_row["range_pos_20"]) else 0.0,
            "range_pos_60": float(feat_row["range_pos_60"]) if pd.notna(feat_row["range_pos_60"]) else 0.0,
            "volume_z20": float(feat_row["volume_z20"]) if pd.notna(feat_row["volume_z20"]) else 0.0,
            "volume_rank20": float(feat_row["volume_rank20"]) if pd.notna(feat_row["volume_rank20"]) else 0.0,
            "up_count_5": float(feat_row["up_count_5"]) if pd.notna(feat_row["up_count_5"]) else 0.0,
            "down_count_5": float(feat_row["down_count_5"]) if pd.notna(feat_row["down_count_5"]) else 0.0,
            "up_count_10": float(feat_row["up_count_10"]) if pd.notna(feat_row["up_count_10"]) else 0.0,
            "down_count_10": float(feat_row["down_count_10"]) if pd.notna(feat_row["down_count_10"]) else 0.0,
            "ma_bull_alignment": bool(feat_row["ma_bull_alignment"]),
            "ma_bear_alignment": bool(feat_row["ma_bear_alignment"]),
            "rsi14": float(feat_row["rsi14"]) if pd.notna(feat_row["rsi14"]) else 0.0,
            "seq_map": extract_sequence(feat.iloc[idx - SEQ_LEN + 1 : idx + 1]),
        }
    )
    return out


def _second_top_turn_flag(
    feat: pd.DataFrame,
    buy_idx: int,
    cur_idx: int,
    peak_tolerance: float = 0.03,
    min_gap: int = 5,
) -> bool:
    if cur_idx - buy_idx < min_gap + 2:
        return False
    history_high = feat["high"].iloc[buy_idx:cur_idx].to_numpy(dtype=float)
    if len(history_high) < min_gap + 1:
        return False
    prior_peak_rel = int(np.argmax(history_high))
    prior_peak_idx = buy_idx + prior_peak_rel
    if cur_idx - prior_peak_idx < min_gap:
        return False
    prior_peak_high = float(feat.iloc[prior_peak_idx]["high"])
    recent_high = float(feat.iloc[max(buy_idx, cur_idx - 2) : cur_idx + 1]["high"].max())
    if not np.isfinite(prior_peak_high) or prior_peak_high <= 0:
        return False
    near_prior_peak = recent_high >= prior_peak_high * (1.0 - peak_tolerance)
    if not near_prior_peak:
        return False
    cur = feat.iloc[cur_idx]
    prev = feat.iloc[cur_idx - 1]
    return bool(
        (cur["close"] < prev["close"])
        and (cur["signal_ret"] < -0.01)
        and ((cur["close"] < prev["low"]) or (cur["close"] < cur["trend_line"]))
    )


def build_daily_sell_semantic_features(feat: pd.DataFrame, buy_idx: int, cur_idx: int) -> Dict[str, Any]:
    feat_row = feat.iloc[cur_idx]
    buy_close = float(feat.iloc[buy_idx]["close"])
    cur_close = float(feat_row["close"])
    peak_high = float(feat.iloc[buy_idx : cur_idx + 1]["high"].max())
    profit_since_buy = cur_close / buy_close - 1.0 if buy_close > 0 else np.nan
    drawdown_from_peak = cur_close / peak_high - 1.0 if peak_high > 0 else np.nan

    sub = feat.iloc[buy_idx:cur_idx]
    bullish_vols = sub.loc[sub["close"] > sub["open"], "volume"]
    max_bullish_volume_since_buy = float(bullish_vols.max()) if len(bullish_vols) else np.nan
    heavy_bear_over_all_bulls = bool(
        (feat_row["close"] < feat_row["open"])
        and np.isfinite(max_bullish_volume_since_buy)
        and (float(feat_row["volume"]) > max_bullish_volume_since_buy)
    )
    break_trend_sell = bool(
        (feat_row["close"] < feat_row["trend_line"])
        and (feat.iloc[cur_idx - 1]["close"] >= feat.iloc[cur_idx - 1]["trend_line"])
    ) if cur_idx > 0 else False
    double_top_turn = _second_top_turn_flag(feat, buy_idx, cur_idx)

    out = {
        "hold_day_idx": int(cur_idx - buy_idx),
        "profit_since_buy": profit_since_buy,
        "drawdown_from_peak": drawdown_from_peak,
        "J": float(feat_row["J"]) if pd.notna(feat_row["J"]) else np.nan,
        "ret1": float(feat_row["ret1"]) if pd.notna(feat_row["ret1"]) else np.nan,
        "ret3": float(feat_row["ret3"]) if pd.notna(feat_row["ret3"]) else np.nan,
        "ret5": float(feat_row["ret5"]) if pd.notna(feat_row["ret5"]) else np.nan,
        "ret10": float(feat_row["ret10"]) if pd.notna(feat_row["ret10"]) else np.nan,
        "ret20": float(feat_row["ret20"]) if pd.notna(feat_row["ret20"]) else np.nan,
        "ret30": float(feat_row["ret30"]) if pd.notna(feat_row["ret30"]) else np.nan,
        "signal_ret": float(feat_row["signal_ret"]) if pd.notna(feat_row["signal_ret"]) else np.nan,
        "trend_spread": float(feat_row["trend_spread"]) if pd.notna(feat_row["trend_spread"]) else np.nan,
        "close_to_trend": float(feat_row["close_to_trend"]) if pd.notna(feat_row["close_to_trend"]) else np.nan,
        "close_to_long": float(feat_row["close_to_long"]) if pd.notna(feat_row["close_to_long"]) else np.nan,
        "signal_vs_ma5": float(feat_row["signal_vs_ma5"]) if pd.notna(feat_row["signal_vs_ma5"]) else np.nan,
        "vol_vs_prev": float(feat_row["vol_vs_prev"]) if pd.notna(feat_row["vol_vs_prev"]) else np.nan,
        "ret_std_5": float(feat_row["ret_std_5"]) if pd.notna(feat_row["ret_std_5"]) else np.nan,
        "ret_std_10": float(feat_row["ret_std_10"]) if pd.notna(feat_row["ret_std_10"]) else np.nan,
        "ret_std_20": float(feat_row["ret_std_20"]) if pd.notna(feat_row["ret_std_20"]) else np.nan,
        "ret_skew_10": float(feat_row["ret_skew_10"]) if pd.notna(feat_row["ret_skew_10"]) else np.nan,
        "ret_skew_20": float(feat_row["ret_skew_20"]) if pd.notna(feat_row["ret_skew_20"]) else np.nan,
        "ret_kurt_20": float(feat_row["ret_kurt_20"]) if pd.notna(feat_row["ret_kurt_20"]) else np.nan,
        "price_skew_20": float(feat_row["price_skew_20"]) if pd.notna(feat_row["price_skew_20"]) else np.nan,
        "atr14_pct": float(feat_row["atr14_pct"]) if pd.notna(feat_row["atr14_pct"]) else np.nan,
        "dist_20d_high": float(feat_row["dist_20d_high"]) if pd.notna(feat_row["dist_20d_high"]) else np.nan,
        "dist_20d_low": float(feat_row["dist_20d_low"]) if pd.notna(feat_row["dist_20d_low"]) else np.nan,
        "dist_60d_high": float(feat_row["dist_60d_high"]) if pd.notna(feat_row["dist_60d_high"]) else np.nan,
        "dist_60d_low": float(feat_row["dist_60d_low"]) if pd.notna(feat_row["dist_60d_low"]) else np.nan,
        "range_pos_20": float(feat_row["range_pos_20"]) if pd.notna(feat_row["range_pos_20"]) else np.nan,
        "range_pos_60": float(feat_row["range_pos_60"]) if pd.notna(feat_row["range_pos_60"]) else np.nan,
        "volume_z20": float(feat_row["volume_z20"]) if pd.notna(feat_row["volume_z20"]) else np.nan,
        "volume_rank20": float(feat_row["volume_rank20"]) if pd.notna(feat_row["volume_rank20"]) else np.nan,
        "up_count_5": float(feat_row["up_count_5"]) if pd.notna(feat_row["up_count_5"]) else np.nan,
        "down_count_5": float(feat_row["down_count_5"]) if pd.notna(feat_row["down_count_5"]) else np.nan,
        "up_count_10": float(feat_row["up_count_10"]) if pd.notna(feat_row["up_count_10"]) else np.nan,
        "down_count_10": float(feat_row["down_count_10"]) if pd.notna(feat_row["down_count_10"]) else np.nan,
        "ma_bull_alignment": float(feat_row["ma_bull_alignment"]) if pd.notna(feat_row["ma_bull_alignment"]) else np.nan,
        "ma_bear_alignment": float(feat_row["ma_bear_alignment"]) if pd.notna(feat_row["ma_bear_alignment"]) else np.nan,
        "rsi14": float(feat_row["rsi14"]) if pd.notna(feat_row["rsi14"]) else np.nan,
        "body_ratio": float(feat_row["body_ratio"]) if pd.notna(feat_row["body_ratio"]) else np.nan,
        "upper_shadow_pct": float(feat_row["upper_shadow_pct"]) if pd.notna(feat_row["upper_shadow_pct"]) else np.nan,
        "lower_shadow_pct": float(feat_row["lower_shadow_pct"]) if pd.notna(feat_row["lower_shadow_pct"]) else np.nan,
        "close_location": float(feat_row["close_location"]) if pd.notna(feat_row["close_location"]) else np.nan,
        "ma5_slope_5": float(feat_row["ma5_slope_5"]) if pd.notna(feat_row["ma5_slope_5"]) else np.nan,
        "ma10_slope_5": float(feat_row["ma10_slope_5"]) if pd.notna(feat_row["ma10_slope_5"]) else np.nan,
        "ma20_slope_5": float(feat_row["ma20_slope_5"]) if pd.notna(feat_row["ma20_slope_5"]) else np.nan,
        "trend_slope_5": float(feat_row["trend_slope_5"]) if pd.notna(feat_row["trend_slope_5"]) else np.nan,
        "long_slope_5": float(feat_row["long_slope_5"]) if pd.notna(feat_row["long_slope_5"]) else np.nan,
        "double_top_turn_sell": int(double_top_turn),
        "heavy_bear_over_all_bulls": int(heavy_bear_over_all_bulls),
        "break_trend_sell": int(break_trend_sell),
        "max_bullish_volume_since_buy": max_bullish_volume_since_buy,
    }
    for col in [
        "recent_heavy_bear_top_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
        "recent_stair_bear_20",
        "risk_fast_rise_10d_40",
        "risk_segment_rise_slope_10_006",
        "risk_distribution_any_20",
    ]:
        out[f"risk_{col}" if not col.startswith("risk_") else col] = int(bool(feat_row[col]))
    return out


SELL_FEATURE_COLS_V2 = [
    "hold_day_idx",
    "profit_since_buy",
    "drawdown_from_peak",
    "J",
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "ret30",
    "signal_ret",
    "trend_spread",
    "close_to_trend",
    "close_to_long",
    "signal_vs_ma5",
    "vol_vs_prev",
    "ret_std_5",
    "ret_std_10",
    "ret_std_20",
    "ret_skew_10",
    "ret_skew_20",
    "ret_kurt_20",
    "price_skew_20",
    "atr14_pct",
    "dist_20d_high",
    "dist_20d_low",
    "dist_60d_high",
    "dist_60d_low",
    "range_pos_20",
    "range_pos_60",
    "volume_z20",
    "volume_rank20",
    "up_count_5",
    "down_count_5",
    "up_count_10",
    "down_count_10",
    "ma_bull_alignment",
    "ma_bear_alignment",
    "rsi14",
    "body_ratio",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "close_location",
    "ma5_slope_5",
    "ma10_slope_5",
    "ma20_slope_5",
    "trend_slope_5",
    "long_slope_5",
    "double_top_turn_sell",
    "heavy_bear_over_all_bulls",
    "break_trend_sell",
    "risk_recent_heavy_bear_top_20",
    "risk_recent_failed_breakout_20",
    "risk_top_distribution_20",
    "risk_recent_stair_bear_20",
    "risk_fast_rise_10d_40",
    "risk_segment_rise_slope_10_006",
    "risk_distribution_any_20",
]


def score_sell_rule_v2(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    score += df["double_top_turn_sell"].astype(float) * 2.2
    score += df["heavy_bear_over_all_bulls"].astype(float) * 2.0
    score += df["break_trend_sell"].astype(float) * 1.6
    score += (df["drawdown_from_peak"] <= -0.08).astype(float) * 0.8
    score += (df["signal_ret"] <= -0.025).astype(float) * 0.8
    score += (df["close_location"] <= 0.30).astype(float) * 0.5
    score += (df["risk_distribution_any_20"] > 0).astype(float) * 0.5
    return score
