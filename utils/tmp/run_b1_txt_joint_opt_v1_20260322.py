from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BASE_SIGNAL_DIR = ROOT / "results" / "b1_full_factor_signal_v6_full_20260321_102049"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_20260320.py"
SEMANTIC_SCRIPT = ROOT / "utils" / "tmp" / "b1_semantic_shared_20260320.py"
TXT_POS_PATH = ROOT / "data" / "完美图" / "B1" / "正例.txt"
TXT_NEG_PATH = ROOT / "data" / "完美图" / "B1" / "反例.txt"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESULT_DIR = ROOT / "results" / f"b1_txt_joint_signal_v1_{RUN_TS}"
DEFAULT_TOPN_LIST = [3, 5, 8]
NAME_ALIASES = {
    "昂立康": "昂利康",
    "淮柴动力": "潍柴动力",
}

SELL_REASON_MARKERS = [
    "高位收盘价跌破趋势线",
    "收盘价跌破趋势线",
    "高位放量阴线",
    "放量阴线",
    "连续2日收盘价小于前一天最低价",
    "连续两天收盘价小于前一天最低价",
    "连续两天收盘价跌破前一天最低价",
    "高位阴线连续3日阴线",
    "连续快速上涨",
]
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base_mod = load_module(BASE_SCRIPT, "b1_txt_joint_base")
sem_mod = load_module(SEMANTIC_SCRIPT, "b1_txt_joint_sem")

HAS_SKLEARN = False
HAS_LGB = False
HAS_XGB = False
try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    HAS_SKLEARN = True
except Exception:
    ExtraTreesClassifier = None  # type: ignore
    LogisticRegression = None  # type: ignore

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    lgb = None  # type: ignore

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    xgb = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 文本标签监督版定向联合优化")
    parser.add_argument("--base-signal-dir", type=Path, default=BASE_SIGNAL_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--topn-list", type=str, default="")
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(result_dir / "progress.json", payload)


def parse_topn_list(raw: str) -> List[int]:
    if not raw.strip():
        return DEFAULT_TOPN_LIST
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return sorted(set(v for v in vals if v > 0))


def load_candidate_df(base_signal_dir: Path, file_limit: int, must_keep_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(base_signal_dir / "candidate_enriched.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    mask = (df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)
    df = df.loc[mask].copy()
    if file_limit > 0:
        keep_codes = sorted(df["code"].astype(str).drop_duplicates().tolist())[:file_limit]
        if must_keep_codes is not None:
            keep_codes = sorted(set(keep_codes) | {str(x) for x in must_keep_codes if str(x)})
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    return df.reset_index(drop=True)


def split_reason_tail(rest: str) -> Tuple[str, str]:
    rest = rest.strip()
    # 优先使用用户在 txt 中明确补的句号分隔。
    for delim in ("。", "."):
        if delim in rest:
            left, right = rest.split(delim, 1)
            buy_reason = left.strip()
            sell_reason = right.strip()
            if buy_reason and sell_reason:
                return buy_reason, sell_reason
    hit_pos: Optional[int] = None
    hit_marker: Optional[str] = None
    for marker in SELL_REASON_MARKERS:
        idx = rest.rfind(marker)
        if idx > 0 and (hit_pos is None or idx < hit_pos):
            hit_pos = idx
            hit_marker = marker
    if hit_pos is None or hit_marker is None:
        return rest, ""
    buy_reason = rest[:hit_pos].strip()
    sell_reason = rest[hit_pos:].strip()
    return buy_reason, sell_reason


def parse_positive_txt(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("注意："):
            continue
        parts = text.split(maxsplit=3)
        if len(parts) < 4:
            bad_rows.append({"line_no": line_no, "raw": text, "reason": "token不足"})
            continue
        stock_name, buy_date_raw, sell_date_raw, rest = parts
        if not (buy_date_raw.isdigit() and len(buy_date_raw) == 8):
            bad_rows.append({"line_no": line_no, "raw": text, "reason": "买入日期格式错误"})
            continue
        if not (sell_date_raw.isdigit() and len(sell_date_raw) == 8):
            bad_rows.append({"line_no": line_no, "raw": text, "reason": "卖出日期格式错误"})
            continue
        buy_reason, sell_reason = split_reason_tail(rest)
        if not buy_reason or not sell_reason:
            bad_rows.append({"line_no": line_no, "raw": text, "reason": "买卖原因拆分失败"})
            continue
        rows.append(
            {
                "line_no": line_no,
                "stock_name": stock_name,
                "buy_date": pd.Timestamp(buy_date_raw),
                "sell_date": pd.Timestamp(sell_date_raw),
                "buy_reason": buy_reason,
                "sell_reason": sell_reason,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bad_rows)


def parse_negative_txt(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("注意："):
            continue
        parts = text.split(maxsplit=2)
        if len(parts) < 3:
            continue
        stock_name, b1_date_raw, reason = parts
        if not (b1_date_raw.isdigit() and len(b1_date_raw) == 8):
            continue
        rows.append(
            {
                "line_no": line_no,
                "stock_name": stock_name,
                "b1_date": pd.Timestamp(b1_date_raw),
                "no_buy_reason": reason.strip(),
            }
        )
    return pd.DataFrame(rows)


def extract_label_codes(pos_txt: pd.DataFrame, neg_txt: pd.DataFrame, mapping: Dict[str, str]) -> List[str]:
    codes = set()
    for name in list(pos_txt.get("stock_name", pd.Series(dtype=str))) + list(neg_txt.get("stock_name", pd.Series(dtype=str))):
        code = resolve_code(str(name), mapping)
        if code:
            codes.add(str(code))
    return sorted(codes)


def collect_label_target_dates(pos_txt: pd.DataFrame, neg_txt: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, List[pd.Timestamp]]:
    out: Dict[str, List[pd.Timestamp]] = {}
    for _, row in pos_txt.iterrows():
        code = resolve_code(str(row["stock_name"]), mapping)
        if code:
            out.setdefault(str(code), []).append(pd.Timestamp(row["buy_date"]))
    for _, row in neg_txt.iterrows():
        code = resolve_code(str(row["stock_name"]), mapping)
        if code:
            out.setdefault(str(code), []).append(pd.Timestamp(row["b1_date"]))
    return out


def add_reason_flags(df: pd.DataFrame, reason_col: str) -> pd.DataFrame:
    x = df.copy()
    reason_s = x[reason_col].fillna("")
    x["reason_trend_pullback"] = reason_s.str.contains("回踩趋势线", regex=False)
    x["reason_long_pullback"] = reason_s.str.contains("回踩多空线", regex=False)
    x["reason_key_k"] = reason_s.str.contains("关键K", regex=False)
    x["reason_half_volume"] = reason_s.str.contains("缩半量", regex=False)
    x["reason_double_bull"] = reason_s.str.contains("倍量柱", regex=False)
    x["reason_break_trend_sell"] = reason_s.str.contains("跌破趋势线", regex=False)
    x["reason_heavy_bear_sell"] = reason_s.str.contains("放量阴线", regex=False)
    x["reason_double_top_sell"] = reason_s.str.contains("双头", regex=False)
    return x


def resolve_code(stock_name: str, mapping: Dict[str, str]) -> Optional[str]:
    code = base_mod.resolve_code(stock_name, mapping)
    if code:
        return code
    alias = NAME_ALIASES.get(stock_name)
    if alias:
        return base_mod.resolve_code(alias, mapping)
    return None


def infer_split_windows(candidate_df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in ["research", "validation", "final_test"]:
        sub = candidate_df[candidate_df["split"] == split]
        if sub.empty:
            continue
        windows[split] = (pd.Timestamp(sub["signal_date"].min()), pd.Timestamp(sub["signal_date"].max()))
    return windows


def assign_split_by_date(date_val: pd.Timestamp, windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    for split, (start, end) in windows.items():
        if start <= date_val <= end:
            return split
    return ""


def build_label_feature_df(
    codes: Iterable[str],
    split_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    target_dates_by_code: Optional[Dict[str, List[pd.Timestamp]]] = None,
    day_window: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keep_codes = sorted({str(c) for c in codes if str(c)})
    total = len(keep_codes)
    for i, code in enumerate(keep_codes, 1):
        path = base_mod.DATA_DIR / f"{code}.txt"
        if not path.exists():
            continue
        df = base_mod.load_stock_data(str(path))
        if df is None or df.empty:
            continue
        feat = sem_mod.add_semantic_buy_features(df)
        target_dates = [pd.Timestamp(x) for x in (target_dates_by_code or {}).get(code, [])]
        start_idx = max(base_mod.MIN_BARS, sem_mod.SEQ_LEN - 1)
        end_idx = len(feat) - (sem_mod.BUY_DELAY_DAYS + 1)
        if target_dates:
            ds = pd.to_datetime(feat["date"])
            mask = pd.Series(False, index=feat.index)
            for dt in target_dates:
                mask = mask | ds.between(dt - pd.Timedelta(days=day_window + 2), dt + pd.Timedelta(days=day_window))
            idx_list = [int(i) for i in np.flatnonzero(mask.to_numpy()) if start_idx <= int(i) < end_idx]
        else:
            idx_list = list(range(start_idx, end_idx))
        for idx in idx_list:
            metrics = sem_mod.future_metrics(feat, idx)
            if not metrics:
                continue
            row = feat.iloc[idx]
            signal_date = pd.Timestamp(row["date"])
            entry_date = pd.Timestamp(metrics["entry_date"])
            rows.append(
                {
                    "code": code,
                    "signal_date": signal_date,
                    "signal_idx": int(idx),
                    "entry_date": entry_date,
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
                    "split": assign_split_by_date(signal_date, split_windows),
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
                    "buy_semantic_score": float(row["buy_semantic_score"]) if pd.notna(row["buy_semantic_score"]) else 0.0,
                }
            )
        if i % 10 == 0 or i == total:
            print(f"文本标签特征构建进度: {i}/{total}")
    return pd.DataFrame(rows)


def choose_best_label_hit(hit: pd.DataFrame, target_date: pd.Timestamp, prefer_trend: bool, prefer_long: bool) -> Tuple[Optional[pd.Series], str]:
    if hit.empty:
        return None, ""
    hit = hit.copy()
    hit["entry_gap"] = (hit["entry_date"] - target_date).abs().dt.days
    hit["signal_gap"] = (hit["signal_date"] - target_date).abs().dt.days
    hit["date_gap"] = hit[["entry_gap", "signal_gap"]].min(axis=1)
    hit["reason_match"] = 0.0
    if prefer_trend and "near_trend_pullback" in hit.columns:
        hit["reason_match"] += hit["near_trend_pullback"].fillna(False).astype(float)
    if prefer_long and "near_long_pullback" in hit.columns:
        hit["reason_match"] += hit["near_long_pullback"].fillna(False).astype(float)
    hit = hit.sort_values(["date_gap", "reason_match", "buy_semantic_score"], ascending=[True, False, False])
    best = hit.iloc[0]
    mode = "entry_date" if pd.Timestamp(best["entry_date"]) == target_date else "signal_date" if pd.Timestamp(best["signal_date"]) == target_date else "nearest_date"
    return best, mode


def map_positive_cases(pos_txt: pd.DataFrame, label_df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for _, r in pos_txt.iterrows():
        code = resolve_code(str(r["stock_name"]), mapping)
        if not code:
            skipped.append({**r.to_dict(), "skip_reason": "股票名未映射"})
            continue
        sub = label_df[label_df["code"].astype(str) == str(code)].copy()
        if sub.empty:
            skipped.append({**r.to_dict(), "skip_reason": "候选池无该股票"})
            continue
        target_date = pd.Timestamp(r["buy_date"])
        hit = sub[(sub["entry_date"] == target_date) | (sub["signal_date"] == target_date)].copy()
        if hit.empty:
            hit = sub[
                sub["entry_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
                | sub["signal_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
            ].copy()
        best_row, mode = choose_best_label_hit(
            hit,
            target_date,
            bool(r.get("reason_trend_pullback", False)),
            bool(r.get("reason_long_pullback", False)),
        )
        if best_row is None:
            skipped.append({**r.to_dict(), "resolved_code": code, "skip_reason": "买入日期无法映射"})
            continue
        best = best_row.to_dict()
        best.update(r.to_dict())
        best["resolved_code"] = code
        best["mapping_mode"] = mode
        rows.append(best)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=list(label_df.columns) + list(pos_txt.columns) + ["resolved_code", "mapping_mode"])
    return out, pd.DataFrame(skipped)


def map_negative_cases(neg_txt: pd.DataFrame, label_df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for _, r in neg_txt.iterrows():
        code = resolve_code(str(r["stock_name"]), mapping)
        if not code:
            skipped.append({**r.to_dict(), "skip_reason": "股票名未映射"})
            continue
        sub = label_df[label_df["code"].astype(str) == str(code)].copy()
        if sub.empty:
            skipped.append({**r.to_dict(), "skip_reason": "候选池无该股票"})
            continue
        target_date = pd.Timestamp(r["b1_date"])
        hit = sub[(sub["signal_date"] == target_date) | (sub["entry_date"] == target_date)].copy()
        if hit.empty:
            hit = sub[
                sub["signal_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
                | sub["entry_date"].between(target_date - pd.Timedelta(days=3), target_date + pd.Timedelta(days=3))
            ].copy()
        best_row, mode = choose_best_label_hit(hit, target_date, False, False)
        if best_row is None:
            skipped.append({**r.to_dict(), "resolved_code": code, "skip_reason": "反例日期无法映射"})
            continue
        best = best_row.to_dict()
        best.update(r.to_dict())
        best["resolved_code"] = code
        best["mapping_mode"] = mode
        rows.append(best)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=list(label_df.columns) + list(neg_txt.columns) + ["resolved_code", "mapping_mode"])
    return out, pd.DataFrame(skipped)


def assign_target_pools(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["txt_core_trend"] = (
        (x["J"] < 13)
        & x["near_trend_pullback"].fillna(False)
        & x["semantic_uptrend_pullback"].fillna(False)
    )
    x["txt_core_long"] = (
        (x["J"] < 13)
        & x["near_long_pullback"].fillna(False)
    )
    x["txt_core_dual"] = x["txt_core_trend"] | x["txt_core_long"]
    x["txt_confirm_bonus"] = (
        x["key_k_support"].fillna(False).astype(int) * 1.0
        + x["half_volume"].fillna(False).astype(int) * 0.8
        + x["double_bull_exist_60"].fillna(False).astype(int) * 0.7
        + x["semi_shrink"].fillna(False).astype(int) * 0.4
    ).astype(float)
    x["pool_txt_trend"] = x["txt_core_trend"]
    x["pool_txt_long"] = x["txt_core_long"]
    x["pool_txt_dual"] = x["txt_core_dual"]
    x["pool_txt_confirmed"] = x["txt_core_dual"] & (x["txt_confirm_bonus"] >= 0.8)
    x["pool_txt_strict"] = x["txt_core_dual"] & (x["txt_confirm_bonus"] >= 1.5)
    return x


def ecdf_score(train_vals: pd.Series, values: pd.Series) -> pd.Series:
    arr = np.sort(pd.to_numeric(train_vals, errors="coerce").dropna().astype(float).to_numpy())
    if len(arr) == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    pos = np.searchsorted(arr, x, side="right")
    score = pos / float(len(arr))
    score[~np.isfinite(x)] = 0.5
    return pd.Series(score, index=values.index, dtype=float)


def add_supervised_scores(candidate_df: pd.DataFrame, pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    x = candidate_df.copy()
    if "split" not in pos_df.columns:
        pos_df = pos_df.copy()
        pos_df["split"] = pd.Series(dtype="object")
    if "split" not in neg_df.columns:
        neg_df = neg_df.copy()
        neg_df["split"] = pd.Series(dtype="object")
    research_pos = pos_df[pos_df["split"] == "research"].copy()
    research_neg = neg_df[neg_df["split"] == "research"].copy()

    sim_cols = [
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_spearman_close_vol_concat_contrast",
        "sim_corr_close_norm",
        "sim_cosine_close_norm",
    ]
    rank_cols: List[str] = []
    for col in [c for c in sim_cols + ["discovery_factor_score", "txt_confirm_bonus"] if c in x.columns]:
        x[f"rank_{col}_txt"] = ecdf_score(x.loc[x["split"] == "research", col], x[col])
        rank_cols.append(f"rank_{col}_txt")

    sep_weights: List[Tuple[str, float]] = []
    for col in sim_cols:
        if (
            col not in x.columns
            or research_pos.empty
            or research_neg.empty
            or col not in research_pos.columns
            or col not in research_neg.columns
        ):
            continue
        pos_mean = float(pd.to_numeric(research_pos[col], errors="coerce").mean())
        neg_mean = float(pd.to_numeric(research_neg[col], errors="coerce").mean())
        delta = pos_mean - neg_mean
        if abs(delta) < 1e-9:
            continue
        sep_weights.append((col, delta))
    if sep_weights:
        pos_parts = []
        neg_parts = []
        total = sum(abs(w) for _, w in sep_weights)
        for col, w in sep_weights:
            part = x[f"rank_{col}_txt"]
            if w >= 0:
                pos_parts.append(part * abs(w))
            else:
                neg_parts.append((1.0 - part) * abs(w))
        x["txt_similarity_score"] = (sum(pos_parts, pd.Series(0.0, index=x.index)) + sum(neg_parts, pd.Series(0.0, index=x.index))) / total
    else:
        x["txt_similarity_score"] = x.get("rank_sim_corr_close_vol_concat_txt", 0.5)

    feature_cols = [
        "discovery_factor_score",
        "close_to_trend",
        "close_to_long",
        "trend_slope_5",
        "long_slope_5",
        "ma5_slope_5",
        "signal_vs_ma5",
        "vol_vs_prev",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "rsi14",
        "ret3",
        "ret5",
        "ret10",
        "ret20",
        "dist_60d_high",
        "txt_confirm_bonus",
        "semantic_uptrend_pullback",
        "semantic_low_cross_pullback",
        "near_trend_pullback",
        "near_long_pullback",
        "key_k_support",
        "half_volume",
        "semi_shrink",
        "double_bull_exist_60",
        "risk_distribution_any_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
        "sim_corr_close_vol_concat",
        "sim_cosine_close_vol_concat",
        "sim_spearman_close_vol_concat_contrast",
    ]
    feature_cols = [c for c in feature_cols if c in x.columns]

    label_rows: List[pd.DataFrame] = []
    if not research_pos.empty:
        tmp = research_pos.copy()
        tmp["label"] = 1
        label_rows.append(tmp)
    if not research_neg.empty:
        tmp = research_neg.copy()
        tmp["label"] = 0
        label_rows.append(tmp)

    auto_neg = x[
        (x["split"] == "research")
        & x["pool_txt_dual"].fillna(False)
        & (~x.index.isin(research_pos.index))
        & (~x.index.isin(research_neg.index))
        & (
            x["negative_30d"].fillna(False)
            | (pd.to_numeric(x["ret_20d"], errors="coerce") < -0.08)
        )
    ].copy()
    auto_neg = auto_neg.sort_values(["ret_20d", "min_close_ret_30"]).head(max(40, len(research_pos) * 4))
    if not auto_neg.empty:
        auto_neg["label"] = 0
        label_rows.append(auto_neg)

    model_map: Dict[str, Any] = {}
    if label_rows:
        train_df = pd.concat(label_rows, ignore_index=True)
        train_df = train_df.copy()
        if not train_df.empty and train_df["label"].nunique() >= 2:
            X = train_df.reindex(columns=feature_cols).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
            y = train_df["label"].to_numpy(dtype=int)
            if HAS_SKLEARN:
                try:
                    lr = LogisticRegression(max_iter=500, class_weight="balanced")
                    lr.fit(X, y)
                    model_map["txt_lr_score"] = (lr, feature_cols)
                except Exception:
                    pass
                try:
                    et = ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    )
                    et.fit(X, y)
                    model_map["txt_et_score"] = (et, feature_cols)
                except Exception:
                    pass
            if HAS_LGB:
                try:
                    clf = lgb.LGBMClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        num_leaves=15,
                        min_child_samples=8,
                        random_state=42,
                    )
                    clf.fit(X, y)
                    model_map["txt_lgb_score"] = (clf, feature_cols)
                except Exception:
                    pass
            if HAS_XGB:
                try:
                    clf = xgb.XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                        eval_metric="logloss",
                        verbosity=0,
                    )
                    clf.fit(X, y)
                    model_map["txt_xgb_score"] = (clf, feature_cols)
                except Exception:
                    pass

    for score_col, (model, cols) in model_map.items():
        X_all = x[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        x[score_col] = model.predict_proba(X_all)[:, 1]
        x[f"rank_{score_col}_txt"] = ecdf_score(x.loc[x["split"] == "research", score_col], x[score_col])

    ml_rank_cols = [c for c in ["rank_txt_xgb_score_txt", "rank_txt_lgb_score_txt", "rank_txt_et_score_txt", "rank_txt_lr_score_txt"] if c in x.columns]
    if ml_rank_cols:
        x["txt_ml_score"] = np.mean(np.column_stack([x[c] for c in ml_rank_cols]), axis=1)
    else:
        fallback_cols = [c for c in ["xgb_full_score", "lgb_full_score", "et_full_score"] if c in x.columns]
        if fallback_cols:
            for col in fallback_cols:
                x[f"rank_{col}_txt"] = ecdf_score(x.loc[x["split"] == "research", col], x[col])
            x["txt_ml_score"] = np.mean(np.column_stack([x[f"rank_{col}_txt"] for col in fallback_cols]), axis=1)
        else:
            x["txt_ml_score"] = 0.5

    x["txt_confirm_score"] = (
        0.45 * ecdf_score(x.loc[x["split"] == "research", "txt_confirm_bonus"], x["txt_confirm_bonus"])
        + 0.25 * x["semantic_low_cross_pullback"].fillna(False).astype(float)
        + 0.20 * x["semantic_uptrend_pullback"].fillna(False).astype(float)
        + 0.10 * x["target_no_risk"].fillna(True).astype(float) if "target_no_risk" in x.columns else 0.0
    )
    x["txt_joint_similarity_score"] = (
        x["rank_discovery_factor_score_txt"] * 0.50
        + x["txt_similarity_score"] * 0.30
        + x["txt_confirm_score"] * 0.20
    )
    x["txt_joint_ml_score"] = (
        x["rank_discovery_factor_score_txt"] * 0.45
        + x["txt_ml_score"] * 0.35
        + x["txt_confirm_score"] * 0.20
    )
    x["txt_joint_full_score"] = (
        x["rank_discovery_factor_score_txt"] * 0.35
        + x["txt_similarity_score"] * 0.25
        + x["txt_ml_score"] * 0.25
        + x["txt_confirm_score"] * 0.15
    )
    return x


def signal_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sample_count": int(len(df))}
    for horizon in [3, 5, 10, 20, 30]:
        out[f"ret_{horizon}d_mean"] = float(pd.to_numeric(df[f"ret_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
        out[f"up_{horizon}d_rate"] = float(pd.to_numeric(df[f"up_{horizon}d"], errors="coerce").mean()) if not df.empty else np.nan
    return out


def evaluate_strategies(df: pd.DataFrame, topn_list: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    strategy_specs = [
        ("baseline", "champion_discovery", "pool_low_cross", "discovery_factor_score"),
        ("similarity", "txt_similarity", "pool_txt_dual", "txt_similarity_score"),
        ("ml", "txt_ml", "pool_txt_confirmed", "txt_ml_score"),
        ("fusion", "txt_joint_similarity", "pool_txt_confirmed", "txt_joint_similarity_score"),
        ("fusion", "txt_joint_ml", "pool_txt_confirmed", "txt_joint_ml_score"),
        ("fusion", "txt_joint_full", "pool_txt_confirmed", "txt_joint_full_score"),
    ]
    for family, variant, pool_name, score_col in strategy_specs:
        if pool_name not in df.columns or score_col not in df.columns:
            continue
        for topn in topn_list:
            for split in ["validation", "final_test"]:
                part = df[(df["split"] == split) & (df[pool_name].fillna(False))].copy()
                if part.empty:
                    continue
                selected = part.sort_values(["signal_date", score_col], ascending=[True, False]).groupby("signal_date").head(topn).copy()
                rows.append(
                    {
                        "split": split,
                        "family": family,
                        "variant": variant,
                        "pool": pool_name,
                        "topn": int(topn),
                        "score_col": score_col,
                        **signal_metrics(selected),
                    }
                )
    return pd.DataFrame(rows)


def choose_validation_family_best(leader_df: pd.DataFrame) -> pd.DataFrame:
    val = leader_df[leader_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()
    val = val.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False])
    return val.groupby("family", as_index=False).head(1).reset_index(drop=True)


def build_final_test_report(leader_df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    final_df = leader_df[leader_df["split"] == "final_test"].copy()
    rows = []
    for _, r in family_best.iterrows():
        hit = final_df[
            (final_df["family"] == r["family"])
            & (final_df["variant"] == r["variant"])
            & (final_df["pool"] == r["pool"])
            & (final_df["topn"] == r["topn"])
        ]
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_selected_rows(df: pd.DataFrame, family_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    final_df = df[df["split"] == "final_test"].copy()
    for _, r in family_best.iterrows():
        part = final_df[final_df[r["pool"]].fillna(False)].copy()
        if part.empty:
            continue
        selected = part.sort_values(["signal_date", r["score_col"]], ascending=[True, False]).groupby("signal_date").head(int(r["topn"])).copy()
        selected["strategy_tag"] = f"{r['family']}_{r['variant']}_{r['pool']}_top{int(r['topn'])}"
        rows.append(selected[["strategy_tag", "code", "signal_date", "entry_date", "entry_price", "split"]])
    if not rows:
        return pd.DataFrame(columns=["strategy_tag", "code", "signal_date", "entry_date", "entry_price", "split"])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    topn_list = parse_topn_list(args.topn_list)

    name_code_map = base_mod.load_name_code_map()
    pos_txt, pos_bad = parse_positive_txt(TXT_POS_PATH)
    neg_txt = parse_negative_txt(TXT_NEG_PATH)
    pos_txt = add_reason_flags(pos_txt, "buy_reason")
    neg_txt = add_reason_flags(neg_txt, "no_buy_reason")
    must_keep_codes = extract_label_codes(pos_txt, neg_txt, name_code_map)
    target_dates_by_code = collect_label_target_dates(pos_txt, neg_txt, name_code_map)

    update_progress(result_dir, "loading_candidates", file_limit=args.file_limit, must_keep_code_count=len(must_keep_codes))
    candidate_df = load_candidate_df(args.base_signal_dir, args.file_limit, must_keep_codes)
    candidate_df = assign_target_pools(candidate_df)
    candidate_df.to_csv(result_dir / "candidate_rows.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "candidate_ready", candidate_count=int(len(candidate_df)))

    split_windows = infer_split_windows(candidate_df)
    label_feature_df = build_label_feature_df(must_keep_codes, split_windows, target_dates_by_code=target_dates_by_code, day_window=5)
    label_feature_df.to_csv(result_dir / "label_feature_rows.csv", index=False, encoding="utf-8-sig")

    pos_df, pos_skipped = map_positive_cases(pos_txt, label_feature_df, name_code_map)
    neg_df, neg_skipped = map_negative_cases(neg_txt, label_feature_df, name_code_map)
    if not pos_df.empty:
        pos_df = add_reason_flags(pos_df, "buy_reason")
    if not neg_df.empty:
        neg_df = add_reason_flags(neg_df, "no_buy_reason")
    pos_df.to_csv(result_dir / "txt_positive_manifest.csv", index=False, encoding="utf-8-sig")
    neg_df.to_csv(result_dir / "txt_negative_manifest.csv", index=False, encoding="utf-8-sig")
    pos_bad.to_csv(result_dir / "txt_positive_bad_rows.csv", index=False, encoding="utf-8-sig")
    pos_skipped.to_csv(result_dir / "txt_positive_skipped.csv", index=False, encoding="utf-8-sig")
    neg_skipped.to_csv(result_dir / "txt_negative_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(
        result_dir,
        "txt_labels_ready",
        positive_count=int(len(pos_df)),
        negative_count=int(len(neg_df)),
        positive_bad_count=int(len(pos_bad)),
        positive_skipped_count=int(len(pos_skipped)),
        negative_skipped_count=int(len(neg_skipped)),
    )

    candidate_df = add_supervised_scores(candidate_df, pos_df, neg_df)
    candidate_df.to_csv(result_dir / "candidate_enriched.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "scores_ready", candidate_count=int(len(candidate_df)))

    leader_df = evaluate_strategies(candidate_df, topn_list)
    leader_df.to_csv(result_dir / "signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    family_best = choose_validation_family_best(leader_df)
    family_best.to_csv(result_dir / "validation_family_best.csv", index=False, encoding="utf-8-sig")
    final_report = build_final_test_report(leader_df, family_best)
    final_report.to_csv(result_dir / "final_test_report.csv", index=False, encoding="utf-8-sig")
    selected_rows = build_selected_rows(candidate_df, family_best)
    selected_rows.to_csv(result_dir / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "base_signal_dir": str(args.base_signal_dir),
        "result_dir": str(result_dir),
        "file_limit": int(args.file_limit),
        "candidate_count": int(len(candidate_df)),
        "positive_count": int(len(pos_df)),
        "negative_count": int(len(neg_df)),
        "positive_bad_count": int(len(pos_bad)),
        "positive_skipped_count": int(len(pos_skipped)),
        "negative_skipped_count": int(len(neg_skipped)),
        "leaderboard_rows": int(len(leader_df)),
        "validation_family_best_rows": int(len(family_best)),
        "final_test_report_rows": int(len(final_report)),
        "selected_signal_count": int(len(selected_rows)),
        "best_validation_row": family_best.iloc[0].to_dict() if not family_best.empty else {},
        "best_final_row": final_report.iloc[0].to_dict() if not final_report.empty else {},
    }
    write_json(result_dir / "summary.json", summary)
    update_progress(result_dir, "finished", selected_signal_count=int(len(selected_rows)))


if __name__ == "__main__":
    main()
