from __future__ import annotations

"""
BRICK case-first 筛选器
======================

目标不是先优化账户层收益，而是先优化：

1. 在对应日期尽可能把你的完美砖型图案例筛出来；
2. 让这些案例尽可能排到更靠前的位置；
3. 再在这个基础上讨论收益。

这条线和 relaxed_fusion 的区别：
- relaxed_fusion 更像“历史回测成功样本 + 因子 + ML”的收益导向排序器；
- case-first 更像“完美案例模板库 + 分型语义 + 早期红砖语义 + 出货风险”的案例导向排序器。
"""

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RELAXED_PATH = ROOT / "utils" / "brickfilter_relaxed_fusion.py"
SIMILARITY_PATH = ROOT / "utils" / "tmp" / "similarity_filter_research.py"
ROLLING_PATH = ROOT / "utils" / "tmp" / "run_brick_buypoint_rolling_compare_v1_20260326.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "brick_optimize" / "brick_case_semantics_v1_20260326.py"
INPUT_DIR = ROOT / "data" / "20260324"
STRATEGY_NAME = "BRICK_CASE_FIRST"
DEFAULT_TOPN = 20
DEFAULT_SEQ_LEN = 5
DEFAULT_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
CASE_SEQ_LENS = [3, 5, 8]
CASE_RESERVED_SLOTS = 10

CASE_WEIGHT = 0.82
HIST_WEIGHT = 0.03
FACTOR_WEIGHT = 0.05
TYPE_WEIGHT = 0.10
CASE_SIM_GATE = 0.62

_MODULE_CACHE: dict[str, Any] = {}
_PERFECT_CASE_FEATURE_CACHE: dict[str, pd.DataFrame] = {}
_PERFECT_TEMPLATE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


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


relaxed = _load_module(RELAXED_PATH, "brick_case_first_relaxed")
sim = _load_module(SIMILARITY_PATH, "brick_case_first_similarity")
rolling = _load_module(ROLLING_PATH, "brick_case_first_rolling")
case_semantics = _load_module(CASE_SEMANTICS_PATH, "brick_case_first_case_semantics")


def strategy_name() -> str:
    return STRATEGY_NAME


def strategy_description() -> str:
    return (
        "BRICK case-first：以完美砖型案例为主模板，相似度和分型优先，"
        "只保留很轻的历史收益信息辅助排序。"
    )


def operation_suggestion() -> str:
    return (
        "这条线优先保证完美案例在对应日期尽可能排到前面，更适合做案例召回和人工复盘，"
        "不是当前正式收益冠军。"
    )


def execution_rule_summary() -> str:
    return (
        "执行规则仍参考当前统一冠军卖法：次日开盘买；次日高开4%及以上不买；"
        "买入当天不能卖；跌破买入K实体低点当日止损；达到5.5%后次日开盘止盈；最多持有3天。"
    )


def _load_perfect_case_feature_df(data_dir: str | Path) -> pd.DataFrame:
    daily_dir = _resolve_daily_dir(data_dir)
    cache_key = str(daily_dir.resolve())
    if cache_key in _PERFECT_CASE_FEATURE_CACHE:
        return _PERFECT_CASE_FEATURE_CACHE[cache_key].copy()
    feat_df = case_semantics.load_case_day_features(case_semantics.PERFECT_CASE_DIR, daily_dir)
    _PERFECT_CASE_FEATURE_CACHE[cache_key] = feat_df.copy()
    return feat_df.copy()


def _build_perfect_case_template_bundle(
    data_dir: str | Path,
    latest_signal_date: pd.Timestamp,
    rep: str = "close_norm",
) -> dict[str, Any]:
    daily_dir = _resolve_daily_dir(data_dir)
    cache_key = (str(daily_dir.resolve()), str(pd.Timestamp(latest_signal_date).date()))
    if cache_key in _PERFECT_TEMPLATE_CACHE:
        return _PERFECT_TEMPLATE_CACHE[cache_key]

    feat_df = _load_perfect_case_feature_df(daily_dir)
    if feat_df.empty:
        bundle = {"all": {}, "by_type": {}}
        _PERFECT_TEMPLATE_CACHE[cache_key] = bundle
        return bundle

    feat_df = feat_df[
        (pd.to_datetime(feat_df["signal_date"]) < pd.Timestamp(latest_signal_date))
        & (
            (pd.to_datetime(feat_df["signal_date"]) < relaxed.EXCLUDE_START)
            | (pd.to_datetime(feat_df["signal_date"]) > relaxed.EXCLUDE_END)
        )
    ].copy()
    if feat_df.empty:
        bundle = {"all": {}, "by_type": {}}
        _PERFECT_TEMPLATE_CACHE[cache_key] = bundle
        return bundle

    all_templates: dict[int, list[np.ndarray]] = {}
    by_type_templates: dict[int, dict[int, list[np.ndarray]]] = {}
    for seq_len in CASE_SEQ_LENS:
        all_templates[seq_len] = relaxed._build_perfect_case_templates(str(daily_dir), pd.Timestamp(latest_signal_date), seq_len, rep)
    for case_type in sorted(pd.to_numeric(feat_df["brick_case_type"], errors="coerce").dropna().astype(int).unique()):
        sub_df = feat_df[feat_df["brick_case_type"] == case_type].copy()
        records: list[dict[str, Any]] = []
        for row in sub_df.itertuples(index=False):
            code = str(getattr(row, "code"))
            file_path = daily_dir / f"{code}.txt"
            if not file_path.exists():
                continue
            df = sim.load_stock_data(str(file_path))
            if df is None or df.empty:
                continue
            x = sim.compute_relaxed_brick_features(df).reset_index(drop=True)
            if x.empty:
                continue
            match = x.index[x["date"] == pd.Timestamp(getattr(row, "signal_date"))]
            if len(match) == 0:
                continue
            signal_idx = int(match[-1])
            seq_map: dict[int, dict[str, np.ndarray]] = {}
            for use_len in CASE_SEQ_LENS:
                if signal_idx < use_len:
                    continue
                seq_map[use_len] = sim.extract_sequence(x.iloc[signal_idx - use_len + 1 : signal_idx + 1], use_len)
            if not seq_map:
                continue
            records.append(
                {
                    "code": code,
                    "date": pd.Timestamp(getattr(row, "signal_date")),
                    "seq_map": seq_map,
                }
            )
        by_type_templates[int(case_type)] = {}
        for seq_len in CASE_SEQ_LENS:
            eligible = [r for r in records if seq_len in r["seq_map"] and rep in r["seq_map"][seq_len]]
            by_type_templates[int(case_type)][seq_len] = sim.build_templates(eligible, seq_len, rep, "recent_100") if eligible else []

    bundle = {"all": all_templates, "by_type": by_type_templates}
    _PERFECT_TEMPLATE_CACHE[cache_key] = bundle
    return bundle


def _resolve_daily_dir(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    if list(data_path.glob("*.txt")):
        return data_path
    normal_dir = data_path / "normal"
    if list(normal_dir.glob("*.txt")):
        return normal_dir
    raise FileNotFoundError(f"未找到可用日线目录: {data_dir}")


def _compute_signal_vs_ma5_proxy(x: pd.DataFrame) -> pd.Series:
    vol_prev_ma5 = pd.to_numeric(x["volume"], errors="coerce").shift(1).rolling(5).mean()
    return pd.to_numeric(x["volume"], errors="coerce") / vol_prev_ma5.replace(0, np.nan)


def _case_first_signal_flag(latest: pd.Series) -> bool:
    if bool(latest.get("signal_relaxed", False)):
        return True
    trend_ok = bool(latest.get("trend_ok", False))
    prev_green_streak = float(latest.get("prev_green_streak", 0.0) or 0.0)
    signal_ret = float(latest.get("signal_ret", 0.0) or 0.0)
    body_ratio = float(latest.get("body_ratio", 0.0) or 0.0)
    upper_shadow_pct = float(latest.get("upper_shadow_pct", 0.0) or 0.0)
    rebound_ratio = float(latest.get("rebound_ratio", 0.0) or 0.0)
    return (
        trend_ok
        and prev_green_streak >= 1.0
        and signal_ret > 0.0
        and body_ratio >= 0.45
        and upper_shadow_pct <= 0.45
        and rebound_ratio >= 1.0
    )


def _normalize_required_lens(required_lens: Optional[list[int] | tuple[int, ...]]) -> list[int]:
    if required_lens:
        return sorted({int(x) for x in required_lens})
    return sorted(set(list(sim.SEQUENCE_LENS) + [DEFAULT_SEQ_LEN] + CASE_SEQ_LENS))


def _record_for_date(
    file_path_str: str,
    target_date_str: str,
    required_lens: Optional[list[int] | tuple[int, ...]] = None,
) -> Optional[dict[str, Any]]:
    target_date = pd.Timestamp(target_date_str)
    df = sim.load_stock_data(file_path_str)
    if df is None or df.empty:
        return None
    x = sim.compute_relaxed_brick_features(df).reset_index(drop=True)
    if x.empty:
        return None
    x["signal_vs_ma5_proxy"] = _compute_signal_vs_ma5_proxy(x)
    match_idx = x.index[x["date"] == target_date]
    if len(match_idx) == 0:
        return None
    signal_idx = int(match_idx[-1])
    if signal_idx < DEFAULT_SEQ_LEN:
        return None
    latest = x.iloc[signal_idx]
    if not _case_first_signal_flag(latest):
        return None

    path = Path(file_path_str)
    daily_root = path.parent.parent if path.parent.name == "normal" else path.parent
    risk_profile = case_semantics.build_risk_profile(daily_root)

    seq_map = {}
    use_lens = _normalize_required_lens(required_lens)
    for seq_len in use_lens:
        seq_map[seq_len] = sim.extract_sequence(x.iloc[signal_idx - seq_len + 1 : signal_idx + 1], seq_len)

    code = str(latest["code"])
    signal_date = pd.Timestamp(latest["date"])
    prev_green_streak = float(latest.get("prev_green_streak", 0.0) or 0.0)
    prev_red_streak = float(latest.get("prev_red_streak", 0.0) or 0.0)
    trend_layer = "high" if bool(latest.get("trend_ok", False)) else "low"
    green4_flag = prev_green_streak == 4
    red4_flag = prev_red_streak == 4

    record = {
        "code": code,
        "signal_date": signal_date,
        "entry_date": signal_date + pd.Timedelta(days=1),
        "exit_date": signal_date + pd.Timedelta(days=3),
        "entry_price": float(latest["close"]),
        "signal_low": float(latest["low"]),
        "signal_open": float(latest["open"]),
        "signal_close": float(latest["close"]),
        "ret1": float(latest.get("ret1", 0.0) or 0.0),
        "ret5": float(latest.get("ret5", 0.0) or 0.0),
        "ret10": float(latest.get("ret10", 0.0) or 0.0),
        "signal_ret": float(latest.get("signal_ret", 0.0) or 0.0),
        "trend_spread": float(latest.get("trend_spread", 0.0) or 0.0),
        "close_to_trend": float(latest.get("close_to_trend", 0.0) or 0.0),
        "close_to_long": float(latest.get("close_to_long", 0.0) or 0.0),
        "ma10_slope_5": float(latest.get("ma10_slope_5", 0.0) or 0.0),
        "ma20_slope_5": float(latest.get("ma20_slope_5", 0.0) or 0.0),
        "brick_red_len": float(latest.get("brick_red_len", 0.0) or 0.0),
        "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[signal_idx] or 0.0),
        "rebound_ratio": float(latest.get("rebound_ratio", 0.0) or 0.0),
        "RSI14": float(latest.get("RSI14", 0.0) or 0.0),
        "MACD_hist": float(latest.get("MACD_hist", 0.0) or 0.0),
        "KDJ_J": float(latest.get("KDJ_J", 0.0) or 0.0),
        "body_ratio": float(latest.get("body_ratio", 0.0) or 0.0),
        "upper_shadow_pct": float(latest.get("upper_shadow_pct", 0.0) or 0.0),
        "lower_shadow_pct": float(latest.get("lower_shadow_pct", 0.0) or 0.0),
        "signal_vs_ma5_proxy": float(latest.get("signal_vs_ma5_proxy", 0.0) or 0.0),
        "prev_green_streak": prev_green_streak,
        "prev_red_streak": prev_red_streak,
        "trend_ok": bool(latest.get("trend_ok", False)),
        "green4_flag": bool(green4_flag),
        "red4_flag": bool(red4_flag),
        "green4_flag_num": float(green4_flag),
        "red4_flag_num": float(red4_flag),
        "trend_layer": trend_layer,
        "trend_layer_num": 1.0 if trend_layer == "high" else 0.0,
        "green4_low_flag": bool(green4_flag and trend_layer == "low"),
        "green4_low_flag_num": float(green4_flag and trend_layer == "low"),
        "candidate_pool": "brick.case_first",
        "pool_bonus": 0.0,
        "seq_map": seq_map,
    }
    record.update(
        case_semantics.enrich_case_type_and_risk_from_values(
            stock_name=None,
            signal_date=signal_date,
            prev_green_streak=prev_green_streak,
            prev_red_streak=prev_red_streak,
            rebound_ratio=float(latest.get("rebound_ratio", 0.0) or 0.0),
            signal_ret=float(latest.get("signal_ret", 0.0) or 0.0),
            upper_shadow_pct=float(latest.get("upper_shadow_pct", 0.0) or 0.0),
            body_ratio=float(latest.get("body_ratio", 0.0) or 0.0),
            close_to_trend=float(latest.get("close_to_trend", 0.0) or 0.0),
            close_to_long=float(latest.get("close_to_long", 0.0) or 0.0),
            feature_df=x,
            signal_idx=signal_idx,
            risk_profile=risk_profile,
        )
    )
    return record


def _records_for_file(
    file_path_str: str,
    target_date_set: Optional[set[str]] = None,
    required_lens: Optional[list[int] | tuple[int, ...]] = None,
) -> list[dict[str, Any]]:
    df = sim.load_stock_data(file_path_str)
    if df is None or df.empty:
        return []
    x = sim.compute_relaxed_brick_features(df).reset_index(drop=True)
    if x.empty:
        return []
    x["signal_vs_ma5_proxy"] = _compute_signal_vs_ma5_proxy(x)
    path = Path(file_path_str)
    daily_root = path.parent.parent if path.parent.name == "normal" else path.parent
    risk_profile = case_semantics.build_risk_profile(daily_root)
    out: list[dict[str, Any]] = []
    use_lens = _normalize_required_lens(required_lens)
    min_len = min(use_lens)
    for signal_idx in range(min_len, len(x)):
        latest = x.iloc[signal_idx]
        signal_date = pd.Timestamp(latest["date"])
        signal_key = signal_date.strftime("%Y-%m-%d")
        if target_date_set is not None and signal_key not in target_date_set:
            continue
        if not _case_first_signal_flag(latest):
            continue
        seq_map = {}
        for seq_len in use_lens:
            if signal_idx < seq_len:
                continue
            seq_map[seq_len] = sim.extract_sequence(x.iloc[signal_idx - seq_len + 1 : signal_idx + 1], seq_len)
        if DEFAULT_SEQ_LEN not in seq_map:
            fallback_len = min(use_lens)
            if fallback_len not in seq_map:
                continue
        code = str(latest["code"])
        prev_green_streak = float(latest.get("prev_green_streak", 0.0) or 0.0)
        prev_red_streak = float(latest.get("prev_red_streak", 0.0) or 0.0)
        trend_layer = "high" if bool(latest.get("trend_ok", False)) else "low"
        green4_flag = prev_green_streak == 4
        red4_flag = prev_red_streak == 4
        record = {
            "code": code,
            "signal_date": signal_date,
            "entry_date": signal_date + pd.Timedelta(days=1),
            "exit_date": signal_date + pd.Timedelta(days=3),
            "entry_price": float(latest["close"]),
            "signal_low": float(latest["low"]),
            "signal_open": float(latest["open"]),
            "signal_close": float(latest["close"]),
            "ret1": float(latest.get("ret1", 0.0) or 0.0),
            "ret5": float(latest.get("ret5", 0.0) or 0.0),
            "ret10": float(latest.get("ret10", 0.0) or 0.0),
            "signal_ret": float(latest.get("signal_ret", 0.0) or 0.0),
            "trend_spread": float(latest.get("trend_spread", 0.0) or 0.0),
            "close_to_trend": float(latest.get("close_to_trend", 0.0) or 0.0),
            "close_to_long": float(latest.get("close_to_long", 0.0) or 0.0),
            "ma10_slope_5": float(latest.get("ma10_slope_5", 0.0) or 0.0),
            "ma20_slope_5": float(latest.get("ma20_slope_5", 0.0) or 0.0),
            "brick_red_len": float(latest.get("brick_red_len", 0.0) or 0.0),
            "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[signal_idx] or 0.0),
            "rebound_ratio": float(latest.get("rebound_ratio", 0.0) or 0.0),
            "RSI14": float(latest.get("RSI14", 0.0) or 0.0),
            "MACD_hist": float(latest.get("MACD_hist", 0.0) or 0.0),
            "KDJ_J": float(latest.get("KDJ_J", 0.0) or 0.0),
            "body_ratio": float(latest.get("body_ratio", 0.0) or 0.0),
            "upper_shadow_pct": float(latest.get("upper_shadow_pct", 0.0) or 0.0),
            "lower_shadow_pct": float(latest.get("lower_shadow_pct", 0.0) or 0.0),
            "signal_vs_ma5_proxy": float(latest.get("signal_vs_ma5_proxy", 0.0) or 0.0),
            "prev_green_streak": prev_green_streak,
            "prev_red_streak": prev_red_streak,
            "trend_ok": bool(latest.get("trend_ok", False)),
            "green4_flag": bool(green4_flag),
            "red4_flag": bool(red4_flag),
            "green4_flag_num": float(green4_flag),
            "red4_flag_num": float(red4_flag),
            "trend_layer": trend_layer,
            "trend_layer_num": 1.0 if trend_layer == "high" else 0.0,
            "green4_low_flag": bool(green4_flag and trend_layer == "low"),
            "green4_low_flag_num": float(green4_flag and trend_layer == "low"),
            "candidate_pool": "brick.case_first",
            "pool_bonus": 0.0,
            "seq_map": seq_map,
        }
        record.update(
            case_semantics.enrich_case_type_and_risk_from_values(
                stock_name=None,
                signal_date=signal_date,
                prev_green_streak=prev_green_streak,
                prev_red_streak=prev_red_streak,
                rebound_ratio=float(latest.get("rebound_ratio", 0.0) or 0.0),
                signal_ret=float(latest.get("signal_ret", 0.0) or 0.0),
                upper_shadow_pct=float(latest.get("upper_shadow_pct", 0.0) or 0.0),
                body_ratio=float(latest.get("body_ratio", 0.0) or 0.0),
                close_to_trend=float(latest.get("close_to_trend", 0.0) or 0.0),
                close_to_long=float(latest.get("close_to_long", 0.0) or 0.0),
                feature_df=x,
                signal_idx=signal_idx,
                risk_profile=risk_profile,
            )
        )
        out.append(record)
    return out


def _records_for_file_chunk(
    file_path_list: list[str],
    target_date_list: list[str],
    required_lens: Optional[list[int] | tuple[int, ...]] = None,
) -> list[dict[str, Any]]:
    target_date_set = set(target_date_list)
    rows: list[dict[str, Any]] = []
    for file_path_str in file_path_list:
        rows.extend(_records_for_file(file_path_str, target_date_set, required_lens))
    return rows


def build_candidates_for_date(
    target_date: pd.Timestamp,
    data_dir: str | Path,
    file_limit: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
    required_lens: Optional[list[int] | tuple[int, ...]] = None,
) -> pd.DataFrame:
    daily_dir = _resolve_daily_dir(data_dir)
    file_paths = sorted(daily_dir.glob("*.txt"))
    if file_limit > 0:
        file_paths = file_paths[:file_limit]
    rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for path in file_paths:
            item = _record_for_date(str(path), str(target_date.date()))
            if item is not None:
                rows.append(item)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for item in executor.map(
                    _record_for_date,
                    [str(path) for path in file_paths],
                    [str(target_date.date())] * len(file_paths),
                    [required_lens] * len(file_paths),
                    chunksize=16,
                ):
                    if item is not None:
                        rows.append(item)
        except Exception:
            raise
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)


def build_candidate_cache_for_dates(
    data_dir: str | Path,
    target_dates: list[pd.Timestamp] | list[str],
    file_limit: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
    required_lens: Optional[list[int] | tuple[int, ...]] = None,
) -> pd.DataFrame:
    daily_dir = _resolve_daily_dir(data_dir)
    file_paths = sorted(daily_dir.glob("*.txt"))
    if file_limit > 0:
        file_paths = file_paths[:file_limit]
    target_date_set = {pd.Timestamp(x).strftime("%Y-%m-%d") for x in target_dates}
    rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for path in file_paths:
            rows.extend(_records_for_file(str(path), target_date_set, required_lens))
    else:
        file_path_strs = [str(path) for path in file_paths]
        chunk_size = max(8, len(file_path_strs) // max_workers)
        chunks = [file_path_strs[i : i + chunk_size] for i in range(0, len(file_path_strs), chunk_size)]
        target_date_list = sorted(target_date_set)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for items in executor.map(
                _records_for_file_chunk,
                chunks,
                [target_date_list] * len(chunks),
                [required_lens] * len(chunks),
                chunksize=1,
            ):
                if items:
                    rows.extend(items)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)


def score_candidates_for_date(
    target_date: pd.Timestamp,
    candidate_df: pd.DataFrame,
    data_dir: str | Path,
    topn: int = DEFAULT_TOPN,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    _, hist_df = relaxed._load_best_and_history()
    train_df, val_df = relaxed._build_train_val_frames(hist_df, target_date)
    if train_df.empty or val_df.empty:
        return pd.DataFrame()
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    q1, q2 = relaxed._turn_layer_thresholds(trainval_df)
    trainval_df = relaxed._apply_turn_strength_features(trainval_df, q1, q2)
    current_df = relaxed._apply_turn_strength_features(candidate_df, q1, q2)

    factor_model = rolling.build_factor_model(trainval_df)
    trainval_with_factor = rolling.apply_factor_model(trainval_df, factor_model)
    current_with_factor = rolling.apply_factor_model(current_df, factor_model)

    stage_records = relaxed._build_stage_records(current_with_factor)
    sim_cfg = sim.BaseConfig(
        builder="recent_100",
        seq_len=DEFAULT_SEQ_LEN,
        rep="close_norm",
        scorer="pipeline_corr_dtw",
    )
    hist_best = {"builder": "recent_100", "seq_len": DEFAULT_SEQ_LEN, "rep": "close_norm", "scorer": "pipeline_corr_dtw"}
    hist_templates = relaxed._build_similarity_templates(trainval_with_factor, hist_best)
    hist_sim_df = sim.build_scored_df_normal(stage_records, hist_templates, sim_cfg).rename(
        columns={"date": "signal_date", "score": "sim_score"}
    )
    current_with_factor = current_with_factor.merge(
        hist_sim_df[["code", "signal_date", "sim_score"]], on=["code", "signal_date"], how="left"
    )
    current_with_factor["sim_score"] = pd.to_numeric(current_with_factor["sim_score"], errors="coerce").fillna(-1.0)

    template_bundle = _build_perfect_case_template_bundle(data_dir, target_date, rep="close_norm")
    current_with_factor["perfect_case_sim_score"] = -1.0
    current_with_factor["same_type_case_sim_score"] = -1.0
    key_cols = ["code", "signal_date"]
    for seq_len in CASE_SEQ_LENS:
        cfg = sim.BaseConfig(builder="recent_100", seq_len=int(seq_len), rep="close_norm", scorer="pipeline_corr_dtw")
        all_templates = template_bundle["all"].get(int(seq_len), [])
        if all_templates:
            all_sim_df = sim.build_scored_df_normal(stage_records, all_templates, cfg).rename(
                columns={"date": "signal_date", "score": f"perfect_case_sim_{seq_len}"}
            )
            current_with_factor = current_with_factor.merge(
                all_sim_df[["code", "signal_date", f"perfect_case_sim_{seq_len}"]],
                on=key_cols,
                how="left",
            )
            score_col = pd.to_numeric(current_with_factor[f"perfect_case_sim_{seq_len}"], errors="coerce").fillna(-1.0)
            current_with_factor["perfect_case_sim_score"] = np.maximum(current_with_factor["perfect_case_sim_score"], score_col)
        for case_type in sorted(pd.to_numeric(current_with_factor["brick_case_type"], errors="coerce").dropna().astype(int).unique()):
            type_templates = template_bundle["by_type"].get(int(case_type), {}).get(int(seq_len), [])
            if not type_templates:
                continue
            mask = pd.to_numeric(current_with_factor["brick_case_type"], errors="coerce").fillna(4).astype(int) == int(case_type)
            if not mask.any():
                continue
            subset_records = relaxed._build_stage_records(current_with_factor.loc[mask].copy())
            type_sim_df = sim.build_scored_df_normal(subset_records, type_templates, cfg).rename(
                columns={"date": "signal_date", "score": f"same_type_case_sim_{seq_len}"}
            )
            current_with_factor = current_with_factor.merge(
                type_sim_df[["code", "signal_date", f"same_type_case_sim_{seq_len}"]],
                on=key_cols,
                how="left",
            )
        type_col = f"same_type_case_sim_{seq_len}"
        if type_col in current_with_factor.columns:
            type_score_col = pd.to_numeric(current_with_factor[type_col], errors="coerce").fillna(-1.0)
            current_with_factor["same_type_case_sim_score"] = np.maximum(
                current_with_factor["same_type_case_sim_score"], type_score_col
            )

    current_with_factor["perfect_case_sim_score"] = pd.to_numeric(
        current_with_factor["perfect_case_sim_score"], errors="coerce"
    ).fillna(-1.0)
    current_with_factor["same_type_case_sim_score"] = pd.to_numeric(
        current_with_factor["same_type_case_sim_score"], errors="coerce"
    ).fillna(-1.0)

    risk_penalty = (
        pd.to_numeric(current_with_factor["risk_distribution_recent_20"], errors="coerce").fillna(0.0) * 0.15
        + pd.to_numeric(current_with_factor["risk_distribution_recent_30"], errors="coerce").fillna(0.0) * 0.10
        + pd.to_numeric(current_with_factor["risk_distribution_recent_60"], errors="coerce").fillna(0.0) * 0.05
    )
    type_raw = (
        pd.to_numeric(current_with_factor["brick_case_type_score"], errors="coerce").fillna(0.45)
        + pd.to_numeric(current_with_factor["early_red_stage_flag_num"], errors="coerce").fillna(0.0) * 0.25
        - risk_penalty
    )

    case_rank = rolling.normalize_rank(current_with_factor["perfect_case_sim_score"])
    hist_rank = rolling.normalize_rank(current_with_factor["sim_score"])
    factor_rank = pd.to_numeric(current_with_factor["factor_score"], errors="coerce").fillna(0.0)
    type_rank = rolling.normalize_rank(type_raw)
    same_type_rank = rolling.normalize_rank(current_with_factor["same_type_case_sim_score"])

    current_with_factor["case_rank"] = case_rank
    current_with_factor["hist_rank"] = hist_rank
    current_with_factor["type_rank"] = type_rank
    current_with_factor["same_type_rank"] = same_type_rank
    current_with_factor["case_primary_score"] = (
        0.65 * case_rank
        + 0.20 * same_type_rank
        + 0.10 * type_rank
        + 0.05 * factor_rank
    )
    current_with_factor["rank_score"] = (
        CASE_WEIGHT * case_rank
        + HIST_WEIGHT * hist_rank
        + FACTOR_WEIGHT * factor_rank
        + TYPE_WEIGHT * type_rank
    )

    selected = current_with_factor[
        (current_with_factor["perfect_case_sim_score"] >= CASE_SIM_GATE)
        | (current_with_factor["same_type_case_sim_score"] >= CASE_SIM_GATE - 0.03)
        | (
            (pd.to_numeric(current_with_factor["brick_case_type_score"], errors="coerce").fillna(0.0) >= 0.92)
            & (pd.to_numeric(current_with_factor["early_red_stage_flag_num"], errors="coerce").fillna(0.0) >= 1.0)
            & (pd.to_numeric(current_with_factor["sim_score"], errors="coerce").fillna(-1.0) >= 0.70)
        )
    ].copy()
    if selected.empty:
        return selected
    selected["sort_signal_date"] = selected["signal_date"]
    primary_pick = (
        selected.sort_values(
            ["sort_signal_date", "case_primary_score", "perfect_case_sim_score", "same_type_case_sim_score", "code"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .groupby("sort_signal_date", group_keys=False)
        .head(min(int(topn), CASE_RESERVED_SLOTS))
        .reset_index(drop=True)
    )
    primary_keys = set(zip(primary_pick["code"], pd.to_datetime(primary_pick["signal_date"])))
    remaining = selected[
        ~selected.apply(lambda r: (r["code"], pd.Timestamp(r["signal_date"])) in primary_keys, axis=1)
    ].copy()
    secondary_pick = (
        remaining.sort_values(
            ["sort_signal_date", "rank_score", "case_primary_score", "code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .groupby("sort_signal_date", group_keys=False)
        .head(max(0, int(topn) - min(int(topn), CASE_RESERVED_SLOTS)))
        .reset_index(drop=True)
    )
    final = (
        pd.concat([primary_pick, secondary_pick], ignore_index=True)
        .sort_values(
            ["signal_date", "case_primary_score", "rank_score", "code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .groupby("signal_date", group_keys=False)
        .head(int(topn))
        .reset_index(drop=True)
    )
    return final


def _format_output(df: pd.DataFrame) -> list[list[str]]:
    out: list[list[str]] = []
    for row in df.itertuples(index=False):
        code = str(getattr(row, "code")).split("#")[-1]
        stop_ref = round(float(min(getattr(row, "signal_open"), getattr(row, "signal_close"))), 3)
        close_price = round(float(getattr(row, "signal_close")), 3)
        rank_score = round(float(getattr(row, "rank_score")), 4)
        note = (
            f"case:{float(getattr(row, 'perfect_case_sim_score')):.3f} "
            f"same:{float(getattr(row, 'same_type_case_sim_score', -1.0)):.3f} "
            f"hist:{float(getattr(row, 'sim_score')):.3f} "
            f"type:{int(getattr(row, 'brick_case_type', 4))} "
            f"factor:{float(getattr(row, 'factor_score')):.3f} "
            f"r20:{float(getattr(row, 'risk_distribution_recent_20', 0.0)):.2f}"
        )
        out.append([code, f"{stop_ref:.3f}", f"{close_price:.3f}", f"{rank_score:.4f}", note])
    return out


def scan_dir(data_dir: str, hold_list=None, file_limit: int = 0, max_workers: int = DEFAULT_MAX_WORKERS) -> list[list[str]]:
    del hold_list
    daily_dir = _resolve_daily_dir(data_dir)
    current_df = relaxed._build_current_candidates(str(daily_dir), file_limit=file_limit, max_workers=max_workers)
    if current_df.empty:
        return []
    latest_signal_date = pd.Timestamp(current_df["signal_date"].max())
    current_df = build_candidates_for_date(latest_signal_date, daily_dir, file_limit=file_limit, max_workers=max_workers)
    if current_df.empty:
        return []
    selected = score_candidates_for_date(latest_signal_date, current_df, daily_dir, topn=DEFAULT_TOPN)
    if selected.empty:
        return []
    return _format_output(selected)


def print_selected(selected: list[list[str]]) -> None:
    print(f"【{STRATEGY_NAME}】")
    print(strategy_description())
    print(operation_suggestion())
    print(execution_rule_summary())
    if not selected:
        print("当前没有筛出 case-first 候选。")
        return
    print(f"共筛选出 {len(selected)} 只股票：")
    for code, stop_ref, close_price, score, note in selected:
        print(
            f"股票代码{code:<6} | 止损价(参考)：{stop_ref:<8} | "
            f"当日收盘价：{close_price:<8} | 综合分：{score:<8} | 备注：{note}"
        )


def main() -> None:
    selected = scan_dir(str(INPUT_DIR), max_workers=DEFAULT_MAX_WORKERS)
    print_selected(selected)


if __name__ == "__main__":
    main()
