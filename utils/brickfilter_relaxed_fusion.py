from __future__ import annotations

"""
BRICK relaxed_fusion 实盘封装
=============================

这不是 `brick_filter.py` 那种“单只股票规则检查器”，而是：

1. 先在当天全市场里找出 `relaxed_base` 候选；
2. 再用 BRICK 综合实验里验证过的：
   - 相似度
   - 因子
   - RandomForest
   做融合评分；
3. 最后按日内总分取 `top10`。

为什么要单独做成一个模块
------------------------
`relaxed_fusion` 的语义不是“看单票像不像”，而是“同一天候选里谁更强”。
所以它不能硬塞进 `main.py` 的单文件 `check()` 流程里，不然会把：

- 全市场相对排序
- 单票绝对条件判断

混成一件事，语义会坏掉。

当前实现口径
-----------
这里做的是“当前扫描日可直接调用的实盘版”：

- 买点语义：`brick.relaxed_base`
- 相似度模板：沿用综合实验冠军配置
  - `sample_300`
  - `len21`
  - `close_norm`
  - `pipeline_corr_dtw`
- 融合权重：
  - `sim 0.4`
  - `factor 0.2`
  - `ml 0.4`
- 训练方式：
  - 以当前扫描日之前的数据为训练样本
  - 默认剔除 `2015-06-01 ~ 2024-09-30`
  - 最近 `24个月 train + 6个月 validation` 训练实时模型

统一执行规则
-----------
这条买点在项目里对应的当前冠军执行，不是“看到名单后随便买卖”，而是：

- `signal_date` 出信号；
- `entry_date = signal_date + 1`，次日开盘买入；
- 如果 `entry_date` 高开达到 `4%` 及以上，则不买；
- 买入当天不能卖；
- 若后续跌破买入 K 线实体低点，则当日止损；
- 若后续相对买入价最高达到 `5.5%`，则在次日开盘止盈；
- 最长持有 `3天`。

注意
----
这里在扫描阶段展示的止损价，是“信号 K 实体低点”的参考价：

- `min(signal_open, signal_close)`

而我们当前账户层冠军卖法的真实止损语义是：

- 买入后，如果后续跌破“买入 K 实体低点”则当日止损

也就是说，主扫描里这里只能给你一个“信号日参考止损位”，
不能把它误当成买入后的真实执行价。
"""

import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
BEST_RESULT_DIR = ROOT / "results" / "brick_comprehensive_lab_full_20260325_r1"
SIMILARITY_PATH = ROOT / "utils" / "tmp" / "similarity_filter_research.py"
ROLLING_PATH = ROOT / "utils" / "tmp" / "run_brick_buypoint_rolling_compare_v1_20260326.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "tmp" / "brick_case_semantics_v1_20260326.py"
INPUT_DIR = ROOT / "data" / "20260324" / "normal"
PERFECT_CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
STRATEGY_NAME = "BRICKFILTER_RELAXED_FUSION"

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
DEFAULT_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
PERFECT_CASE_WEIGHT = 0.35

_MODULE_CACHE: dict[str, Any] = {}
_HISTORICAL_CACHE: Optional[tuple[dict[str, Any], pd.DataFrame]] = None
_PERFECT_CASE_TEMPLATE_CACHE: dict[tuple[str, str, int, str], list[np.ndarray]] = {}


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


sim = _load_module(SIMILARITY_PATH, "brick_relaxed_fusion_similarity")
rolling = _load_module(ROLLING_PATH, "brick_relaxed_fusion_rolling")
case_semantics = _load_module(CASE_SEMANTICS_PATH, "brick_relaxed_fusion_case_semantics")


def strategy_description() -> str:
    return (
        "BRICK relaxed_fusion：先找 relaxed_base 候选，再用历史相似度 + 完美案例相似度 + "
        "因子 + RandomForest 做全市场融合排序，默认取当日 top10。"
    )


def operation_suggestion() -> str:
    return (
        "这条线更适合当成 BRICK 的二次排序器使用，而不是单票绝对规则过滤器；"
        "如果主扫描里出现结果，优先把它和 formal_best 并排看；"
        "当前版本会额外参考完美砖型案例模板，优先提升更像你主观高质量案例的候选。"
    )


def strategy_name() -> str:
    return STRATEGY_NAME


def execution_rule_summary() -> str:
    return (
        "执行规则：次日开盘买；次日高开4%及以上不买；买入当天不能卖；"
        "跌破买入K实体低点当日止损；达到5.5%后次日开盘止盈；最多持有3天。"
    )


def _safe_eval_seq_map(text: str) -> dict[int, dict[str, np.ndarray]]:
    if not isinstance(text, str) or not text.strip():
        return {}
    return eval(text, {"__builtins__": {}}, {"array": np.array, "nan": np.nan})  # noqa: S307


def _load_best_and_history() -> tuple[dict[str, Any], pd.DataFrame]:
    global _HISTORICAL_CACHE
    if _HISTORICAL_CACHE is not None:
        best, df = _HISTORICAL_CACHE
        return best, df.copy()

    best = json.loads((BEST_RESULT_DIR / "best_config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(
        BEST_RESULT_DIR / "candidate_scored.csv",
        parse_dates=["signal_date", "entry_date", "exit_date"],
    )
    df = df[df["candidate_pool"].astype(str) == str(best["candidate_pool"])].copy()
    df = df[(df["signal_date"] < EXCLUDE_START) | (df["signal_date"] > EXCLUDE_END)].copy()
    for col in [
        "sim_score",
        "factor_score",
        "pool_bonus",
        "signal_vs_ma5_proxy",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "ret1",
        "ret5",
        "brick_green_len_prev",
        "brick_red_len",
        "rebound_ratio",
        "RSI14",
        "MACD_hist",
        "body_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "green4_flag_num",
        "green4_low_flag_num",
        "red4_flag_num",
        "turn_strength_layer_num",
        "trend_layer_num",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    enriched = []
    for row in df.itertuples(index=False):
        extra = case_semantics.enrich_case_type_and_risk_from_values(
            prev_green_streak=float(getattr(row, "prev_green_streak", 0.0) or 0.0),
            prev_red_streak=float(getattr(row, "prev_red_streak", 0.0) or 0.0),
            rebound_ratio=float(getattr(row, "rebound_ratio", 0.0) or 0.0),
            signal_ret=float(getattr(row, "signal_ret", 0.0) or 0.0),
            upper_shadow_pct=float(getattr(row, "upper_shadow_pct", 0.0) or 0.0),
            body_ratio=float(getattr(row, "body_ratio", 0.0) or 0.0),
            close_to_trend=float(getattr(row, "close_to_trend", 0.0) or 0.0),
            close_to_long=float(getattr(row, "close_to_long", 0.0) or 0.0),
        )
        enriched.append(extra)
    if enriched:
        extra_df = pd.DataFrame(enriched)
        for col in extra_df.columns:
            df[col] = extra_df[col].to_numpy()
        risk_penalty = (
            pd.to_numeric(df["risk_distribution_recent_20"], errors="coerce").fillna(0.0) * 0.08
            + pd.to_numeric(df["risk_distribution_recent_30"], errors="coerce").fillna(0.0) * 0.05
            + pd.to_numeric(df["risk_distribution_recent_60"], errors="coerce").fillna(0.0) * 0.03
        )
        case_bonus = (
            (pd.to_numeric(df["brick_case_type_score"], errors="coerce").fillna(0.45) - 0.45) * 0.12
            + pd.to_numeric(df["early_red_stage_flag_num"], errors="coerce").fillna(0.0) * 0.04
        )
        df["pool_bonus"] = pd.to_numeric(df["pool_bonus"], errors="coerce").fillna(0.0) + case_bonus - risk_penalty
    _HISTORICAL_CACHE = (best, df.copy())
    return best, df


def _build_name_code_map(data_dir: str) -> dict[str, str]:
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


def _parse_perfect_cases() -> pd.DataFrame:
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    rows: list[dict[str, Any]] = []
    for path in sorted(PERFECT_CASE_DIR.glob("*.png")):
        if "反例" in path.name or path.stem == "案例图":
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
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _compute_signal_vs_ma5_proxy(x: pd.DataFrame) -> pd.Series:
    vol_prev_ma5 = pd.to_numeric(x["volume"], errors="coerce").shift(1).rolling(5).mean()
    return pd.to_numeric(x["volume"], errors="coerce") / vol_prev_ma5.replace(0, np.nan)


def _current_record_from_file(file_path_str: str) -> Optional[dict[str, Any]]:
    df = sim.load_stock_data(file_path_str)
    if df is None or df.empty:
        return None
    x = sim.compute_relaxed_brick_features(df).reset_index(drop=True)
    if x.empty:
        return None
    x["signal_vs_ma5_proxy"] = _compute_signal_vs_ma5_proxy(x)
    latest_idx = len(x) - 1
    if latest_idx < max(sim.SEQUENCE_LENS):
        return None
    latest = x.iloc[latest_idx]
    if not bool(latest.get("signal_relaxed", False)):
        return None
    risk_profile = case_semantics.build_risk_profile(Path(file_path_str).parent.parent if Path(file_path_str).parent.name == "normal" else Path(file_path_str).parent)

    seq_map: dict[int, dict[str, np.ndarray]] = {}
    for seq_len in sim.SEQUENCE_LENS:
        seq_map[seq_len] = sim.extract_sequence(x.iloc[latest_idx - seq_len + 1 : latest_idx + 1], seq_len)

    code = str(latest["code"])
    signal_date = pd.Timestamp(latest["date"])
    trend_layer = "high" if bool(latest.get("trend_ok", False)) else "low"
    prev_green_streak = float(latest.get("prev_green_streak", 0.0) or 0.0)
    prev_red_streak = float(latest.get("prev_red_streak", 0.0) or 0.0)
    green4_flag = prev_green_streak == 4
    red4_flag = prev_red_streak == 4

    record = {
        "code": code,
        "signal_idx": int(latest_idx),
        "signal_date": signal_date,
        "entry_date": signal_date + timedelta(days=1),
        "exit_date": signal_date + timedelta(days=3),
        "entry_price": float(latest["close"]),
        "signal_low": float(latest["low"]),
        "signal_open": float(latest["open"]),
        "signal_close": float(latest["close"]),
        "label": 0,
        "result": "pending",
        "ret": 0.0,
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
        "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[latest_idx] or 0.0),
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
        "candidate_pool": "brick.relaxed_base",
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
            signal_idx=latest_idx,
            risk_profile=risk_profile,
        )
    )
    risk_penalty = (
        float(record["risk_distribution_recent_20"]) * 0.08
        + float(record["risk_distribution_recent_30"]) * 0.05
        + float(record["risk_distribution_recent_60"]) * 0.03
    )
    case_bonus = (float(record["brick_case_type_score"]) - 0.45) * 0.12 + float(record["early_red_stage_flag_num"]) * 0.04
    record["pool_bonus"] = case_bonus - risk_penalty
    return record


def _build_current_candidates(data_dir: str, file_limit: int = 0, max_workers: int = DEFAULT_MAX_WORKERS) -> pd.DataFrame:
    file_paths = sorted(Path(data_dir).glob("*.txt"))
    if file_limit > 0:
        file_paths = file_paths[:file_limit]
    if not file_paths:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for path in file_paths:
            item = _current_record_from_file(str(path))
            if item is not None:
                rows.append(item)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for item in executor.map(_current_record_from_file, [str(path) for path in file_paths], chunksize=16):
                    if item is not None:
                        rows.append(item)
        except Exception:
            for path in file_paths:
                item = _current_record_from_file(str(path))
                if item is not None:
                    rows.append(item)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "code"]).reset_index(drop=True)


def _build_train_val_frames(hist_df: pd.DataFrame, latest_signal_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_df = hist_df[hist_df["signal_date"] < latest_signal_date].copy()
    if hist_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    val_start = latest_signal_date - DateOffset(months=6)
    train_start = latest_signal_date - DateOffset(months=30)
    train_df = hist_df[(hist_df["signal_date"] >= train_start) & (hist_df["signal_date"] < val_start)].copy()
    val_df = hist_df[(hist_df["signal_date"] >= val_start) & (hist_df["signal_date"] < latest_signal_date)].copy()

    # 如果最近 30 个月不足以形成稳定训练，回退到所有历史的 80/20 切分。
    if train_df.empty or val_df.empty or train_df["label"].nunique() < 2:
        hist_df = hist_df.sort_values("signal_date").reset_index(drop=True)
        cut = max(1, int(len(hist_df) * 0.8))
        train_df = hist_df.iloc[:cut].copy()
        val_df = hist_df.iloc[cut:].copy()
    return train_df, val_df


def _turn_layer_thresholds(trainval_df: pd.DataFrame) -> tuple[float, float]:
    source = pd.to_numeric(trainval_df["brick_green_len_prev"], errors="coerce").fillna(0.0)
    q1 = float(source.quantile(0.33))
    q2 = float(source.quantile(0.67))
    if abs(q2 - q1) < 1e-12:
        q1 = float(source.median())
        q2 = q1
    return q1, q2


def _apply_turn_strength_features(df: pd.DataFrame, q1: float, q2: float) -> pd.DataFrame:
    out = df.copy()
    source = pd.to_numeric(out["brick_green_len_prev"], errors="coerce").fillna(0.0)
    layer = np.where(source <= q1, "low", np.where(source >= q2, "high", "mid"))
    out["turn_strength_layer"] = layer
    out["turn_strength_layer_num"] = np.where(layer == "low", 0.0, np.where(layer == "mid", 0.5, 1.0))
    return out


def _build_similarity_templates(
    hist_df: pd.DataFrame,
    best: dict[str, Any],
) -> list[np.ndarray]:
    success_df = hist_df[hist_df["label"] == 1].copy()
    if success_df.empty:
        return []
    builder = str(best["builder"])
    if builder == "recent_100":
        success_df = success_df.sort_values("signal_date").tail(100).copy()
    elif builder == "sample_300":
        success_df = success_df.sample(n=min(300, len(success_df)), random_state=42).copy()
    # cluster_100 保持全量，交给原始模板逻辑做聚类。

    records: list[dict[str, Any]] = []
    target_seq_len = int(best["seq_len"])
    target_rep = str(best["rep"])
    for row in success_df.itertuples(index=False):
        seq_map = _safe_eval_seq_map(getattr(row, "seq_map", ""))
        if not seq_map:
            continue
        if target_seq_len not in seq_map:
            larger_lens = sorted(k for k in seq_map if int(k) > target_seq_len and target_rep in seq_map[k])
            if larger_lens:
                use_len = int(larger_lens[0])
                use_vec = np.asarray(seq_map[use_len][target_rep], dtype=float)
                if target_rep in ("close_norm", "vol_norm") and len(use_vec) >= target_seq_len:
                    tail = use_vec[-target_seq_len:]
                    lo = float(np.nanmin(tail))
                    hi = float(np.nanmax(tail))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        rebuilt = (tail - lo) / (hi - lo)
                    else:
                        rebuilt = np.zeros(target_seq_len, dtype=float)
                    seq_map[target_seq_len] = {target_rep: rebuilt}
        records.append(
            {
                "code": getattr(row, "code"),
                "date": pd.Timestamp(getattr(row, "signal_date")),
                "seq_map": seq_map,
            }
        )
    cfg = sim.BaseConfig(
        builder=str(best["builder"]),
        seq_len=target_seq_len,
        rep=target_rep,
        scorer=str(best["scorer"]),
    )
    return sim.build_templates(records, cfg.seq_len, cfg.rep, cfg.builder)


def _build_perfect_case_templates(
    data_dir: str,
    latest_signal_date: pd.Timestamp,
    seq_len: int,
    rep: str,
) -> list[np.ndarray]:
    cache_key = (str(Path(data_dir).resolve()), str(latest_signal_date.date()), int(seq_len), str(rep))
    if cache_key in _PERFECT_CASE_TEMPLATE_CACHE:
        return _PERFECT_CASE_TEMPLATE_CACHE[cache_key]

    case_df = _parse_perfect_cases()
    if case_df.empty:
        _PERFECT_CASE_TEMPLATE_CACHE[cache_key] = []
        return []

    case_df = case_df[
        (case_df["signal_date"] < latest_signal_date)
        & ((case_df["signal_date"] < EXCLUDE_START) | (case_df["signal_date"] > EXCLUDE_END))
    ].copy()
    if case_df.empty:
        _PERFECT_CASE_TEMPLATE_CACHE[cache_key] = []
        return []

    name_code_map = _build_name_code_map(data_dir)
    case_df["code"] = case_df["stock_name"].map(name_code_map)
    case_df = case_df.dropna(subset=["code"]).copy()
    if case_df.empty:
        _PERFECT_CASE_TEMPLATE_CACHE[cache_key] = []
        return []

    records: list[dict[str, Any]] = []
    required_lens = sorted(set(list(sim.SEQUENCE_LENS) + [int(seq_len)]))
    min_required = max(required_lens)
    normal_dir = Path(data_dir)
    for row in case_df.itertuples(index=False):
        code = str(getattr(row, "code"))
        file_path = normal_dir / f"{code}.txt"
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
        if signal_idx < min_required:
            continue
        seq_map: dict[int, dict[str, np.ndarray]] = {}
        for use_len in required_lens:
            seq_map[use_len] = sim.extract_sequence(x.iloc[signal_idx - use_len + 1 : signal_idx + 1], use_len)
        records.append(
            {
                "code": code,
                "date": pd.Timestamp(getattr(row, "signal_date")),
                "seq_map": seq_map,
            }
        )
    if not records:
        _PERFECT_CASE_TEMPLATE_CACHE[cache_key] = []
        return []

    templates = sim.build_templates(records, seq_len, rep, "recent_100")
    _PERFECT_CASE_TEMPLATE_CACHE[cache_key] = templates
    return templates


def _build_stage_records(current_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in current_df.itertuples(index=False):
        records.append(
            {
                "code": getattr(row, "code"),
                "date": pd.Timestamp(getattr(row, "signal_date")),
                "label": 0,
                "result": "pending",
                "ret": 0.0,
                "entry_date": pd.Timestamp(getattr(row, "entry_date")),
                "exit_date": pd.Timestamp(getattr(row, "exit_date")),
                "entry_price": float(getattr(row, "entry_price")),
                "ret1": float(getattr(row, "ret1")),
                "ret5": float(getattr(row, "ret5")),
                "ret10": float(getattr(row, "ret10")),
                "signal_ret": float(getattr(row, "signal_ret")),
                "trend_spread": float(getattr(row, "trend_spread")),
                "close_to_trend": float(getattr(row, "close_to_trend")),
                "close_to_long": float(getattr(row, "close_to_long")),
                "ma10_slope_5": float(getattr(row, "ma10_slope_5")),
                "ma20_slope_5": float(getattr(row, "ma20_slope_5")),
                "brick_red_len": float(getattr(row, "brick_red_len")),
                "brick_green_len_prev": float(getattr(row, "brick_green_len_prev")),
                "rebound_ratio": float(getattr(row, "rebound_ratio")),
                "RSI14": float(getattr(row, "RSI14")),
                "MACD_hist": float(getattr(row, "MACD_hist")),
                "KDJ_J": float(getattr(row, "KDJ_J")),
                "body_ratio": float(getattr(row, "body_ratio")),
                "upper_shadow_pct": float(getattr(row, "upper_shadow_pct")),
                "lower_shadow_pct": float(getattr(row, "lower_shadow_pct")),
                "seq_map": getattr(row, "seq_map"),
            }
        )
    return records


def _format_output(df: pd.DataFrame) -> list[list[str]]:
    out: list[list[str]] = []
    for row in df.itertuples(index=False):
        code = str(getattr(row, "code")).split("#")[-1]
        stop_ref = round(float(min(getattr(row, "signal_open"), getattr(row, "signal_close"))), 3)
        close_price = round(float(getattr(row, "signal_close")), 3)
        rank_score = round(float(getattr(row, "rank_score")), 4)
        note = (
            f"sim:{float(getattr(row, 'sim_score')):.3f} "
            f"case:{float(getattr(row, 'perfect_case_sim_score')):.3f} "
            f"factor:{float(getattr(row, 'factor_score')):.3f} "
            f"ml:{float(getattr(row, 'ml_score')):.3f} "
            f"type:{int(getattr(row, 'brick_case_type', 4))} "
            f"r20:{float(getattr(row, 'risk_distribution_recent_20', 0.0)):.2f}"
        )
        out.append([code, f"{stop_ref:.3f}", f"{close_price:.3f}", f"{rank_score:.4f}", note])
    return out


def scan_dir(
    data_dir: str,
    hold_list=None,
    file_limit: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[list[str]]:
    del hold_list
    best, hist_df = _load_best_and_history()
    current_df = _build_current_candidates(data_dir, file_limit=file_limit, max_workers=max_workers)
    if current_df.empty:
        return []

    latest_signal_date = pd.Timestamp(current_df["signal_date"].max())
    current_df = current_df[current_df["signal_date"] == latest_signal_date].copy().reset_index(drop=True)

    train_df, val_df = _build_train_val_frames(hist_df, latest_signal_date)
    if train_df.empty or val_df.empty:
        return []

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    q1, q2 = _turn_layer_thresholds(trainval_df)
    current_df = _apply_turn_strength_features(current_df, q1, q2)
    if "turn_strength_layer_num" not in trainval_df.columns:
        trainval_df = _apply_turn_strength_features(trainval_df, q1, q2)

    factor_model = rolling.build_factor_model(trainval_df)
    trainval_with_factor = rolling.apply_factor_model(trainval_df, factor_model)
    current_with_factor = rolling.apply_factor_model(current_df, factor_model)

    stage_records = _build_stage_records(current_with_factor)
    sim_cfg = sim.BaseConfig(
        builder=str(best["builder"]),
        seq_len=int(best["seq_len"]),
        rep=str(best["rep"]),
        scorer=str(best["scorer"]),
    )
    templates = _build_similarity_templates(trainval_df, best)
    sim_df = sim.build_scored_df_normal(stage_records, templates, sim_cfg)
    sim_df = sim_df.rename(columns={"date": "signal_date", "score": "sim_score"})
    current_with_factor = current_with_factor.merge(
        sim_df[["code", "signal_date", "sim_score"]],
        on=["code", "signal_date"],
        how="left",
    )
    current_with_factor["sim_score"] = pd.to_numeric(current_with_factor["sim_score"], errors="coerce").fillna(-1.0)

    perfect_templates = _build_perfect_case_templates(data_dir, latest_signal_date, sim_cfg.seq_len, sim_cfg.rep)
    if perfect_templates:
        perfect_sim_df = sim.build_scored_df_normal(stage_records, perfect_templates, sim_cfg)
        perfect_sim_df = perfect_sim_df.rename(
            columns={"date": "signal_date", "score": "perfect_case_sim_score"}
        )
        current_with_factor = current_with_factor.merge(
            perfect_sim_df[["code", "signal_date", "perfect_case_sim_score"]],
            on=["code", "signal_date"],
            how="left",
        )
    current_with_factor["perfect_case_sim_score"] = pd.to_numeric(
        current_with_factor.get("perfect_case_sim_score"), errors="coerce"
    ).fillna(-1.0)

    rf_model = rolling.fit_rf_model(trainval_with_factor)
    current_prob = rolling.predict_rf_prob(current_with_factor, rf_model)
    current_with_factor["ml_score_raw"] = current_prob
    current_with_factor["ml_score"] = rolling.normalize_rank(current_prob)

    base_rank_score = (
        rolling.normalize_rank(current_with_factor["sim_score"]) * float(best["sim_weight"])
        + pd.to_numeric(current_with_factor["factor_score"], errors="coerce").fillna(0.0) * float(best["factor_weight"])
        + pd.to_numeric(current_with_factor["ml_score"], errors="coerce").fillna(0.0) * float(best["ml_weight"])
        + pd.to_numeric(current_with_factor["pool_bonus"], errors="coerce").fillna(0.0)
    )
    perfect_case_rank = rolling.normalize_rank(current_with_factor["perfect_case_sim_score"])
    current_with_factor["rank_score"] = (
        (1.0 - PERFECT_CASE_WEIGHT) * base_rank_score
        + PERFECT_CASE_WEIGHT * perfect_case_rank
    )
    current_with_factor["perfect_case_rank"] = perfect_case_rank
    selected = current_with_factor[current_with_factor["sim_score"] >= float(best["sim_gate"])].copy()
    if selected.empty:
        return []
    selected = (
        selected.sort_values(
            ["signal_date", "perfect_case_rank", "rank_score", "code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .groupby("signal_date", group_keys=False)
        .head(int(best["daily_topn"]))
        .reset_index(drop=True)
    )
    return _format_output(selected)


def print_selected(selected: list[list[str]]) -> None:
    print(f"【{STRATEGY_NAME}】")
    print(strategy_description())
    print(operation_suggestion())
    print(execution_rule_summary())
    if not selected:
        print("当前没有筛出 relaxed_fusion 候选。")
        return
    print(f"共筛选出 {len(selected)} 只股票：")
    for row in selected:
        code, stop_ref, close_price, score, note = row
        print(
            f"股票代码{code:<6} | 止损价(参考)：{stop_ref:<8} | "
            f"当日收盘价：{close_price:<8} | 综合分：{score:<8} | 备注：{note}"
        )


def check(file_path: str, hold_list=None, mode: str = "latest", feature_cache=None):
    """
    提供一个与 `brick_filter.py` 同风格的兼容接口，但不建议主流程逐文件调用。

    relaxed_fusion 的真实语义是“全市场排序”，这里只是为了接口兼容保留：
    - 若最新 K 线满足 relaxed 信号，返回参考止损和收盘价；
    - 分数固定为 0，因为单票模式下无法体现全市场相对强弱。
    """
    del hold_list, mode
    if feature_cache is not None:
        df = feature_cache.raw_df()
    else:
        df = sim.load_stock_data(file_path)
    if df is None or df.empty:
        return [-1]
    x = sim.compute_relaxed_brick_features(df).reset_index(drop=True)
    if x.empty:
        return [-1]
    latest = x.iloc[-1]
    if not bool(latest.get("signal_relaxed", False)):
        return [-1]
    stop_ref = round(float(min(latest["open"], latest["close"])), 3)
    close_price = float(latest["close"])
    return [1, stop_ref, close_price, 0.0, "需进入全市场排序"]


def main() -> None:
    """
    单独运行时，像其他 filter 一样直接打印今日候选。

    注意：
    - 这里输出的是“排序后的候选清单”
    - 不是独立于执行规则的最终买卖指令
    """
    selected = scan_dir(str(INPUT_DIR))
    print_selected(selected)


if __name__ == "__main__":
    main()
