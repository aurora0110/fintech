from __future__ import annotations

"""
BRICK case-recall 筛选器
=======================

目标：
1. 尽可能在对应日期把完美砖型图案例召回出来；
2. 优先优化 Recall@20 / Recall@50 / Recall@100；
3. 不把账户层收益当主目标。
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
SIMILARITY_PATH = ROOT / "utils" / "tmp" / "similarity_filter_research.py"
CASE_FIRST_PATH = ROOT / "utils" / "tmp" / "brickfilter_case_first_v1_20260326.py"
CASE_SEMANTICS_PATH = ROOT / "utils" / "tmp" / "brick_case_semantics_v1_20260326.py"

STRATEGY_NAME = "BRICK_CASE_RECALL"
INPUT_DIR = ROOT / "data" / "20260324"
DEFAULT_TOPN = 100
CASE_SEQ_LENS = [3, 5, 8]

_MODULE_CACHE: dict[str, Any] = {}


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


sim = _load_module(SIMILARITY_PATH, "brick_case_recall_similarity")
case_first = _load_module(CASE_FIRST_PATH, "brick_case_recall_case_first")
case_semantics = _load_module(CASE_SEMANTICS_PATH, "brick_case_recall_case_semantics")


def strategy_name() -> str:
    return STRATEGY_NAME


def strategy_description() -> str:
    return "BRICK case_recall：以完美案例覆盖率为第一目标，优先优化 Recall@20/50/100。"


def operation_suggestion() -> str:
    return "这条线是案例覆盖器，不是收益冠军选股器；更适合找全、找像，再让人工或第二层交易规则筛。"


def _build_type_quota_map(topn: int, latest_signal_date: pd.Timestamp, daily_dir: Path) -> dict[int, int]:
    feat_df = case_first._load_perfect_case_feature_df(daily_dir)
    if feat_df.empty:
        return {1: topn // 3, 2: topn // 3, 3: max(1, topn // 10), 4: topn - 2 * (topn // 3) - max(1, topn // 10)}
    feat_df = feat_df[
        (pd.to_datetime(feat_df["signal_date"]) < pd.Timestamp(latest_signal_date))
        & (
            (pd.to_datetime(feat_df["signal_date"]) < case_first.relaxed.EXCLUDE_START)
            | (pd.to_datetime(feat_df["signal_date"]) > case_first.relaxed.EXCLUDE_END)
        )
    ].copy()
    if feat_df.empty:
        return {1: topn // 3, 2: topn // 3, 3: max(1, topn // 10), 4: topn - 2 * (topn // 3) - max(1, topn // 10)}

    counts = feat_df["brick_case_type"].value_counts().to_dict()
    total = sum(int(v) for v in counts.values()) or 1
    raw = {k: topn * float(counts.get(k, 0)) / total for k in (1, 2, 3, 4)}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in (1, 2, 4):
        if counts.get(k, 0) > 0 and base[k] <= 0:
            base[k] = 1
    remain = topn - sum(base.values())
    remainders = sorted(((raw[k] - base[k], k) for k in raw), reverse=True)
    idx = 0
    while remain > 0 and remainders:
        _, k = remainders[idx % len(remainders)]
        base[k] += 1
        remain -= 1
        idx += 1
    return base


def _series_or_default(current: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in current.columns:
        return pd.to_numeric(current[col], errors="coerce").fillna(default)
    return pd.Series(default, index=current.index, dtype=float)


def _build_perfect_case_quality_score(current: pd.DataFrame) -> pd.Series:
    rebound_rank = case_first.rolling.normalize_rank(_series_or_default(current, "rebound_ratio", 0.0))
    close_rank = case_first.rolling.normalize_rank(_series_or_default(current, "close_location", 0.0))
    body_rank = case_first.rolling.normalize_rank(_series_or_default(current, "body_ratio", 0.0))
    upper_shadow_penalty = case_first.rolling.normalize_rank(_series_or_default(current, "upper_shadow_pct", 1.0))
    pattern_a_bonus = _series_or_default(current, "pattern_a_relaxed", 0.0) * 0.08
    green_streak = _series_or_default(current, "prev_green_streak", 0.0)
    green_window_bonus = green_streak.between(4.0, 7.0, inclusive="both").astype(float) * 0.07
    return (
        0.40 * rebound_rank
        + 0.25 * close_rank
        + 0.20 * body_rank
        - 0.15 * upper_shadow_penalty
        + pattern_a_bonus
        + green_window_bonus
    )


def enrich_candidates_for_date(
    target_date: pd.Timestamp,
    candidate_df: pd.DataFrame,
    data_dir: str | Path,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    daily_dir = case_first._resolve_daily_dir(data_dir)
    current = candidate_df.copy()
    current["perfect_case_sim_score"] = -1.0
    current["same_type_case_sim_score"] = -1.0

    template_bundle = case_first._build_perfect_case_template_bundle(daily_dir, target_date, rep="close_norm")
    stage_records = case_first.relaxed._build_stage_records(current)

    key_cols = ["code", "signal_date"]
    for seq_len in CASE_SEQ_LENS:
        type_col = f"same_type_case_sim_{seq_len}"
        cfg = sim.BaseConfig(builder="recent_100", seq_len=int(seq_len), rep="close_norm", scorer="pipeline_corr_dtw")
        all_templates = template_bundle["all"].get(int(seq_len), [])
        if all_templates:
            all_sim_df = sim.build_scored_df_normal(stage_records, all_templates, cfg).rename(
                columns={"date": "signal_date", "score": f"perfect_case_sim_{seq_len}"}
            )
            current = current.merge(all_sim_df[["code", "signal_date", f"perfect_case_sim_{seq_len}"]], on=key_cols, how="left")
            current["perfect_case_sim_score"] = np.maximum(
                current["perfect_case_sim_score"],
                pd.to_numeric(current[f"perfect_case_sim_{seq_len}"], errors="coerce").fillna(-1.0),
            )
        if type_col not in current.columns:
            current[type_col] = -1.0
        for case_type in sorted(pd.to_numeric(current["brick_case_type"], errors="coerce").fillna(4).astype(int).unique()):
            type_templates = template_bundle["by_type"].get(int(case_type), {}).get(int(seq_len), [])
            if not type_templates:
                continue
            mask = pd.to_numeric(current["brick_case_type"], errors="coerce").fillna(4).astype(int) == int(case_type)
            if not mask.any():
                continue
            subset_records = case_first.relaxed._build_stage_records(current.loc[mask].copy())
            type_sim_df = sim.build_scored_df_normal(subset_records, type_templates, cfg).rename(
                columns={"date": "signal_date", "score": type_col}
            )
            subset_scored = current.loc[mask, key_cols].merge(
                type_sim_df[["code", "signal_date", type_col]],
                on=key_cols,
                how="left",
            )
            current.loc[mask, type_col] = np.maximum(
                pd.to_numeric(current.loc[mask, type_col], errors="coerce").fillna(-1.0).to_numpy(),
                pd.to_numeric(subset_scored[type_col], errors="coerce").fillna(-1.0).to_numpy(),
            )
        current["same_type_case_sim_score"] = np.maximum(
            current["same_type_case_sim_score"],
            pd.to_numeric(current[type_col], errors="coerce").fillna(-1.0),
        )

    risk_penalty = (
        pd.to_numeric(current.get("risk_distribution_recent_20"), errors="coerce").fillna(0.0) * 0.08
        + pd.to_numeric(current.get("risk_distribution_recent_30"), errors="coerce").fillna(0.0) * 0.05
        + pd.to_numeric(current.get("risk_distribution_recent_60"), errors="coerce").fillna(0.0) * 0.03
    )
    current["perfect_case_quality_score"] = _build_perfect_case_quality_score(current)
    current["type_rank"] = (
        pd.to_numeric(current.get("brick_case_type_score"), errors="coerce").fillna(0.45)
        + pd.to_numeric(current.get("early_red_stage_flag_num"), errors="coerce").fillna(0.0) * 0.25
        - risk_penalty
    )
    current["recall_score"] = (
        0.50 * case_first.rolling.normalize_rank(current["same_type_case_sim_score"])
        + 0.25 * case_first.rolling.normalize_rank(current["perfect_case_sim_score"])
        + 0.15 * case_first.rolling.normalize_rank(current["perfect_case_quality_score"])
        + 0.10 * case_first.rolling.normalize_rank(current["type_rank"])
    )
    return current


def score_candidates_for_date(
    target_date: pd.Timestamp,
    candidate_df: pd.DataFrame,
    data_dir: str | Path,
    topn: int = DEFAULT_TOPN,
) -> pd.DataFrame:
    current = enrich_candidates_for_date(
        target_date=target_date,
        candidate_df=candidate_df,
        data_dir=data_dir,
    )
    if current.empty:
        return current

    gated = current[
        (pd.to_numeric(current["same_type_case_sim_score"], errors="coerce").fillna(-1.0) >= 0.58)
        | (pd.to_numeric(current["perfect_case_sim_score"], errors="coerce").fillna(-1.0) >= 0.62)
        | (
            (pd.to_numeric(current.get("brick_case_type_score"), errors="coerce").fillna(0.0) >= 0.92)
            & (pd.to_numeric(current.get("early_red_stage_flag_num"), errors="coerce").fillna(0.0) >= 1.0)
        )
    ].copy()
    if gated.empty:
        return gated

    quotas = _build_type_quota_map(int(topn), pd.Timestamp(target_date), daily_dir)
    picked_parts: list[pd.DataFrame] = []
    picked_keys: set[tuple[str, pd.Timestamp]] = set()
    for case_type, quota in quotas.items():
        if quota <= 0:
            continue
        sub = gated[pd.to_numeric(gated["brick_case_type"], errors="coerce").fillna(4).astype(int) == int(case_type)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["signal_date", "same_type_case_sim_score", "perfect_case_sim_score", "recall_score", "code"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        ).groupby("signal_date", group_keys=False).head(int(quota))
        picked_parts.append(sub)
        for row in sub.itertuples(index=False):
            picked_keys.add((str(row.code), pd.Timestamp(row.signal_date)))

    remainder = gated[
        ~gated.apply(lambda r: (str(r["code"]), pd.Timestamp(r["signal_date"])) in picked_keys, axis=1)
    ].copy()
    remain_n = max(0, int(topn) - sum(len(x) for x in picked_parts))
    if remain_n > 0 and not remainder.empty:
        remainder = remainder.sort_values(
            ["signal_date", "recall_score", "same_type_case_sim_score", "perfect_case_sim_score", "code"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        ).groupby("signal_date", group_keys=False).head(remain_n)
        picked_parts.append(remainder)

    out = pd.concat(picked_parts, ignore_index=True) if picked_parts else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(
        ["signal_date", "recall_score", "same_type_case_sim_score", "perfect_case_sim_score", "code"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).groupby("signal_date", group_keys=False).head(int(topn)).reset_index(drop=True)
    return out


def _format_output(df: pd.DataFrame) -> list[list[str]]:
    out: list[list[str]] = []
    for row in df.itertuples(index=False):
        code = str(getattr(row, "code")).split("#")[-1]
        stop_ref = round(float(min(getattr(row, "signal_open"), getattr(row, "signal_close"))), 3)
        close_price = round(float(getattr(row, "signal_close")), 3)
        score = round(float(getattr(row, "recall_score")), 4)
        note = (
            f"same:{float(getattr(row, 'same_type_case_sim_score', -1.0)):.3f} "
            f"case:{float(getattr(row, 'perfect_case_sim_score', -1.0)):.3f} "
            f"q:{float(getattr(row, 'perfect_case_quality_score', -1.0)):.3f} "
            f"type:{int(getattr(row, 'brick_case_type', 4))} "
            f"r20:{float(getattr(row, 'risk_distribution_recent_20', 0.0)):.2f}"
        )
        out.append([code, f"{stop_ref:.3f}", f"{close_price:.3f}", f"{score:.4f}", note])
    return out


def scan_dir(data_dir: str, hold_list=None, file_limit: int = 0, max_workers: int = 8) -> list[list[str]]:
    del hold_list
    daily_dir = case_first._resolve_daily_dir(data_dir)
    current_df = case_first.build_candidates_for_date(
        pd.Timestamp(case_first.relaxed._build_current_candidates(str(daily_dir), file_limit=file_limit, max_workers=max_workers)["signal_date"].max()),
        daily_dir,
        file_limit=file_limit,
        max_workers=max_workers,
    )
    if current_df.empty:
        return []
    latest_signal_date = pd.Timestamp(current_df["signal_date"].max())
    selected = score_candidates_for_date(latest_signal_date, current_df, daily_dir, topn=DEFAULT_TOPN)
    if selected.empty:
        return []
    return _format_output(selected)


def print_selected(selected: list[list[str]]) -> None:
    print(f"【{STRATEGY_NAME}】")
    print(strategy_description())
    print(operation_suggestion())
    if not selected:
        print("当前没有筛出 case_recall 候选。")
        return
    print(f"共筛选出 {len(selected)} 只股票：")
    for code, stop_ref, close_price, score, note in selected:
        print(
            f"股票代码{code:<6} | 止损价(参考)：{stop_ref:<8} | "
            f"当日收盘价：{close_price:<8} | 召回分：{score:<8} | 备注：{note}"
        )


def main() -> None:
    selected = scan_dir(str(INPUT_DIR), max_workers=max(1, os.cpu_count() or 1))
    print_selected(selected)


if __name__ == "__main__":
    main()
