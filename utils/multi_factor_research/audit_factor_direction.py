from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.factor_calculator import (
    FACTOR_COLUMNS,
    FACTOR_NAME_MAP,
    PENALTY_COLUMNS,
    build_trend_start_candidate_mask,
    build_prepared_stock_data,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        return iterable


HORIZONS = [5, 10, 20, 30, 60, 120, 240]
ALL_FACTOR_COLUMNS = FACTOR_COLUMNS + PENALTY_COLUMNS


def _build_candidate_dataset(
    prepared_stock_data: dict[str, pd.DataFrame],
    candidate_j_threshold: float,
    require_trend_above: bool,
    mode: str,
    start_gain_10: float,
    start_gain_20: float,
    use_trend_start_pool: bool,
    min_confirmation_hits: int,
    min_support_hits: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    base_columns = [
        "date",
        "board",
        "J",
        "trend_line",
        "bull_bear_line",
        *ALL_FACTOR_COLUMNS,
    ]

    for code, df in tqdm(prepared_stock_data.items(), desc="构建候选信号样本", unit="股"):
        if use_trend_start_pool:
            signal_mask = build_trend_start_candidate_mask(
                df,
                j_threshold=candidate_j_threshold,
                require_trend_above=require_trend_above,
                min_confirmation_hits=min_confirmation_hits,
                min_support_hits=min_support_hits,
            )
        else:
            signal_mask = df["J"].lt(candidate_j_threshold)
            if require_trend_above:
                signal_mask = signal_mask & df["trend_line"].gt(df["bull_bear_line"])
        signal_idx = np.flatnonzero(signal_mask.to_numpy(dtype=bool))
        signal_idx = signal_idx[signal_idx + 1 < len(df)]
        if signal_idx.size == 0:
            continue

        entry_idx = signal_idx + 1
        entry_price = df["open"].to_numpy(dtype=float)[entry_idx]
        close_arr = df["close"].to_numpy(dtype=float)
        high_arr = df["high"].to_numpy(dtype=float)

        chunk = df.iloc[signal_idx][base_columns].copy().reset_index(drop=True)
        chunk.rename(columns={"date": "信号日期", "board": "板块", "J": "J值"}, inplace=True)
        chunk["股票代码"] = code
        chunk["信号序号"] = signal_idx
        chunk["买入日期"] = df["date"].to_numpy()[entry_idx]
        chunk["买入价"] = entry_price
        chunk["年份"] = pd.to_datetime(chunk["信号日期"]).dt.year

        for horizon in HORIZONS:
            ret = np.full(signal_idx.size, np.nan, dtype=float)
            max_ret = np.full(signal_idx.size, np.nan, dtype=float)
            valid = entry_idx + horizon - 1 < len(df)
            if valid.any():
                exit_idx = entry_idx[valid] + horizon - 1
                ret[valid] = close_arr[exit_idx] / entry_price[valid] - 1.0
                for local_idx, src_idx in enumerate(np.flatnonzero(valid)):
                    end_idx = entry_idx[src_idx] + horizon
                    max_ret[src_idx] = high_arr[entry_idx[src_idx]:end_idx].max() / entry_price[src_idx] - 1.0
            chunk[f"{horizon}日收益率"] = ret
            chunk[f"{horizon}日上涨"] = (ret > 0).astype(float)
            chunk[f"{horizon}日最高涨幅"] = max_ret

        if mode == "trend":
            chunk["10日趋势启动"] = (chunk["10日最高涨幅"] >= start_gain_10).astype(float)
            chunk["20日强趋势启动"] = (chunk["20日最高涨幅"] >= start_gain_20).astype(float)

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 20:
        return 0.0
    x_std = float(data.iloc[:, 0].std())
    y_std = float(data.iloc[:, 1].std())
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(data.iloc[:, 0].corr(data.iloc[:, 1]))


def _group_sign(value: float, threshold: float = 1e-4) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _calc_stability(dataset: pd.DataFrame, factor: str, overall_sign: int, target_column: str) -> tuple[float, int]:
    if overall_sign == 0:
        return 0.0, 0

    checks: list[int] = []

    for year, sub_df in dataset.groupby("年份"):
        if len(sub_df) < 1000:
            continue
        sign = _group_sign(_safe_corr(sub_df[factor], sub_df[target_column]))
        if sign != 0:
            checks.append(int(sign == overall_sign))

    for board, sub_df in dataset.groupby("板块"):
        if len(sub_df) < 1000:
            continue
        sign = _group_sign(_safe_corr(sub_df[factor], sub_df[target_column]))
        if sign != 0:
            checks.append(int(sign == overall_sign))

    if not checks:
        return 0.0, 0
    return float(np.mean(checks)), len(checks)


def _calc_factor_rows(dataset: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    for factor in tqdm(ALL_FACTOR_COLUMNS, desc="分析单因子方向", unit="因子"):
        values = dataset[factor].fillna(0.0)
        if mode == "trend":
            corr_up_10 = _safe_corr(values, dataset["10日趋势启动"])
            corr_up_30 = _safe_corr(values, dataset["20日强趋势启动"])
            corr_ret_30 = _safe_corr(values, dataset["20日最高涨幅"])
            low_target_10 = "10日趋势启动"
            high_target_10 = "10日趋势启动"
            low_target_30 = "20日强趋势启动"
            high_target_30 = "20日强趋势启动"
            ret_target_30 = "20日最高涨幅"
            stability_target = "20日最高涨幅"
        else:
            corr_up_10 = _safe_corr(values, dataset["10日上涨"])
            corr_up_30 = _safe_corr(values, dataset["30日上涨"])
            corr_ret_30 = _safe_corr(values, dataset["30日收益率"])
            low_target_10 = "10日上涨"
            high_target_10 = "10日上涨"
            low_target_30 = "30日上涨"
            high_target_30 = "30日上涨"
            ret_target_30 = "30日收益率"
            stability_target = "30日收益率"

        low_cut = float(values.quantile(0.30))
        high_cut = float(values.quantile(0.70))
        low_mask = values <= low_cut
        high_mask = values >= high_cut

        low_up_10 = float(dataset.loc[low_mask, low_target_10].mean()) if low_mask.any() else 0.0
        high_up_10 = float(dataset.loc[high_mask, high_target_10].mean()) if high_mask.any() else 0.0
        low_up_30 = float(dataset.loc[low_mask, low_target_30].mean()) if low_mask.any() else 0.0
        high_up_30 = float(dataset.loc[high_mask, high_target_30].mean()) if high_mask.any() else 0.0
        low_ret_30 = float(dataset.loc[low_mask, ret_target_30].mean()) if low_mask.any() else 0.0
        high_ret_30 = float(dataset.loc[high_mask, ret_target_30].mean()) if high_mask.any() else 0.0

        signed_signal = 0.25 * corr_up_10 + 0.45 * corr_up_30 + 0.30 * corr_ret_30
        direction_sign = _group_sign(signed_signal)
        stability_ratio, stability_checks = _calc_stability(dataset, factor, direction_sign, stability_target)

        if direction_sign > 0:
            direction = "正贡献"
        elif direction_sign < 0:
            direction = "负贡献"
        else:
            direction = "待观察"

        rows.append(
            {
                "因子代码": factor,
                "因子名称": FACTOR_NAME_MAP.get(factor, factor),
                "因子类型": "原加分因子" if factor in FACTOR_COLUMNS else "原扣分因子",
                "10日上涨相关系数": corr_up_10,
                "30日上涨相关系数": corr_up_30,
                "30日收益相关系数": corr_ret_30,
                "低分组10日上涨概率": low_up_10,
                "高分组10日上涨概率": high_up_10,
                "低分组30日上涨概率": low_up_30,
                "高分组30日上涨概率": high_up_30,
                "低分组30日平均收益率": low_ret_30,
                "高分组30日平均收益率": high_ret_30,
                "10日上涨概率提升": high_up_10 - low_up_10,
                "30日上涨概率提升": high_up_30 - low_up_30,
                "30日收益率提升": high_ret_30 - low_ret_30,
                "方向综合信号": signed_signal,
                "稳定性比例": stability_ratio,
                "稳定性样本组数": stability_checks,
                "建议方向": direction,
            }
        )

    result = pd.DataFrame(rows)
    for source_col, rank_col in [
        ("10日上涨相关系数", "10日上涨强度分位"),
        ("30日上涨相关系数", "30日上涨强度分位"),
        ("30日收益相关系数", "30日收益强度分位"),
        ("30日上涨概率提升", "30日上涨提升分位"),
        ("稳定性比例", "稳定性分位"),
    ]:
        result[rank_col] = result[source_col].abs().rank(method="average", pct=True)

    result["贡献原始强度"] = (
        0.20 * result["10日上涨强度分位"]
        + 0.30 * result["30日上涨强度分位"]
        + 0.25 * result["30日收益强度分位"]
        + 0.15 * result["30日上涨提升分位"]
        + 0.10 * result["稳定性分位"]
    )
    return result.sort_values(["贡献原始强度", "方向综合信号"], ascending=[False, False]).reset_index(drop=True)


def _allocate_integer_scores(weights: pd.Series, total_score: int = 100) -> list[int]:
    if weights.sum() <= 0:
        base = [0] * len(weights)
        if len(base) > 0:
            base[0] = total_score
        return base

    raw = weights / weights.sum() * total_score
    floors = np.floor(raw).astype(int)
    remainder = int(total_score - floors.sum())
    if remainder > 0:
        order = np.argsort(-(raw - floors))
        floors[order[:remainder]] += 1
    return floors.tolist()


def _build_scorecard(result: pd.DataFrame) -> pd.DataFrame:
    scorecard = result.copy()
    scorecard["评分方向"] = scorecard["建议方向"].map(
        {
            "正贡献": "加分",
            "负贡献": "扣分",
            "待观察": "观察",
        }
    )
    scorecard["百分制评分"] = _allocate_integer_scores(scorecard["贡献原始强度"], total_score=100)
    return scorecard.sort_values(
        ["评分方向", "百分制评分", "贡献原始强度"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="审计各因子对后续上涨与收益的方向贡献")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/20260226")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_direction_audit_v1")
    parser.add_argument("--mode", choices=["repair", "trend"], default="repair")
    parser.add_argument("--candidate-j-threshold", type=float, default=13.0)
    parser.add_argument("--require-trend-above", action="store_true", default=True)
    parser.add_argument("--start-gain-10", type=float, default=0.08)
    parser.add_argument("--start-gain-20", type=float, default=0.12)
    parser.add_argument("--use-trend-start-pool", action="store_true")
    parser.add_argument("--min-confirmation-hits", type=int, default=1)
    parser.add_argument("--min-support-hits", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(str(Path(args.data_dir)))
    prepared_stock_data = build_prepared_stock_data(stock_data, burst_window=20)
    dataset = _build_candidate_dataset(
        prepared_stock_data,
        candidate_j_threshold=args.candidate_j_threshold,
        require_trend_above=args.require_trend_above,
        mode=args.mode,
        start_gain_10=args.start_gain_10,
        start_gain_20=args.start_gain_20,
        use_trend_start_pool=args.use_trend_start_pool,
        min_confirmation_hits=args.min_confirmation_hits,
        min_support_hits=args.min_support_hits,
    )
    dataset.to_csv(output_root / "候选信号样本.csv", index=False, encoding="utf-8-sig")

    result = _calc_factor_rows(dataset, mode=args.mode)
    result.to_csv(output_root / "因子方向分析.csv", index=False, encoding="utf-8-sig")

    scorecard = _build_scorecard(result)
    scorecard.to_csv(output_root / "百分制评分卡.csv", index=False, encoding="utf-8-sig")

    summary = {
        "样本定义": (
            f"候选信号定义为J<{args.candidate_j_threshold}"
            + ("且趋势线>多空线" if args.require_trend_above else "")
            + (
                f"，并满足趋势启动候选池：至少{args.min_confirmation_hits}个启动确认因子、至少{args.min_support_hits}个动量/结构因子"
                if args.use_trend_start_pool
                else ""
            )
            + (
                f"，趋势模式评价10日最高涨幅是否达到{args.start_gain_10:.0%}、20日最高涨幅是否达到{args.start_gain_20:.0%}"
                if args.mode == "trend"
                else "，买点按信号后下一交易日开盘价，评价后续5/10/20/30/60/120/240日表现。"
            )
        ),
        "审计模式": "趋势型" if args.mode == "trend" else "修复型",
        "样本数量": int(len(dataset)),
        "正贡献因子": scorecard.loc[scorecard["评分方向"] == "加分", ["因子名称", "百分制评分"]].to_dict(orient="records"),
        "负贡献因子": scorecard.loc[scorecard["评分方向"] == "扣分", ["因子名称", "百分制评分"]].to_dict(orient="records"),
        "待观察因子": scorecard.loc[scorecard["评分方向"] == "观察", ["因子名称", "百分制评分"]].to_dict(orient="records"),
        "前五名重要因子": scorecard.head(5)[["因子名称", "评分方向", "百分制评分"]].to_dict(orient="records"),
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
