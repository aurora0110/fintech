from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.multi_factor_research.data_processor import load_stock_directory
from utils.multi_factor_research.exit_models import FixedDaysExit, FixedTakeProfitExit, simulate_signal_exits
from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, FACTOR_NAME_MAP, build_prepared_stock_data, build_signal_candidates
from utils.multi_factor_research.weight_optimizer import analyze_factor_contributions


def _summarize_contributions(dataset: pd.DataFrame, label: str) -> pd.DataFrame:
    contributions = analyze_factor_contributions(dataset).copy()
    contributions["模型"] = label
    contributions["中文因子"] = contributions["factor"].map(FACTOR_NAME_MAP)
    return contributions


def main() -> None:
    root = Path("/Users/lidongyang/Desktop/Qstrategy")
    output_root = root / "results/factor_stability_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    stock_data = load_stock_directory(str(root / "data/20260226"))
    prepared = build_prepared_stock_data(stock_data, burst_window=20)
    signal_candidates = build_signal_candidates(prepared)

    fixed_tp = simulate_signal_exits(prepared, signal_candidates, FixedTakeProfitExit(0.40), 0.0)
    fixed_days = simulate_signal_exits(prepared, signal_candidates, FixedDaysExit(30), 0.0)

    contrib_tp = _summarize_contributions(fixed_tp, "固定涨幅止盈40%")
    contrib_days = _summarize_contributions(fixed_days, "固定持有30天")
    contributions = pd.concat([contrib_tp, contrib_days], ignore_index=True)
    contributions.to_csv(output_root / "因子边际贡献.csv", index=False, encoding="utf-8-sig")

    factor_rows = []
    for df in prepared.values():
        factor_rows.append(df[FACTOR_COLUMNS])
    factor_matrix = pd.concat(factor_rows, axis=0, ignore_index=True).fillna(0.0)
    corr = factor_matrix.corr()
    corr.to_csv(output_root / "因子相关性矩阵.csv", encoding="utf-8-sig")

    avg_contrib = (
        contributions.groupby(["factor", "中文因子"], as_index=False)[["combined_contribution", "return_contribution", "success_contribution"]]
        .mean()
        .sort_values("combined_contribution", ascending=False)
    )
    avg_contrib["平均绝对相关性"] = [
        float(corr.loc[factor].drop(index=factor).abs().mean()) for factor in avg_contrib["factor"]
    ]
    avg_contrib["建议保留"] = (
        (avg_contrib["combined_contribution"] >= avg_contrib["combined_contribution"].median())
        & (avg_contrib["平均绝对相关性"] <= avg_contrib["平均绝对相关性"].median())
    )
    avg_contrib.to_csv(output_root / "因子稳定性汇总.csv", index=False, encoding="utf-8-sig")

    core = avg_contrib[avg_contrib["建议保留"]].to_dict(orient="records")
    summary = {
        "候选核心因子": core,
        "建议说明": "优先保留平均边际贡献较高且平均绝对相关性较低的因子，作为后续6到8个核心因子的初始候选。",
    }
    (output_root / "汇总结果.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
