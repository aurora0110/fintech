from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
CASE_SEMANTICS_PATH = ROOT / "utils" / "tmp" / "brick_case_semantics_v1_20260326.py"
DEFAULT_RESULT_DIR = ROOT / "results" / "brick_relaxed_perfect_case_coverage_check_v1_20260326_r6"
DEFAULT_DATA_DIR = ROOT / "data" / "20260324"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


semantics = load_module(CASE_SEMANTICS_PATH, "brick_case_postprocess_semantics")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_summary_md(result_dir: Path, merged: pd.DataFrame) -> str:
    stage_counts = merged["stage"].value_counts().to_dict()
    selected = merged[merged["stage"] == "selected"].copy()
    topn = merged[merged["stage"] == "topn"].copy()
    sim_gate = merged[merged["stage"] == "sim_gate"].copy()
    candidate_pool = merged[merged["stage"] == "candidate_pool"].copy()
    lines = [
        "# BRICK relaxed_fusion 完美案例逐案例解释",
        "",
        f"- 总案例数：`{len(merged)}`",
        f"- 直接入选：`{stage_counts.get('selected', 0)}`",
        f"- 卡在 top10：`{stage_counts.get('topn', 0)}`",
        f"- 卡在 sim_gate：`{stage_counts.get('sim_gate', 0)}`",
        f"- 卡在候选池：`{stage_counts.get('candidate_pool', 0)}`",
        "",
        "## 直接被选中的 3 个案例",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- `{row.stock_name} {pd.Timestamp(row.signal_date).strftime('%Y-%m-%d')}`"
            f" | 分型：`{row.brick_case_type_name}`"
            f" | 日内排名：`{int(row.daily_rank) if pd.notna(row.daily_rank) else 'NA'}`"
            f" | sim=`{float(row.sim_score):.3f}`"
            f" | case=`{float(row.perfect_case_sim_score):.3f}`"
            f" | rank=`{float(row.rank_score):.3f}`"
        )
    lines.extend(
        [
            "",
            "## 主要失败原因",
            f"- `top10` 是当前最大瓶颈，绝大多数完美案例已经进入候选并通过 `sim_gate`，但排名不够靠前。",
            f"- `sim_gate` 只挡掉了 `{len(sim_gate)}` 个案例，说明相似度门槛不是当前主要问题。",
            f"- 候选池当前只错杀了 `{len(candidate_pool)}` 个案例，主要原因是 `signal_relaxed=False`。",
            "",
            "## 候选池错杀案例",
        ]
    )
    for row in candidate_pool.itertuples(index=False):
        lines.append(
            f"- `{row.stock_name} {pd.Timestamp(row.signal_date).strftime('%Y-%m-%d')}`"
            f" | 分型：`{row.brick_case_type_name}`"
            f" | 原因：`{row.reason}`"
            f" | prev_green=`{row.prev_green_streak}`"
        )
    lines.extend(["", "## sim_gate 失败案例"])
    for row in sim_gate.itertuples(index=False):
        lines.append(
            f"- `{row.stock_name} {pd.Timestamp(row.signal_date).strftime('%Y-%m-%d')}`"
            f" | 分型：`{row.brick_case_type_name}`"
            f" | 原因：`{row.reason}`"
            f" | case=`{float(row.perfect_case_sim_score):.3f}`"
        )
    lines.extend(["", "## top10 失败案例（按日内排名最靠前列前 20 个）"])
    topn_sorted = topn.sort_values(["daily_rank", "signal_date", "stock_name"]).head(20)
    for row in topn_sorted.itertuples(index=False):
        lines.append(
            f"- `{row.stock_name} {pd.Timestamp(row.signal_date).strftime('%Y-%m-%d')}`"
            f" | 分型：`{row.brick_case_type_name}`"
            f" | 日内排名：`{int(row.daily_rank)}`"
            f" | sim=`{float(row.sim_score):.3f}`"
            f" | case=`{float(row.perfect_case_sim_score):.3f}`"
            f" | rank=`{float(row.rank_score):.3f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="BRICK relaxed_fusion 完美案例覆盖结果后处理")
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    daily_dir = Path(args.data_dir)
    if (daily_dir / "normal").exists():
        daily_dir = daily_dir / "normal"

    coverage_path = result_dir / "perfect_case_coverage_results.csv"
    if not coverage_path.exists():
        raise FileNotFoundError(f"未找到结果文件: {coverage_path}")
    coverage_df = pd.read_csv(coverage_path, parse_dates=["signal_date"])

    typed_df = semantics.load_case_day_features(semantics.PERFECT_CASE_DIR, daily_dir)
    if typed_df.empty:
        raise RuntimeError("未能生成完美案例分型表")

    merged = coverage_df.merge(
        typed_df[
            [
                "stock_name",
                "signal_date",
                "brick_case_type",
                "brick_case_type_name",
                "brick_case_type_score",
                "brick_case_type_source",
                "early_red_stage_flag",
                "early_red_stage_flag_num",
            ]
        ],
        on=["stock_name", "signal_date"],
        how="left",
    )
    merged.to_csv(result_dir / "perfect_case_coverage_with_types.csv", index=False, encoding="utf-8-sig")
    typed_df.to_csv(result_dir / "perfect_case_type_table.csv", index=False, encoding="utf-8-sig")

    selected = merged[merged["stage"] == "selected"].copy()
    top10_missed = merged[merged["stage"] == "topn"].copy()
    sim_gate_failed = merged[merged["stage"] == "sim_gate"].copy()
    candidate_pool_failed = merged[merged["stage"] == "candidate_pool"].copy()

    selected.to_csv(result_dir / "selected_cases.csv", index=False, encoding="utf-8-sig")
    top10_missed.to_csv(result_dir / "top10_missed_cases.csv", index=False, encoding="utf-8-sig")
    sim_gate_failed.to_csv(result_dir / "sim_gate_failed_cases.csv", index=False, encoding="utf-8-sig")
    candidate_pool_failed.to_csv(result_dir / "candidate_pool_failed_cases.csv", index=False, encoding="utf-8-sig")

    summary = {
        "selected_cases": int(len(selected)),
        "top10_missed_cases": int(len(top10_missed)),
        "sim_gate_failed_cases": int(len(sim_gate_failed)),
        "candidate_pool_failed_cases": int(len(candidate_pool_failed)),
        "case_type_counts": merged["brick_case_type_name"].value_counts(dropna=False).to_dict(),
        "stage_counts": merged["stage"].value_counts(dropna=False).to_dict(),
    }
    write_json(result_dir / "postprocess_summary.json", summary)
    (result_dir / "summary_cn.md").write_text(build_summary_md(result_dir, merged), encoding="utf-8")


if __name__ == "__main__":
    main()
