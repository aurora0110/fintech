from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.multi_factor_research.weighted_combo_search import (
    search_weighted_score_combinations,
    summarize_best_weighted_combos,
)


DEFAULT_TARGETS = [
    "固定涨幅止盈_40pct",
    "固定持有_30天",
    "分批顺序止盈",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine weighted factor score search on existing signal datasets")
    parser.add_argument("--input-root", default="results/multi_factor_research_v4")
    parser.add_argument("--output-root", default="results/multi_factor_research_v4_refined")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--top-quantile", type=float, default=0.30)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    final_summary: dict[str, dict] = {}
    for target in args.targets:
        dataset_path = input_root / target / "signal_scores.csv"
        if not dataset_path.exists():
            continue
        dataset = pd.read_csv(dataset_path)
        weighted_df = search_weighted_score_combinations(
            dataset=dataset,
            top_quantile=args.top_quantile,
            min_samples=args.min_samples,
            weight_step=args.weight_step,
        )
        best = summarize_best_weighted_combos(weighted_df)

        run_dir = output_root / target
        run_dir.mkdir(parents=True, exist_ok=True)
        weighted_df.to_csv(run_dir / "weighted_score_ranking.csv", index=False, encoding="utf-8-sig")
        payload = {
            "target": target,
            "weight_step": args.weight_step,
            "top_quantile": args.top_quantile,
            "min_samples": args.min_samples,
            "best_weighted_combos": best,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        final_summary[target] = payload

    (output_root / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
