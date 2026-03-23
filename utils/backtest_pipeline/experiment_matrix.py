from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.backtest_pipeline.catalog import register_builtin_modules
from utils.backtest_pipeline.compatibility import family_spec, supported_families
from utils.backtest_pipeline.registry import CONFIRMER_REGISTRY, EXIT_REGISTRY, RANKER_REGISTRY

LEDGER_JSON_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/experiment_ledger.json")


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    return obj


def _combo_key(combo: dict[str, Any]) -> str:
    payload = {
        "family": combo["strategy_family"],
        "candidate_pool": combo["candidate_pool"],
        "confirmer": combo.get("confirmer"),
        "ranker": combo["ranker"],
        "ranker_top_n": combo["ranker_top_n"],
        "ranker_params": _canonical(combo.get("ranker_params", {})),
        "exit": combo["exit"],
        "exit_params": _canonical(combo.get("exit_params", {})),
        "account_policy": combo["account_policy"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_experiment_ledger(path: Path = LEDGER_JSON_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("experiments", [])


def completed_combo_keys(path: Path = LEDGER_JSON_PATH) -> set[str]:
    keys: set[str] = set()
    for item in load_experiment_ledger(path):
        combo = item.get("combo")
        if combo:
            keys.add(_combo_key(combo))
    return keys


def generate_matrix(families: list[str] | None = None) -> list[dict[str, Any]]:
    register_builtin_modules()
    rows: list[dict[str, Any]] = []
    for family in (families or supported_families()):
        spec = family_spec(family)
        for candidate_pool, pool_spec in spec["candidate_pools"].items():
            for confirmer in pool_spec["confirmers"]:
                if confirmer and confirmer not in CONFIRMER_REGISTRY.names():
                    continue
                for ranker in pool_spec["rankers"]:
                    if ranker["name"] not in RANKER_REGISTRY.names():
                        continue
                    for top_n in ranker["top_n_values"]:
                        for exit_item in spec["exits"]:
                            if exit_item["name"] not in EXIT_REGISTRY.names():
                                continue
                            rows.append(
                                {
                                    "strategy_family": family,
                                    "candidate_pool": candidate_pool,
                                    "confirmer": confirmer,
                                    "ranker": ranker["name"],
                                    "ranker_top_n": int(top_n),
                                    "ranker_params": deepcopy(ranker.get("params", {})),
                                    "exit": exit_item["name"],
                                    "exit_params": deepcopy(exit_item.get("params", {})),
                                    "account_policy": "portfolio.equal_weight",
                                }
                            )
    return rows


def coverage_rows(families: list[str] | None = None, ledger_path: Path = LEDGER_JSON_PATH) -> list[dict[str, Any]]:
    done_keys = completed_combo_keys(ledger_path)
    rows: list[dict[str, Any]] = []
    for combo in generate_matrix(families):
        row = deepcopy(combo)
        row["status"] = "done" if _combo_key(combo) in done_keys else "pending"
        rows.append(row)
    return rows


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "done": sum(1 for row in rows if row["status"] == "done"),
        "pending": sum(1 for row in rows if row["status"] != "done"),
        "by_family": {},
    }
    for family in sorted({row["strategy_family"] for row in rows}):
        family_rows = [row for row in rows if row["strategy_family"] == family]
        summary["by_family"][family] = {
            "total": len(family_rows),
            "done": sum(1 for row in family_rows if row["status"] == "done"),
            "pending": sum(1 for row in family_rows if row["status"] != "done"),
        }
    return summary


def render_coverage_markdown(rows: list[dict[str, Any]]) -> str:
    summary = coverage_summary(rows)
    lines = [
        "# Experiment Coverage Report",
        "",
        f"- 总组合数：`{summary['total']}`",
        f"- 已登记完成：`{summary['done']}`",
        f"- 待覆盖：`{summary['pending']}`",
        "",
        "## 按策略家族汇总",
        "",
        "| 策略 | 总数 | 已做 | 待做 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, item in summary["by_family"].items():
        lines.append(f"| `{family}` | {item['total']} | {item['done']} | {item['pending']} |")
    lines.extend(
        [
            "",
            "## 待做组合",
            "",
            "| 策略 | 候选池 | 确认因子 | 排序器 | TopN | 卖法 | 参数 |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in rows:
        if row["status"] == "done":
            continue
        params = {"ranker": row.get("ranker_params", {}), "exit": row.get("exit_params", {})}
        lines.append(
            "| `{family}` | `{candidate_pool}` | `{confirmer}` | `{ranker}` | {top_n} | `{exit}` | `{params}` |".format(
                family=row["strategy_family"],
                candidate_pool=row["candidate_pool"],
                confirmer=row.get("confirmer") or "-",
                ranker=row["ranker"],
                top_n=row["ranker_top_n"],
                exit=row["exit"],
                params=json.dumps(params, ensure_ascii=False, sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def write_coverage_outputs(rows: list[dict[str, Any]], json_path: Path, markdown_path: Path) -> None:
    payload = {"summary": coverage_summary(rows), "rows": rows}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(render_coverage_markdown(rows))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成 backtest pipeline 实验矩阵与覆盖报告")
    parser.add_argument("--family", action="append", help="只生成指定策略家族，可多次传入")
    parser.add_argument("--json-output", default="/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/coverage_report.json")
    parser.add_argument("--md-output", default="/Users/lidongyang/Desktop/Qstrategy/utils/backtest_pipeline/docs/coverage_report.md")
    args = parser.parse_args()

    rows = coverage_rows(args.family)
    write_coverage_outputs(rows, Path(args.json_output), Path(args.md_output))
    print(json.dumps(coverage_summary(rows), ensure_ascii=False, indent=2))
