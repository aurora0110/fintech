from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import load_price_directory
from utils.multi_factor_research.factor_calculator import FACTOR_COLUMNS, FACTOR_NAME_MAP, PENALTY_COLUMNS
from utils.multi_factor_research.run_weighted_portfolio_backtest import _calc_net_score_series, _load_refined_weights, _parse_weight_spec, _prepare_stock_frames
from utils.multi_factor_research.weight_optimizer import PENALTY_WEIGHTS


def _read_stock_name(data_dir: Path, code: str) -> str:
    path = data_dir / f"{code}.txt"
    if not path.exists():
        return code
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    best_name = code
    best_score = -1
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="ignore") as fh:
                first_line = fh.readline().strip()
        except Exception:
            continue
        chinese_parts = re.findall(r"[\u4e00-\u9fffA-Za-z]+", first_line)
        for part in chinese_parts:
            if part in {"日线", "前复权"} or part == code.replace("#", ""):
                continue
            chinese_count = sum(1 for ch in part if "\u4e00" <= ch <= "\u9fff")
            score = chinese_count * 10 + len(part)
            if score > best_score:
                best_score = score
                best_name = part
    return best_name


def main() -> None:
    parser = argparse.ArgumentParser(description="导出指定净分区间的信号因子明细")
    parser.add_argument("data_dir")
    parser.add_argument("--refined-root", default="results/multi_factor_research_v18_constrained_card")
    parser.add_argument("--strategy", default="tiered", choices=["tiered", "fixed_take_profit", "fixed_days"])
    parser.add_argument("--min-score", type=float, default=40.0)
    parser.add_argument("--max-score", type=float, default=70.0)
    parser.add_argument("--output-csv", default="results/signal_breakdown_40_70/信号明细.csv")
    parser.add_argument("--output-json", default="results/signal_breakdown_40_70/汇总结果.json")
    args = parser.parse_args()

    refined_root = Path(args.refined_root)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    weights_payload = _load_refined_weights(refined_root)
    weights = _parse_weight_spec(weights_payload[args.strategy]["combo"])
    data_dir = Path(args.data_dir)

    stock_data, _ = load_price_directory(str(data_dir))
    prepared = _prepare_stock_frames(stock_data)

    records: list[dict] = []
    for code, df in prepared.items():
        stock_name = _read_stock_name(data_dir, code)
        net_scores = _calc_net_score_series(df, weights)
        signal_mask = df["J"].lt(13) & df["trend_line"].gt(df["bull_bear_line"])
        selected = df.loc[signal_mask].copy()
        if selected.empty:
            continue
        selected["净分"] = net_scores.loc[selected.index]
        selected = selected[(selected["净分"] >= args.min_score) & (selected["净分"] <= args.max_score)].copy()
        if selected.empty:
            continue

        for signal_date, row in selected.iterrows():
            reward_total = 0.0
            penalty_total = 0.0
            payload = {
                "股票代码": code,
                "股票名称": stock_name,
                "信号日期": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "净分": float(row["净分"]),
                "总加分": 0.0,
                "总扣分": 0.0,
                "J值": float(row["J"]),
                "趋势线": float(row["trend_line"]),
                "多空线": float(row["bull_bear_line"]),
            }
            for factor in FACTOR_COLUMNS:
                factor_cn = FACTOR_NAME_MAP[factor]
                raw_value = float(row.get(factor, 0.0))
                contrib = raw_value * float(weights.get(factor, 0.0))
                payload[f"{factor_cn}原始值"] = raw_value
                payload[f"{factor_cn}加分"] = contrib
                reward_total += contrib
            for factor in PENALTY_COLUMNS:
                factor_cn = FACTOR_NAME_MAP[factor]
                raw_value = float(row.get(factor, 0.0))
                contrib = raw_value * float(PENALTY_WEIGHTS.get(factor, 0.0))
                payload[f"{factor_cn}原始值"] = raw_value
                payload[f"{factor_cn}扣分"] = contrib
                penalty_total += contrib
            payload["总加分"] = reward_total
            payload["总扣分"] = penalty_total
            records.append(payload)

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["净分", "信号日期", "股票代码"], ascending=[False, True, True]).reset_index(drop=True)
        result.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = {
        "策略": args.strategy,
        "最小净分": args.min_score,
        "最大净分": args.max_score,
        "信号数量": int(len(result)),
        "输出文件": str(output_csv),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
