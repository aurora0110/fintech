from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="根据方向审计结果生成正式评分卡")
    parser.add_argument("--audit-root", default="/Users/lidongyang/Desktop/Qstrategy/results/factor_direction_audit_v1")
    parser.add_argument("--output-root", default="/Users/lidongyang/Desktop/Qstrategy/results/formal_scorecard_v1")
    parser.add_argument("--scorecard-name", default="方向审计正式评分卡V1")
    parser.add_argument("--method", choices=["weighted", "binary"], default="weighted")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_root / "百分制评分卡.csv")
    add_df = df[df["评分方向"] == "加分"].copy()
    penalty_df = df[df["评分方向"] == "扣分"].copy()
    watch_df = df[df["评分方向"] == "观察"].copy()

    if args.method == "binary":
        add_df["百分制评分"] = add_df["百分制评分"].apply(lambda v: 1 if int(v) > 0 else 0)
        penalty_df["百分制评分"] = penalty_df["百分制评分"].apply(lambda v: 1 if int(v) > 0 else 0)

    add_spec = "; ".join(f"{row.因子名称}={int(row.百分制评分)}分" for row in add_df.itertuples(index=False) if int(row.百分制评分) > 0)
    penalty_spec = "; ".join(f"{row.因子名称}={int(row.百分制评分)}分" for row in penalty_df.itertuples(index=False) if int(row.百分制评分) > 0)

    payload = {
        "评分卡名称": args.scorecard_name,
        "评分方法": "二值法" if args.method == "binary" else "加权法",
        "样本说明": "基于J<13且趋势线>多空线候选池，按后续5/10/30/60/120/240日表现方向审计生成。",
        "加分项": add_df[["因子名称", "百分制评分"]].to_dict(orient="records"),
        "扣分项": penalty_df[["因子名称", "百分制评分"]].to_dict(orient="records"),
        "观察项": watch_df[["因子名称", "百分制评分"]].to_dict(orient="records"),
        "加分组合字符串": add_spec,
        "扣分组合字符串": penalty_spec,
    }

    (output_root / "正式评分卡.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(output_root / "正式评分卡明细.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
