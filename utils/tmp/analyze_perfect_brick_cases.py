from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/lidongyang/Desktop/Qstrategy")
from utils import brick_filter


RAW_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260312")
NORMAL_DIR = RAW_DIR / "normal"
CASE_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图")
OUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/perfect_brick_case_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_case_files() -> pd.DataFrame:
    rows: List[dict] = []
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    for path in sorted(CASE_DIR.glob("*.png")):
        m = pat.match(path.name)
        if not m:
            continue
        stock_name, ds = m.groups()
        rows.append(
            {
                "stock_name": stock_name,
                "signal_date": pd.to_datetime(ds, format="%Y%m%d"),
                "case_file": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_name_code_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in RAW_DIR.glob("*.txt"):
        if path.parent.name == "normal":
            continue
        try:
            with open(path, "r", encoding="gbk", errors="ignore") as f:
                first_line = f.readline().strip()
        except Exception:
            continue
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[1]] = path.stem
    return mapping


def get_row_with_extra_features(code_path: Path, signal_date: pd.Timestamp) -> pd.Series | None:
    df = brick_filter.load_one_csv(str(code_path))
    if df is None or df.empty:
        return None
    x = brick_filter.add_features(df)
    x["upper_shadow_ratio"] = (x["high"] - np.maximum(x["open"], x["close"])) / (
        (x["high"] - x["low"]).replace(0, np.nan)
    )
    x["close_position"] = (x["close"] - x["low"]) / ((x["high"] - x["low"]).replace(0, np.nan))
    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["dist_20d_high"] = x["close"] / x["high"].rolling(20).max() - 1.0
    x["dist_20d_low"] = x["close"] / x["low"].rolling(20).min() - 1.0
    x["ret_5d_before"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret_10d_before"] = x["close"] / x["close"].shift(10) - 1.0
    match = x.loc[x["date"] == signal_date]
    if match.empty:
        return None
    return match.iloc[-1]


def condition_flags(row: pd.Series) -> Tuple[bool, List[str]]:
    mask_a = bool(row.get("pattern_a", False)) and float(row.get("rebound_ratio", np.nan)) >= 1.2
    mask_b = bool(row.get("pattern_b", False)) and float(row.get("rebound_ratio", np.nan)) >= 1.0
    failures: List[str] = []
    if not (mask_a or mask_b):
        failures.append("砖型形态或反包阈值不满足")
    if not bool(row.get("pullback_shrinking", False)):
        failures.append("回调未缩量")
    if not bool(row.get("signal_vs_ma5_valid", False)):
        failures.append("量比五日不在[1.3,2.2]")
    if not bool(row.get("not_sideways", False)):
        failures.append("横盘过滤未通过")
    if not pd.notna(row.get("ret1")) or float(row.get("ret1", 0.0)) > 0.08:
        failures.append("当日涨幅超过8%")
    if not (float(row.get("trend_line", np.nan)) > float(row.get("long_line", np.nan))):
        failures.append("趋势线不高于多空线")
    return len(failures) == 0, failures


def main() -> None:
    cases = parse_case_files()
    name_code_map = build_name_code_map()
    if cases.empty:
        raise SystemExit("未找到可解析案例文件")

    signal_df = brick_filter.build_signal_df(NORMAL_DIR)
    selected_df = signal_df[
        (signal_df["score_pct_rank"] >= brick_filter.PCT_RANK_THRESHOLD)
        & (signal_df["daily_rank"] <= brick_filter.TOP_N)
    ].copy()

    results: List[dict] = []
    feature_rows: List[dict] = []

    for _, case in cases.iterrows():
        stock_name = str(case["stock_name"])
        code = name_code_map.get(stock_name, "")
        signal_date = pd.Timestamp(case["signal_date"])
        res = {
            "股票名称": stock_name,
            "股票代码": code,
            "信号日期": signal_date.strftime("%Y-%m-%d"),
            "案例文件": case["case_file"],
            "名称匹配成功": bool(code),
            "命中信号池": False,
            "命中最终筛选": False,
            "最终日内排名": None,
            "最终评分": None,
            "未命中原因": "",
        }
        if not code:
            res["未命中原因"] = "无法从原始数据映射股票代码"
            results.append(res)
            continue

        code_path = NORMAL_DIR / f"{code}.txt"
        row = get_row_with_extra_features(code_path, signal_date)
        if row is None:
            res["未命中原因"] = "标准化数据中无该日期"
            results.append(res)
            continue

        ok, failures = condition_flags(row)
        day_signal = signal_df[(signal_df["date"] == signal_date) & (signal_df["code"] == code)]
        day_selected = selected_df[(selected_df["date"] == signal_date) & (selected_df["code"] == code)]

        res["命中信号池"] = not day_signal.empty
        res["命中最终筛选"] = not day_selected.empty
        if not day_selected.empty:
            res["最终日内排名"] = int(day_selected.iloc[0]["daily_rank"])
            res["最终评分"] = round(float(day_selected.iloc[0]["sort_score"]), 4)
        elif not day_signal.empty:
            res["最终日内排名"] = int(day_signal.iloc[0]["daily_rank"])
            res["最终评分"] = round(float(day_signal.iloc[0]["sort_score"]), 4)

        if not ok:
            res["未命中原因"] = "；".join(failures)
        elif day_signal.empty:
            res["未命中原因"] = "未进入信号池(交叉日期对齐失败)"
        elif day_selected.empty:
            res["未命中原因"] = "进入信号池但未进前50%或前10"

        results.append(res)
        feature_rows.append(
            {
                "股票名称": stock_name,
                "股票代码": code,
                "信号日期": signal_date,
                "rebound_ratio": float(row.get("rebound_ratio", np.nan)),
                "signal_vs_ma5": float(row.get("signal_vs_ma5", np.nan)),
                "pullback_shrink_ratio": float(row.get("pullback_shrink_ratio", np.nan)),
                "ret1": float(row.get("ret1", np.nan)),
                "trend_spread": float((row.get("trend_line", np.nan) - row.get("long_line", np.nan)) / row.get("close", np.nan))
                if pd.notna(row.get("close"))
                else np.nan,
                "close_position": float(row.get("close_position", np.nan)),
                "upper_shadow_ratio": float(row.get("upper_shadow_ratio", np.nan)),
                "trend_slope_3": float(row.get("trend_slope_3", np.nan)),
                "trend_slope_5": float(row.get("trend_slope_5", np.nan)),
                "dist_20d_high": float(row.get("dist_20d_high", np.nan)),
                "dist_20d_low": float(row.get("dist_20d_low", np.nan)),
                "ret_5d_before": float(row.get("ret_5d_before", np.nan)),
                "ret_10d_before": float(row.get("ret_10d_before", np.nan)),
                "命中信号池": res["命中信号池"],
                "命中最终筛选": res["命中最终筛选"],
            }
        )

    result_df = pd.DataFrame(results).sort_values(["信号日期", "股票名称"]).reset_index(drop=True)
    feature_df = pd.DataFrame(feature_rows).sort_values(["信号日期", "股票名称"]).reset_index(drop=True)

    recent_min = feature_df["信号日期"].min()
    recent_max = feature_df["信号日期"].max()
    benchmark = signal_df[(signal_df["date"] >= recent_min) & (signal_df["date"] <= recent_max)].copy()

    if not benchmark.empty:
        code_to_name = {v: k for k, v in name_code_map.items()}
        bench_features = []
        for _, r in benchmark.iterrows():
            code = str(r["code"])
            row = get_row_with_extra_features(NORMAL_DIR / f"{code}.txt", pd.Timestamp(r["date"]))
            if row is None:
                continue
            bench_features.append(
                {
                    "股票名称": code_to_name.get(code, code),
                    "股票代码": code,
                    "信号日期": pd.Timestamp(r["date"]),
                    "rebound_ratio": float(row.get("rebound_ratio", np.nan)),
                    "signal_vs_ma5": float(row.get("signal_vs_ma5", np.nan)),
                    "pullback_shrink_ratio": float(row.get("pullback_shrink_ratio", np.nan)),
                    "ret1": float(row.get("ret1", np.nan)),
                    "trend_spread": float((row.get("trend_line", np.nan) - row.get("long_line", np.nan)) / row.get("close", np.nan))
                    if pd.notna(row.get("close"))
                    else np.nan,
                    "close_position": float(row.get("close_position", np.nan)),
                    "upper_shadow_ratio": float(row.get("upper_shadow_ratio", np.nan)),
                    "trend_slope_3": float(row.get("trend_slope_3", np.nan)),
                    "trend_slope_5": float(row.get("trend_slope_5", np.nan)),
                    "dist_20d_high": float(row.get("dist_20d_high", np.nan)),
                    "dist_20d_low": float(row.get("dist_20d_low", np.nan)),
                    "ret_5d_before": float(row.get("ret_5d_before", np.nan)),
                    "ret_10d_before": float(row.get("ret_10d_before", np.nan)),
                }
            )
        bench_df = pd.DataFrame(bench_features)
    else:
        bench_df = pd.DataFrame()

    summary = {
        "案例数": int(len(result_df)),
        "名称匹配成功数": int(result_df["名称匹配成功"].sum()) if not result_df.empty else 0,
        "命中信号池数": int(result_df["命中信号池"].sum()) if not result_df.empty else 0,
        "命中最终筛选数": int(result_df["命中最终筛选"].sum()) if not result_df.empty else 0,
        "案例日期范围": [
            str(feature_df["信号日期"].min().date()) if not feature_df.empty else None,
            str(feature_df["信号日期"].max().date()) if not feature_df.empty else None,
        ],
    }

    key_cols = [
        "rebound_ratio",
        "signal_vs_ma5",
        "pullback_shrink_ratio",
        "ret1",
        "trend_spread",
        "close_position",
        "upper_shadow_ratio",
        "trend_slope_3",
        "trend_slope_5",
        "dist_20d_high",
        "dist_20d_low",
        "ret_5d_before",
        "ret_10d_before",
    ]
    compare_rows: List[dict] = []
    if not feature_df.empty and not bench_df.empty:
        for col in key_cols:
            compare_rows.append(
                {
                    "指标": col,
                    "案例中位数": float(feature_df[col].median()),
                    "同期全部砖型信号中位数": float(bench_df[col].median()),
                    "案例均值": float(feature_df[col].mean()),
                    "同期全部砖型信号均值": float(bench_df[col].mean()),
                }
            )
    compare_df = pd.DataFrame(compare_rows)

    result_df.to_csv(OUT_DIR / "case_match_results.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUT_DIR / "case_features.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUT_DIR / "case_vs_benchmark_compare.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
