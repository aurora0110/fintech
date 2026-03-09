from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_loader import _read_txt
from strategies.common import base_prepare


@dataclass(frozen=True)
class TestSpec:
    name: str
    hold_days: int
    target_return: float


def classify_j_bucket(j_value: float) -> str:
    if pd.isna(j_value):
        return "nan"
    if j_value <= 0:
        return "very_low"
    if j_value <= 30:
        return "mid_low"
    if j_value <= 60:
        return "neutral"
    return "high"


def permutation_pvalue(x: np.ndarray, y: np.ndarray, seed: int, n_iter: int = 1000) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    observed = abs(float(np.mean(x) - np.mean(y)))
    combined = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_iter):
        rng.shuffle(combined)
        diff = abs(float(np.mean(combined[: len(x)]) - np.mean(combined[len(x) :])))
        if diff >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_iter + 1)


def two_prop_ztest(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    if min(total_a, total_b) == 0:
        return math.nan
    p1 = success_a / total_a
    p2 = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    denom = math.sqrt(max(pooled * (1 - pooled) * (1 / total_a + 1 / total_b), 1e-12))
    z = abs(p1 - p2) / denom
    return math.erfc(z / math.sqrt(2))


def build_events(stock_data: dict[str, pd.DataFrame], specs: list[TestSpec]) -> pd.DataFrame:
    events: list[dict[str, object]] = []
    min_hold = max(spec.hold_days for spec in specs)

    for code, raw_df in stock_data.items():
        df = raw_df.reset_index().rename(
            columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close"}
        )
        df = base_prepare(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        if len(df) <= 130 + min_hold:
            continue

        df["prev_close"] = df["close"].shift(1)
        df["prev_short"] = df["short_trend"].shift(1)
        df["prev_long"] = df["long_trend"].shift(1)
        df["trend_pullback"] = (
            (df["prev_close"] > df["prev_short"])
            & (df["low"] <= df["short_trend"])
            & (df["close"] >= df["short_trend"])
            & (df["low"] > df["long_trend"])
            & (df["short_trend"] > df["long_trend"])
        )
        df["long_pullback"] = (
            (df["prev_close"] > df["prev_long"])
            & (df["low"] <= df["long_trend"])
            & (df["close"] >= df["long_trend"])
            & (df["short_trend"] > df["long_trend"])
        )

        for idx in range(130, len(df) - min_hold):
            row = df.iloc[idx]
            if pd.isna(row["close"]) or float(row["close"]) <= 0:
                continue
            event_type = None
            line_value = math.nan
            if bool(row["long_pullback"]):
                event_type = "long_line"
                line_value = float(row["long_trend"])
            elif bool(row["trend_pullback"]):
                event_type = "trend_line"
                line_value = float(row["short_trend"])
            if event_type is None or pd.isna(row["J"]):
                continue

            future = df.iloc[idx + 1 : idx + 1 + min_hold]
            if len(future) < min_hold:
                continue

            item: dict[str, object] = {
                "code": code,
                "date": row["date"],
                "event_type": event_type,
                "J": float(row["J"]),
                "j_bucket": classify_j_bucket(float(row["J"])),
            }

            for spec in specs:
                window = df.iloc[idx + 1 : idx + 1 + spec.hold_days]
                max_return = float(window["close"].max() / row["close"] - 1.0)
                no_breakdown = bool((window["close"] >= line_value * 0.98).all())
                item[f"{spec.name}_success"] = bool(max_return >= spec.target_return and no_breakdown)
                item[f"{spec.name}_max_return"] = max_return
            events.append(item)

    return pd.DataFrame(events)


def load_stock_data(data_dir: str, stock_limit: int, seed: int) -> dict[str, pd.DataFrame]:
    stock_data: dict[str, pd.DataFrame] = {}
    paths = sorted(Path(data_dir).glob("*.txt"))
    if stock_limit > 0:
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(paths), size=min(stock_limit, len(paths)), replace=False).tolist())
        paths = [paths[idx] for idx in pick]
    for path in paths:
        code = path.stem
        df = _read_txt(str(path))
        if df is None or len(df) < 160:
            continue
        df["code"] = code
        df = df.set_index("date")
        stock_data[code] = df
    return stock_data


def summarize_by_bucket(events: pd.DataFrame, success_col: str) -> pd.DataFrame:
    summary = (
        events.groupby(["event_type", "j_bucket"], observed=True)
        .agg(
            samples=("J", "size"),
            mean_j=("J", "mean"),
            success_rate=(success_col, "mean"),
            median_j=("J", "median"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["success_rate"] * 100
    return summary.sort_values(["event_type", "j_bucket"])


def compare_core_claim(events: pd.DataFrame, success_col: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_type in ["trend_line", "long_line"]:
        sub = events[events["event_type"] == event_type].copy()
        low = sub[sub["j_bucket"] == "very_low"][success_col].astype(int).to_numpy()
        mid = sub[sub["j_bucket"] == "mid_low"][success_col].astype(int).to_numpy()
        if len(low) == 0 or len(mid) == 0:
            continue
        rows.append(
            {
                "event_type": event_type,
                "very_low_samples": len(low),
                "mid_low_samples": len(mid),
                "very_low_success_rate": low.mean() * 100,
                "mid_low_success_rate": mid.mean() * 100,
                "rate_diff_pct": (low.mean() - mid.mean()) * 100,
                "two_prop_pvalue": two_prop_ztest(int(low.sum()), len(low), int(mid.sum()), len(mid)),
                "permutation_pvalue": permutation_pvalue(low, mid, seed=seed + (1 if event_type == "trend_line" else 2)),
            }
        )
    return pd.DataFrame(rows)


def robustness_vote(claim_df: pd.DataFrame) -> tuple[int, int]:
    if claim_df.empty:
        return 0, 0
    passes = 0
    total = 0
    for _, row in claim_df.iterrows():
        total += 1
        if row["event_type"] == "trend_line":
            if row["two_prop_pvalue"] >= 0.05 and row["rate_diff_pct"] <= 5:
                passes += 1
        elif row["event_type"] == "long_line":
            if row["two_prop_pvalue"] < 0.05 and row["rate_diff_pct"] > 0:
                passes += 1
    return passes, total


def main() -> None:
    parser = argparse.ArgumentParser(description="检验回踩趋势线/多空线时 J 值说法的统计显著性")
    parser.add_argument("--data-dir", default="/Users/lidongyang/Desktop/Qstrategy/data/forward_data")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stock-limit", type=int, default=0)
    args = parser.parse_args()

    stock_data = load_stock_data(args.data_dir, args.stock_limit, args.seed)
    print(f"loaded_stocks={len(stock_data)}", flush=True)
    specs = [
        TestSpec("h10_r06", hold_days=10, target_return=0.06),
        TestSpec("h15_r08", hold_days=15, target_return=0.08),
        TestSpec("h20_r10", hold_days=20, target_return=0.10),
    ]
    events = build_events(stock_data, specs)
    if events.empty:
        raise SystemExit("未生成任何事件，请检查数据目录或事件定义。")

    print(f"total_events={len(events)}")
    print(events["event_type"].value_counts().sort_index().to_string())

    overall_votes = []
    for spec in specs:
        success_col = f"{spec.name}_success"
        print(f"\n=== {spec.name} | hold={spec.hold_days}d | target={spec.target_return:.0%} ===")
        print("\n[bucket_summary]")
        print(summarize_by_bucket(events, success_col).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print("\n[core_claim_test]")
        claim_df = compare_core_claim(events, success_col, seed=args.seed)
        if claim_df.empty:
            print("no comparable samples")
        else:
            print(claim_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        passes, total = robustness_vote(claim_df)
        overall_votes.append((spec.name, passes, total))

    print("\n[robustness_vote]")
    for name, passes, total in overall_votes:
        print(f"{name}: {passes}/{total}")


if __name__ == "__main__":
    main()
