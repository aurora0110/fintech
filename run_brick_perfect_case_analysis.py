from __future__ import annotations

import importlib.util
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=FutureWarning)


BRICK_FILTER_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/utils/brick_filter.py")
IMAGE_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图")
DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/brick_perfect_case_analysis_20260312")


NAME_TO_CODE = {
    "中信重工": "SH#601608",
    "中材节能": "SH#603126",
    "中来股份": "SZ#300393",
    "中国化学": "SH#601117",
    "亿帆医药": "SZ#002019",
    "冠农股份": "SH#600251",
    "创意通": "SZ#300991",
    "凯格精机": "SZ#301338",
    "利民股份": "SZ#002734",
    "华懋科技": "SH#603306",
    "南网能源": "SZ#003035",
    "厚普股份": "SZ#300471",
    "大连电瓷": "SZ#002606",
    "威尔高": "SZ#301251",
    "建设机械": "SH#600984",
    "德福科技": "SZ#301511",
    "成地香江": "SH#603887",
    "晶盛机电": "SZ#300316",
    "沧州大化": "SH#600230",
    "润邦股份": "SZ#002483",
    "盈峰环境": "SZ#000967",
    "珈伟新能": "SZ#300317",
    "科瑞技术": "SZ#002957",
    "科翔股份": "SZ#300903",
    "衢州东峰": "SH#601515",
    "诺德股份": "SH#600110",
    "鹏鼎控股": "SZ#002938",
    "双象股份": "SZ#002395",
}


DATE_RE = re.compile(r"^(?P<name>.+?)(?P<date>20\d{6})(?:-[a-z0-9]+)?\.png$")


@dataclass
class CaseSample:
    name: str
    date: str
    code: str
    label: str
    source_file: str


def load_brick_filter_module():
    spec = importlib.util.spec_from_file_location("brick_filter", BRICK_FILTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def calc_kdj(df: pd.DataFrame) -> pd.DataFrame:
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k, "D": d, "J": j})


def calc_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def parse_case_samples() -> List[CaseSample]:
    samples: Dict[tuple[str, str], CaseSample] = {}
    for path in sorted(IMAGE_DIR.glob("*.png")):
        if path.name.startswith("案例图"):
            continue
        match = DATE_RE.match(path.name)
        if not match:
            continue
        name = match.group("name")
        date = match.group("date")
        label = "counterexample" if "反例" in name else "success"
        clean_name = name.replace("反例", "")
        code = NAME_TO_CODE.get(clean_name)
        if code is None:
            continue
        key = (clean_name, date)
        samples.setdefault(
            key,
            CaseSample(
                name=clean_name,
                date=f"{date[:4]}-{date[4:6]}-{date[6:]}",
                code=code,
                label=label,
                source_file=path.name,
            ),
        )
    return list(samples.values())


def enrich_stock_df(brick_filter, code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{code}.txt"
    df = brick_filter.load_one_csv(str(path))
    x = brick_filter.add_features(df)

    kdj = calc_kdj(x)
    x = pd.concat([x, kdj], axis=1)
    x["RSI14"] = calc_rsi(x["close"], 14)
    x["RSI28"] = calc_rsi(x["close"], 28)
    x["RSI57"] = calc_rsi(x["close"], 57)

    x["trend_spread"] = x["trend_line"] / x["long_line"] - 1.0
    x["close_to_trend"] = x["close"] / x["trend_line"] - 1.0
    x["close_to_long"] = x["close"] / x["long_line"] - 1.0
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["vol_ma10_prev"] = x["volume"].shift(1).rolling(10).mean()
    x["signal_vs_ma10"] = x["volume"] / x["vol_ma10_prev"]
    x["hhv20_prev"] = x["high"].shift(1).rolling(20).max()
    x["breakout_gap20"] = x["close"] / x["hhv20_prev"] - 1.0
    x["J_turn_up"] = x["J"] > x["J"].shift(1)
    x["RSI_stack"] = (x["RSI14"] > x["RSI28"]) & (x["RSI28"] > x["RSI57"])

    body = (x["close"] - x["open"]).abs()
    rng = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_abs"] = body
    x["body_pct"] = body / x["close"]
    x["upper_shadow"] = (x["high"] - x[["open", "close"]].max(axis=1)).clip(lower=0)
    x["lower_shadow"] = (x[["open", "close"]].min(axis=1) - x["low"]).clip(lower=0)
    x["upper_shadow_pct"] = x["upper_shadow"] / rng
    x["lower_shadow_pct"] = x["lower_shadow"] / rng
    x["close_location"] = (x["close"] - x["low"]) / rng

    x["touch_trend"] = x["low"] <= x["trend_line"] * 1.015
    x["touch_long"] = x["low"] <= x["long_line"] * 1.015
    x["trend_support_hold"] = x["low"] <= x["trend_line"] * 1.015
    x["long_support_hold"] = x["low"] <= x["long_line"] * 1.015
    x["trend_riding"] = x["close_to_trend"].between(0.0, 0.06, inclusive="both")

    x["green_bar"] = x["close"] > x["open"]
    x["double_vol_prev"] = x["volume"] >= x["volume"].shift(1) * 2.0
    x["double_bull_bar"] = x["green_bar"] & x["double_vol_prev"]
    x["prior_double_bull_20"] = x["double_bull_bar"].shift(1).rolling(20).max().fillna(0).astype(bool)

    x["double_bar_high"] = np.nan
    x["double_bar_low"] = np.nan
    x["double_bar_close"] = np.nan
    last_high = np.nan
    last_low = np.nan
    last_close = np.nan
    for idx, row in x.iterrows():
        if bool(row["double_bull_bar"]):
            last_high = float(row["high"])
            last_low = float(row["low"])
            last_close = float(row["close"])
        x.at[idx, "double_bar_high"] = last_high
        x.at[idx, "double_bar_low"] = last_low
        x.at[idx, "double_bar_close"] = last_close

    x["support_above_double_low"] = x["close"] >= x["double_bar_low"]
    x["support_above_double_close"] = x["close"] >= x["double_bar_close"]
    x["support_above_double_high"] = x["close"] >= x["double_bar_high"]
    x["dist_to_double_high"] = x["close"] / x["double_bar_high"] - 1.0
    x["dist_to_double_close"] = x["close"] / x["double_bar_close"] - 1.0
    x["dist_to_double_low"] = x["close"] / x["double_bar_low"] - 1.0

    x["prior5_green_ratio"] = x["green_bar"].shift(1).rolling(5).mean()
    x["prior5_avg_close_to_trend"] = x["close_to_trend"].shift(1).rolling(5).mean()
    x["price_vol_trend_sync"] = (x["ret5"] > 0) & (x["signal_vs_ma10"] > 0.8) & (x["signal_vs_ma10"] < 1.8)
    return x


def build_case_rows(
    brick_filter,
    samples: List[CaseSample],
    signal_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    stock_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[dict] = []
    for sample in samples:
        x = stock_cache[sample.code]
        dt = pd.Timestamp(sample.date)
        hit = x.loc[x["date"] == dt]
        if hit.empty:
            continue
        row = hit.iloc[0]
        in_signal = signal_df[(signal_df["code"] == sample.code) & (pd.to_datetime(signal_df["date"]) == dt)]
        in_selected = selected_df[(selected_df["code"] == sample.code) & (pd.to_datetime(selected_df["date"]) == dt)]

        mask_a = bool(row["pattern_a"]) and float(row["rebound_ratio"]) >= 1.2
        mask_b = bool(row["pattern_b"]) and float(row["rebound_ratio"]) >= 1.0
        raw_signal = bool(row["signal_base"]) and float(row["ret1"]) <= 0.08 and (mask_a or mask_b) and float(row["trend_line"]) > float(row["long_line"])

        fail_reasons: List[str] = []
        if not bool(row["pullback_shrinking"]):
            fail_reasons.append("pullback_shrinking")
        if not bool(row["signal_vs_ma5_valid"]):
            fail_reasons.append("signal_vs_ma5_valid")
        if not bool(row["not_sideways"]):
            fail_reasons.append("not_sideways")
        if not (bool(row["pattern_a"]) or bool(row["pattern_b"])):
            fail_reasons.append("pattern_ab")
        if bool(row["pattern_a"]) and not mask_a:
            fail_reasons.append("pattern_a_rebound_ratio")
        if bool(row["pattern_b"]) and not mask_b:
            fail_reasons.append("pattern_b_rebound_ratio")
        if float(row["ret1"]) > 0.08:
            fail_reasons.append("ret1_gt_8pct")
        if not (float(row["trend_line"]) > float(row["long_line"])):
            fail_reasons.append("trend_line_le_long_line")

        pos = x.index[x["date"] == dt][0]

        def future_ret(n: int) -> float:
            if pos + n >= len(x):
                return np.nan
            return float(x.iloc[pos + n]["close"] / x.iloc[pos]["close"] - 1.0)

        rows.append(
            {
                "code": sample.code,
                "name": sample.name,
                "date": sample.date,
                "label": sample.label,
                "source_file": sample.source_file,
                "raw_signal": raw_signal,
                "selected_top10": not in_selected.empty,
                "daily_rank": int(in_selected["daily_rank"].iloc[0]) if not in_selected.empty else (int(in_signal["daily_rank"].iloc[0]) if not in_signal.empty else np.nan),
                "sort_score": float(in_signal["sort_score"].iloc[0]) if not in_signal.empty else np.nan,
                "fail_reasons": "|".join(fail_reasons),
                "pattern_a": bool(row["pattern_a"]),
                "pattern_b": bool(row["pattern_b"]),
                "rebound_ratio": float(row["rebound_ratio"]) if pd.notna(row["rebound_ratio"]) else np.nan,
                "pullback_shrink_ratio": float(row["pullback_avg_vol"] / row["up_leg_avg_vol"]) if pd.notna(row["up_leg_avg_vol"]) and row["up_leg_avg_vol"] > 0 else np.nan,
                "signal_vs_ma5": float(row["signal_vs_ma5"]) if pd.notna(row["signal_vs_ma5"]) else np.nan,
                "signal_vs_ma10": float(row["signal_vs_ma10"]) if pd.notna(row["signal_vs_ma10"]) else np.nan,
                "trend_spread": float(row["trend_spread"]) if pd.notna(row["trend_spread"]) else np.nan,
                "close_to_trend": float(row["close_to_trend"]) if pd.notna(row["close_to_trend"]) else np.nan,
                "close_to_long": float(row["close_to_long"]) if pd.notna(row["close_to_long"]) else np.nan,
                "ret1": float(row["ret1"]) if pd.notna(row["ret1"]) else np.nan,
                "ret3": float(row["ret3"]) if pd.notna(row["ret3"]) else np.nan,
                "ret5": float(row["ret5"]) if pd.notna(row["ret5"]) else np.nan,
                "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else np.nan,
                "ret20": float(row["ret20"]) if pd.notna(row["ret20"]) else np.nan,
                "J": float(row["J"]) if pd.notna(row["J"]) else np.nan,
                "J_turn_up": bool(row["J_turn_up"]) if pd.notna(row["J_turn_up"]) else False,
                "RSI14": float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan,
                "RSI28": float(row["RSI28"]) if pd.notna(row["RSI28"]) else np.nan,
                "RSI57": float(row["RSI57"]) if pd.notna(row["RSI57"]) else np.nan,
                "RSI_stack": bool(row["RSI_stack"]) if pd.notna(row["RSI_stack"]) else False,
                "breakout_gap20": float(row["breakout_gap20"]) if pd.notna(row["breakout_gap20"]) else np.nan,
                "body_pct": float(row["body_pct"]) if pd.notna(row["body_pct"]) else np.nan,
                "upper_shadow_pct": float(row["upper_shadow_pct"]) if pd.notna(row["upper_shadow_pct"]) else np.nan,
                "lower_shadow_pct": float(row["lower_shadow_pct"]) if pd.notna(row["lower_shadow_pct"]) else np.nan,
                "close_location": float(row["close_location"]) if pd.notna(row["close_location"]) else np.nan,
                "touch_trend": bool(row["touch_trend"]) if pd.notna(row["touch_trend"]) else False,
                "touch_long": bool(row["touch_long"]) if pd.notna(row["touch_long"]) else False,
                "trend_riding": bool(row["trend_riding"]) if pd.notna(row["trend_riding"]) else False,
                "prior_double_bull_20": bool(row["prior_double_bull_20"]) if pd.notna(row["prior_double_bull_20"]) else False,
                "support_above_double_low": bool(row["support_above_double_low"]) if pd.notna(row["support_above_double_low"]) else False,
                "support_above_double_close": bool(row["support_above_double_close"]) if pd.notna(row["support_above_double_close"]) else False,
                "support_above_double_high": bool(row["support_above_double_high"]) if pd.notna(row["support_above_double_high"]) else False,
                "dist_to_double_high": float(row["dist_to_double_high"]) if pd.notna(row["dist_to_double_high"]) else np.nan,
                "dist_to_double_close": float(row["dist_to_double_close"]) if pd.notna(row["dist_to_double_close"]) else np.nan,
                "dist_to_double_low": float(row["dist_to_double_low"]) if pd.notna(row["dist_to_double_low"]) else np.nan,
                "prior5_green_ratio": float(row["prior5_green_ratio"]) if pd.notna(row["prior5_green_ratio"]) else np.nan,
                "prior5_avg_close_to_trend": float(row["prior5_avg_close_to_trend"]) if pd.notna(row["prior5_avg_close_to_trend"]) else np.nan,
                "price_vol_trend_sync": bool(row["price_vol_trend_sync"]) if pd.notna(row["price_vol_trend_sync"]) else False,
                "next1": future_ret(1),
                "next3": future_ret(3),
                "next5": future_ret(5),
                "next10": future_ret(10),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_rows(signal_df: pd.DataFrame, stock_cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    signal_2026 = signal_df[pd.to_datetime(signal_df["date"]).dt.year.eq(2026)].copy()
    for _, signal in signal_2026.iterrows():
        code = signal["code"]
        x = stock_cache.get(code)
        if x is None:
            continue
        hit = x.loc[x["date"] == pd.Timestamp(signal["date"])]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "code": code,
                "date": signal["date"],
                "rebound_ratio": row["rebound_ratio"],
                "pullback_shrink_ratio": float(row["pullback_avg_vol"] / row["up_leg_avg_vol"]) if pd.notna(row["up_leg_avg_vol"]) and row["up_leg_avg_vol"] > 0 else np.nan,
                "signal_vs_ma5": row["signal_vs_ma5"],
                "signal_vs_ma10": row["signal_vs_ma10"],
                "trend_spread": row["trend_spread"],
                "close_to_trend": row["close_to_trend"],
                "close_to_long": row["close_to_long"],
                "ret1": row["ret1"],
                "ret3": row["ret3"],
                "ret5": row["ret5"],
                "ret10": row["ret10"],
                "ret20": row["ret20"],
                "J": row["J"],
                "J_turn_up": bool(row["J_turn_up"]),
                "RSI14": row["RSI14"],
                "RSI28": row["RSI28"],
                "RSI57": row["RSI57"],
                "RSI_stack": bool(row["RSI_stack"]),
                "breakout_gap20": row["breakout_gap20"],
                "body_pct": row["body_pct"],
                "upper_shadow_pct": row["upper_shadow_pct"],
                "lower_shadow_pct": row["lower_shadow_pct"],
                "close_location": row["close_location"],
                "touch_trend": bool(row["touch_trend"]),
                "touch_long": bool(row["touch_long"]),
                "trend_riding": bool(row["trend_riding"]),
                "prior_double_bull_20": bool(row["prior_double_bull_20"]),
                "support_above_double_low": bool(row["support_above_double_low"]),
                "support_above_double_close": bool(row["support_above_double_close"]),
                "support_above_double_high": bool(row["support_above_double_high"]),
                "dist_to_double_high": row["dist_to_double_high"],
                "dist_to_double_close": row["dist_to_double_close"],
                "dist_to_double_low": row["dist_to_double_low"],
                "prior5_green_ratio": row["prior5_green_ratio"],
                "prior5_avg_close_to_trend": row["prior5_avg_close_to_trend"],
                "price_vol_trend_sync": bool(row["price_vol_trend_sync"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(case_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    success_df = case_df[case_df["label"] == "success"].copy()

    numeric_cols = [
        "rebound_ratio",
        "pullback_shrink_ratio",
        "signal_vs_ma5",
        "signal_vs_ma10",
        "trend_spread",
        "close_to_trend",
        "close_to_long",
        "ret1",
        "ret3",
        "ret5",
        "ret10",
        "ret20",
        "J",
        "RSI14",
        "RSI28",
        "RSI57",
        "breakout_gap20",
        "body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "close_location",
        "dist_to_double_high",
        "dist_to_double_close",
        "dist_to_double_low",
        "prior5_green_ratio",
        "prior5_avg_close_to_trend",
    ]
    bool_cols = [
        "pattern_a",
        "pattern_b",
        "raw_signal",
        "selected_top10",
        "J_turn_up",
        "RSI_stack",
        "touch_trend",
        "touch_long",
        "trend_riding",
        "prior_double_bull_20",
        "support_above_double_low",
        "support_above_double_close",
        "support_above_double_high",
        "price_vol_trend_sync",
    ]

    numeric_summary = pd.DataFrame(
        {
            "success_mean": success_df[numeric_cols].mean(),
            "all_signal_mean": baseline_df[numeric_cols].mean(),
        }
    )
    numeric_summary["diff"] = numeric_summary["success_mean"] - numeric_summary["all_signal_mean"]

    bool_summary = pd.DataFrame(
        {
            "success_rate": success_df[bool_cols].mean(),
            "all_signal_rate": baseline_df[[c for c in bool_cols if c in baseline_df.columns]].mean(),
        }
    )
    bool_summary["diff"] = bool_summary["success_rate"] - bool_summary["all_signal_rate"]

    fail_counts: Dict[str, int] = {}
    for raw in success_df["fail_reasons"].fillna(""):
        for reason in [item for item in raw.split("|") if item]:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

    return {
        "sample_count": int(len(case_df)),
        "success_case_count": int(len(success_df)),
        "raw_signal_hit_count": int(success_df["raw_signal"].sum()),
        "selected_top10_hit_count": int(success_df["selected_top10"].sum()),
        "avg_next1": float(success_df["next1"].mean()),
        "avg_next3": float(success_df["next3"].mean()),
        "avg_next5": float(success_df["next5"].mean()),
        "avg_next10": float(success_df["next10"].mean()),
        "top_fail_reasons": fail_counts,
        "numeric_summary": numeric_summary.round(6).to_dict(orient="index"),
        "bool_summary": bool_summary.round(6).to_dict(orient="index"),
    }


def main() -> None:
    brick_filter = load_brick_filter_module()
    samples = parse_case_samples()

    signal_df = brick_filter.build_signal_df(DATA_DIR)
    selected_df = brick_filter.apply_selection(signal_df)

    needed_codes = sorted(
        set(sample.code for sample in samples)
        | set(signal_df.loc[pd.to_datetime(signal_df["date"]).dt.year.eq(2026), "code"].tolist())
    )
    stock_cache = {code: enrich_stock_df(brick_filter, code) for code in needed_codes if (DATA_DIR / f"{code}.txt").exists()}

    case_df = build_case_rows(brick_filter, samples, signal_df, selected_df, stock_cache)
    baseline_df = build_baseline_rows(signal_df, stock_cache)
    summary = summarize(case_df, baseline_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(OUTPUT_DIR / "case_review.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIR / "baseline_2026_signals.csv", index=False)
    pd.DataFrame(summary["numeric_summary"]).T.to_csv(OUTPUT_DIR / "numeric_commonality.csv")
    pd.DataFrame(summary["bool_summary"]).T.to_csv(OUTPUT_DIR / "bool_commonality.csv")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k not in {"numeric_summary", "bool_summary"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
