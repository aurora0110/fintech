#!/usr/bin/env python3
"""
BRICK 连续砖转色统计分析
验证"连续4根红砖后更容易转绿、连续4根绿砖后更容易转红"是否成立

输入目录: /Users/lidongyang/Desktop/Qstrategy/data/20260317/normal
输出目录: /Users/lidongyang/Desktop/Qstrategy/results/brick_turning_point_YYYYMMDD_HHMMSS/
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.brick_filter import (
    add_features,
    calc_green_streak,
    load_one_csv,
    read_csv_auto,
    pick_col,
    safe_div,
    tdx_sma,
    MIN_BARS,
)

INPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/20260317/normal"
OUTPUT_DIR = os.path.join(
    "/Users/lidongyang/Desktop/Qstrategy/results",
    f"brick_turning_point_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
MAX_N = 8


def safe_div_f(a, b, default=np.nan):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, default, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def compute_brick(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().reset_index(drop=True)
    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["trend_line"] = (
        x["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    )
    x["long_line"] = (
        x["close"].rolling(14).mean()
        + x["close"].rolling(28).mean()
        + x["close"].rolling(57).mean()
        + x["close"].rolling(114).mean()
    ) / 4.0
    x["trend_spread"] = safe_div_f(x["trend_line"] - x["long_line"], x["close"])
    x["close_to_long"] = safe_div_f(x["close"] - x["long_line"], x["long_line"])

    hhv4 = x["high"].rolling(4).max()
    llv4 = x["low"].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div_f((hhv4 - x["close"]), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=x.index), 4, 1) + 100
    var3a = safe_div_f((x["close"] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=x.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a
    x["brick"] = np.where(var6a > 4, var6a - 4, 0.0)
    x["brick_prev"] = x["brick"].shift(1)
    x["brick_red_len"] = np.where(x["brick"] > x["brick_prev"], x["brick"] - x["brick_prev"], 0.0)
    x["brick_green_len"] = np.where(x["brick"] < x["brick_prev"], x["brick_prev"] - x["brick"], 0.0)
    x["brick_red"] = x["brick_red_len"] > 0
    x["brick_green"] = x["brick_green_len"] > 0

    green_flag = x["brick_green"].to_numpy()
    green_streak = calc_green_streak(green_flag)
    x["green_streak"] = green_streak
    x["prev_green_streak"] = pd.Series(green_streak, index=x.index).shift(1)

    red_flag = x["brick_red"].to_numpy()
    red_streak = calc_green_streak(red_flag)
    x["red_streak"] = red_streak
    x["prev_red_streak"] = pd.Series(red_streak, index=x.index).shift(1)

    x["next_brick_red"] = x["brick_red"].shift(-1)
    x["next_brick_green"] = x["brick_green"].shift(-1)
    x["next2_brick_red"] = x["brick_red"].shift(-2)
    x["next2_brick_green"] = x["brick_green"].shift(-2)
    x["next3_brick_red"] = x["brick_red"].shift(-3)
    x["next3_brick_green"] = x["brick_green"].shift(-3)

    x["valid"] = (
        x["brick"].notna()
        & x["brick_prev"].notna()
        & x["close"].notna()
        & np.isfinite(x["close"])
        & np.isfinite(x["brick"])
        & np.isfinite(x["brick_prev"])
    )
    return x


def load_data(input_dir: str):
    paths = list(Path(input_dir).glob("*.txt"))
    print(f"加载 {len(paths)} 个文件...")
    rows = []
    for p in tqdm(paths, desc="加载数据"):
        df = load_one_csv(str(p))
        if df is None:
            continue
        try:
            x = compute_brick(df)
            x = x[x["valid"]].copy()
            if len(x) < MIN_BARS:
                continue
            rows.append(x)
        except Exception:
            continue
    if not rows:
        raise ValueError("没有有效数据")
    return pd.concat(rows, ignore_index=True)


def event_stats(df: pd.DataFrame, streak_col: str, target_red: bool, n_max: int = MAX_N):
    records = []
    for n in range(1, n_max + 1):
        mask = (df[streak_col] == n) & df["valid"]
        total = int(mask.sum())
        if total == 0:
            continue

        if target_red:
            p1 = df.loc[mask, "next_brick_red"].mean()
            p2 = df.loc[mask, ["next_brick_red", "next2_brick_red"]].any(axis=1).mean()
            p3 = df.loc[mask, ["next_brick_red", "next2_brick_red", "next3_brick_red"]].any(axis=1).mean()
        else:
            p1 = df.loc[mask, "next_brick_green"].mean()
            p2 = df.loc[mask, ["next_brick_green", "next2_brick_green"]].any(axis=1).mean()
            p3 = df.loc[mask, ["next_brick_green", "next2_brick_green", "next3_brick_green"]].any(axis=1).mean()

        records.append(
            {
                "streak_n": n,
                "count": total,
                "p_next_turn": round(p1, 4),
                "p_within2": round(p2, 4),
                "p_within3": round(p3, 4),
            }
        )
    return pd.DataFrame(records)


def significance_test(df: pd.DataFrame):
    rows = []
    pairs = [
        ("red", 4, 3),
        ("red", 4, 5),
        ("green", 4, 3),
        ("green", 4, 5),
    ]
    for color, focal, other in pairs:
        streak_col = f"prev_{color}_streak"
        target_col = f"next_brick_{'red' if color == 'red' else 'green'}"
        for n, n2 in [(focal, other)]:
            m1 = (df[streak_col] == n) & df["valid"] & df[target_col].notna()
            m2 = (df[streak_col] == n2) & df["valid"] & df[target_col].notna()
            x1 = df.loc[m1, target_col].dropna().astype(int)
            x2 = df.loc[m2, target_col].dropna().astype(int)
            if len(x1) < 10 or len(x2) < 10:
                continue
            p1 = x1.sum() / len(x1)
            p2 = x2.sum() / len(x2)
            n1, n2 = len(x1), len(x2)
            p_pool = (x1.sum() + x2.sum()) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            stat = (p1 - p2) / se if se > 1e-12 else 0.0
            p_val = 2 * (1 - stats.norm.cdf(abs(stat)))
            rows.append(
                {
                    "color": color,
                    "focal_n": n,
                    "compare_n": n2,
                    "focal_count": len(x1),
                    "focal_rate": round(x1.mean(), 4),
                    "compare_count": len(x2),
                    "compare_rate": round(x2.mean(), 4),
                    "z_stat": round(stat, 4),
                    "p_value": round(p_val, 6),
                    "significant_05": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")),
                }
            )
    return pd.DataFrame(rows)


def forward_returns(df: pd.DataFrame, condition_mask, label: str, horizons=(1, 3, 5)):
    m = condition_mask & df["valid"]
    records = []
    for h in horizons:
        ret_col = f"ret{h}"
        if ret_col not in df.columns:
            df[ret_col] = df["close"].pct_change(h)
        rets = df.loc[m, ret_col].dropna()
        if len(rets) == 0:
            continue
        records.append(
            {
                "label": label,
                "horizon": h,
                "count": len(rets),
                "mean": round(rets.mean() * 100, 4),
                "median": round(rets.median() * 100, 4),
                "std": round(rets.std() * 100, 4),
                "positive_rate": round((rets > 0).mean(), 4),
                "min": round(rets.min() * 100, 4),
                "max": round(rets.max() * 100, 4),
            }
        )
    return pd.DataFrame(records)


def brick_filter_backtest(df: pd.DataFrame):
    records = []
    configs = [
        ("baseline", (df["brick_red"] | df["brick_green"]) & df["valid"]),
        (
            "red4_filter",
            (df["brick_red"] | df["brick_green"])
            & df["valid"]
            & ~((df["prev_red_streak"] >= 4) & df["brick_red"]),
        ),
        (
            "green4_enhance",
            (df["brick_red"] | df["brick_green"])
            & df["valid"]
            & ~((df["prev_green_streak"] >= 4) & df["brick_green"]),
        ),
    ]
    for name, base_mask in configs:
        for h in [1, 3, 5]:
            col = f"ret{h}"
            rets = df.loc[base_mask, col].dropna()
            if len(rets) == 0:
                continue
            records.append(
                {
                    "filter": name,
                    "horizon": h,
                    "count": len(rets),
                    "mean_pct": round(rets.mean() * 100, 4),
                    "median_pct": round(rets.median() * 100, 4),
                    "positive_rate": round((rets > 0).mean(), 4),
                    "std_pct": round(rets.std() * 100, 4),
                }
            )
    return pd.DataFrame(records)


def layered_stats(df: pd.DataFrame, streak_col: str, target_red: bool, layer_col: str, layer_name: str):
    records = []
    layers = sorted(df[layer_col].dropna().unique())
    n_levels = min(3, len(layers))
    if n_levels == 3:
        boundaries = [df[layer_col].quantile(q) for q in [0.33, 0.67]]
        def assign_bin(v):
            if pd.isna(v):
                return "mid"
            if v <= boundaries[0]:
                return "low"
            if v >= boundaries[1]:
                return "high"
            return "mid"
    else:
        med = df[layer_col].median()
        def assign_bin(v):
            if pd.isna(v):
                return "mid"
            return "low" if v <= med else "high"

    df["_layer_bin"] = df[layer_col].apply(assign_bin)
    for n in range(1, MAX_N + 1):
        for bin_label in ["low", "mid", "high"]:
            mask = (df[streak_col] == n) & (df["_layer_bin"] == bin_label) & df["valid"]
            total = int(mask.sum())
            if total < 30:
                continue
            if target_red:
                p1 = df.loc[mask, "next_brick_red"].mean()
                p2 = df.loc[mask, ["next_brick_red", "next2_brick_red"]].any(axis=1).mean()
                p3 = df.loc[mask, ["next_brick_red", "next2_brick_red", "next3_brick_red"]].any(axis=1).mean()
            else:
                p1 = df.loc[mask, "next_brick_green"].mean()
                p2 = df.loc[mask, ["next_brick_green", "next2_brick_green"]].any(axis=1).mean()
                p3 = df.loc[mask, ["next_brick_green", "next2_brick_green", "next3_brick_green"]].any(axis=1).mean()
            records.append(
                {
                    "streak_n": n,
                    "layer": bin_label,
                    "layer_name": layer_name,
                    "count": total,
                    "p_next_turn": round(p1, 4),
                    "p_within2": round(p2, 4),
                    "p_within3": round(p3, 4),
                }
            )
    df.drop(columns=["_layer_bin"], inplace=True)
    return pd.DataFrame(records)


def build_report(
    red_summary,
    green_summary,
    sig,
    g2r_fwd,
    r2g_fwd,
    filter_bt,
    red_trend,
    green_trend,
    red_pos,
    green_pos,
    red_str,
    green_str,
) -> str:
    def make_table(df, cols):
        if df.empty:
            return "*数据不足*"
        return df[cols].to_markdown(index=False)

    lines = [
        "# BRICK 转色拐点统计分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据来源**: {INPUT_DIR}",
        "",
        "## 1. 实验目的",
        "",
        "验证 BRICK 策略中「连续4根红砖后更容易转绿、连续4根绿砖后更容易转红」这一经验规律是否具有统计显著性，",
        "以及该规律是否在特定市场环境下更为显著、是否具有交易价值。",
        "",
        "## 2. 核心结论",
        "",
    ]

    try:
        r4 = red_summary[red_summary["streak_n"] == 4].iloc[0] if 4 in red_summary["streak_n"].values else None
        g4 = green_summary[green_summary["streak_n"] == 4].iloc[0] if 4 in green_summary["streak_n"].values else None
        if r4 is not None:
            lines.append(f"- 红4时，下一根转绿概率 = **{r4['p_next_turn']:.2%}**（样本数={r4['count']}）")
        if g4 is not None:
            lines.append(f"- 绿4时，下一根转红概率 = **{g4['p_next_turn']:.2%}**（样本数={g4['count']}）")
    except Exception:
        pass

    lines.append("")
    lines.append("## 3. 红砖侧统计（连续红砖后转绿）")
    lines.append("")
    lines.append(make_table(red_summary[["streak_n", "count", "p_next_turn", "p_within2", "p_within3"]], ["streak_n", "count", "p_next_turn", "p_within2", "p_within3"]))
    lines.append("")
    lines.append("## 4. 绿砖侧统计（连续绿砖后转红）")
    lines.append("")
    lines.append(make_table(green_summary[["streak_n", "count", "p_next_turn", "p_within2", "p_within3"]], ["streak_n", "count", "p_next_turn", "p_within2", "p_within3"]))
    lines.append("")
    lines.append("## 5. 显著性检验（红4/绿4是否特殊）")
    lines.append("")
    if not sig.empty:
        lines.append(make_table(sig, ["color", "focal_n", "compare_n", "focal_count", "focal_rate", "compare_rate", "z_stat", "p_value", "significant_05"]))
    else:
        lines.append("*样本不足，无法进行显著性检验*")
    lines.append("")
    lines.append("## 6. 分层统计")
    for tag, red_df, green_df, name in [
        ("趋势环境", red_trend, green_trend, "trend"),
        ("位置分层", red_pos, green_pos, "position"),
        ("砖强度", red_str, green_str, "brick_strength"),
    ]:
        lines.append(f"\n### 6.x {tag}\n")
        lines.append("**红砖侧**")
        if not red_df.empty:
            lines.append(make_table(red_df[red_df["streak_n"] == 4], ["streak_n", "layer", "count", "p_next_turn", "p_within2", "p_within3"]))
        else:
            lines.append("*数据不足*")
        lines.append("**绿砖侧**")
        if not green_df.empty:
            lines.append(make_table(green_df[green_df["streak_n"] == 4], ["streak_n", "layer", "count", "p_next_turn", "p_within2", "p_within3"]))
        else:
            lines.append("*数据不足*")
    lines.append("")
    lines.append("## 7. 交易验证：绿4转红后持有收益")
    lines.append("")
    lines.append(make_table(g2r_fwd, ["label", "horizon", "count", "mean", "median", "positive_rate"]))
    lines.append("")
    lines.append("## 8. 交易验证：红4转绿后持有收益")
    lines.append("")
    lines.append(make_table(r2g_fwd, ["label", "horizon", "count", "mean", "median", "positive_rate"]))
    lines.append("")
    lines.append("## 9. BRICK 过滤回测对比")
    lines.append("")
    lines.append(make_table(filter_bt, ["filter", "horizon", "count", "mean_pct", "positive_rate"]))
    lines.append("")
    lines.append("## 10. 综合结论")
    lines.append("")
    lines.append("### 4 是否真的特殊？")
    try:
        sig_row = sig[sig["focal_n"] == 4]
        if not sig_row.empty:
            for _, row in sig_row.iterrows():
                star = row["significant_05"]
                lines.append(
                    f"- {row['color']}4 vs {row['color']}{row['compare_n']}: "
                    f"z={row['z_stat']:.3f}, p={row['p_value']:.4f} {star}"
                )
        else:
            lines.append("- 样本不足，无法判断")
    except Exception:
        lines.append("- 数据不足以得出结论")
    lines.append("")
    lines.append("### 红砖侧是否成立？")
    try:
        r4_row = red_summary[red_summary["streak_n"] == 4].iloc[0]
        baseline = red_summary["p_next_turn"].mean()
        lines.append(f"- 红4时转绿概率 {r4_row['p_next_turn']:.2%}，全样本平均 {baseline:.2%}，{'高于' if r4_row['p_next_turn'] > baseline else '低于'}平均")
    except Exception:
        lines.append("- 数据不足")
    lines.append("")
    lines.append("### 绿砖侧是否成立？")
    try:
        g4_row = green_summary[green_summary["streak_n"] == 4].iloc[0]
        baseline = green_summary["p_next_turn"].mean()
        lines.append(f"- 绿4时转红概率 {g4_row['p_next_turn']:.2%}，全样本平均 {baseline:.2%}，{'高于' if g4_row['p_next_turn'] > baseline else '低于'}平均")
    except Exception:
        lines.append("- 数据不足")
    lines.append("")
    lines.append("### 是否只在特定环境成立？")
    try:
        if not red_trend.empty:
            r4_trend = red_trend[red_trend["streak_n"] == 4]
            if not r4_trend.empty:
                for _, row in r4_trend.iterrows():
                    lines.append(f"- 红4+{row['layer']}层: 转绿概率={row['p_next_turn']:.2%}（n={row['count']}）")
    except Exception:
        pass
    try:
        if not green_trend.empty:
            g4_trend = green_trend[green_trend["streak_n"] == 4]
            if not g4_trend.empty:
                for _, row in g4_trend.iterrows():
                    lines.append(f"- 绿4+{row['layer']}层: 转红概率={row['p_next_turn']:.2%}（n={row['count']}）")
    except Exception:
        pass
    lines.append("")
    lines.append("### 是否有交易价值？")
    try:
        if not g2r_fwd.empty:
            g4_ret = g2r_fwd[g2r_fwd["label"].str.contains("绿4")]
            if not g4_ret.empty:
                for _, row in g4_ret.iterrows():
                    lines.append(
                        f"- 绿4转红后持有{row['horizon']}天: 均值={row['mean']:.2f}%, "
                        f"胜率={row['positive_rate']:.2%}（n={row['count']}）"
                    )
    except Exception:
        pass
    lines.append("")
    lines.append("## 11. 风险提示")
    lines.append("")
    lines.append("- 本分析基于历史数据，不代表未来表现")
    lines.append("- 样本量较小的分层统计结论需谨慎解读")
    lines.append("- 实际交易需考虑滑点、流动性、涨跌停等限制")
    lines.append("- 显著性检验结果仅供参考，0.05显著性阈值是人为设定")
    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    start = time.time()

    print("\n[1/9] 加载数据...")
    df = load_data(INPUT_DIR)
    print(f"  有效记录数: {len(df)}")

    print("\n[2/9] 红砖侧事件统计...")
    red_summary = event_stats(df, "prev_red_streak", target_red=False)
    path = os.path.join(OUTPUT_DIR, "red_reversal_summary.csv")
    red_summary.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存: {path}")

    print("\n[3/9] 绿砖侧事件统计...")
    green_summary = event_stats(df, "prev_green_streak", target_red=True)
    path = os.path.join(OUTPUT_DIR, "green_reversal_summary.csv")
    green_summary.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存: {path}")

    print("\n[4/9] 分层统计...")
    df["trend_ok"] = df["trend_line"] > df["long_line"]
    for col, label in [("trend_ok", "trend")]:
        r = layered_stats(df, "prev_red_streak", False, col, label)
        g = layered_stats(df, "prev_green_streak", True, col, label)
        r.to_csv(os.path.join(OUTPUT_DIR, f"red_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")
        g.to_csv(os.path.join(OUTPUT_DIR, f"green_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")

    pos_col = "close_to_long"
    for col, label in [(pos_col, "position")]:
        r = layered_stats(df, "prev_red_streak", False, col, label)
        g = layered_stats(df, "prev_green_streak", True, col, label)
        r.to_csv(os.path.join(OUTPUT_DIR, f"red_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")
        g.to_csv(os.path.join(OUTPUT_DIR, f"green_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")

    for col, label in [("brick_green_len", "brick_strength")]:
        r = layered_stats(df, "prev_red_streak", False, col, label)
        g = layered_stats(df, "prev_green_streak", True, col, label)
        r.to_csv(os.path.join(OUTPUT_DIR, f"red_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")
        g.to_csv(os.path.join(OUTPUT_DIR, f"green_reversal_by_{label}.csv"), index=False, encoding="utf-8-sig")
    print("  保存: red_reversal_by_*.csv, green_reversal_by_*.csv")

    print("\n[5/9] 显著性检验...")
    sig = significance_test(df)
    path = os.path.join(OUTPUT_DIR, "brick_turning_point_significance.csv")
    sig.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存: {path}")

    print("\n[6/9] 交易验证：绿4转红后持有收益...")
    g2r_mask = (df["prev_green_streak"] == 4) & df["brick_red"] & df["valid"]
    g2r_fwd = forward_returns(df, g2r_mask, "绿4转红", horizons=(1, 3, 5))
    g2r_fwd.to_csv(os.path.join(OUTPUT_DIR, "green_to_red_forward_returns.csv"), index=False, encoding="utf-8-sig")

    r2g_mask = (df["prev_red_streak"] == 4) & df["brick_green"] & df["valid"]
    r2g_fwd = forward_returns(df, r2g_mask, "红4转绿", horizons=(1, 3, 5))
    r2g_fwd.to_csv(os.path.join(OUTPUT_DIR, "red_to_green_forward_returns.csv"), index=False, encoding="utf-8-sig")
    print(f"  保存: green_to_red_forward_returns.csv, red_to_green_forward_returns.csv")

    print("\n[7/9] BRICK 过滤回测...")
    filter_bt = brick_filter_backtest(df)
    path = os.path.join(OUTPUT_DIR, "brick_turning_filter_backtest.csv")
    filter_bt.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存: {path}")

    print("\n[8/9] 生成报告...")
    red_trend = pd.read_csv(os.path.join(OUTPUT_DIR, "red_reversal_by_trend.csv"))
    green_trend = pd.read_csv(os.path.join(OUTPUT_DIR, "green_reversal_by_trend.csv"))
    red_pos = pd.read_csv(os.path.join(OUTPUT_DIR, "red_reversal_by_position.csv"))
    green_pos = pd.read_csv(os.path.join(OUTPUT_DIR, "green_reversal_by_position.csv"))
    red_str = pd.read_csv(os.path.join(OUTPUT_DIR, "red_reversal_by_brick_strength.csv"))
    green_str = pd.read_csv(os.path.join(OUTPUT_DIR, "green_reversal_by_brick_strength.csv"))

    report = build_report(
        red_summary, green_summary, sig,
        g2r_fwd, r2g_fwd, filter_bt,
        red_trend, green_trend, red_pos, green_pos, red_str, green_str,
    )
    report_path = os.path.join(OUTPUT_DIR, "brick_turning_point_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  保存: {report_path}")

    print("\n[9/9] 完成!")
    print(f"总耗时: {time.time() - start:.1f}s")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
