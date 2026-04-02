#!/usr/bin/env python3
"""
B1趋势线偏离度分析实验
=========================

研究内容：
1. 只研究 知行趋势线 > 知行多空线 且 close > 两条线 的样本
2. 计算收盘价相对知行趋势线、知行多空线的上偏离百分比
3. 统计未来3/5/10日最大回撤、跌破概率、终点收益在不同偏离区间下的变化
4. 找出可作为卖点依据的偏离阈值

输出：
- 单变量分箱表
- 二维联合分箱表
- 热力图
- 文字结论
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import stoploss
from utils import technical_indicators

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except ImportError:
    PLOTTING_AVAILABLE = False
    print("警告: matplotlib/seaborn 未安装，将跳过热力图生成")

DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260312")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b1_trend_deviation_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_BARS = 160

TREND_BINS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 100]
LONG_BINS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 100]

HORIZONS = [3, 5, 10]


def load_stock_data(file_path: str) -> pd.DataFrame | None:
    """加载股票数据"""
    df, load_error = stoploss.load_data(file_path)
    if load_error or df is None or len(df) < MIN_HISTORY_BARS:
        return None
    
    df = df.sort_values("日期").reset_index(drop=True)
    
    numeric_cols = ["收盘", "最低", "最高", "开盘"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["收盘", "最低"]).copy()
    
    if len(df) < MIN_HISTORY_BARS:
        return None
    
    return df


def calculate_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """计算偏离度和未来收益指标"""
    df = df.copy()
    
    df = technical_indicators.calculate_trend(df)
    
    df["trend_deviation_pct"] = (df["收盘"] - df["知行短期趋势线"]) / df["知行短期趋势线"] * 100
    df["long_deviation_pct"] = (df["收盘"] - df["知行多空线"]) / df["知行多空线"] * 100
    
    for horizon in HORIZONS:
        df[f"future_{horizon}d_max_drawdown"] = np.nan
        df[f"future_{horizon}d_below_trend"] = np.nan
        df[f"future_{horizon}d_below_long"] = np.nan
        df[f"future_{horizon}d_return"] = np.nan
        df[f"future_{horizon}d_dd_gt_3"] = np.nan
        df[f"future_{horizon}d_dd_gt_5"] = np.nan
        df[f"future_{horizon}d_dd_gt_8"] = np.nan
        
        for i in range(len(df) - horizon):
            current_close = float(df["收盘"].iloc[i])
            future_slice = df.iloc[i+1:i+1+horizon]
            
            if future_slice.empty:
                continue
                
            future_lows = future_slice["最低"].values.astype(float)
            future_closes = future_slice["收盘"].values.astype(float)
            
            max_drawdown = (future_lows.min() / current_close - 1) * 100
            
            future_trend = future_slice["知行短期趋势线"].values.astype(float)
            future_long = future_slice["知行多空线"].values.astype(float)
            
            below_trend = float(np.any(future_lows < future_trend))
            below_long = float(np.any(future_lows < future_long))
            
            end_return = (future_closes[-1] / current_close - 1) * 100
            
            dd_gt_3 = float(max_drawdown <= -3.0)
            dd_gt_5 = float(max_drawdown <= -5.0)
            dd_gt_8 = float(max_drawdown <= -8.0)
            
            df.at[df.index[i], f"future_{horizon}d_max_drawdown"] = max_drawdown
            df.at[df.index[i], f"future_{horizon}d_below_trend"] = below_trend
            df.at[df.index[i], f"future_{horizon}d_below_long"] = below_long
            df.at[df.index[i], f"future_{horizon}d_return"] = end_return
            df.at[df.index[i], f"future_{horizon}d_dd_gt_3"] = dd_gt_3
            df.at[df.index[i], f"future_{horizon}d_dd_gt_5"] = dd_gt_5
            df.at[df.index[i], f"future_{horizon}d_dd_gt_8"] = dd_gt_8
    
    return df


def filter_samples(df: pd.DataFrame) -> pd.DataFrame:
    """筛选符合条件的样本：趋势线 > 多空线 且 close > 两条线"""
    mask = (
        (df["知行短期趋势线"] > 0) &
        (df["知行多空线"] > 0) &
        (df["知行短期趋势线"] > df["知行多空线"]) &
        (df["收盘"] > df["知行短期趋势线"]) &
        (df["收盘"] > df["知行多空线"])
    )
    return df[mask].copy().reset_index(drop=True)


def analyze_single_variable(samples: pd.DataFrame, deviation_col: str, bins: List[float], horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """单变量分箱分析，返回原始表和风险排序表"""
    df = samples.copy()
    
    valid_cols = [
        f"future_{horizon}d_max_drawdown",
        f"future_{horizon}d_below_trend",
        f"future_{horizon}d_below_long",
        f"future_{horizon}d_return",
        f"future_{horizon}d_dd_gt_3",
        f"future_{horizon}d_dd_gt_5",
        f"future_{horizon}d_dd_gt_8"
    ]
    df = df.dropna(subset=valid_cols).copy()
    
    df[f"{deviation_col}_bin"] = pd.cut(df[deviation_col], bins=bins, right=False)
    
    result = df.groupby(f"{deviation_col}_bin").agg(
        max_drawdown_mean=(f"future_{horizon}d_max_drawdown", "mean"),
        max_drawdown_median=(f"future_{horizon}d_max_drawdown", "median"),
        max_drawdown_min=(f"future_{horizon}d_max_drawdown", "min"),
        below_trend_prob=(f"future_{horizon}d_below_trend", "mean"),
        below_long_prob=(f"future_{horizon}d_below_long", "mean"),
        return_mean=(f"future_{horizon}d_return", "mean"),
        return_median=(f"future_{horizon}d_return", "median"),
        dd_gt_3_prob=(f"future_{horizon}d_dd_gt_3", "mean"),
        dd_gt_5_prob=(f"future_{horizon}d_dd_gt_5", "mean"),
        dd_gt_8_prob=(f"future_{horizon}d_dd_gt_8", "mean"),
        sample_count=(f"future_{horizon}d_return", "count")
    ).round(4)
    
    result_sorted = result.copy().reset_index()
    result_sorted = result_sorted.sort_values(
        by=["dd_gt_5_prob", "sample_count"],
        ascending=[False, False]
    ).set_index(f"{deviation_col}_bin")
    
    return result, result_sorted


def analyze_joint(samples: pd.DataFrame, trend_bins: List[float], long_bins: List[float], horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """二维联合分箱分析"""
    df = samples.copy()
    
    valid_cols = [
        f"future_{horizon}d_max_drawdown",
        f"future_{horizon}d_below_trend",
        f"future_{horizon}d_below_long",
        f"future_{horizon}d_return",
        f"future_{horizon}d_dd_gt_5"
    ]
    df = df.dropna(subset=valid_cols).copy()
    
    df["trend_bin"] = pd.cut(df["trend_deviation_pct"], bins=trend_bins, right=False)
    df["long_bin"] = pd.cut(df["long_deviation_pct"], bins=long_bins, right=False)
    
    pivot_drawdown = df.pivot_table(
        values=f"future_{horizon}d_max_drawdown",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    ).round(2)
    
    pivot_below_trend = df.pivot_table(
        values=f"future_{horizon}d_below_trend",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    ).round(4)
    
    pivot_return = df.pivot_table(
        values=f"future_{horizon}d_return",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    ).round(2)
    
    pivot_dd_gt_5 = df.pivot_table(
        values=f"future_{horizon}d_dd_gt_5",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    ).round(4)
    
    pivot_count = df.pivot_table(
        values=f"future_{horizon}d_return",
        index="trend_bin",
        columns="long_bin",
        aggfunc="count"
    ).fillna(0).astype(int)
    
    return pivot_drawdown, pivot_below_trend, pivot_return, pivot_dd_gt_5, pivot_count


def plot_heatmaps(samples: pd.DataFrame, trend_bins: List[float], long_bins: List[float], horizon: int, output_dir: Path):
    """绘制热力图"""
    if not PLOTTING_AVAILABLE:
        return
    
    df = samples.copy()
    
    valid_cols = [
        f"future_{horizon}d_max_drawdown",
        f"future_{horizon}d_below_trend",
        f"future_{horizon}d_return",
        f"future_{horizon}d_dd_gt_5"
    ]
    df = df.dropna(subset=valid_cols).copy()
    
    df["trend_bin"] = pd.cut(df["trend_deviation_pct"], bins=trend_bins, right=False)
    df["long_bin"] = pd.cut(df["long_deviation_pct"], bins=long_bins, right=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pivot_count = df.pivot_table(
        values=f"future_{horizon}d_return",
        index="trend_bin",
        columns="long_bin",
        aggfunc="count"
    ).fillna(0).astype(int)
    mask = pivot_count < 20
    
    pivot_drawdown = df.pivot_table(
        values=f"future_{horizon}d_max_drawdown",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    )
    sns.heatmap(pivot_drawdown.mask(mask), annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[0, 0])
    axes[0, 0].set_title(f"未来{horizon}日最大回撤(%)热力图 (样本<20已掩码)")
    
    pivot_dd_gt_5 = df.pivot_table(
        values=f"future_{horizon}d_dd_gt_5",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    )
    sns.heatmap(pivot_dd_gt_5.mask(mask), annot=True, fmt=".4f", cmap="RdYlGn_r", ax=axes[0, 1])
    axes[0, 1].set_title(f"未来{horizon}日回撤>5%概率热力图 (样本<20已掩码)")
    
    pivot_return = df.pivot_table(
        values=f"future_{horizon}d_return",
        index="trend_bin",
        columns="long_bin",
        aggfunc="mean"
    )
    sns.heatmap(pivot_return.mask(mask), annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1, 0])
    axes[1, 0].set_title(f"未来{horizon}日收益率(%)热力图 (样本<20已掩码)")
    
    sns.heatmap(pivot_count, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[1, 1])
    axes[1, 1].set_title(f"样本数量热力图")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmaps_{horizon}d.png", dpi=150, bbox_inches="tight")
    plt.close()


def find_min_threshold(valid_samples: pd.DataFrame, deviation_col: str, horizon: int, 
                       baseline_dd_gt_5: float, baseline_return: float) -> Tuple[float | None, pd.DataFrame]:
    """
    扫描并找出最小有效卖点阈值
    
    参数:
        valid_samples: 已过滤的有效样本DataFrame
        deviation_col: 偏离度列名
        horizon: 未来天数
        baseline_dd_gt_5: 全样本基准回撤>5%概率
        baseline_return: 全样本基准平均收益
    
    返回:
        (最小有效阈值, 阈值扫描结果表)
    """
    candidate_thresholds = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    
    df = valid_samples.copy()
    
    scan_results = []
    valid_thresholds = []
    
    for threshold in candidate_thresholds:
        mask = df[deviation_col] >= threshold
        threshold_samples = df[mask]
        
        if len(threshold_samples) < 200:
            scan_results.append({
                "threshold": threshold,
                "sample_count": len(threshold_samples),
                "dd_gt_5_prob": np.nan,
                "mean_drawdown": np.nan,
                "mean_return": np.nan,
                "is_valid": False,
                "reason": "样本数不足"
            })
            continue
        
        dd_gt_5_prob = threshold_samples[f"future_{horizon}d_dd_gt_5"].mean()
        mean_drawdown = threshold_samples[f"future_{horizon}d_max_drawdown"].mean()
        mean_return = threshold_samples[f"future_{horizon}d_return"].mean()
        
        is_valid = (
            dd_gt_5_prob >= baseline_dd_gt_5 + 0.05 and
            mean_return < baseline_return
        )
        
        scan_results.append({
            "threshold": threshold,
            "sample_count": len(threshold_samples),
            "dd_gt_5_prob": dd_gt_5_prob,
            "mean_drawdown": mean_drawdown,
            "mean_return": mean_return,
            "is_valid": is_valid,
            "reason": "有效" if is_valid else "未满足条件"
        })
        
        if is_valid:
            valid_thresholds.append(threshold)
    
    scan_df = pd.DataFrame(scan_results)
    
    min_threshold = valid_thresholds[0] if valid_thresholds else None
    
    return min_threshold, scan_df


def generate_conclusion(samples: pd.DataFrame, trend_bins: List[float], long_bins: List[float], 
                       output_dir: Path) -> str:
    """生成文字结论"""
    lines = ["=" * 80, "趋势线偏离度分析结论", "=" * 80, ""]
    
    horizon = 5
    valid_cols = [
        f"future_{horizon}d_max_drawdown",
        f"future_{horizon}d_below_trend",
        f"future_{horizon}d_below_long",
        f"future_{horizon}d_return",
        f"future_{horizon}d_dd_gt_3",
        f"future_{horizon}d_dd_gt_5",
        f"future_{horizon}d_dd_gt_8"
    ]
    valid_samples = samples.dropna(subset=valid_cols).copy()
    
    lines.append(f"总样本数: {len(samples)}")
    lines.append(f"有效样本数(有完整未来{horizon}日数据): {len(valid_samples)}")
    
    if len(valid_samples) == 0:
        lines.append("")
        lines.append("警告: 无有效样本可用于分析！")
        return "\n".join(lines)
    
    lines.append(f"趋势线偏离度范围: {valid_samples['trend_deviation_pct'].min():.2f}% ~ {valid_samples['trend_deviation_pct'].max():.2f}%")
    lines.append(f"多空线偏离度范围: {valid_samples['long_deviation_pct'].min():.2f}% ~ {valid_samples['long_deviation_pct'].max():.2f}%")
    lines.append("")
    
    baseline_dd_gt_5 = valid_samples[f"future_{horizon}d_dd_gt_5"].mean()
    baseline_mean_drawdown = valid_samples[f"future_{horizon}d_max_drawdown"].mean()
    baseline_mean_return = valid_samples[f"future_{horizon}d_return"].mean()
    baseline_below_trend = valid_samples[f"future_{horizon}d_below_trend"].mean()
    baseline_below_long = valid_samples[f"future_{horizon}d_below_long"].mean()
    
    lines.append("=" * 80)
    lines.append(f"【全样本基准（未来{horizon}日）】")
    lines.append("=" * 80)
    lines.append(f"平均最大回撤: {baseline_mean_drawdown:.2f}%")
    lines.append(f"回撤>3%概率: {valid_samples[f'future_{horizon}d_dd_gt_3'].mean():.2%}")
    lines.append(f"回撤>5%概率: {baseline_dd_gt_5:.2%}")
    lines.append(f"回撤>8%概率: {valid_samples[f'future_{horizon}d_dd_gt_8'].mean():.2%}")
    lines.append(f"跌破趋势线概率: {baseline_below_trend:.2%}")
    lines.append(f"跌破多空线概率: {baseline_below_long:.2%}")
    lines.append(f"平均终点收益: {baseline_mean_return:.2f}%")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("【最小有效阈值扫描】")
    lines.append("=" * 80)
    
    min_threshold_trend, scan_df_trend = find_min_threshold(
        valid_samples, "trend_deviation_pct", horizon, baseline_dd_gt_5, baseline_mean_return
    )
    min_threshold_long, scan_df_long = find_min_threshold(
        valid_samples, "long_deviation_pct", horizon, baseline_dd_gt_5, baseline_mean_return
    )
    
    scan_df_trend.to_csv(output_dir / f"threshold_scan_trend_{horizon}d.csv", encoding="utf-8-sig", index=False)
    scan_df_long.to_csv(output_dir / f"threshold_scan_long_{horizon}d.csv", encoding="utf-8-sig", index=False)
    
    lines.append(f"趋势线偏离最小有效阈值: {min_threshold_trend}%")
    lines.append(f"多空线偏离最小有效阈值: {min_threshold_long}%")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append(f"【高风险偏离区间（未来{horizon}日，样本≥100）】")
    lines.append("=" * 80)
    
    high_risk_thresholds = []
    for deviation_col in ["trend_deviation_pct", "long_deviation_pct"]:
        name = "趋势线" if deviation_col == "trend_deviation_pct" else "多空线"
        
        df = valid_samples.copy()
        bins = TREND_BINS if deviation_col == "trend_deviation_pct" else LONG_BINS
        df["bin"] = pd.cut(df[deviation_col], bins=bins, right=False)
        
        for bin_val in df["bin"].cat.categories:
            if pd.isna(bin_val):
                continue
            bin_samples = df[df["bin"] == bin_val]
            if len(bin_samples) < 100:
                continue
                
            bin_dd_gt_5 = bin_samples[f"future_{horizon}d_dd_gt_5"].mean()
            bin_drawdown = bin_samples[f"future_{horizon}d_max_drawdown"].mean()
            bin_return = bin_samples[f"future_{horizon}d_return"].mean()
            
            if (bin_dd_gt_5 >= baseline_dd_gt_5 + 0.05 and 
                bin_return < baseline_mean_return):
                high_risk_thresholds.append((name, bin_val, len(bin_samples), bin_dd_gt_5, bin_drawdown, bin_return))
    
    if high_risk_thresholds:
        lines.append("满足条件的区间（回撤>5%概率≥基准+5% 且 平均收益<基准）:")
        for item in sorted(high_risk_thresholds, key=lambda x: x[3], reverse=True):
            name, bin_val, cnt, dd_gt_5, dd, ret = item
            lines.append(f"  {name} {bin_val}: 样本={cnt}, 回撤>5%={dd_gt_5:.2%}, 平均回撤={dd:.2f}%, 平均收益={ret:.2f}%")
    else:
        lines.append("  未发现满足条件的高风险区间")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("说明:")
    lines.append("=" * 80)
    lines.append("  1. 最大回撤定义: min(未来N日最低价 / T日收盘价 - 1) * 100")
    lines.append("  2. 跌破定义: 未来N日最低价 < 当日对应趋势线/多空线")
    lines.append("  3. 最小有效阈值条件: 样本≥200, 回撤>5%概率≥基准+5%, 平均收益<基准")
    lines.append("  4. 热力图中样本数<20的格子已掩码，避免随机噪声干扰")
    lines.append("  5. 建议优先参考最小有效阈值和高风险区间综合判断")
    
    return "\n".join(lines)


def main():
    """主函数"""
    print("=" * 80)
    print("开始B1趋势线偏离度分析实验")
    print("=" * 80)
    
    all_samples = []
    file_paths = sorted(DATA_DIR.glob("*.txt"))
    
    print(f"\n发现 {len(file_paths)} 个股票文件，开始处理...")
    
    for idx, file_path in enumerate(file_paths[:]):
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(file_paths)}")
        
        try:
            df = load_stock_data(str(file_path))
            if df is None:
                continue
                
            df = calculate_deviations(df)
            filtered_df = filter_samples(df)
            
            if not filtered_df.empty:
                filtered_df["stock_code"] = file_path.stem
                all_samples.append(filtered_df)
        except Exception as e:
            print(f"处理 {file_path.name} 时出错: {e}")
            continue
    
    if not all_samples:
        print("\n未找到符合条件的样本！")
        return
    
    samples = pd.concat(all_samples, ignore_index=True)
    print(f"\n处理完成！共获得 {len(samples)} 个符合条件的样本")
    
    samples.to_csv(OUTPUT_DIR / "all_samples.csv", index=False, encoding="utf-8-sig")
    print(f"样本数据已保存到: {OUTPUT_DIR / 'all_samples.csv'}")
    
    print("\n开始分析...")
    
    for horizon in HORIZONS:
        print(f"\n分析未来{horizon}日指标...")
        
        trend_analysis, trend_analysis_sorted = analyze_single_variable(samples, "trend_deviation_pct", TREND_BINS, horizon)
        long_analysis, long_analysis_sorted = analyze_single_variable(samples, "long_deviation_pct", LONG_BINS, horizon)
        
        trend_analysis.to_csv(OUTPUT_DIR / f"single_variable_trend_{horizon}d.csv", encoding="utf-8-sig")
        trend_analysis_sorted.to_csv(OUTPUT_DIR / f"single_variable_trend_{horizon}d_sorted.csv", encoding="utf-8-sig")
        long_analysis.to_csv(OUTPUT_DIR / f"single_variable_long_{horizon}d.csv", encoding="utf-8-sig")
        long_analysis_sorted.to_csv(OUTPUT_DIR / f"single_variable_long_{horizon}d_sorted.csv", encoding="utf-8-sig")
        
        pivot_drawdown, pivot_below_trend, pivot_return, pivot_dd_gt_5, pivot_count = analyze_joint(
            samples, TREND_BINS, LONG_BINS, horizon
        )
        
        pivot_drawdown.to_csv(OUTPUT_DIR / f"joint_drawdown_{horizon}d.csv", encoding="utf-8-sig")
        pivot_below_trend.to_csv(OUTPUT_DIR / f"joint_below_trend_{horizon}d.csv", encoding="utf-8-sig")
        pivot_return.to_csv(OUTPUT_DIR / f"joint_return_{horizon}d.csv", encoding="utf-8-sig")
        pivot_dd_gt_5.to_csv(OUTPUT_DIR / f"joint_dd_gt_5_{horizon}d.csv", encoding="utf-8-sig")
        pivot_count.to_csv(OUTPUT_DIR / f"joint_count_{horizon}d.csv", encoding="utf-8-sig")
        
        plot_heatmaps(samples, TREND_BINS, LONG_BINS, horizon, OUTPUT_DIR)
        if PLOTTING_AVAILABLE:
            print(f"热力图已保存: heatmaps_{horizon}d.png")
        else:
            print("已跳过热力图生成（缺少 matplotlib/seaborn）")
    
    print("\n生成结论...")
    conclusion = generate_conclusion(samples, TREND_BINS, LONG_BINS, OUTPUT_DIR)
    
    with open(OUTPUT_DIR / "conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion)
    
    print("\n" + "=" * 80)
    print(conclusion)
    print("\n" + "=" * 80)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("分析完成！")


if __name__ == "__main__":
    main()
