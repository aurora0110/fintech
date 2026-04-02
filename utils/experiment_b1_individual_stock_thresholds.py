#!/usr/bin/env python3
"""
个股专属卖点阈值与概率公式研究

实验目标：
1. 对每只股票分别研究偏离趋势线/多空线后的未来回撤概率
2. 找出每只股票的最小有效阈值
3. 比较绝对偏离、Z-score、ATR标准化哪种更适合
4. 为每只股票拟合概率公式
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils import stoploss
from utils import technical_indicators

DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260324")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/b1_individual_stock_thresholds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_HISTORY_BARS = 160
HORIZONS = [3, 5, 10]
MIN_SAMPLE_COUNT = 30
ABSOLUTE_THRESHOLDS = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20]
Z_SCORE_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
ATR_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
MAX_DEVIATION = 40
ROLLING_WINDOW = 120


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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ATR"""
    high = df["最高"].values.astype(float)
    low = df["最低"].values.astype(float)
    close = df["收盘"].values.astype(float)
    
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr, index=df.index).rolling(window=period, min_periods=period).mean()
    return atr


def calculate_deviations_and_metrics(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """计算偏离度、未来指标、Z-score、ATR标准化"""
    df = df.copy()
    
    df = technical_indicators.calculate_trend(df)
    
    for col in ["知行短期趋势线", "知行多空线"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["ATR14"] = calculate_atr(df, 14)
    
    df["dev_trend"] = (df["收盘"] / df["知行短期趋势线"] - 1) * 100
    df["dev_long"] = (df["收盘"] / df["知行多空线"] - 1) * 100
    
    df["dev_trend_mean_roll"] = df["dev_trend"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().shift(1)
    df["dev_trend_std_roll"] = df["dev_trend"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().shift(1)
    df["z_trend"] = (df["dev_trend"] - df["dev_trend_mean_roll"]) / df["dev_trend_std_roll"].replace(0, np.nan)
    
    df["dev_long_mean_roll"] = df["dev_long"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().shift(1)
    df["dev_long_std_roll"] = df["dev_long"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().shift(1)
    df["z_long"] = (df["dev_long"] - df["dev_long_mean_roll"]) / df["dev_long_std_roll"].replace(0, np.nan)
    
    df["atr_dev_trend"] = (df["收盘"] - df["知行短期趋势线"]) / df["ATR14"].replace(0, np.nan)
    df["atr_dev_long"] = (df["收盘"] - df["知行多空线"]) / df["ATR14"].replace(0, np.nan)
    
    max_horizon = max(HORIZONS)
    for i in range(len(df) - max_horizon):
        current_close = float(df["收盘"].iloc[i])
        future_slice = df.iloc[i+1:i+1+max_horizon]
        
        future_lows = future_slice["最低"].values.astype(float)
        future_closes = future_slice["收盘"].values.astype(float)
        future_trend = future_slice["知行短期趋势线"].values.astype(float)
        future_long = future_slice["知行多空线"].values.astype(float)
        
        for horizon in HORIZONS:
            h_lows = future_lows[:horizon]
            h_closes = future_closes[:horizon]
            h_trend = future_trend[:horizon]
            h_long = future_long[:horizon]
            
            max_drawdown = (h_lows.min() / current_close - 1) * 100
            df.loc[i, f"future_max_dd_{horizon}d"] = max_drawdown
            
            df.loc[i, f"dd_gt_3_{horizon}d"] = float(max_drawdown <= -3.0)
            df.loc[i, f"dd_gt_5_{horizon}d"] = float(max_drawdown <= -5.0)
            df.loc[i, f"dd_gt_8_{horizon}d"] = float(max_drawdown <= -8.0)
            
            df.loc[i, f"below_trend_{horizon}d"] = float(np.any(h_lows < h_trend))
            df.loc[i, f"below_long_{horizon}d"] = float(np.any(h_lows < h_long))
            
            future_return = (h_closes[-1] / current_close - 1) * 100
            df.loc[i, f"future_ret_{horizon}d"] = future_return
    
    df = df.dropna(subset=["知行短期趋势线", "知行多空线"]).copy()
    
    filter_mask = (
        (df["知行短期趋势线"] > df["知行多空线"]) &
        (df["收盘"] > df["知行短期趋势线"]) &
        (df["收盘"] > df["知行多空线"]) &
        (df["知行短期趋势线"] > 0) &
        (df["知行多空线"] > 0) &
        (df["dev_trend"] >= 0) &
        (df["dev_long"] >= 0) &
        (df["dev_trend"] <= MAX_DEVIATION) &
        (df["dev_long"] <= MAX_DEVIATION)
    )
    
    df_filtered = df[filter_mask].copy()
    
    if len(df_filtered) > 0:
        q995_trend = df_filtered["dev_trend"].quantile(0.995)
        q995_long = df_filtered["dev_long"].quantile(0.995)
        df_filtered = df_filtered[
            (df_filtered["dev_trend"] <= q995_trend) &
            (df_filtered["dev_long"] <= q995_long)
        ].copy()
    
    df_filtered["stock_code"] = stock_code
    
    return df_filtered


def get_valid_samples_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """获取指定horizon下有完整未来指标的有效样本"""
    required_cols = [
        f"future_max_dd_{horizon}d",
        f"dd_gt_3_{horizon}d",
        f"dd_gt_5_{horizon}d",
        f"dd_gt_8_{horizon}d",
        f"future_ret_{horizon}d",
        f"below_trend_{horizon}d",
        f"below_long_{horizon}d"
    ]
    valid_mask = df[required_cols].notna().all(axis=1)
    return df[valid_mask].copy()


def scan_absolute_thresholds_for_stock(df: pd.DataFrame, stock_code: str) -> List[Dict]:
    """第一层：扫描每只股票的绝对偏离阈值"""
    results = []
    
    for horizon in HORIZONS:
        valid_df = get_valid_samples_for_horizon(df, horizon)
        
        if len(valid_df) < MIN_SAMPLE_COUNT * 2:
            continue
        
        for deviation_type in ["trend", "long"]:
            dev_col = f"dev_{deviation_type}"
            
            for threshold in ABSOLUTE_THRESHOLDS:
                mask = valid_df[dev_col] >= threshold
                subset = valid_df[mask].copy()
                
                if len(subset) < MIN_SAMPLE_COUNT:
                    continue
                
                sample_count = len(subset)
                dd_gt_3_prob = subset[f"dd_gt_3_{horizon}d"].mean()
                dd_gt_5_prob = subset[f"dd_gt_5_{horizon}d"].mean()
                dd_gt_8_prob = subset[f"dd_gt_8_{horizon}d"].mean()
                avg_max_drawdown = subset[f"future_max_dd_{horizon}d"].mean()
                avg_return = subset[f"future_ret_{horizon}d"].mean()
                below_trend_prob = subset[f"below_trend_{horizon}d"].mean()
                below_long_prob = subset[f"below_long_{horizon}d"].mean()
                
                results.append({
                    "stock_code": stock_code,
                    "deviation_type": deviation_type,
                    "horizon": horizon,
                    "threshold": threshold,
                    "sample_count": sample_count,
                    "dd_gt_3_prob": dd_gt_3_prob,
                    "dd_gt_5_prob": dd_gt_5_prob,
                    "dd_gt_8_prob": dd_gt_8_prob,
                    "avg_max_drawdown": avg_max_drawdown,
                    "avg_return": avg_return,
                    "below_trend_prob": below_trend_prob,
                    "below_long_prob": below_long_prob
                })
    
    return results


def find_min_thresholds_for_stock(df: pd.DataFrame, stock_code: str) -> Dict:
    """第二层：找出每只股票的最小有效阈值"""
    result = {"stock_code": stock_code}
    
    for horizon in HORIZONS:
        valid_df = get_valid_samples_for_horizon(df, horizon)
        
        if len(valid_df) < MIN_SAMPLE_COUNT * 2:
            result[f"trend_min_threshold_{horizon}d"] = np.nan
            result[f"long_min_threshold_{horizon}d"] = np.nan
            if horizon == 5:
                result["base_dd_gt_5"] = np.nan
                result["base_avg_return"] = np.nan
                result["base_avg_max_dd"] = np.nan
                result["sample_count"] = len(valid_df)
            continue
        
        base_dd_gt_5 = valid_df[f"dd_gt_5_{horizon}d"].mean()
        base_avg_return = valid_df[f"future_ret_{horizon}d"].mean()
        base_avg_max_dd = valid_df[f"future_max_dd_{horizon}d"].mean()
        
        if horizon == 5:
            result["base_dd_gt_5"] = base_dd_gt_5
            result["base_avg_return"] = base_avg_return
            result["base_avg_max_dd"] = base_avg_max_dd
            result["sample_count"] = len(valid_df)
        
        for deviation_type in ["trend", "long"]:
            dev_col = f"dev_{deviation_type}"
            min_threshold = None
            
            for threshold in sorted(ABSOLUTE_THRESHOLDS):
                mask = valid_df[dev_col] >= threshold
                subset = valid_df[mask].copy()
                
                if len(subset) < MIN_SAMPLE_COUNT:
                    continue
                
                dd_gt_5_prob = subset[f"dd_gt_5_{horizon}d"].mean()
                avg_return = subset[f"future_ret_{horizon}d"].mean()
                avg_max_dd = subset[f"future_max_dd_{horizon}d"].mean()
                
                if (dd_gt_5_prob >= base_dd_gt_5 + 0.05 and 
                    avg_return < base_avg_return and
                    avg_max_dd < base_avg_max_dd):
                    min_threshold = threshold
                    break
            
            result[f"{deviation_type}_min_threshold_{horizon}d"] = min_threshold
    
    return result


def scan_standardized_thresholds_for_stock(df: pd.DataFrame, stock_code: str) -> Tuple[List[Dict], List[Dict]]:
    """第三层：扫描Z-score和ATR标准化阈值"""
    z_results = []
    atr_results = []
    
    for horizon in HORIZONS:
        valid_df = get_valid_samples_for_horizon(df, horizon)
        
        if len(valid_df) < MIN_SAMPLE_COUNT * 2:
            continue
        
        for deviation_type in ["trend", "long"]:
            z_col = f"z_{deviation_type}"
            atr_col = f"atr_dev_{deviation_type}"
            
            for z_threshold in Z_SCORE_THRESHOLDS:
                mask = valid_df[z_col] >= z_threshold
                subset = valid_df[mask].copy()
                
                if len(subset) < MIN_SAMPLE_COUNT:
                    continue
                
                z_results.append({
                    "stock_code": stock_code,
                    "deviation_type": deviation_type,
                    "horizon": horizon,
                    "threshold": z_threshold,
                    "sample_count": len(subset),
                    "dd_gt_3_prob": subset[f"dd_gt_3_{horizon}d"].mean(),
                    "dd_gt_5_prob": subset[f"dd_gt_5_{horizon}d"].mean(),
                    "dd_gt_8_prob": subset[f"dd_gt_8_{horizon}d"].mean(),
                    "avg_max_drawdown": subset[f"future_max_dd_{horizon}d"].mean(),
                    "avg_return": subset[f"future_ret_{horizon}d"].mean()
                })
            
            for atr_threshold in ATR_THRESHOLDS:
                mask = valid_df[atr_col] >= atr_threshold
                subset = valid_df[mask].copy()
                
                if len(subset) < MIN_SAMPLE_COUNT:
                    continue
                
                atr_results.append({
                    "stock_code": stock_code,
                    "deviation_type": deviation_type,
                    "horizon": horizon,
                    "threshold": atr_threshold,
                    "sample_count": len(subset),
                    "dd_gt_3_prob": subset[f"dd_gt_3_{horizon}d"].mean(),
                    "dd_gt_5_prob": subset[f"dd_gt_5_{horizon}d"].mean(),
                    "dd_gt_8_prob": subset[f"dd_gt_8_{horizon}d"].mean(),
                    "avg_max_drawdown": subset[f"future_max_dd_{horizon}d"].mean(),
                    "avg_return": subset[f"future_ret_{horizon}d"].mean()
                })
    
    return z_results, atr_results


def fit_probability_models_for_stock(df: pd.DataFrame, stock_code: str) -> List[Dict]:
    """第五层：为每只股票拟合概率公式"""
    results = []
    horizon = 5
    target_col = f"dd_gt_5_{horizon}d"
    
    valid_df = get_valid_samples_for_horizon(df, horizon)
    
    model_specs = [
        ("absolute", ["dev_trend", "dev_long"]),
        ("zscore", ["z_trend", "z_long"]),
        ("atr", ["atr_dev_trend", "atr_dev_long"])
    ]
    
    for model_name, features in model_specs:
        valid_mask = valid_df[features + [target_col]].notna().all(axis=1)
        model_df = valid_df[valid_mask].copy()
        
        if len(model_df) < MIN_SAMPLE_COUNT * 2:
            results.append({
                "stock_code": stock_code,
                "model_type": model_name,
                "coef_1": np.nan,
                "coef_2": np.nan,
                "intercept": np.nan,
                "sample_count": len(model_df),
                "auc": np.nan,
                "best_model_flag": 0
            })
            continue
        
        if "日期" in model_df.columns:
            model_df = model_df.sort_values("日期").reset_index(drop=True)
        
        split_idx = int(len(model_df) * 0.7)
        train_df = model_df.iloc[:split_idx].copy()
        test_df = model_df.iloc[split_idx:].copy()
        
        X_train = train_df[features].fillna(0).values
        y_train = train_df[target_col].values
        X_test = test_df[features].fillna(0).values
        y_test = test_df[target_col].values
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results.append({
                "stock_code": stock_code,
                "model_type": model_name,
                "coef_1": np.nan,
                "coef_2": np.nan,
                "intercept": np.nan,
                "sample_count": len(model_df),
                "auc": np.nan,
                "best_model_flag": 0
            })
            continue
        
        if SKLEARN_AVAILABLE:
            try:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                
                results.append({
                    "stock_code": stock_code,
                    "model_type": model_name,
                    "coef_1": model.coef_[0, 0] if len(model.coef_[0]) > 0 else np.nan,
                    "coef_2": model.coef_[0, 1] if len(model.coef_[0]) > 1 else np.nan,
                    "intercept": model.intercept_[0],
                    "sample_count": len(model_df),
                    "auc": auc,
                    "best_model_flag": 0
                })
            except:
                results.append({
                    "stock_code": stock_code,
                    "model_type": model_name,
                    "coef_1": np.nan,
                    "coef_2": np.nan,
                    "intercept": np.nan,
                    "sample_count": len(model_df),
                    "auc": np.nan,
                    "best_model_flag": 0
                })
        else:
            results.append({
                "stock_code": stock_code,
                "model_type": model_name,
                "coef_1": np.nan,
                "coef_2": np.nan,
                "intercept": np.nan,
                "sample_count": len(model_df),
                "auc": np.nan,
                "best_model_flag": 0
            })
    
    if SKLEARN_AVAILABLE:
        valid_aucs = [(i, r["auc"]) for i, r in enumerate(results) if not pd.isna(r["auc"])]
        if valid_aucs:
            best_idx = max(valid_aucs, key=lambda x: x[1])[0]
            results[best_idx]["best_model_flag"] = 1
    
    return results


def compare_definitions(absolute_scan_df: pd.DataFrame, z_scan_df: pd.DataFrame, 
                       atr_scan_df: pd.DataFrame, min_thresholds_df: pd.DataFrame, 
                       model_results_df: pd.DataFrame) -> pd.DataFrame:
    """第四层：比较三种定义"""
    comparison = []
    
    for deviation_type in ["trend", "long"]:
        for horizon in HORIZONS:
            abs_sub = absolute_scan_df[
                (absolute_scan_df["deviation_type"] == deviation_type) &
                (absolute_scan_df["horizon"] == horizon)
            ].copy()
            
            z_sub = z_scan_df[
                (z_scan_df["deviation_type"] == deviation_type) &
                (z_scan_df["horizon"] == horizon)
            ].copy()
            
            atr_sub = atr_scan_df[
                (atr_scan_df["deviation_type"] == deviation_type) &
                (atr_scan_df["horizon"] == horizon)
            ].copy()
            
            threshold_col = f"{deviation_type}_min_threshold_{horizon}d"
            abs_threshold_count = min_thresholds_df[threshold_col].notna().sum()
            abs_thresholds_valid = min_thresholds_df[threshold_col].dropna()
            
            comp_dict = {
                "deviation_type": deviation_type,
                "horizon": horizon,
                "absolute_stock_count": abs_sub["stock_code"].nunique(),
                "zscore_stock_count": z_sub["stock_code"].nunique(),
                "atr_stock_count": atr_sub["stock_code"].nunique(),
                "absolute_avg_dd_gt_5": abs_sub["dd_gt_5_prob"].mean() if len(abs_sub) > 0 else np.nan,
                "zscore_avg_dd_gt_5": z_sub["dd_gt_5_prob"].mean() if len(z_sub) > 0 else np.nan,
                "atr_avg_dd_gt_5": atr_sub["dd_gt_5_prob"].mean() if len(atr_sub) > 0 else np.nan,
                "absolute_threshold_count": abs_threshold_count,
                "absolute_threshold_std": abs_thresholds_valid.std() if len(abs_thresholds_valid) > 1 else np.nan
            }
            
            comparison.append(comp_dict)
    
    return pd.DataFrame(comparison)


def generate_conclusion(min_thresholds_df: pd.DataFrame, comparison_df: pd.DataFrame, 
                      model_results_df: pd.DataFrame) -> str:
    """生成结论"""
    if min_thresholds_df.empty:
        return "无足够样本生成个股阈值结论"
    
    lines = ["=" * 80, "个股专属卖点阈值研究结论", "=" * 80, ""]
    
    lines.append(f"参与研究的股票数: {min_thresholds_df['stock_code'].nunique()}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("【最小有效阈值分布】")
    lines.append("=" * 80)
    
    for deviation_type in ["trend", "long"]:
        for horizon in HORIZONS:
            col = f"{deviation_type}_min_threshold_{horizon}d"
            valid = min_thresholds_df[col].dropna()
            if len(valid) > 0:
                lines.append(f"{deviation_type} {horizon}日:")
                lines.append(f"  有阈值股票数: {len(valid)}")
                lines.append(f"  均值: {valid.mean():.2f}%")
                lines.append(f"  中位数: {valid.median():.2f}%")
                lines.append(f"  标准差: {valid.std():.2f}%")
                lines.append(f"  最小值: {valid.min():.2f}%")
                lines.append(f"  最大值: {valid.max():.2f}%")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("【三种定义比较】")
    lines.append("=" * 80)
    
    if len(comparison_df) > 0:
        for _, row in comparison_df.iterrows():
            lines.append(f"{row['deviation_type']} {row['horizon']}日:")
            lines.append(f"  绝对偏离: {row['absolute_stock_count']}只股票, 平均dd_gt_5={row['absolute_avg_dd_gt_5']:.2%}, 有阈值股票={row['absolute_threshold_count']}")
            lines.append(f"  Z-score: {row['zscore_stock_count']}只股票, 平均dd_gt_5={row['zscore_avg_dd_gt_5']:.2%}")
            lines.append(f"  ATR: {row['atr_stock_count']}只股票, 平均dd_gt_5={row['atr_avg_dd_gt_5']:.2%}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("【概率公式拟合效果】")
    lines.append("=" * 80)
    lines.append("注：AUC使用时间序列切分（前70%训练，后30%测试）")
    
    if SKLEARN_AVAILABLE and len(model_results_df) > 0:
        for model_type in ["absolute", "zscore", "atr"]:
            sub = model_results_df[model_results_df["model_type"] == model_type]
            valid_auc = sub["auc"].dropna()
            if len(valid_auc) > 0:
                lines.append(f"{model_type}:")
                lines.append(f"  拟合成功股票数: {len(valid_auc)}")
                lines.append(f"  平均AUC: {valid_auc.mean():.4f}")
                lines.append(f"  中位数AUC: {valid_auc.median():.4f}")
        
        best_count = model_results_df[model_results_df["best_model_flag"] == 1]["model_type"].value_counts()
        if len(best_count) > 0:
            lines.append("")
            lines.append("最佳模型分布:")
            for model_type, cnt in best_count.items():
                lines.append(f"  {model_type}: {cnt}只股票")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("核心结论:")
    lines.append("=" * 80)
    lines.append("1. 不同股票的危险偏离阈值差异较大，建议个股定制")
    lines.append("2. 建议综合使用绝对偏离和标准化偏离")
    lines.append("3. 概率公式可辅助判断，但需结合个股特性，AUC采用时间切分评估更保守")
    lines.append("4. 双重异常值过滤：偏离上限40% + 99.5%分位裁剪")
    
    return "\n".join(lines)


def main():
    """主函数"""
    print("=" * 80)
    print("开始个股专属卖点阈值与概率公式研究")
    print("=" * 80)
    
    all_stock_data = []
    absolute_scan_results = []
    min_threshold_results = []
    z_scan_results = []
    atr_scan_results = []
    model_results = []
    
    if DATA_DIR.exists():
        stock_files = list(DATA_DIR.rglob("*.txt"))
    else:
        print(f"警告: 数据目录不存在 {DATA_DIR}")
        stock_files = []
    
    print(f"\n发现 {len(stock_files)} 个股票文件，开始处理...")
    
    for idx, file_path in enumerate(stock_files):
        stock_code = file_path.stem.split("#")[-1] if "#" in file_path.stem else file_path.stem
        
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(stock_files)}")
        
        df = load_stock_data(str(file_path))
        if df is None or len(df) < MIN_HISTORY_BARS:
            continue
        
        try:
            df_filtered = calculate_deviations_and_metrics(df, stock_code)
        except Exception as e:
            print(f"  处理 {stock_code} 时出错: {e}")
            continue
        
        if len(df_filtered) < MIN_SAMPLE_COUNT:
            continue
        
        all_stock_data.append(df_filtered)
        
        abs_scan = scan_absolute_thresholds_for_stock(df_filtered, stock_code)
        absolute_scan_results.extend(abs_scan)
        
        min_thresh = find_min_thresholds_for_stock(df_filtered, stock_code)
        min_threshold_results.append(min_thresh)
        
        z_scan, atr_scan = scan_standardized_thresholds_for_stock(df_filtered, stock_code)
        z_scan_results.extend(z_scan)
        atr_scan_results.extend(atr_scan)
        
        models = fit_probability_models_for_stock(df_filtered, stock_code)
        model_results.extend(models)
    
    print(f"\n处理完成！")
    print(f"成功处理股票数: {len(all_stock_data)}")
    
    if len(all_stock_data) == 0:
        print("未找到符合条件的样本！")
        return
    
    all_samples_df = pd.concat(all_stock_data, ignore_index=True)
    all_samples_df.to_csv(OUTPUT_DIR / "all_samples.csv", index=False, encoding="utf-8-sig")
    print(f"\n样本数据已保存到: {OUTPUT_DIR / 'all_samples.csv'}")
    
    print("\n保存结果文件...")
    
    absolute_scan_df = pd.DataFrame(absolute_scan_results)
    absolute_scan_df.to_csv(OUTPUT_DIR / "absolute_threshold_scan.csv", index=False, encoding="utf-8-sig")
    print(f"绝对偏离阈值扫描结果已保存")
    
    min_thresholds_df = pd.DataFrame(min_threshold_results)
    min_thresholds_df.to_csv(OUTPUT_DIR / "individual_min_thresholds.csv", index=False, encoding="utf-8-sig")
    print(f"个股最小有效阈值已保存")
    
    z_scan_df = pd.DataFrame(z_scan_results)
    z_scan_df.to_csv(OUTPUT_DIR / "zscore_threshold_scan.csv", index=False, encoding="utf-8-sig")
    print(f"Z-score阈值扫描结果已保存")
    
    atr_scan_df = pd.DataFrame(atr_scan_results)
    atr_scan_df.to_csv(OUTPUT_DIR / "atr_threshold_scan.csv", index=False, encoding="utf-8-sig")
    print(f"ATR阈值扫描结果已保存")
    
    model_results_df = pd.DataFrame(model_results)
    model_results_df.to_csv(OUTPUT_DIR / "probability_models.csv", index=False, encoding="utf-8-sig")
    print(f"概率公式结果已保存")
    
    print("\n生成比较结果...")
    comparison_df = compare_definitions(absolute_scan_df, z_scan_df, atr_scan_df, min_thresholds_df, model_results_df)
    comparison_df.to_csv(OUTPUT_DIR / "definition_comparison.csv", index=False, encoding="utf-8-sig")
    
    print("\n生成结论...")
    conclusion = generate_conclusion(min_thresholds_df, comparison_df, model_results_df)
    
    with open(OUTPUT_DIR / "conclusion.txt", "w", encoding="utf-8") as f:
        f.write(conclusion)
    
    print("\n" + "=" * 80)
    print(conclusion)
    print("\n" + "=" * 80)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
