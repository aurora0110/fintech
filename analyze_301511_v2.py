#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析股票301511（使用最新数据）
"""

import sys
sys.path.insert(0, '/Users/lidongyang/Desktop/Qstrategy/utils')
from brick_filter import (
    load_one_csv, add_features, 
    calc_green_streak, triangle_quality
)
import numpy as np
import pandas as pd

# 加载股票数据 - 使用20260311的最新数据
file_path = "/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal/SZ#301511.txt"
df = load_one_csv(file_path)

if df is None or df.empty:
    print("数据加载失败")
    sys.exit(1)

print("=" * 80)
print(f"股票代码: 301511")
print(f"数据总行数: {len(df)}")
print("=" * 80)

# 添加特征
x = add_features(df)

# 获取最新一天的数据
latest_idx = len(x) - 1
latest = x.iloc[latest_idx]
print(f"\n最新日期: {latest['date']}")
print(f"最新收盘价: {latest['close']}")
print(f"最新开盘价: {latest['open']}")
print(f"最新最高价: {latest['high']}")
print(f"最新最低价: {latest['low']}")
print(f"当日涨跌幅: {latest['ret1']:.4%}")

print("\n" + "=" * 80)
print("逐条检查条件")
print("=" * 80)

# 检查pattern_a
pattern_a = bool(latest["pattern_a"])
print(f"\n【形态A条件】prev_green_streak >= 3:")
print(f"  prev_green_streak = {latest['prev_green_streak']}")
print(f"  brick_red = {latest['brick_red']}")
print(f"  close_pullback_white.shift(1) = {latest['close_pullback_white']}")
print(f"  close_above_white = {latest['close_above_white']}")
print(f"  结果: {pattern_a}")

# 检查pattern_b
pattern_b = bool(latest["pattern_b"])
print(f"\n【形态B条件】绿砖连续3天后红砖:")
print(f"  brick_red = {latest['brick_red']}")
print(f"  brick_green.shift(1) = {latest['brick_green']}")
print(f"  brick_red.shift(2) = {latest['brick_red']}")
print(f"  close_pullback_white.shift(1) = {latest['close_pullback_white']}")
print(f"  close_above_white = {latest['close_above_white']}")
print(f"  结果: {pattern_b}")

# 检查rebound_ratio
rebound_ratio = float(latest["rebound_ratio"])
print(f"\n【反弹比例】:")
print(f"  brick_red_len = {latest['brick_red_len']}")
print(f"  brick_green_len.shift(1) = {latest['brick_green_len']}")
print(f"  rebound_ratio = {rebound_ratio}")
print(f"  形态A要求 >= 1.2: {rebound_ratio >= 1.2}")
print(f"  形态B要求 >= 1.0: {rebound_ratio >= 1.0}")

# 检查signal_base
print(f"\n【基础信号条件】signal_base:")
print(f"  (pattern_a | pattern_b) = {pattern_a or pattern_b}")
print(f"  pullback_shrinking = {latest['pullback_shrinking']}")
print(f"  signal_vs_ma5_valid = {latest['signal_vs_ma5_valid']}")
print(f"    signal_vs_ma5 = {latest['signal_vs_ma5']}")
print(f"  not_sideways = {latest['not_sideways']}")
print(f"  ret1.notna() = {pd.notna(latest['ret1'])}")

signal_base = (
    (pattern_a | pattern_b)
    & latest["pullback_shrinking"]
    & latest["signal_vs_ma5_valid"]
    & latest["not_sideways"]
    & pd.notna(latest["ret1"])
)
print(f"  结果: {signal_base}")

# 检查ret1条件
ret1_condition = float(latest["ret1"]) <= 0.08
print(f"\n【涨幅限制】ret1 <= 0.08:")
print(f"  ret1 = {latest['ret1']:.4%}")
print(f"  结果: {ret1_condition}")

# 检查趋势条件
trend_condition = float(latest["trend_line"]) > float(latest["long_line"])
print(f"\n【趋势条件】trend_line > long_line:")
print(f"  trend_line = {latest['trend_line']:.4f}")
print(f"  long_line = {latest['long_line']:.4f}")
print(f"  结果: {trend_condition}")

# 汇总
print("\n" + "=" * 80)
print("最终检查结果")
print("=" * 80)

mask_a = pattern_a and (rebound_ratio >= 1.2)
mask_b = pattern_b and (rebound_ratio >= 1.0)
signal_ok = (
    signal_base
    and ret1_condition
    and (mask_a or mask_b)
    and trend_condition
)

print(f"\n最终信号: {'通过' if signal_ok else '不通过'}")

if not signal_ok:
    print("\n未通过的原因分析:")
    if not signal_base:
        print("  ❌ signal_base 基础信号条件未满足")
    if not ret1_condition:
        print(f"  ❌ ret1 <= 0.08 涨幅限制未满足 (ret1={latest['ret1']:.4%})")
    if not (mask_a or mask_b):
        print(f"  ❌ 形态条件未满足")
        print(f"     - 形态A: {pattern_a}, rebound_ratio >= 1.2: {rebound_ratio >= 1.2}")
        print(f"     - 形态B: {pattern_b}, rebound_ratio >= 1.0: {rebound_ratio >= 1.0}")
    if not trend_condition:
        print(f"  ❌ 趋势条件未满足 (trend_line <= long_line)")

# 显示关键指标
print("\n" + "=" * 80)
print("关键指标详情")
print("=" * 80)
print(f"砖块值: {latest['brick']:.4f}")
print(f"前一日砖块值: {latest['brick_prev']:.4f}")
print(f"砖块变化: {latest['brick'] - latest['brick_prev']:.4f}")
print(f"红砖长度: {latest['brick_red_len']:.4f}")
print(f"绿砖长度: {latest['brick_green_len']:.4f}")
print(f"成交量/5日均量: {latest['signal_vs_ma5']:.4f}")
print(f"回撤期成交量: {latest['pullback_avg_vol']:.0f}")
print(f"上涨期成交量: {latest['up_leg_avg_vol']:.0f}")
pullback_shrink = latest['pullback_avg_vol'] / latest['up_leg_avg_vol'] if latest['up_leg_avg_vol'] > 0 else float('nan')
print(f"回撤缩量比: {pullback_shrink:.4f}")

# 显示最近10天的砖块状态
print("\n" + "=" * 80)
print("最近10天砖块状态")
print("=" * 80)
for i in range(max(0, latest_idx-9), latest_idx+1):
    row = x.iloc[i]
    print(f"{row['date']} | brick={row['brick']:.2f} | red={row['brick_red']} | green={row['brick_green']} | close={row['close']:.2f} | ret1={row['ret1']:.2%}")
