"""
使用现有缓存生成优化所需的完整缓存数据
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

CACHE_FILE = "/tmp/b1_cache_full.pkl"
OUTPUT_CACHE = "/tmp/b1_full_optimization_cache.pkl"

print("="*60)
print("生成优化缓存数据")
print("="*60)

print("\n加载现有缓存...")
with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

stock_data = cache['stock_data']
all_dates = cache['all_dates']
daily_scores_original = cache['daily_scores']

print(f"股票数: {len(stock_data)}")
print(f"交易日: {len(all_dates)}")

print("\n计算J值百分位...")
j_values_by_date = {d: [] for d in all_dates}

for stock, df in tqdm(stock_data.items()):
    for idx, date in enumerate(df.index):
        if date in j_values_by_date:
            j_val = df.iloc[idx].get('J')
            if not pd.isna(j_val):
                j_values_by_date[date].append(j_val)

j_percentile_index = {}
for date, j_values in tqdm(j_values_by_date.items(), desc="百分位"):
    if len(j_values) >= 10:
        j_percentile_index[date] = {
            'p10': np.percentile(j_values, 10),
            'p5': np.percentile(j_values, 5),
            'p90': np.percentile(j_values, 90),
            'p95': np.percentile(j_values, 95),
        }

print(f"\nJ百分位数据: {len(j_percentile_index)} 天")

print("\n计算新增因子...")
def calculate_extra_factors(df, idx):
    """计算新增的因子"""
    row = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else row
    
    factors = {}
    
    change_pct = row.get('涨跌幅')
    vol = row.get('VOLUME')
    prev_vol = prev.get('VOLUME')
    
    if idx >= 1:
        prev_trend = prev.get('知行短期趋势线')
        curr_trend = row.get('知行短期趋势线')
        if not pd.isna(prev_trend) and not pd.isna(curr_trend):
            if prev['是否阴线'] and prev['CLOSE'] <= prev_trend and row['CLOSE'] > curr_trend:
                if not pd.isna(vol) and not pd.isna(prev_vol) and vol < prev_vol:
                    factors['前一日阴线回踩趋势线'] = True
    
    if idx >= 60:
        for i in range(max(0, idx-60), idx):
            if i > 0:
                if df.iloc[i]['是否阳线'] and df.iloc[i]['VOLUME'] >= df.iloc[i-1]['VOLUME'] * 2:
                    factors['倍量柱'] = True
                    break
    
    if idx >= 1:
        prev_change = prev.get('涨跌幅')
        if not pd.isna(change_pct) and not pd.isna(prev_change):
            if abs(change_pct) > abs(prev_change):
                factors['跌幅扩大缩量'] = True
    
    if idx >= 60:
        consecutive = 0
        for i in range(max(0, idx-60), idx):
            if i > 0:
                if df.iloc[i]['是否阳线'] and df.iloc[i]['VOLUME'] >= df.iloc[i-1]['VOLUME'] * 2:
                    consecutive += 1
                else:
                    consecutive = 0
        if consecutive >= 2:
            factors['连续倍量柱_阳线'] = True
    
    if idx >= 1:
        for i in range(max(0, idx-60), idx-1):
            if i > 0:
                if df.iloc[i]['是否阳线'] and df.iloc[i+1]['是否阴线']:
                    if df.iloc[i]['VOLUME'] >= df.iloc[i+1]['VOLUME'] * 2:
                        factors['阳量后接小阴量'] = True
                        break
    
    return factors

print("\n合并因子到daily_scores...")
daily_scores = {}

for date in tqdm(all_dates, desc="处理日期"):
    if date not in daily_scores_original:
        continue
    
    signals = []
    for stock, old_score, details in daily_scores_original[date]:
        if not details:
            continue
        
        df = stock_data.get(stock)
        if df is None or date not in df.index:
            continue
        
        idx = df.index.get_loc(date)
        extra_factors = calculate_extra_factors(df, idx)
        
        all_details = dict(details)
        all_details.update(extra_factors)
        
        signals.append((stock, 1.0, all_details))
    
    if signals:
        daily_scores[date] = signals

print(f"\n有信号日: {len(daily_scores)}")

print("\n保存缓存...")
output_cache = {
    'stock_data': stock_data,
    'all_dates': all_dates,
    'daily_scores': daily_scores,
    'j_percentile_index': j_percentile_index,
}

with open(OUTPUT_CACHE, 'wb') as f:
    pickle.dump(output_cache, f)

print(f"\n缓存已保存到: {OUTPUT_CACHE}")
print("完成!")
