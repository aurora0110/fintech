"""
生成优化所需的缓存数据
- J值历史百分位
- 所有因子的评分
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
CACHE_FILE = "/tmp/b1_full_optimization_cache.pkl"

print("="*60)
print("生成优化缓存数据")
print("="*60)

def load_stock_data():
    """加载所有股票数据"""
    print("\n加载股票数据...")
    stock_data = {}
    
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return None
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"找到 {len(files)} 个文件")
    
    for filename in tqdm(files, desc="加载股票"):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
            
            df = df.sort_index()
            stock_data[filename] = df
        except Exception as e:
            continue
    
    return stock_data

def calculate_indicators(df):
    """计算技术指标"""
    df = df.copy()
    
    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()
    
    df['MA14'] = df['CLOSE'].rolling(window=14).mean()
    df['MA28'] = df['CLOSE'].rolling(window=28).mean()
    df['MA57'] = df['CLOSE'].rolling(window=57).mean()
    df['MA114'] = df['CLOSE'].rolling(window=114).mean()
    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
    
    df['HHV9'] = df['HIGH'].rolling(9).max()
    df['LLV9'] = df['LOW'].rolling(9).min()
    rng = df['HHV9'] - df['LLV9']
    df['RSV'] = (df['CLOSE'] - df['LLV9']) / rng * 100
    df['RSV'] = df['RSV'].fillna(50)
    df['K'] = df['RSV'].ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    exp1 = df['CLOSE'].ewm(span=12, adjust=False).mean()
    exp2 = df['CLOSE'].ewm(span=26, adjust=False).mean()
    df['MACD_DIF'] = exp1 - exp2
    
    delta = df['CLOSE'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    gain28 = delta.where(delta > 0, 0).rolling(28).mean()
    loss28 = (-delta.where(delta < 0, 0)).rolling(28).mean()
    rs28 = gain28 / loss28
    df['RSI28'] = 100 - (100 / (1 + rs28))
    
    gain57 = delta.where(delta > 0, 0).rolling(57).mean()
    loss57 = (-delta.where(delta < 0, 0)).rolling(57).mean()
    rs57 = gain57 / loss57
    df['RSI57'] = 100 - (100 / (1 + rs57))
    
    df['是否阳线'] = df['CLOSE'] > df['OPEN']
    df['是否阴线'] = df['CLOSE'] < df['OPEN']
    df['涨跌幅'] = df['CLOSE'].pct_change() * 100
    
    return df

def calculate_factors(df, idx):
    """计算因子"""
    row = df.iloc[idx]
    prev = df.iloc[idx - 1] if idx > 0 else row
    
    factors = {}
    
    if not pd.isna(row.get('MACD_DIF')) and row['MACD_DIF'] > 0:
        factors['MACD_DIF>0'] = True
    
    if not (pd.isna(row.get('RSI14')) or pd.isna(row.get('RSI28')) or pd.isna(row.get('RSI57'))):
        if row['RSI14'] > row['RSI28'] > row['RSI57']:
            factors['RSI_14_28_57'] = True
    
    change_pct = row.get('涨跌幅')
    vol = row.get('VOLUME')
    prev_vol = prev.get('VOLUME')
    if not pd.isna(change_pct) and not pd.isna(vol) and not pd.isna(prev_vol):
        if -3.5 < change_pct < 2 and vol < prev_vol:
            factors['涨幅缩量'] = True
    
    if idx >= 1:
        prev_trend = prev.get('知行短期趋势线')
        curr_trend = row.get('知行短期趋势线')
        if not pd.isna(prev_trend) and not pd.isna(curr_trend):
            if prev['是否阴线'] and prev['CLOSE'] <= prev_trend and row['CLOSE'] > curr_trend and row.get('是否阳线', False):
                if not pd.isna(vol) and not pd.isna(prev_vol) and vol < prev_vol:
                    factors['前一日阴线回踩趋势线'] = True
    
    if idx >= 60:
        rolling_vol = df.iloc[max(0, idx-60):idx]['VOLUME']
        for i in range(max(0, idx-60), idx):
            if i > 0 and not pd.isna(df.iloc[i]['VOLUME']) and not pd.isna(df.iloc[i-1]['VOLUME']):
                if df.iloc[i]['是否阳线'] and df.iloc[i]['VOLUME'] >= df.iloc[i-1]['VOLUME'] * 2:
                    factors['倍量柱'] = True
                    break
    
    dk = row.get('知行多空线')
    trend = row.get('知行短期趋势线')
    close = row.get('CLOSE')
    if not pd.isna(dk) and not pd.isna(trend) and not pd.isna(close):
        if dk <= close <= trend:
            factors['多空线_趋势线区间'] = True
    
    if idx >= 1:
        prev_change = prev.get('涨跌幅')
        if not pd.isna(change_pct) and not pd.isna(prev_change) and not pd.isna(vol) and not pd.isna(prev_vol):
            if abs(change_pct) > abs(prev_change) and vol < prev_vol:
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
    
    dk = row.get('知行多空线')
    close = row.get('CLOSE')
    if not pd.isna(dk) and not pd.isna(close):
        if close < dk:
            factors['收盘<多空线'] = True
    
    return factors

def build_cache():
    """构建缓存"""
    stock_data = load_stock_data()
    if stock_data is None:
        return
    
    print(f"\n计算所有股票的技术指标...")
    for stock in tqdm(stock_data.keys(), desc="计算指标"):
        stock_data[stock] = calculate_indicators(stock_data[stock])
    
    print(f"\n收集所有交易日...")
    all_dates_set = set()
    for df in stock_data.values():
        all_dates_set.update(df.index.tolist())
    all_dates = sorted(list(all_dates_set))
    print(f"总交易日: {len(all_dates)}")
    
    print(f"\n计算J值百分位...")
    j_values_by_date = {d: [] for d in all_dates}
    
    for stock, df in tqdm(stock_data.items(), desc="J值分布"):
        for idx, row in df.iterrows():
            if idx in j_values_by_date and not pd.isna(row.get('J')):
                j_values_by_date[idx].append(row['J'])
    
    j_percentile_index = {}
    for date, j_values in tqdm(j_values_by_date.items(), desc="百分位"):
        if len(j_values) >= 10:
            j_percentile_index[date] = {
                'p10': np.percentile(j_values, 10),
                'p5': np.percentile(j_values, 5),
                'p90': np.percentile(j_values, 90),
                'p95': np.percentile(j_values, 95),
            }
    
    print(f"\n计算每日因子信号...")
    daily_scores = {}
    
    for date in tqdm(all_dates, desc="因子信号"):
        signals = []
        
        for stock, df in stock_data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 2:
                continue
            
            row = df.iloc[idx]
            
            if pd.isna(row.get('J')) or pd.isna(row.get('知行短期趋势线')) or pd.isna(row.get('知行多空线')):
                continue
            
            j_buy = row['J'] < 13
            if not j_buy:
                continue
            
            trend_above = row['知行短期趋势线'] > row['知行多空线']
            if not trend_above:
                continue
            
            factors = calculate_factors(df, idx)
            
            signals.append((stock, 1.0, factors))
        
        if signals:
            daily_scores[date] = signals
    
    print(f"\n保存缓存...")
    cache = {
        'stock_data': stock_data,
        'all_dates': all_dates,
        'daily_scores': daily_scores,
        'j_percentile_index': j_percentile_index,
    }
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    
    print(f"\n缓存已保存到: {CACHE_FILE}")
    print(f"  股票数: {len(stock_data)}")
    print(f"  交易日: {len(all_dates)}")
    print(f"  有信号日: {len(daily_scores)}")

if __name__ == "__main__":
    build_cache()
