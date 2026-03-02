import os
import sys
import pandas as pd
import pickle
import time
import numpy as np

data_dir = '/Users/lidongyang/Desktop/Qstrategy/data/forward_data'
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

print(f'总共 {len(files)} 个文件', flush=True)

encodings_list = ['gbk', 'utf-8', 'gb18030', 'latin-1']

def load_stock(path):
    df = None
    for encoding in encodings_list:
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                header=1,
                encoding=encoding
            )
            df.columns = ["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "成交额"][:len(df.columns)]
            df = df[["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df = df[pd.to_numeric(df["OPEN"], errors="coerce").notna()]
            df["日期"] = pd.to_datetime(df["日期"], errors='coerce')
            df = df[df["日期"].notna()]
            df = df.sort_values("日期")
            df.set_index("日期", inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df
        except:
            continue
    return None

def calculate_indicators(df):
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
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    gain28 = (delta.where(delta > 0, 0)).rolling(window=28).mean()
    loss28 = (-delta.where(delta < 0, 0)).rolling(window=28).mean()
    rs28 = gain28 / loss28
    df['RSI28'] = 100 - (100 / (1 + rs28))

    gain57 = (delta.where(delta > 0, 0)).rolling(window=57).mean()
    loss57 = (-delta.where(delta < 0, 0)).rolling(window=57).mean()
    rs57 = gain57 / loss57
    df['RSI57'] = 100 - (100 / (1 + rs57))

    df['是否阳线'] = df['CLOSE'] > df['OPEN']
    df['是否阴线'] = df['CLOSE'] < df['OPEN']
    df['涨跌幅'] = df['CLOSE'].pct_change() * 100

    return df

stock_data = {}
daily_signals = {}
daily_scores = {}

loaded_count = 0
start_time = time.time()

for idx, file in enumerate(files):
    if idx % 100 == 0:
        elapsed = time.time() - start_time
        print(f'[{idx}/{len(files)}] 已加载 {loaded_count} 只，耗时 {elapsed:.1f}s', flush=True)
    
    df = load_stock(os.path.join(data_dir, file))
    if df is None or len(df) < 130:
        continue
    
    loaded_count += 1
    
    df = calculate_indicators(df)
    stock_data[file] = df
    
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        if pd.isna(row['J']) or pd.isna(row['知行短期趋势线']) or pd.isna(row['知行多空线']):
            continue
        
        if row['J'] >= 13:
            continue
        
        if row['知行短期趋势线'] <= row['知行多空线']:
            continue
        
        score = 0
        details = {}
        
        if not pd.isna(row['MACD_DIF']) and row['MACD_DIF'] > 0:
            score += 0.5
            details['MACD_DIF>0'] = 0.5
        
        if not (pd.isna(row['RSI14']) or pd.isna(row['RSI28']) or pd.isna(row['RSI57'])):
            if row['RSI14'] > row['RSI28'] > row['RSI57']:
                score += 0.5
                details['RSI14>28>57'] = 0.5
        
        change_pct = row['涨跌幅']
        if not pd.isna(change_pct) and not pd.isna(row['VOLUME']) and not pd.isna(prev['VOLUME']):
            if -3.5 < change_pct < 2 and row['VOLUME'] < prev['VOLUME']:
                score += 2
                details['涨幅-3.5~2%且缩量'] = 2
        
        if score > 0:
            date = df.index[i]
            if date not in daily_signals:
                daily_signals[date] = []
                daily_scores[date] = []
            daily_signals[date].append(file)
            daily_scores[date].append((file, score, details))

elapsed = time.time() - start_time
print(f'完成! 成功加载 {loaded_count} 只，总耗时 {elapsed:.1f}s', flush=True)

all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
print(f'总交易日: {len(all_dates)}', flush=True)
print(f'有信号的天数: {len(daily_signals)}', flush=True)

# Save cache
print('保存缓存...', flush=True)
cache_file = "/tmp/b1_cache_full.pkl"
with open(cache_file, 'wb') as f:
    pickle.dump({
        'stock_data': stock_data,
        'daily_signals': daily_signals,
        'daily_scores': daily_scores,
        'all_dates': all_dates
    }, f)
print('完成!', flush=True)
