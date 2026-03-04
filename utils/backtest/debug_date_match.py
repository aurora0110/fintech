import os
import pandas as pd
import numpy as np
import datetime

data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

def calculate_indicators(df):
    close = df['CLOSE']
    open_p = df['OPEN']
    high = df['HIGH']
    low = df['LOW']
    volume = df['VOLUME']
    
    df['涨跌幅'] = close.pct_change() * 100
    
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA30'] = close.rolling(30).mean()
    df['MA60'] = close.rolling(60).mean()
    
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def check_data_anomaly(df):
    anomaly_dates = set()
    
    if len(df) < 2:
        return anomaly_dates
    
    for i in range(len(df)):
        row = df.iloc[i]
        open_p = row['OPEN']
        high = row['HIGH']
        low = row['LOW']
        close = row['CLOSE']
        volume = row['VOLUME']
        
        if pd.isna(open_p) or pd.isna(high) or pd.isna(low) or pd.isna(close):
            anomaly_dates.add(df.index[i])
            continue
        
        if high == low and low == close:
            anomaly_dates.add(df.index[i])
            continue
        
        if i > 0:
            prev_close = df.iloc[i-1]['CLOSE']
            if not pd.isna(prev_close) and prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 20 or change_pct < -20:
                    anomaly_dates.add(df.index[i])
                    continue
        
        if volume <= 0:
            anomaly_dates.add(df.index[i])
            continue
        
        if close > 0 and open_p > 0:
            if close > high or close < low or open_p > high or open_p < low:
                anomaly_dates.add(df.index[i])
                continue
        
        if i > 0:
            prev_open = df.iloc[i-1]['OPEN']
            prev_close = df.iloc[i-1]['CLOSE']
            if pd.isna(prev_open) or pd.isna(prev_close):
                anomaly_dates.add(df.index[i])
                continue
    
    return anomaly_dates

# 加载数据
print("加载数据...")
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
loaded_count = 0
stock_data = {}
daily_signals = {}
daily_scores = {}

for file in files[:100]:
    stock_code = file.replace('.txt', '')
    path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df.sort_index()
        
        anomaly_dates = check_data_anomaly(df)
        if len(anomaly_dates) > len(df) * 0.1:
            continue
        
        df = calculate_indicators(df)
        stock_data[stock_code] = df
        loaded_count += 1
        
        for i in range(2, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['J']):
                continue
            
            if row['J'] >= -5:
                continue
            
            date = df.index[i]
            if date not in daily_signals:
                daily_signals[date] = []
                daily_scores[date] = []
            daily_signals[date].append(stock_code)
            daily_scores[date].append((stock_code, 1, {'J<-5': 1}))
        
        if loaded_count >= 100:
            print(f"已达到测试数量 100，停止加载")
            break
    
    except Exception as e:
        continue

print(f"成功加载 {loaded_count} 只股票")

# 收集所有日期
all_dates = []
for df in stock_data.values():
    all_dates.extend(df.index.tolist())
all_dates = sorted(set(all_dates))

print(f"all_dates 类型: {type(all_dates[0])}")
print(f"all_dates 数量: {len(all_dates)}")
print(f"all_dates 前5个: {all_dates[:5]}")

print(f"\ndaily_signals keys 类型: {type(list(daily_signals.keys())[0])}")
print(f"daily_signals keys 前5个: {sorted(daily_signals.keys())[:5]}")

# 检查匹配
date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
print(f"\n检查日期匹配:")
matched = 0
unmatched = 0
for date in sorted(daily_signals.keys())[:10]:
    if date in date_to_idx:
        matched += 1
    else:
        unmatched += 1
        print(f"  未匹配日期: {date}, type: {type(date)}")

print(f"前10天信号中匹配数: {matched}, 未匹配数: {unmatched}")

# 进一步检查
print("\n进一步检查:")
sample_date = sorted(daily_signals.keys())[0]
print(f"sample_date: {sample_date}")
print(f"sample_date in date_to_idx: {sample_date in date_to_idx}")

# 尝试精确匹配
for d in all_dates[:100]:
    if d == sample_date:
        print(f"找到匹配: {d}")
        break
else:
    print(f"未找到精确匹配，尝试找相近日期:")
    for d in all_dates[:100]:
        if abs((d - sample_date).days) < 5:
            print(f"  相近日期: {d}")
