import os
import pandas as pd
import numpy as np

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
print(f"找到 {len(files)} 个文件")

stock_data = {}
daily_signals = {}
daily_scores = {}
loaded_count = 0

for file in files[:10]:  # 先测试10只股票
    stock_code = file.replace('.txt', '')
    path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df.sort_index()
        
        anomaly_dates = check_data_anomaly(df)
        print(f"{file}: 总行数={len(df)}, 异常日期数={len(anomaly_dates)}")
        
        if len(anomaly_dates) > len(df) * 0.1:
            print(f"  跳过: 异常数据过多")
            continue
        
        df = calculate_indicators(df)
        stock_data[stock_code] = df
        loaded_count += 1
        
        signal_count = 0
        for i in range(2, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['J']):
                continue
            
            if row['J'] >= -5:
                continue
            
            signal_count += 1
            date = df.index[i]
            if date not in daily_signals:
                daily_signals[date] = []
                daily_scores[date] = []
            daily_signals[date].append(stock_code)
            daily_scores[date].append((stock_code, 1, {'J<-5': 1}))
        
        print(f"  信号数: {signal_count}")
        
        if loaded_count >= 10:
            break
    
    except Exception as e:
        print(f"加载失败 {file}: {e}")
        continue

print(f"\n成功加载 {loaded_count} 只股票")
print(f"总信号天数: {len(daily_signals)}")
print(f"总信号数: {sum(len(v) for v in daily_signals.values())}")

# 显示前10天的信号
print("\n前10天信号:")
for i, (date, signals) in enumerate(sorted(daily_signals.items())[:10]):
    print(f"  {date}: {signals}")
