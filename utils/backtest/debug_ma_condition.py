import os
import pandas as pd
import numpy as np

data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

def calculate_indicators(df):
    close = df['CLOSE']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA30'] = close.rolling(30).mean()
    df['MA60'] = close.rolling(60).mean()
    
    low_9 = df['LOW'].rolling(9).min()
    high_9 = df['HIGH'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df['知行短期趋势线'] = close.ewm(span=5, adjust=False).mean()
    df['知行多空线'] = close.ewm(span=10, adjust=False).mean()
    
    return df

def is_bullish_ma(df, idx):
    if idx < 1:
        return False
    
    row = df.iloc[idx]
    ma5 = row.get('MA5')
    ma10 = row.get('MA10')
    ma30 = row.get('MA30')
    close = row.get('CLOSE')
    
    if any(pd.isna(x) for x in [ma5, ma10, ma30, close]):
        return False
    
    cond1 = ma5 > ma10
    cond2 = ma10 > ma30
    cond3 = close > ma30
    
    return cond1 and cond2 and cond3

# 加载数据
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
stock_data = {}
daily_signals = {}
loaded_count = 0

for file in files[:100]:
    stock_code = file.replace('.txt', '')
    path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(path, sep='\t', encoding='utf-8')
        df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        df = df.sort_index()
        
        df = calculate_indicators(df)
        stock_data[stock_code] = df
        loaded_count += 1
        
        for i in range(2, len(df)):
            row = df.iloc[i]
            if pd.isna(row['J']):
                continue
            if row['J'] >= -5:
                continue
            
            # 检查日线多头条件
            if not is_bullish_ma(df, i):
                continue
            
            # 检查前一日多空线<趋势线
            prev_row = df.iloc[i - 1]
            prev_dk = prev_row.get('知行多空线')
            prev_trend = prev_row.get('知行短期趋势线')
            if pd.isna(prev_dk) or pd.isna(prev_trend):
                continue
            if prev_dk >= prev_trend:
                continue
            
            date = df.index[i]
            if date not in daily_signals:
                daily_signals[date] = []
            daily_signals[date].append(stock_code)
        
        if loaded_count >= 100:
            break
    
    except Exception as e:
        continue

print(f"加载股票数: {loaded_count}")
print(f"有信号的天数: {len(daily_signals)}")
print(f"总信号数: {sum(len(v) for v in daily_signals.values())}")

# 显示前10天信号
print("\n前10天信号:")
for i, (date, signals) in enumerate(sorted(daily_signals.items())[:10]):
    print(f"  {date}: {len(signals)}个")
