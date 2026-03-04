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

# 测试一只股票
file = "SH#600000.txt"
path = os.path.join(data_dir, file)
df = pd.read_csv(path, sep='\t', encoding='utf-8')
df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
df['日期'] = pd.to_datetime(df['日期'])
df = df.set_index('日期')
df = df.sort_index()

df = calculate_indicators(df)

# 统计前一日多空线<趋势线的天数
count = 0
for i in range(1, len(df)):
    prev_row = df.iloc[i - 1]
    curr_row = df.iloc[i]
    prev_dk = prev_row.get('知行多空线')
    prev_trend = prev_row.get('知行短期趋势线')
    if pd.isna(prev_dk) or pd.isna(prev_trend):
        continue
    if prev_dk < prev_trend:
        count += 1
        if count <= 10:
            print(f"{df.index[i]}: 前一日多空线={prev_dk:.4f}, 趋势线={prev_trend:.4f}")

print(f"\n总天数: {len(df)}")
print(f"前一日多空线<趋势线的天数: {count}")
print(f"占比: {count/len(df)*100:.2f}%")

# 统计J<-5的天数
j_count = 0
for i in range(2, len(df)):
    row = df.iloc[i]
    if pd.isna(row['J']):
        continue
    if row['J'] < -5:
        j_count += 1

print(f"J<-5的天数: {j_count}")

# 统计同时满足J<-5和前一日多空线<趋势线的天数
both_count = 0
for i in range(2, len(df)):
    row = df.iloc[i]
    if pd.isna(row['J']):
        continue
    if row['J'] >= -5:
        continue
    
    prev_row = df.iloc[i - 1]
    prev_dk = prev_row.get('知行多空线')
    prev_trend = prev_row.get('知行短期趋势线')
    if pd.isna(prev_dk) or pd.isna(prev_trend):
        continue
    if prev_dk < prev_trend:
        both_count += 1
        if both_count <= 10:
            print(f"{df.index[i]}: J={row['J']:.2f}, 前一日多空线={prev_dk:.4f}, 趋势线={prev_trend:.4f}")

print(f"\n同时满足J<-5和前一日多空线<趋势线的天数: {both_count}")
