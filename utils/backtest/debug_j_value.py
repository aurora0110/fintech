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

# 测试一只股票
file = "SH#600000.txt"
path = os.path.join(data_dir, file)
df = pd.read_csv(path, sep='\t', encoding='utf-8')
df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
df['日期'] = pd.to_datetime(df['日期'])
df = df.set_index('日期')
df = df.sort_index()

df = calculate_indicators(df)

# 检查J值分布
print("=" * 50)
print(f"股票: {file}")
print(f"总行数: {len(df)}")
print(f"J值统计:")
print(df['J'].describe())

# 统计J<-5的信号数量
j_signals = df[df['J'] < -5]
print(f"\nJ < -5 的天数: {len(j_signals)}")

# 显示部分J<-5的信号
if len(j_signals) > 0:
    print("\n部分J<-5的信号:")
    print(j_signals[['CLOSE', 'J']].head(20))
