import os
import pandas as pd
import numpy as np

def tongdaxin_sma(series, n, m=1):
    """
    通达信SMA公式: SMA(X, N, M) = (X * M + Y' * (N-M)) / N
    其中Y'是上一周期的SMA值
    """
    result = np.zeros(len(series))
    prev_sma = 0
    for i in range(len(series)):
        val = series.iloc[i]
        if i < n - 1:
            # 前n-1个数据使用简单平均
            sma = series.iloc[:i+1].sum() / (i + 1)
        else:
            sma = (val * m + prev_sma * (n - m)) / n
        result[i] = sma
        prev_sma = sma
    return pd.Series(result, index=series.index)

# 加载920533数据
data_dir = r'C:\Users\lidon\Desktop\backtest_data'
file_path = os.path.join(data_dir, 'BJ#920533.txt')

# 读取文件
with open(file_path, 'r', encoding='gbk') as f:
    lines = f.readlines()

# 解析数据
data = []
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    if i == 0 and '开盘' in line:
        continue
    parts = line.split()
    if len(parts) >= 6:
        try:
            date_str = parts[0]
            open_price = float(parts[1])
            high_price = float(parts[2])
            low_price = float(parts[3])
            close_price = float(parts[4])
            volume = float(parts[5])
            data.append([date_str, open_price, high_price, low_price, close_price, volume])
        except ValueError:
            continue

df = pd.DataFrame(data, columns=['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期').reset_index(drop=True)

# 使用通达信SMA公式计算
df['HHV_H4'] = df['HIGH'].rolling(window=4).max()
df['LLV_L4'] = df['LOW'].rolling(window=4).min()
df['VAR1A'] = (df['HHV_H4'] - df['CLOSE']) / (df['HHV_H4'] - df['LLV_L4']) * 100 - 90
df['VAR2A'] = tongdaxin_sma(df['VAR1A'], 4, 1) + 100
df['VAR3A'] = (df['CLOSE'] - df['LLV_L4']) / (df['HHV_H4'] - df['LLV_L4']) * 100
df['VAR4A'] = tongdaxin_sma(df['VAR3A'], 6, 1)
df['VAR5A'] = tongdaxin_sma(df['VAR4A'], 6, 1) + 100
df['VAR6A'] = df['VAR5A'] - df['VAR2A']
df['砖型图数值'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)

# 按通达信公式判断红绿柱
df['REF_砖型图数值_1'] = df['砖型图数值'].shift(1)
df['REF_砖型图数值_2'] = df['砖型图数值'].shift(2)
df['当日红柱'] = df['REF_砖型图数值_1'] < df['砖型图数值']
df['前日绿柱'] = df['REF_砖型图数值_2'] > df['REF_砖型图数值_1']

# 显示2025年2月28日到3月10日的数据
print('砖型图计算结果（使用通达信SMA公式，2025年2月28日到3月10日）：')
print(f"{'日期':<12} {'砖型图(T-2)':<14} {'砖型图(T-1)':<14} {'砖型图(T)':<14} {'柱色':<6}")
print("-" * 70)
for i, row in df.iterrows():
    if str(row['日期']) >= '2025-02-28' and str(row['日期']) <= '2025-03-10':
        t_2 = row['REF_砖型图数值_2'] if pd.notna(row['REF_砖型图数值_2']) else '-'
        t_1 = row['REF_砖型图数值_1'] if pd.notna(row['REF_砖型图数值_1']) else '-'
        t = row['砖型图数值'] if pd.notna(row['砖型图数值']) else '-'
        red = '红' if row['当日红柱'] else '绿'
        if isinstance(t_2, float) and isinstance(t_1, float) and isinstance(t, float):
            print(f"{row['日期'].strftime('%Y-%m-%d'):<12} {t_2:<14.4f} {t_1:<14.4f} {t:<14.4f} {red:<6}")
        else:
            print(f"{row['日期'].strftime('%Y-%m-%d'):<12} {str(t_2):<14} {str(t_1):<14} {t:<14.4f} {red:<6}")
