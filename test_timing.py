import os
import pandas as pd
import time
import pickle
import numpy as np

data_dir = '/Users/lidongyang/Desktop/Qstrategy/data/forward_data'
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

print(f"总共 {len(files)} 个文件")

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

# Test loading first 10 files
start = time.time()
stock_data = {}
for i, f in enumerate(files[:10]):
    t0 = time.time()
    df = load_stock(os.path.join(data_dir, f))
    t1 = time.time()
    if df is not None and len(df) >= 130:
        df = calculate_indicators(df)
        t2 = time.time()
        stock_data[f] = df
        print(f"{f}: 加载 {t1-t0:.2f}s, 指标 {t2-t1:.2f}s, 共 {t2-t0:.2f}s")
    else:
        print(f"{f}: 数据不足")

print(f"前10个文件耗时: {time.time()-start:.2f}s")

# Save cache
cache_file = "/tmp/test_cache.pkl"
print("保存缓存...")
with open(cache_file, 'wb') as f:
    pickle.dump({'stock_data': stock_data}, f)
print("完成!")
