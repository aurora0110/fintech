import os
import pandas as pd
import sys

data_dir = '/Users/lidongyang/Desktop/Qstrategy/data/forward_data'
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:10]
print(f'测试前10个文件:', file=sys.stderr)

encodings_list = ['gbk', 'utf-8', 'gb18030', 'latin-1']

for idx, f in enumerate(files):
    print(f'处理 {idx+1}/10: {f}', file=sys.stderr)
    df = None

    for encoding in encodings_list:
        try:
            path = os.path.join(data_dir, f)
            df = pd.read_csv(
                path,
                sep=r'\s+',
                engine='python',
                header=1,
                encoding=encoding
            )
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', '成交额'][:len(df.columns)]
            df = df[['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            df = df[pd.to_numeric(df['OPEN'], errors='coerce').notna()]
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df[df['日期'].notna()]
            df = df.sort_values('日期')
            df.set_index('日期', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            break
        except:
            continue

    if df is None or len(df) < 130:
        print(f'{f}: 数据不足', file=sys.stderr)
        continue

    print(f'  开始计算指标...', file=sys.stderr)
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

    print(f'  完成!', file=sys.stderr)

print('测试完成', file=sys.stderr)
