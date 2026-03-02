import os
import pandas as pd

data_dir = '/Users/lidongyang/Desktop/Qstrategy/data/forward_data'
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:5]
print(f'测试前5个文件:')

for f in files:
    encodings = ['gbk', 'utf-8', 'gb18030', 'latin-1']
    df = None

    for encoding in encodings:
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
            print(f'{f}: 成功加载, {len(df)} 行')
            break
        except Exception as e:
            continue
    if df is None:
        print(f'{f}: 加载失败')
