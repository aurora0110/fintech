import os
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

def load_all_data(data_dir):
    data_dict = {}
    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue
        stock_code = file.replace(".txt", "")
        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            if len(df) >= 60:
                data_dict[stock_code] = df
        except:
            continue
    return data_dict

def analyze():
    data_dict = load_all_data(DATA_DIR)
    print(f"加载 {len(data_dict)} 只股票")
    
    results = []
    
    for is_bullish in [True, False]:
        signal = "阳线" if is_bullish else "阴线"
        
        for ratio in [1.0, 1.3]:
            for w in [30, 60]:
                total = 0
                drop = 0
                
                for stock, df in data_dict.items():
                    df = df.copy()
                    df['vol_max'] = df['成交量'].rolling(w, min_periods=w).max()
                    
                    for i in range(w, len(df) - 10):
                        row = df.iloc[i]
                        prev = df.iloc[i - 1]
                        
                        if is_bullish:
                            sig = row['收盘'] > row['开盘']
                        else:
                            sig = row['收盘'] < row['开盘']
                        
                        vol_inc = row['成交量'] > prev['成交量'] * ratio
                        is_max = row['成交量'] >= row['vol_max'] * 0.99
                        
                        if sig and vol_inc and is_max:
                            for fd in [1, 5]:
                                if i + fd < len(df):
                                    if df.iloc[i+fd]['收盘'] < row['收盘']:
                                        drop += 1
                                    total += 1
                
                if total > 0:
                    prob = drop / total * 100
                    print(f"放量{signal} {ratio}x {w}日: {total}样本, {prob:.2f}%下跌")
                    results.append((signal, ratio, w, total, prob))
    
    print("\n=== 对比 ===")
    print(f"{'信号':<8} {'倍数':<6} {'周期':<6} {'下跌概率':<10}")
    for r in results:
        print(f"{r[0]:<8} {r[1]}x    {r[2]}日   {r[4]:.2f}%")

if __name__ == "__main__":
    analyze()
