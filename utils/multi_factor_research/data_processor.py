import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

EXCLUDE_ST = True
EXCLUDE_BOARD = ['科创', '北交所']
LIMIT_UP = True
LIMIT_DOWN = True
COST_RATE = 0.0003
SLIPPAGE = 0.001

def load_stock_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue
        stock_code = file.replace(".txt", "")
        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            df['stock'] = stock_code
            
            if EXCLUDE_ST and ('ST' in stock_code.upper() or '*ST' in stock_code.upper()):
                continue
            
            if len(df) >= 120:
                data_dict[stock_code] = df
        except:
            continue
    return data_dict

def merge_all_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_dfs = []
    for stock, df in data_dict.items():
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['日期', 'stock']).reset_index(drop=True)
    return combined

def calculate_forward_return(df: pd.DataFrame, days: int = 3) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['stock', '日期']).reset_index(drop=True)
    df['forward_return'] = df.groupby('stock')['收盘'].shift(-days) / df['收盘'] - 1
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['stock', '日期']).reset_index(drop=True)
    
    print("计算技术指标...")
    
    for stock in df['stock'].unique():
        mask = df['stock'] == stock
        g = df.loc[mask, '收盘']
        
        ma5 = g.rolling(5).mean()
        ma10 = g.rolling(10).mean()
        ma20 = g.rolling(20).mean()
        ma60 = g.rolling(60).mean()
        
        low_9 = g.rolling(9).min()
        high_9 = g.rolling(9).max()
        rsv = (g - low_9) / (high_9 - low_9 + 1e-6) * 100
        K = rsv.ewm(com=2, adjust=False).mean()
        D = K.ewm(com=2, adjust=False).mean()
        J = 3 * K - 2 * D
        
        exp12 = g.ewm(span=12, adjust=False).mean()
        exp26 = g.ewm(span=26, adjust=False).mean()
        MACD_DIF = exp12 - exp26
        
        delta = g.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        RSI = 100 - (100 / (1 + rs))
        
        vol = df.loc[mask, '成交量']
        vol_ma20 = vol.rolling(20).mean()
        vol_ma60 = vol.rolling(60).mean()
        
        pct_change = g.pct_change() * 100
        
        low_60 = g.rolling(60).min()
        high_60 = g.rolling(60).max()
        
        idx = df[mask].index
        df.loc[idx, 'ma5'] = ma5.values
        df.loc[idx, 'ma10'] = ma10.values
        df.loc[idx, 'ma20'] = ma20.values
        df.loc[idx, 'ma60'] = ma60.values
        df.loc[idx, 'K'] = K.values
        df.loc[idx, 'D'] = D.values
        df.loc[idx, 'J'] = J.values
        df.loc[idx, 'MACD_DIF'] = MACD_DIF.values
        df.loc[idx, 'RSI'] = RSI.values
        df.loc[idx, 'vol_ma20'] = vol_ma20.values
        df.loc[idx, 'vol_ma60'] = vol_ma60.values
        df.loc[idx, 'pct_change'] = pct_change.values
        df.loc[idx, 'low_60'] = low_60.values
        df.loc[idx, 'high_60'] = high_60.values
    
    return df

def filter_limit_up_down(df: pd.DataFrame) -> pd.DataFrame:
    if LIMIT_UP:
        df = df[df['pct_change'] < 9.5]
    if LIMIT_DOWN:
        df = df[df['pct_change'] > -9.5]
    return df

def prepare_data() -> pd.DataFrame:
    print("加载数据...")
    data_dict = load_stock_data(DATA_DIR)
    print(f"加载了 {len(data_dict)} 只股票")
    
    print("合并数据...")
    df = merge_all_data(data_dict)
    
    print("计算技术指标...")
    df = add_technical_indicators(df)
    
    print("计算未来收益...")
    df = calculate_forward_return(df, days=3)
    
    print("过滤涨跌停...")
    df = filter_limit_up_down(df)
    
    df = df.dropna(subset=['forward_return'])
    print(f"有效样本数: {len(df)}")
    
    return df

if __name__ == "__main__":
    df = prepare_data()
    print(df.head())
