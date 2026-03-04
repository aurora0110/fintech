import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['stock', '日期']).reset_index(drop=True)
    
    print("计算因子...")
    
    df['factor_1a'] = (df['J'] < 0).astype(int)
    df['factor_1b'] = (df['J'] < -20).astype(int)
    df['factor_1c'] = (df['J'] < -50).astype(int)
    
    df['factor_2'] = (df['MACD_DIF'] > 0).astype(int)
    
    df['factor_3a'] = ((df['RSI'] > 28) & (df['RSI'] < 57)).astype(int)
    df['factor_3b'] = ((df['RSI'] > 28) & (df['RSI'] < 50)).astype(int)
    df['factor_3c'] = ((df['RSI'] > 20) & (df['RSI'] < 40)).astype(int)
    
    df['vol_yesterday'] = df.groupby('stock')['成交量'].shift(1)
    df['is_shrink'] = df['成交量'] < df['vol_yesterday'] * 0.8
    df['factor_4'] = ((df['pct_change'] > -3.5) & (df['pct_change'] < 2) & df['is_shrink']).astype(int)
    
    df['prev_is_bearish'] = df.groupby('stock')['收盘'].shift(1) < df.groupby('stock')['开盘'].shift(1)
    df['trend_line'] = df['收盘'].ewm(span=10).mean()
    df['prev_touch_trend'] = (df.groupby('stock')['最低'].shift(1) <= df.groupby('stock')['trend_line'].shift(1)) & df['prev_is_bearish']
    df['factor_5'] = (df['prev_touch_trend'] & (df['收盘'] > df['开盘']) & (df['成交量'] < df['vol_yesterday'])).astype(int)
    
    df['vol_60_ma'] = df['成交量'].rolling(60).mean()
    df['factor_6'] = (df['成交量'] > df['vol_60_ma'] * 2).astype(int)
    
    df['ma_line'] = (df['ma5'] + df['ma10'] + df['ma20'] + df['ma60']) / 4
    df['factor_7'] = ((df['收盘'] >= df['ma_line']) & (df['收盘'] <= df['trend_line'])).astype(int)
    
    df['prev_pct'] = df.groupby('stock')['pct_change'].shift(1)
    df['prev_shrink'] = df['成交量'] < df.groupby('stock')['成交量'].shift(1)
    df['factor_8'] = ((df['pct_change'] < df['prev_pct']) & df['prev_shrink']).astype(int)
    
    df['vol_ratio'] = df['成交量'] / df['成交量'].rolling(60).mean()
    df['factor_9'] = (df['vol_ratio'] > 2).astype(int)
    
    df['prev_bull_vol'] = df.groupby('stock')['成交量'].shift(1) > df.groupby('stock')['成交量'].shift(2) * 2
    df['curr_shrink_vol'] = df['成交量'] <= df['成交量'].shift(1) * 0.5
    df['factor_10'] = (df['prev_bull_vol'] & df['curr_shrink_vol']).astype(int)
    
    df['factor_11'] = (df['收盘'] < df['ma_line']).astype(int)
    
    df = df.drop(columns=['vol_yesterday', 'is_shrink', 'prev_is_bearish', 'trend_line', 
                         'prev_touch_trend', 'vol_60_ma', 'ma_line', 'prev_pct', 'prev_shrink',
                         'vol_ratio', 'prev_bull_vol', 'curr_shrink_vol'], errors='ignore')
    
    factor_cols = [c for c in df.columns if c.startswith('factor_')]
    print(f"共计算了 {len(factor_cols)} 个因子: {factor_cols}")
    
    return df

def get_factor_list() -> List[str]:
    return [
        'factor_1a', 'factor_1b', 'factor_1c',
        'factor_2',
        'factor_3a', 'factor_3b', 'factor_3c',
        'factor_4', 'factor_5', 'factor_6', 'factor_7',
        'factor_8', 'factor_9', 'factor_10', 'factor_11'
    ]

if __name__ == "__main__":
    from data_processor import prepare_data
    df = prepare_data()
    df = calculate_all_factors(df)
    print(df[['日期', 'stock', '收盘', 'forward_return', 'factor_1a', 'factor_2', 'factor_11']].head(20))
