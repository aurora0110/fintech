from __future__ import annotations

from typing import List
import os
import numpy as np
import pandas as pd

from config import RunConfig


def _contains_any(text: str, tokens: List[str]) -> bool:
    low = str(text).lower()
    return any(t.lower() in low for t in tokens)


def load_and_prepare_data(cfg: RunConfig) -> pd.DataFrame:
    # 检查是否是目录（用户的数据格式）
    if os.path.isdir(cfg.data_path):
        df = load_from_directory(cfg.data_path)
    else:
        df = pd.read_csv(cfg.data_path)
    
    # 标准化列名
    column_mapping = {
        '日期': 'date',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
    }
    df = df.rename(columns=column_mapping)
    
    # 确保必要的列存在
    required = ['date', 'code', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = apply_universe_filters(df, cfg)
    
    # 计算因子
    print("计算技术指标因子...")
    df = calculate_factors(df)
    print(f"因子计算完成: {list(df.columns)}")

    # 计算下日收益率
    if 'next_return' not in df.columns:
        df['next_return'] = (
            df.groupby('code')['close'].shift(-1) / df['close'] - 1.0
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['next_return']).reset_index(drop=True)
    return df


def load_from_directory(data_dir: str) -> pd.DataFrame:
    """从目录加载所有txt文件"""
    all_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            # 提取股票代码
            code = filename.replace('.txt', '')
            df['code'] = code
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No data loaded from {data_dir}")
    
    result = pd.concat(all_data, ignore_index=True)
    return result


def apply_universe_filters(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    out = df.copy()

    if cfg.st_flag_col in out.columns:
        out = out[~out[cfg.st_flag_col].fillna(False).astype(bool)]
    elif cfg.name_col in out.columns:
        out = out[~out[cfg.name_col].astype(str).str.upper().str.contains("ST", na=False)]

    if cfg.board_col in out.columns and cfg.exclude_boards:
        mask = out[cfg.board_col].astype(str).apply(
            lambda x: not _contains_any(x, cfg.exclude_boards)
        )
        out = out[mask]

    return out.reset_index(drop=True)


def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标作为因子"""
    df = df.copy()
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 按股票分组计算
    result_list = []
    for code, group in df.groupby('code'):
        g = group.copy()
        g = g.sort_values('date')
        
        # RSI
        delta = g['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        g['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = g['close'].ewm(span=12, adjust=False).mean()
        exp26 = g['close'].ewm(span=26, adjust=False).mean()
        g['macd'] = exp12 - exp26
        g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()
        
        # KDJ
        low_n = g['low'].rolling(window=9).min()
        high_n = g['high'].rolling(window=9).max()
        rsv = (g['close'] - low_n) / (high_n - low_n) * 100
        g['k'] = rsv.ewm(com=2, adjust=False).mean()
        g['d'] = g['k'].ewm(com=2, adjust=False).mean()
        g['j'] = 3 * g['k'] - 2 * g['d']
        
        # 均线
        g['ma5'] = g['close'].rolling(window=5).mean()
        g['ma10'] = g['close'].rolling(window=10).mean()
        g['ma20'] = g['close'].rolling(window=20).mean()
        
        # 量能
        g['vol_ma20'] = g['volume'].rolling(window=20).mean()
        g['vol_ratio'] = g['volume'] / g['vol_ma20']
        
        # 涨跌幅
        g['pct_change'] = g['close'].pct_change() * 100
        
        # 价格位置
        g['price_position'] = (g['close'] - g['low'].rolling(20).min()) / (g['high'].rolling(20).max() - g['low'].rolling(20).min())
        
        result_list.append(g)
    
    return pd.concat(result_list, ignore_index=True)

