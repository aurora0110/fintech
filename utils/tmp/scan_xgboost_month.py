from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODEL_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")
MODEL_FILE = MODEL_DIR / "xgboost_brick_model.json"
THRESHOLD = 0.8

FEATURE_COLS = [
    'signal_ret', 'brick_red_len', 'brick_green_len_prev', 'rebound_ratio',
    'red_len_vs_1d', 'red_len_vs_3d', 'red_len_vs_5d', 'red_len_vs_10d',
    'green_len_prev_vs_1d', 'green_len_prev_vs_3d', 'green_len_prev_vs_5d', 'green_len_prev_vs_10d',
    'trend_spread', 'close_to_trend', 'close_to_long',
    'trend_slope_3', 'trend_slope_5', 'trend_slope_10',
    'ma10_slope_3', 'ma10_slope_5', 'ma10_slope_10',
    'ma20_slope_3', 'ma20_slope_5', 'ma20_slope_10',
    'signal_vs_ma5', 'ret1', 'ret5', 'ret10',
    'RSI14', 'MACD_DIF', 'MACD_DEA', 'MACD_hist',
    'KDJ_K', 'KDJ_D', 'KDJ_J',
    'body_ratio', 'close_location', 'upper_shadow_pct', 'lower_shadow_pct'
]

_model = None


def _load_model():
    global _model
    if not XGBOOST_AVAILABLE:
        return None
    if _model is not None:
        return _model
    try:
        if MODEL_FILE.exists():
            _model = xgb.Booster()
            _model.load_model(str(MODEL_FILE))
            return _model
        else:
            return None
    except Exception:
        return None


def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        data_lines = lines[1:]
        records = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    records.append({
                        'date': parts[0],
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]),
                        'amount': float(parts[6])
                    })
                except ValueError:
                    continue
        if not records:
            return None
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        file_name = Path(file_path).stem
        code_match = re.search(r'(\d{6})', file_name)
        if code_match:
            df['code'] = code_match.group(1)
        return df
    except Exception:
        return None


def compute_brick(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    def calc_trend_line(prices, window=20):
        if len(prices) < window:
            return np.nan
        y = prices[-window:].values
        x = np.arange(window)
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (window - 1) + intercept
    
    df['trend_line'] = df['close'].rolling(20, min_periods=20).apply(
        lambda x: calc_trend_line(pd.Series(x)), raw=False
    )
    df['long_line'] = df['close'].rolling(60, min_periods=60).mean()
    
    df['brick_red_len'] = 0.0
    df['brick_green_len'] = 0.0
    
    red_len = 0
    green_len = 0
    
    for i in range(len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['open']:
            red_len += (df.iloc[i]['close'] - df.iloc[i]['open'])
            green_len = 0
        else:
            green_len += (df.iloc[i]['open'] - df.iloc[i]['close'])
            red_len = 0
        df.loc[df.index[i], 'brick_red_len'] = red_len
        df.loc[df.index[i], 'brick_green_len'] = green_len
    
    return df


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def extract_features(df: pd.DataFrame, signal_idx: int) -> Optional[Dict]:
    if signal_idx < 60:
        return None
    
    row = df.iloc[signal_idx]
    prev_row = df.iloc[signal_idx - 1]
    
    if row['trend_line'] <= row['long_line']:
        return None
    if row['brick_red_len'] <= 0:
        return None
    if prev_row['brick_green_len'] <= 0:
        return None
    
    rebound_ratio = row['brick_red_len'] / prev_row['brick_green_len']
    if rebound_ratio <= 0.66:
        return None
    
    features = {}
    
    features['signal_ret'] = (row['close'] - row['open']) / row['open']
    features['brick_red_len'] = row['brick_red_len']
    features['brick_green_len_prev'] = prev_row['brick_green_len']
    features['rebound_ratio'] = rebound_ratio
    
    red_lens = df['brick_red_len'].values
    green_lens = df['brick_green_len'].values
    
    for d in [1, 3, 5, 10]:
        if signal_idx - d >= 0:
            features[f'red_len_vs_{d}d'] = row['brick_red_len'] / max(red_lens[signal_idx - d], 0.001)
            features[f'green_len_prev_vs_{d}d'] = prev_row['brick_green_len'] / max(green_lens[signal_idx - d], 0.001)
        else:
            features[f'red_len_vs_{d}d'] = 1.0
            features[f'green_len_prev_vs_{d}d'] = 1.0
    
    features['trend_spread'] = (row['trend_line'] - row['long_line']) / row['long_line']
    features['close_to_trend'] = (row['close'] - row['trend_line']) / row['trend_line']
    features['close_to_long'] = (row['close'] - row['long_line']) / row['long_line']
    
    close_series = df['close']
    for window in [3, 5, 10]:
        if len(close_series) >= window:
            y = close_series.iloc[-window:].values
            x = np.arange(window)
            slope, _ = np.polyfit(x, y, 1)
            features[f'trend_slope_{window}'] = slope / y[-1]
        else:
            features[f'trend_slope_{window}'] = 0
    
    for ma in ['ma10', 'ma20']:
        ma_series = df[ma]
        for window in [3, 5, 10]:
            if len(ma_series) >= window and not ma_series.iloc[-window:].isna().any():
                y = ma_series.iloc[-window:].values
                x = np.arange(window)
                slope, _ = np.polyfit(x, y, 1)
                features[f'{ma}_slope_{window}'] = slope / y[-1]
            else:
                features[f'{ma}_slope_{window}'] = 0
    
    features['signal_vs_ma5'] = (row['close'] - row['ma5']) / row['ma5'] if row['ma5'] > 0 else 0
    
    for d in [1, 5, 10]:
        if signal_idx - d >= 0:
            prev_close = df.iloc[signal_idx - d]['close']
            features[f'ret{d}'] = (row['close'] - prev_close) / prev_close
        else:
            features[f'ret{d}'] = 0
    
    features['RSI14'] = compute_rsi(df['close'], 14).iloc[signal_idx]
    dif, dea, macd_hist = compute_macd(df['close'])
    features['MACD_DIF'] = dif.iloc[signal_idx]
    features['MACD_DEA'] = dea.iloc[signal_idx]
    features['MACD_hist'] = macd_hist.iloc[signal_idx]
    
    k, d, j = compute_kdj(df['high'], df['low'], df['close'])
    features['KDJ_K'] = k.iloc[signal_idx]
    features['KDJ_D'] = d.iloc[signal_idx]
    features['KDJ_J'] = j.iloc[signal_idx]
    
    candle_range = row['high'] - row['low']
    body = abs(row['close'] - row['open'])
    features['body_ratio'] = body / candle_range if candle_range > 0 else 0
    
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    
    if row['close'] > row['open']:
        features['close_location'] = (row['close'] - row['low']) / candle_range if candle_range > 0 else 0.5
    else:
        features['close_location'] = (row['open'] - row['low']) / candle_range if candle_range > 0 else 0.5
    
    features['upper_shadow_pct'] = upper_shadow / candle_range if candle_range > 0 else 0
    features['lower_shadow_pct'] = lower_shadow / candle_range if candle_range > 0 else 0
    
    return features


def check(file_path: str, hold_list=None, feature_cache=None) -> List:
    if not XGBOOST_AVAILABLE:
        return [-1]
    df = load_stock_data(file_path)
    if df is None or len(df) < 150:
        return [-1]
    df = compute_brick(df)
    signal_idx = len(df) - 1
    features = extract_features(df, signal_idx)
    if features is None:
        return [-1]
    X = pd.DataFrame([features])
    X = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = _load_model()
    if model is None:
        return [-1]
    try:
        dtest = xgb.DMatrix(X)
        proba = model.predict(dtest)[0]
    except Exception:
        return [-1]
    if proba < THRESHOLD:
        return [-1]
    row = df.iloc[signal_idx]
    stop_loss_price = round(float(row['low']) * 0.99, 3)
    return [1, stop_loss_price, float(row['close']), round(proba, 4), "XGBoost预测"]


def scan_date(date_str: str) -> List[Dict]:
    data_dir = Path(f"/Users/lidongyang/Desktop/Qstrategy/data/{date_str}/normal")
    if not data_dir.exists():
        return []
    
    file_paths = list(data_dir.glob('*.txt'))
    results = []
    
    for fp in file_paths:
        result = check(str(fp))
        if result[0] == 1:
            code_match = re.search(r'(\d{6})', fp.stem)
            code = code_match.group(1) if code_match else fp.stem
            results.append({
                'date': date_str,
                'code': code,
                'stop_loss': result[1],
                'close': result[2],
                'prob': result[3]
            })
    
    return results


def main():
    dates = [
        '20260317', '20260316', '20260315', '20260313',
        '20260312', '20260310', '20260226', '20260225'
    ]
    
    all_results = []
    
    for date_str in dates:
        print(f"扫描日期: {date_str}")
        results = scan_date(date_str)
        all_results.extend(results)
        print(f"  找到 {len(results)} 条")
    
    print("\n" + "="*80)
    print("XGBoost筛选结果（最近一个月）")
    print("="*80)
    
    if all_results:
        print(f"\n共筛选出 {len(all_results)} 条数据:\n")
        print(f"{'日期':<12} {'代码':<8} {'止损价':<10} {'收盘价':<10} {'概率':<8}")
        print("-"*50)
        for r in all_results:
            print(f"{r['date']:<12} {r['code']:<8} {r['stop_loss']:<10.2f} {r['close']:<10.2f} {r['prob']:<8.4f}")
    else:
        print("未筛选出数据")


if __name__ == "__main__":
    main()
