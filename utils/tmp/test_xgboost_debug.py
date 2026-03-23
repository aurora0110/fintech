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
_model_loaded = False


def _load_model():
    global _model, _model_loaded
    if not XGBOOST_AVAILABLE:
        print("XGBoost不可用")
        return None
    if _model_loaded:
        return _model
    try:
        if MODEL_FILE.exists():
            _model = xgb.Booster()
            _model.load_model(str(MODEL_FILE))
            _model_loaded = True
            print(f"模型加载成功: {MODEL_FILE}")
            return _model
        else:
            print(f"模型文件不存在: {MODEL_FILE}")
            _model_loaded = True
            return None
    except Exception as e:
        print(f"模型加载失败: {e}")
        _model_loaded = True
        return None


def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        data_lines = lines[2:]
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
        first_line = lines[0].strip()
        code_match = re.match(r'(\d+)', first_line)
        if code_match:
            df['code'] = code_match.group(1)
        name_match = re.search(r'[\u4e00-\u9fa5]+', first_line)
        if name_match:
            df['name'] = name_match.group()
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def compute_trend_line(close: pd.Series, span: int = 10) -> pd.Series:
    ema1 = close.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    return ema2


def compute_long_line(close: pd.Series) -> pd.Series:
    ma14 = close.rolling(14).mean()
    ma28 = close.rolling(28).mean()
    ma57 = close.rolling(57).mean()
    ma114 = close.rolling(114).mean()
    return (ma14 + ma28 + ma57 + ma114) / 4


def compute_brick(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    hhv4 = x['high'].rolling(4).max()
    llv4 = x['low'].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = ((hhv4 - x['close']) / den4) * 100 - 90
    var2a = var1a.ewm(alpha=1/4, adjust=False).mean() + 100
    var3a = ((x['close'] - llv4) / den4) * 100
    var4a = var3a.ewm(alpha=1/6, adjust=False).mean()
    var5a = var4a.ewm(alpha=1/6, adjust=False).mean() + 100
    var6a = var5a - var2a
    x['brick'] = np.where(var6a > 4, var6a - 4, 0.0)
    x['brick_prev'] = x['brick'].shift(1)
    x['brick_red_len'] = np.where(x['brick'] > x['brick_prev'], x['brick'] - x['brick_prev'], 0.0)
    x['brick_green_len'] = np.where(x['brick'] < x['brick_prev'], x['brick_prev'] - x['brick'], 0.0)
    x['trend_line'] = compute_trend_line(x['close'])
    x['long_line'] = compute_long_line(x['close'])
    return x


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def compute_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9):
    llv = low.rolling(n).min()
    hhv = high.rolling(n).max()
    rsv = (close - llv) / (hhv - llv + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def compute_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr):
        if len(arr) < window or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(window)
        y = np.asarray(arr[-window:])
        if np.any(np.isnan(y)):
            return np.nan
        slope, _ = np.polyfit(x, y, 1)
        return slope
    return series.rolling(window).apply(_slope, raw=False)


def check_brick_condition(row: pd.Series, prev_row: pd.Series) -> bool:
    if pd.isna(row['trend_line']) or pd.isna(row['long_line']):
        return False
    if row['trend_line'] <= row['long_line']:
        return False
    if pd.isna(row['brick_red_len']) or pd.isna(prev_row['brick_green_len']):
        return False
    if prev_row['brick_green_len'] <= 0:
        return False
    if row['brick_red_len'] <= prev_row['brick_green_len'] * 0.66:
        return False
    return True


def extract_features(df: pd.DataFrame, signal_idx: int) -> Optional[Dict]:
    if signal_idx < 20 or signal_idx >= len(df):
        return None
    row = df.iloc[signal_idx]
    prev_row = df.iloc[signal_idx - 1]
    if not check_brick_condition(row, prev_row):
        return None
    features = {}
    features['signal_ret'] = (row['close'] - row['open']) / row['open'] if row['open'] > 0 else 0
    features['brick_red_len'] = row['brick_red_len']
    features['brick_green_len_prev'] = prev_row['brick_green_len']
    features['rebound_ratio'] = row['brick_red_len'] / prev_row['brick_green_len'] if prev_row['brick_green_len'] > 0 else 0
    red_len_1d_max = df['brick_red_len'].iloc[signal_idx-1:signal_idx].max()
    red_len_3d_max = df['brick_red_len'].iloc[signal_idx-3:signal_idx].max()
    red_len_5d_max = df['brick_red_len'].iloc[signal_idx-5:signal_idx].max()
    red_len_10d_max = df['brick_red_len'].iloc[signal_idx-10:signal_idx].max()
    features['red_len_vs_1d'] = row['brick_red_len'] / red_len_1d_max if red_len_1d_max > 0 else 0
    features['red_len_vs_3d'] = row['brick_red_len'] / red_len_3d_max if red_len_3d_max > 0 else 0
    features['red_len_vs_5d'] = row['brick_red_len'] / red_len_5d_max if red_len_5d_max > 0 else 0
    features['red_len_vs_10d'] = row['brick_red_len'] / red_len_10d_max if red_len_10d_max > 0 else 0
    green_len_1d_max = df['brick_green_len'].iloc[signal_idx-2:signal_idx-1].max()
    green_len_3d_max = df['brick_green_len'].iloc[signal_idx-4:signal_idx-1].max()
    green_len_5d_max = df['brick_green_len'].iloc[signal_idx-6:signal_idx-1].max()
    green_len_10d_max = df['brick_green_len'].iloc[signal_idx-11:signal_idx-1].max()
    features['green_len_prev_vs_1d'] = prev_row['brick_green_len'] / green_len_1d_max if green_len_1d_max > 0 else 0
    features['green_len_prev_vs_3d'] = prev_row['brick_green_len'] / green_len_3d_max if green_len_3d_max > 0 else 0
    features['green_len_prev_vs_5d'] = prev_row['brick_green_len'] / green_len_5d_max if green_len_5d_max > 0 else 0
    features['green_len_prev_vs_10d'] = prev_row['brick_green_len'] / green_len_10d_max if green_len_10d_max > 0 else 0
    features['trend_spread'] = (row['trend_line'] - row['long_line']) / row['close'] if row['close'] > 0 else 0
    features['close_to_trend'] = (row['close'] - row['trend_line']) / row['close'] if row['close'] > 0 else 0
    features['close_to_long'] = (row['close'] - row['long_line']) / row['close'] if row['close'] > 0 else 0
    trend_line_series = df['trend_line']
    features['trend_slope_3'] = compute_slope(trend_line_series, 3).iloc[signal_idx] if signal_idx >= 3 else np.nan
    features['trend_slope_5'] = compute_slope(trend_line_series, 5).iloc[signal_idx] if signal_idx >= 5 else np.nan
    features['trend_slope_10'] = compute_slope(trend_line_series, 10).iloc[signal_idx] if signal_idx >= 10 else np.nan
    ma10 = df['close'].rolling(10).mean()
    ma20 = df['close'].rolling(20).mean()
    features['ma10_slope_3'] = compute_slope(ma10, 3).iloc[signal_idx] if signal_idx >= 3 else np.nan
    features['ma10_slope_5'] = compute_slope(ma10, 5).iloc[signal_idx] if signal_idx >= 5 else np.nan
    features['ma10_slope_10'] = compute_slope(ma10, 10).iloc[signal_idx] if signal_idx >= 10 else np.nan
    features['ma20_slope_3'] = compute_slope(ma20, 3).iloc[signal_idx] if signal_idx >= 3 else np.nan
    features['ma20_slope_5'] = compute_slope(ma20, 5).iloc[signal_idx] if signal_idx >= 5 else np.nan
    features['ma20_slope_10'] = compute_slope(ma20, 10).iloc[signal_idx] if signal_idx >= 10 else np.nan
    vol_ma5 = df['volume'].iloc[signal_idx-5:signal_idx].mean()
    features['signal_vs_ma5'] = row['volume'] / vol_ma5 if vol_ma5 > 0 else 0
    features['ret1'] = (row['close'] - df.iloc[signal_idx-1]['close']) / df.iloc[signal_idx-1]['close'] if df.iloc[signal_idx-1]['close'] > 0 else 0
    features['ret5'] = (row['close'] - df.iloc[signal_idx-5]['close']) / df.iloc[signal_idx-5]['close'] if df.iloc[signal_idx-5]['close'] > 0 else 0
    features['ret10'] = (row['close'] - df.iloc[signal_idx-10]['close']) / df.iloc[signal_idx-10]['close'] if df.iloc[signal_idx-10]['close'] > 0 else 0
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    features['RSI14'] = rsi.iloc[signal_idx] if signal_idx < len(rsi) else np.nan
    dif, dea, macd_hist = compute_macd(df['close'])
    features['MACD_DIF'] = dif.iloc[signal_idx] if signal_idx < len(dif) else np.nan
    features['MACD_DEA'] = dea.iloc[signal_idx] if signal_idx < len(dea) else np.nan
    features['MACD_hist'] = macd_hist.iloc[signal_idx] if signal_idx < len(macd_hist) else np.nan
    k, d, j = compute_kdj(df['high'], df['low'], df['close'])
    features['KDJ_K'] = k.iloc[signal_idx] if signal_idx < len(k) else np.nan
    features['KDJ_D'] = d.iloc[signal_idx] if signal_idx < len(d) else np.nan
    features['KDJ_J'] = j.iloc[signal_idx] if signal_idx < len(j) else np.nan
    candle_range = row['high'] - row['low']
    features['body_ratio'] = abs(row['close'] - row['open']) / candle_range if candle_range > 0 else 0
    features['close_location'] = (row['close'] - row['low']) / candle_range if candle_range > 0 else 0
    upper_shadow = row['high'] - max(row['close'], row['open'])
    lower_shadow = min(row['close'], row['open']) - row['low']
    features['upper_shadow_pct'] = upper_shadow / candle_range if candle_range > 0 else 0
    features['lower_shadow_pct'] = lower_shadow / candle_range if candle_range > 0 else 0
    return features


def check(file_path: str, hold_list=None, feature_cache=None) -> List:
    print(f"\n检查文件: {file_path}")
    if not XGBOOST_AVAILABLE:
        print("  XGBoost不可用")
        return [-1]
    df = load_stock_data(file_path)
    if df is None:
        print("  数据加载失败")
        return [-1]
    if len(df) < 150:
        print(f"  数据不足: {len(df)} < 150")
        return [-1]
    df = compute_brick(df)
    signal_idx = len(df) - 1
    features = extract_features(df, signal_idx)
    if features is None:
        print("  不满足brick条件")
        return [-1]
    print(f"  满足brick条件, rebound_ratio={features['rebound_ratio']:.4f}")
    X = pd.DataFrame([features])
    X = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = _load_model()
    if model is None:
        print("  模型加载失败")
        return [-1]
    try:
        dtest = xgb.DMatrix(X)
        proba = model.predict(dtest)[0]
        print(f"  预测概率: {proba:.4f}")
    except Exception as e:
        print(f"  预测失败: {e}")
        return [-1]
    if proba < THRESHOLD:
        print(f"  概率低于阈值: {proba:.4f} < {THRESHOLD}")
        return [-1]
    row = df.iloc[signal_idx]
    stop_loss_price = round(float(row['low']) * 0.99, 3)
    print(f"  通过! 止损价={stop_loss_price}, 收盘价={row['close']}, 概率={proba:.4f}")
    return [1, stop_loss_price, float(row['close']), round(proba, 4), "XGBoost预测"]


if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y%m%d')
    data_dir = Path(f"/Users/lidongyang/Desktop/Qstrategy/data/{today_str}")
    
    brick_stocks = [
        ('002286', 'SZ'),
        ('600359', 'SH'),
        ('603209', 'SH'),
        ('688047', 'SH'),
    ]
    
    print(f"检查今天brick筛选出的4只股票")
    print("="*80)
    
    for code, market in brick_stocks:
        file_path = data_dir / f"{market}#{code}.txt"
        if not file_path.exists():
            print(f"\n{code}: 文件不存在 {file_path}")
            continue
        result = check(str(file_path))
        print(f"  结果: {result}")
