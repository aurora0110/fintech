import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/20260315")
PERFECT_CASES_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data/完美图/砖型图")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_SAMPLES_TARGET = 500
NEGATIVE_SAMPLES_TARGET = 500
HOLD_DAYS = 3
POSITIVE_RETURN_THRESHOLD = 0.035
STOP_LOSS_RATIO = 0.99


def load_stock_data(file_path: Path) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None
        header_line = lines[1].strip()
        cols = header_line.split()
        if len(cols) < 7:
            return None
        col_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
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


def check_positive_label(df: pd.DataFrame, signal_idx: int) -> bool:
    if signal_idx + HOLD_DAYS >= len(df):
        return False
    buy_price = df.iloc[signal_idx + 1]['open']
    if pd.isna(buy_price) or buy_price <= 0:
        return False
    for i in range(1, HOLD_DAYS + 1):
        high_price = df.iloc[signal_idx + i]['high']
        if pd.notna(high_price) and (high_price - buy_price) / buy_price > POSITIVE_RETURN_THRESHOLD:
            return True
    return False


def check_negative_label(df: pd.DataFrame, signal_idx: int) -> bool:
    if signal_idx + HOLD_DAYS >= len(df):
        return False
    buy_price = df.iloc[signal_idx + 1]['open']
    signal_low = df.iloc[signal_idx]['low']
    stop_loss_price = signal_low * STOP_LOSS_RATIO
    if pd.isna(buy_price) or buy_price <= 0:
        return False
    for i in range(1, HOLD_DAYS + 1):
        close_price = df.iloc[signal_idx + i]['close']
        if pd.notna(close_price) and close_price < stop_loss_price:
            return True
    return False


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def compute_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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


def extract_features(df: pd.DataFrame, signal_idx: int) -> Optional[Dict]:
    if signal_idx < 20 or signal_idx >= len(df) - HOLD_DAYS:
        return None
    row = df.iloc[signal_idx]
    prev_row = df.iloc[signal_idx - 1]
    if not check_brick_condition(row, prev_row):
        return None
    features = {}
    features['code'] = row.get('code', '')
    features['name'] = row.get('name', '')
    features['date'] = row['date']
    features['close'] = row['close']
    features['open'] = row['open']
    features['high'] = row['high']
    features['low'] = row['low']
    features['volume'] = row['volume']
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
    features['trend_line'] = row['trend_line']
    features['long_line'] = row['long_line']
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
    features['upper_shadow'] = row['high'] - max(row['close'], row['open'])
    features['lower_shadow'] = min(row['close'], row['open']) - row['low']
    features['upper_shadow_pct'] = features['upper_shadow'] / candle_range if candle_range > 0 else 0
    features['lower_shadow_pct'] = features['lower_shadow'] / candle_range if candle_range > 0 else 0
    return features


def parse_perfect_cases() -> List[Tuple[str, str]]:
    cases = []
    for file in PERFECT_CASES_DIR.glob('*.png'):
        filename = file.name
        if filename in ['案例图.png', '双象股份反例.png']:
            continue
        match = re.match(r'(.*?)(\d{8})\.png', filename)
        if match:
            stock_name = match.group(1)
            date_str = match.group(2)
            cases.append((stock_name, date_str))
    return cases


def find_stock_code_by_name(stock_name: str, all_files: List[Path]) -> Optional[str]:
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                first_line = f.readline()
            if stock_name in first_line:
                code_match = re.match(r'(\d+)', first_line)
                if code_match:
                    return code_match.group(1)
        except:
            continue
    return None


def build_dataset():
    print("开始构建数据集...")
    all_files = list(DATA_DIR.glob('*.txt'))
    print(f"找到 {len(all_files)} 个股票数据文件")
    perfect_cases = parse_perfect_cases()
    print(f"找到 {len(perfect_cases)} 个完美案例")
    positive_samples = []
    negative_samples = []
    print("\n处理完美案例...")
    for stock_name, date_str in perfect_cases:
        stock_code = find_stock_code_by_name(stock_name, all_files)
        if not stock_code:
            print(f"  未找到股票代码: {stock_name}")
            continue
        file_path = DATA_DIR / f"{stock_code}.txt"
        if not file_path.exists():
            for f in all_files:
                if stock_code in f.name:
                    file_path = f
                    break
        if not file_path.exists():
            continue
        df = load_stock_data(file_path)
        if df is None or df.empty:
            continue
        df = compute_brick(df)
        target_date = pd.to_datetime(date_str)
        signal_idx = df[df['date'] == target_date].index
        if len(signal_idx) == 0:
            continue
        signal_idx = signal_idx[0]
        features = extract_features(df, signal_idx)
        if features is None:
            continue
        if check_positive_label(df, signal_idx):
            features['label'] = 1
            positive_samples.append(features)
            print(f"  正样本: {stock_name} {date_str}")
    print(f"\n从完美案例中获得 {len(positive_samples)} 个正样本")
    print("\n从股票数据中筛选更多样本...")
    random.shuffle(all_files)
    for file_path in all_files:
        if len(positive_samples) >= POSITIVE_SAMPLES_TARGET and len(negative_samples) >= NEGATIVE_SAMPLES_TARGET:
            break
        df = load_stock_data(file_path)
        if df is None or len(df) < 150:
            continue
        df = compute_brick(df)
        for i in range(20, len(df) - HOLD_DAYS - 5):
            if len(positive_samples) >= POSITIVE_SAMPLES_TARGET and len(negative_samples) >= NEGATIVE_SAMPLES_TARGET:
                break
            features = extract_features(df, i)
            if features is None:
                continue
            if check_positive_label(df, i) and len(positive_samples) < POSITIVE_SAMPLES_TARGET:
                features['label'] = 1
                positive_samples.append(features)
            elif check_negative_label(df, i) and len(negative_samples) < NEGATIVE_SAMPLES_TARGET:
                features['label'] = 0
                negative_samples.append(features)
    print(f"\n最终样本数量:")
    print(f"  正样本: {len(positive_samples)}")
    print(f"  负样本: {len(negative_samples)}")
    all_samples = positive_samples + negative_samples
    if not all_samples:
        print("没有找到任何样本!")
        return
    result_df = pd.DataFrame(all_samples)
    output_file = OUTPUT_DIR / "brick_train_dataset.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n数据集已保存到: {output_file}")
    print(f"数据集列: {list(result_df.columns)}")


if __name__ == "__main__":
    build_dataset()
