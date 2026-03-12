from utils import stoploss, technical_indicators
import numpy as np


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) <= 1e-12:
        return np.nan
    if a is None or not np.isfinite(a):
        return np.nan
    return float(a) / float(b)


def check(file_path):
    """
    当前最优单针策略：
    1. 趋势线 > 多空线
    2. 满足单针条件
    3. 下影线占比 <= 0.05
    4. 趋势线近3日斜率 > 0.8%
    """
    df, load_error = stoploss.load_data(file_path)
    if load_error or df is None or len(df) < 25:
        return False

    df = technical_indicators.calculate_trend(df)

    today = df.iloc[-1]
    trend_line = float(today["知行短期趋势线"])
    long_line = float(today["知行多空线"])
    if not np.isfinite(trend_line) or not np.isfinite(long_line):
        return False
    if trend_line <= long_line:
        return False

    if not technical_indicators.caculate_pin(df):
        return False

    today_open = float(today["开盘"])
    today_close = float(today["收盘"])
    today_high = float(today["最高"])
    today_low = float(today["最低"])

    full_range = today_high - today_low
    body_low = min(today_open, today_close)
    lower_shadow_ratio = safe_div(body_low - today_low, full_range)
    if not np.isfinite(lower_shadow_ratio) or lower_shadow_ratio > 0.05:
        return False

    prev_trend = float(df["知行短期趋势线"].iloc[-4])
    trend_slope_3 = safe_div(trend_line, prev_trend) - 1.0
    if not np.isfinite(trend_slope_3) or trend_slope_3 <= 0.008:
        return False

    return True
