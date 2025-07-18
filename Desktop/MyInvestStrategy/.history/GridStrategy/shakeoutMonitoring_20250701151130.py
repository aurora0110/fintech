import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monitor(data, start_date, end_date):
    # 模拟数据（可替换为你的真实数据）
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100)
    df = pd.DataFrame({
        'date': dates,
        'close': np.random.normal(loc=100, scale=5, size=100).cumsum(),
        'low': np.random.normal(loc=95, scale=5, size=100).cumsum()
    })

    df.set_index('date', inplace=True)

    # 参数
    N1 = 5
    N2 = 60

    # 计算函数
    def momentum_indicator(C, L, n):
        return 100 * (C - L.rolling(n).min()) / (C.rolling(n).max() - L.rolling(n).min())

    # 计算短中长期指标
    df['短期'] = momentum_indicator(df['close'], df['low'], N1)
    df['中期'] = momentum_indicator(df['close'], df['low'], 10)
    df['中长期'] = momentum_indicator(df['close'], df['low'], 20)
    df['长期'] = momentum_indicator(df['close'], df['low'], N2)

    # 买点条件
    df['四线归零买'] = np.where(
        (df['短期'] <= 6) & (df['中期'] <= 6) & (df['中长期'] <= 6) & (df['长期'] <= 6),
        -30, 0)

    df['白线下20买'] = np.where(
        (df['短期'] <= 20) & (df['长期'] >= 60),
        -30, 0)

    # 白穿红线买（金叉）
    df['白穿红线买'] = np.where(
        (df['短期'] > df['长期']) & (df['短期'].shift(1) <= df['长期'].shift(1)) & (df['长期'] < 20),
        -30, 0)

    # 白穿黄线买（金叉）
    df['白穿黄线买'] = np.where(
        (df['短期'] > df['中期']) & (df['短期'].shift(1) <= df['中期'].shift(1)) & (df['中期'] < 30),
        -30, 0)

    # 输出最后几行查看
    df[['短期', '中期', '中长期', '长期', '四线归零买', '白线下20买', '白穿红线买', '白穿黄线买']].tail()
