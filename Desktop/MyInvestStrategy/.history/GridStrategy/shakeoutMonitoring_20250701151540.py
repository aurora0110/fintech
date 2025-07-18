import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monitor(df, start_date, end_date):
    # 参数
    N1 = 3 # 短期指标
    N2 = 21 # 长期指标

    df = df[(df['日期'] >= str(start_date)) & (df['日期'] <= str(end_date))]


    # 计算函数
    def momentum_indicator(C, L, n):
        return 100 * (C - L.rolling(n).min()) / (C.rolling(n).max() - L.rolling(n).min())

    # 计算短中长期指标
    df['短期'] = momentum_indicator(df['收盘'], df['最低'], N1)
    df['中期'] = momentum_indicator(df['收盘'], df['最低'], 10)
    df['中长期'] = momentum_indicator(df['收盘'], df['最低'], 20)
    df['长期'] = momentum_indicator(df['收盘'], df['最低'], N2)

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

    return df
