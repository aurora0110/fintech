import numpy as np
import pandas as pd

# 计算价格波动范围 ，计算价格的最大值和最小值之间的差值，用于判断价格波动的大小
def cal_range(data, days):
    data['high_range'] = data['最高'].rolling(days).max()
    data['low_range'] = data['最低'].rolling(days).min()
    data['Range'] = data['high_range'] - data['low_range']

    return data

# 计算波动率 计算价格的标准差，用于判断价格波动的程度
def cal_volatility(data, days, price_type):
    data['Volatility'] = data[price_type].rolling(days).std()

    return data

# 计算ATR 平均真实波幅，计算价格波动的幅度，用于判断价格波动的大小，指标上升，波动程度增加，反之减少
def cal_ATR(data, days):
    data['TR'] = np.maximum(data['high'] - data['low'], np.abs(data['high'] - data['close'].shift(1)))
    data['ATR'] = data['TR'].rolling(window=days).mean()
    return data

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/vscodePython/sh51030020250612.csv'
    data = pd.read_csv(file_path)

    cal_range(data, 100)