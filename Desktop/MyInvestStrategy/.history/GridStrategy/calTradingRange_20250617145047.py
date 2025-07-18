import figure as fig
import numpy as np

# 计算价格波动范围 ，计算价格的最大值和最小值之间的差值，用于判断价格波动的大小
def cal_range(data, days, price_type):
    data['high_range'] = data[price_type].rolling(days).max()
    data['low_range'] = data[price_type].rolling(days).min()
    data['Range'] = data['high_range'] - data['low_range']

    fig.figure_line(data['Date'], data['Range'], 'Range', 'red')
    return data

# 计算波动率 计算价格的标准差，用于判断价格波动的程度
def cal_volatility(data, days, price_type):
    data['Volatility'] = data[price_type].rolling(days).std()

    fig.figure_line(data['Date'], data['Volatility'], 'Volatility', 'red')
    return data

# 计算ATR 平均真实波幅，计算价格波动的幅度，用于判断价格波动的大小，指标上升，波动程度增加，反之减少
def cal_ATR(data, days):
    data['TR'] = np.maximum(data['high'] - data['low'], np.abs(data['high'] - data['close'].shift(1)))
    data['ATR'] = data['TR'].rolling(window=days).mean()
    fig.figure_line(data['Date'], data['ATR'], 'ATR', 'red')
    return data