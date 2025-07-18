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