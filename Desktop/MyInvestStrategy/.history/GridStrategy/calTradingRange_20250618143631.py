import numpy as np
import pandas as pd

# 计算价格波动范围 ，计算价格的最大值和最小值之间的差值，用于判断价格波动的大小
def cal_range(data, days):
    data['high_range'] = data['最高'].rolling(days).max()
    data['low_range'] = data['最低'].rolling(days).min()
    data['Range'] = data['high_range'] - data['low_range']
    value = data['Range'].tolist()[-1]

    print(f'{days}天内，最高价格与最低价格波动差为：{value}')
    return data

# 计算波动率 计算价格的标准差，用于判断价格波动的程度
def cal_volatility(data, days):
    data['high_Volatility'] = data['最高'].rolling(days).std()
    data['low_Volatility'] = data['最低'].rolling(days).std()
    data['close_Volatility'] = data['收盘'].rolling(days).std()
    
    #同时计算多列滚动标准差，滚动标准差通过计算数据在连续时间段内的标准差来衡量数据波动的程度
    std_roll_values = data[['最高', '最低', '收盘']].rolling(days).std()
    std_values = data[['最高', '最低', '收盘']].std()
    print(f'全部数据{days}天滚动价格标准差为：{std_roll_values}，全部数据价格标准差为：{std_values}')
    return data

# 计算ATR 平均真实波幅，计算价格波动的幅度，用于判断价格波动的大小，指标上升，波动程度增加，反之减少
def cal_ATR(data, days):
    data['前日收盘价'] = data['收盘'].shift(1)

    # 计算TR的三个组成部分
    data['TR1'] = data['最高'] - data['最低']
    data['TR2'] = abs(data['最高'] - data['前日收盘价'])
    data['TR3'] = abs(data['最低'] - data['前日收盘价'])

    # 计算真实波幅TR
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    # 删除中间列
    data.drop(['TR1', 'TR2', 'TR3'], axis=1, inplace=True)
    # 计算初始ATR
    data['ATR'] = data['TR'].rolling(window=days).mean()
    # 递推计算后续ATR值
    for i in range(days, len(data)):
        data.loc[i, 'ATR'] = (data.loc[i-1, 'ATR'] * (days-1) + data.loc[i, 'TR']) / days

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    data['ATR'] = data['ATR'].round(2)
    print(data['ATR'])
    return data

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/vscodePython/sh51030020250612.csv'
    data = pd.read_csv(file_path)

    cal_ATR(data, 100)