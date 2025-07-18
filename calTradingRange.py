import numpy as np
import pandas as pd

def cal_range(data, days):
    '''
    计算价格波动范围 ，计算窗口内价格的最大值和最小值之间的差值，用于判断价格波动的大小
    :param data: dataframe格式的数据
    :param days: 计算窗口
    :return: 
    '''
    data['high_range'] = data['最高'].rolling(days).max()
    data['low_range'] = data['最低'].rolling(days).min()
    data['Range'] = data['high_range'] - data['low_range']
    value = round(data['Range'].tolist()[-1], 4)

    print(f'全部数据，{days}个交易日内，最高价格与最低价格波动差为：{value}')
    return value
def cal_volatility(data, days):
    '''
    # 计算波动率 计算价格的标准差，用于判断价格波动的程度，衡量价格与均值的偏离程度   
    # :param data: dataframe格式的数据
    :param days: 计算窗口
    :return: 
    '''
    data['high_Volatility'] = data['最高'].rolling(days).std()
    data['low_Volatility'] = data['最低'].rolling(days).std()
    data['close_Volatility'] = data['收盘'].rolling(days).std()
    
    #同时计算多列滚动标准差，滚动标准差通过计算数据在连续时间段内的标准差来衡量数据波动的程度
    std_roll_values_high = round(data[['最高']].rolling(days).std().mean(), 4)
    std_roll_values_low = round(data[['最低']].rolling(days).std().mean(), 4)
    std_roll_values_close = round(data[['收盘']].rolling(days).std().mean(), 4)

    std_values_high = round(data[['最高']].std().mean(), 4)
    std_values_low = round(data[['最低']].std().mean(), 4)
    std_values_close = round(data[['收盘']].std().mean(), 4)

    print(f'全部数据，{days}个交易日内，最高价格滚动标准差为：{std_roll_values_high}，最低价格滚动标准差为：{std_roll_values_low}，收盘价格滚动标准差为：{std_roll_values_close}，\n最高价格标准差为：{std_values_high}，最低价格标准差为：{std_values_low}，收盘价格标准差为：{std_values_close}')

def cal_ATR(data, days):
    '''
    # 计算ATR 平均真实波幅，计算窗口内价格波动的幅度，用于判断价格波动的大小，指标上升，波动程度增加，反之减少，反应价格真实波动范围
    :param data: dataframe格式的数据
    :param days: 计算窗口
    :return: 
    '''
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
    
    data['ATR'] = data['ATR'].round(4)
    # 打印全部内容
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
        # print(data['ATR'])
    
    result = round(data['ATR'].mean(), 4)
    print(f'全部数据，{days}个交易日内，平均滚动ATR为：{result}')
    return result

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/MYINVESTSTRATEGY/sh51030020250612.csv'
    data = pd.read_csv(file_path)

    cal_ATR(data, 100)
    cal_range(data, 100)
    cal_volatility(data, 100)