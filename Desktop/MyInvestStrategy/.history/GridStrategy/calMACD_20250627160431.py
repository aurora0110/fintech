import figure as fig
import numpy as np
import matplotlib.pyplot as plt

# @Description: 计算MACD 指数，计算价格的变化趋势，DIF(快白线)、DEA（慢黄线）、MACD，用于判断价格趋势的强弱 DIF穿过DEA可能需要买入 反之可能需要是卖出
# @params: data pd.DataFrame
# @params: days int
# @params: price_type str
# @return: data pd.DataFrame
def cal_macd(data, start_date, end_date, days_short=12, days_long=26, price_type='收盘'):
    data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]

    result_kv = {}
    data[f'EMA_{days_short}'] = data[price_type].ewm(span=days_short, adjust=False).mean()
    data[f'EMA_{days_long}'] = data[price_type].ewm(span=days_long, adjust=False).mean()

    data['DIF'] = data[f'EMA_{days_short}'] - data[f'EMA_{days_long}']
    data['DEA'] = data['DIF'].ewm(span=9, adjust=False).mean()
    # 计算MACD柱
    data['MACD'] = 2 * (data['DIF'] - data['DEA'])

    result_kv['DIF'] = data['DIF']
    result_kv['DEA'] = data['DEA']
    result_kv['MACD'] = data['MACD']

    return result_kv
