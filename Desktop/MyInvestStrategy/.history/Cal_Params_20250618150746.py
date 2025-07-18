import figure as fig
import numpy as np
# @Description: 计算MACD 指数，计算价格的变化趋势，DIF、DEA、MACD，用于判断价格趋势的强弱 DIF穿过DEA可能是买入 反之可能是卖出
# @params: data pd.DataFrame
# @params: days int
# @params: price_type str
# @return: data pd.DataFrame
def cal_macd(data, days_short, days_long, price_type):
    data[f'EMA_{days_short}'] = data[price_type].ewm(span=days_short, adjust=False).mean()
    data[f'EMA_{days_long}'] = data[price_type].ewm(span=days_long, adjust=False).mean()
    data['DIF'] = data[f'EMA_{days_short}'] - data[f'EMA_{days_long}']
    data['DEA'] = data['DIF'].ewm(span=9, adjust=False).mean()

    fig.figure_line(data['Date'], data['DIF'], data['DEA'], 'DIF', 'DEA', 'red', 'blue')
    return data

# @Description: 计算RSI 指数，计算价格的变化趋势，>70 为超买，<30 为超卖，用于判断价格趋势的强弱
# @params: data pd.DataFrame
# @params: days int
# @params: price_type str
# @return: data pd.DataFrame
def cal_rsi(data, days, price_type):
    data['delta'] = data[price_type].diff()
    data['gain'] = np.where(data['delta'] > 0, data['delta'], 0)
    data['loss'] = np.where(data['delta'] < 0, data['delta'], 0)
    data['avg_gain'] = data['gain'].rolling(days).mean()
    data['avg_loss'] = abs(data['loss'].rolling(days).mean())
    data['RSI'] = 100 - (100 / (1 + data['avg_gain'] / data['avg_loss']))

    fig.figure_line(data['Date'], data['RSI'], 'RSI', 'red')
    return data

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

# 计算KDJ 随机指标，计算价格与收盘价的比率，用于判断价格趋势的强弱
def cal_KDJ(data, days, m1, m2):
    """
    计算KDJ指标
    :param df: pandas DataFrame，需包含列 '最高', '最低', '收盘'
    :param n: 计算RSV的周期（默认9）
    :param m1: K值的平滑周期（默认3）
    :param m2: D值的平滑周期（默认3）
    :return: 原始DataFrame添加 'K', 'D', 'J' 列
    """
    data['high_kdj'] = data['最高'].rolling(window = days, min_periods = 1).max()
    data['low_kdj'] = data['最低'].rolling(window = days, min_periods = 1).min()
    data['RSV'] = (data['收盘'] - data['low_kdj']) / (data['high_kdj'] - data['low_kdj']) * 100
    data['K'] = data['RSV'].ewm(com=0.5,alpha=1/m1, adjust=False).mean()
    data['D'] = data['K'].ewm(com=0.5, alpha=1/m2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    fig.figure_line(data['Date'], data['K'], data['D'], data['J'], 'K', 'D', 'J', 'red', 'blue', 'green') 
    return data   

# 计算CCI 商品通道指数，计算价格与平均值的偏离程度，用于判断超买超卖或趋势信号 >100超买 <-100超卖
def cal_CCI(data, days, price_type):
    data['Typical Price'] = (data['high'] + data['low'] + data['close']) / 3
    data['Mean Deviation'] = data['Typical Price'].rolling(days).std()
    data['CCI'] = (data['Typical Price'] - data['Typical Price'].rolling(days).mean()) / (0.015 * data['Mean Deviation'])

    fig.figure_line(data['Date'], data['CCI'], 'CCI', 'red')
    return data
# 计算DMI 动量指标，计算价格的变化趋势，DI+、DI-、ADX、ADXR，+DI从下向上穿过-DI是买入，反之是卖出 adx>50是趋势，<50是震荡
def cal_DMI(data, days, price_type):
    data['TR'] = np.maximum(data['high'] - data['low'], np.abs(data['high'] - data['close'].shift(1)))
    data['HD'] = np.maximum(data['high'] - data['high'].shift(1), 0)
    data['LD'] = np.maximum(data['low'].shift(1) - data['low'], 0)
    data['DI+'] = 100 * (data['HD'] / data['TR'])
    data['DI-'] = 100 * (data['LD'] / data['TR'])
    data['ADX'] = data['DI+'].rolling(window=days).mean() / data['DI-'].rolling(window=days).mean()
    data['ADX'].fillna(0, inplace=True)

    fig.figure_line(data['Date'], data['DI+'], data['DI-'], data['ADX'],'DI+', 'DI-', 'ADX', 'red', 'green', 'blue')
    return data
# 计算ATR 平均真实波幅，计算价格波动的幅度，用于判断价格波动的大小，指标上升，波动程度增加，反之减少
def cal_ATR(data, days):
    data['TR'] = np.maximum(data['high'] - data['low'], np.abs(data['high'] - data['close'].shift(1)))
    data['ATR'] = data['TR'].rolling(window=days).mean()
    fig.figure_line(data['Date'], data['ATR'], 'ATR', 'red')
    return data

# 计算DMA 平均线差，计算价格与平均线的差值，用于判断价格趋势的强弱
def cal_DMA(data, days):
    data['MoneyFlow'] = (data['close'] - data['low']) - (data['high'] - data['close'])
    data['MoneyFlow'] = data['MoneyFlow'] / (data['high'] - data['low']) * data['Volume']
    data['AD'] = data['MoneyFlow'].cumsum()
    data['DMA'] = data['AD'].rolling(window=days).mean()
    return data

# 计算BOLL 布林带，计算价格与平均值的偏离程度，用于判断超买超卖或趋势信号
def cal_BOLL(data, days):
    data['SMA'] = data['close'].rolling(window=20).mean()
    data['STD'] = data['close'].rolling(window=20).std()
    data['Upper'] = data['SMA'] + 2 * data['STD']
    data['Lower'] = data['SMA'] - 2 * data['STD']

    fig.figure_line(data['Date'], data['Upper'], data['Lower'], data['close'], 'Upper', 'Lower', 'close', 'red', 'blue', 'green')
    return data

# 计算PE
# 计算PB
