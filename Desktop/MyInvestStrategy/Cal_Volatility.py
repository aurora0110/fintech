import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

stock_data = yf.download('600886.SS', start='2020-01-01', end='2024-01-24')

prices = pd.Series(stock_data['Close'].values.flatten())
data = pd.DataFrame({'Price': prices})
std_dev = 2

# 计算移动平均线，0.2范围内震荡
data['MA'] = data['Price'].rolling(window=20).mean()
data['OSC'] = np.where((data['Price'] >= data['MA'] * 0.98) & (data['Price'] <= data['MA'] * 1.02), 'OSC', 'NOT OSC')

# 使用布林带
data['upper'] = data['MA'] + std_dev * prices.rolling(window=20).std()
data['lower'] = data['MA'] - std_dev * prices.rolling(window=20).std()
# 判断价格是否在布林带内震荡
data['bolling_band'] = np.where((data['Price'] >= data['lower']) & (data['Price'] <= data['upper']), 'BOLLING', 'NOT BOLLING')

# 计算价格在最近一段时间的最高价和最低价，判断价格在此范围内是否波动
n = 20
data['high'] = data['Price'].rolling(window=n).max()
data['low'] = data['Price'].rolling(window=n).min()
data['volatility'] = np.where((data['Price'] >= data['low']) & (data['Price'] <= data['high']), 'VOLATILITY', 'NOT VOLATILITY')

# 计算最近n天的收盘价标准差
n = 20
data['volatility_std'] = data['Price'].rolling(window=n).std()

# 计算最近n天的价格波动平均幅度
data['daily_volatility'] = data['high'] - data['low']
data['avg_volatility'] = data['daily_volatility'].rolling(window=n).mean()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(data['Price'], label='Price')
plt.plot(data['MA'], label='MA')
plt.plot(data['upper'], label='Upper')
plt.plot(data['lower'], label='Lower')
plt.legend(loc='best')
plt.show()