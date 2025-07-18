# 使用相对强弱指数（RSI）来判断市场是否进入超买或超卖状态
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 计算RSI指标
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 示例震荡区间策略
def range_trading_strategy(df):
    rsi = calculate_rsi(df['Close'])
    df['Signal'] = 0  # 默认信号为0
    
    # 设定超卖和超买阈值
    oversold = 30
    overbought = 70

    # 超卖信号：RSI低于30，买入
    df.loc[rsi < oversold, 'Signal'] = 1
    
    # 超买信号：RSI高于70，卖出
    df.loc[rsi > overbought, 'Signal'] = -1
    
    return df

# 假设 df 是历史数据
df = pd.DataFrame({
    'Close': np.random.randn(1000).cumsum() + 100  # 示例收盘价数据
})

df = range_trading_strategy(df)
df.to_excel('RangeTrend_output.xlsx', index=False)

print(df)

# 可视化
plt.plot(df['Close'], label='Price')
plt.plot(df.index, df['Signal'], label='Signal', alpha=0.5)
plt.legend()
plt.show()
