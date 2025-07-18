import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

start_date = '2020-01-01'
end_date = '2024-01-24'
# 生成模拟价格数据
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2024-01-01')
#prices = pd.Series(np.cumsum(np.random.randn(len(dates))), index=dates)
#data = pd.DataFrame({'Price': prices})

stock_data = yf.download('600886.SS', start=start_date, end=end_date)
prices = pd.Series(stock_data['Close'].values.flatten())
data = pd.DataFrame({'Price': prices})

def backtest_strategy(data, short_window, long_window, stop_loss, take_profit):
    """
    回测趋势跟踪策略并返回年化收益率
    """
    df = data.copy()
    
    # 计算均线
    df['SMA_Short'] = df['Price'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Price'].rolling(window=long_window).mean()
    
    # 生成买入和卖出信号
    df['Signal'] = 0
    df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
    df.loc[df['SMA_Short'] <= df['SMA_Long'], 'Signal'] = -1
    
    # 计算每日收益
    df['Return'] = df['Price'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']  # 策略收益
    
    # 模拟止损和止盈
    cum_return = 0  # 累计收益
    for i in range(1, len(df)):
        if df['Signal'].iloc[i-1] == 1:  # 多头持仓
            daily_return = df['Return'].iloc[i]
            cum_return += daily_return
            if cum_return < -stop_loss:  # 触发止损
                # df['Signal'].iloc[i:] = 0
                signal_column_index = df.columns.get_loc('Signal')
                df.iloc[i:, signal_column_index] = 0
                break
            elif cum_return > take_profit:  # 触发止盈
                # df['Signal'].iloc[i:] = 0
                signal_column_index = df.columns.get_loc('Signal')
                df.iloc[i:, signal_column_index] = 0
                break
    
    # 计算累计收益和年化收益率
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    final_cumulative_return = df['Cumulative_Return'].iloc[-1]
    if not np.isnan(final_cumulative_return) and final_cumulative_return > 0:
        annualized_return = (final_cumulative_return) ** (252 / len(df)) - 1
    else:
        annualized_return = np.nan  # 或者进行其他的默认处理
    return annualized_return

# 定义参数网格
param_grid = {
    'short_window': [5, 10, 15],
    'long_window': [30, 50, 70],
    'stop_loss': [0.05, 0.1, 0.15],  # 止损阈值
    'take_profit': [0.1, 0.2, 0.3]   # 止盈阈值
}

grid = ParameterGrid(param_grid)
print('grid:', type(grid))

# 存储回测结果
results = []
for params in grid:
    annualized_return = backtest_strategy(data, 
                                          short_window=params['short_window'], 
                                          long_window=params['long_window'], 
                                          stop_loss=params['stop_loss'], 
                                          take_profit=params['take_profit'])
    results.append({
        'params': params,
        'annualized_return': annualized_return
    })

# 将 params 字典展开为独立列
results_df = pd.DataFrame(results)
results_df = pd.concat([results_df.drop(columns=['params']), results_df['params'].apply(pd.Series)], axis=1)

# 构建数据透视表
heatmap_data = results_df.pivot_table(values='annualized_return', 
                                      index='short_window', 
                                      columns='long_window')

# 输出最佳参数
best_result = results_df.loc[results_df['annualized_return'].idxmax()]
print("最佳参数组合：", best_result[['short_window', 'long_window', 'stop_loss', 'take_profit']].to_dict())
print("最佳年化收益率：", best_result['annualized_return'])

# 可视化
plt.figure(figsize=(10, 6))
plt.title('Annualized Return Heatmap')
plt.xlabel('Long Window')
plt.ylabel('Short Window')
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(label='Annualized Return')
# plt.show()
