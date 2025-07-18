import pandas as pd
import numpy as np
import yfinance as yf  # 需要安装：pip install yfinance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

# 1. 获取数据
def get_data(ticker, start_date, end_date, file_path):
    """
    获取股票数据
    :param ticker: 股票代码
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 包含股票数据的DataFrame
    """
    data = pd.read_excel(file_path, engine=None)
    return data

# 2. 均线计算函数
def calculate_moving_averages(data, windows=[20, 60, 120]):
    """
    计算指定周期的移动平均线
    :param data: 包含'收盘'列的DataFrame
    :param windows: 均线周期列表
    :return: 添加了均线列的DataFrame
    """
    windows.append(0) # 添加0周期，增加一个循环添加上日期
    result_kv = {}
    for window in windows:
        if window != 0:
            value = data['收盘'].rolling(window=window).mean()
            result_kv[f'MA_{window}'] = value
        else:
            result_kv['date'] = data['日期']

    return result_kv

# 3. 可视化函数
def plot_moving_averages(data, ticker, windows, colors):
    plt.figure(figsize=(14, 7))
    x_axis = data['date'].tolist()

    fig, ax = plt.subplots()

    i = 0 
    for key, value in data.items():
        if i < len(colors):
            ax.plot(x_axis, value, label=key, color=colors[i], linewidth=1.5)
            i += 1
        else:
            pass

    formatter = FuncFormatter(format_func)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    plt.show()
    '''
    plt.autoscale(enable=True,axis='x')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.text(2, 5, 'MA show trend', fontsize=10, color='gray') # 增加图片注释
    plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    plt.title(f'{ticker} MA')
    plt.legend(loc='best')
    plt.show()
    '''
def format_func(value, tick_number):
    # 定义一个函数格式化刻度标签
    return f'{value:.2f}'
# 主程序
if __name__ == "__main__":
    # 参数设置
    ticker = '600036.SS'  # 以招商银行A股为例
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 获取1年数据
    
    # 获取数据
    stock_data = get_data(ticker, start_date, end_date)
    
    # 计算均线
    ma_data = calculate_moving_averages(stock_data)
    
    # 显示最新均线值
    latest_date = ma_data.index[-1].strftime('%Y-%m-%d')
    print(f"\n{ticker} 最新均线值（{latest_date}）：")
    print(f"20日均线: {ma_data['MA_20'].iloc[-1]:.2f}")
    print(f"60日均线: {ma_data['MA_60'].iloc[-1]:.2f}")
    print(f"120日均线: {ma_data['MA_120'].iloc[-1]:.2f}")
    
    # 可视化
    plot_moving_averages(ma_data, ticker)
    
    # 保存结果（可选）
    ma_data.to_csv(f'{ticker}_moving_averages.csv')
    print(f"\n数据已保存至：{ticker}_moving_averages.csv")