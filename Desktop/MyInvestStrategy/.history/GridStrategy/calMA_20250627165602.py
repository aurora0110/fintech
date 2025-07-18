import pandas as pd
import numpy as np
import yfinance as yf  # 需要安装：pip install yfinance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import matplotlib.dates as mdates

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
def calculate_moving_averages(data, start_date, end_date, windows=[20, 60, 120]):
    """
    计算指定周期的移动平均线
    :param data: 包含'收盘'列的DataFrame
    :param windows: 均线周期列表
    :return: 添加了均线列的DataFrame
    """
    data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]

    windows.append(0) # 添加0周期，增加一个循环添加上日期
    result_kv = {}
    for window in windows:
        if window != 0:
            value = data['收盘'].rolling(window=window).mean()
            result_kv[f'MA_{window}'] = value
        else:
            result_kv['date'] = data['日期']

    return result_kv

def calculate_bbi(data, start_date, end_date):
    """
    计算股票的BBI指标
    :param stock_code: 股票代码（如'600519'）
    :param start_date: 开始日期（'YYYY-MM-DD'）
    :param end_date: 结束日期
    :return: 包含BBI的DataFrame
    """
    data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
    # 计算日均价
    data.loc[:, 'avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
    
    # 计算不同周期均线
    data.loc[:, 'ma3'] = data['avg_price'].rolling(3).mean()
    data.loc[:, 'ma6'] = data['avg_price'].rolling(6).mean()
    data.loc[:, 'ma12'] = data['avg_price'].rolling(12).mean()
    data.loc[:, 'ma24'] = data['avg_price'].rolling(24).mean()
    
    # 计算BBI
    data.loc[:, 'bbi'] = (data['ma3'] + data['ma6'] + data['ma12'] + data['ma24']) / 4

    result_kv = {}
    result_kv[f'bbi'] = data['bbi']
    result_kv['date'] = data['日期']
    
    return result_kv

def calculate_price(data, start_date, end_date):
    """
    计算股票的价格
    :param stock_code: 股票代码（如'600519'）
    :param start_date: 开始日期（'YYYY-MM-DD'）
    :param end_date: 结束日期
    :return: 包含BBI的DataFrame
    """
    data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
    # 计算日均价
    data.loc[:, 'avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
    data.loc[:, 'close_price'] = data['收盘']

    result_kv = {}
    result_kv[f'avg_price'] = data['avg_price']
    result_kv[f'close_price'] = data['close_price']
    result_kv[f'date'] = data['日期']
    
    return result_kv


def cal_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["收盘"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["收盘"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

# 3. 可视化函数
def plot_moving_averages(data, ticker, colors, type):

    plt.figure(figsize=(14, 8))
    x_axis = data['date'].tolist()

    i = 0 
    for key, value in data.items():
        if i < len(colors):
            plt.plot(x_axis, value, label=key, color=colors[i], linewidth=1.5)
            i += 1
        else:
            pass
    
    # 手动设置步长
    step = 100
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.text(2, 5, f'{type}', fontsize=10, color='gray') # 增加图片注释
    plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    plt.title(f'{ticker} {type}')
    plt.legend(loc='best')
    plt.show()

# 4.多张图汇集在一张画板上
def plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, ticker, windows):
    #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
    # 创建画板
    plt.figure(figsize=(14, 10))
    plt.suptitle(f'{ticker} ma bbi prices', fontsize = 16)

    x_axis = data_ma['date'].tolist()

    # 第一张图 ma
    ax1 = plt.subplot2grid((4,2),(0,0))

    ax1.plot(x_axis, data_ma[f'MA_{windows[0]}'], 'r-')
    ax1.plot(x_axis, data_ma[f'MA_{windows[1]}'], 'b-')
    ax1.plot(x_axis, data_ma[f'MA_{windows[2]}'], 'g-')
    ax1.set_title(f'MA {windows[0]} {windows[1]} {windows[2]}')
    #ax1.set_xlabel('date')
    ax1.set_ylabel('price')
    ax1.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    # 手动设置步长
    step = 50
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    ax1.legend()

    # 第二张图 bbi
    ax2 = plt.subplot2grid((4,2),(0,1))
    ax2.plot(x_axis, data_bbi['bbi'], 'r-')
    ax2.set_title('bbi')
    #ax2.set_xlabel('date')
    ax2.set_ylabel('price')
    ax2.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    # 手动设置步长
    step = 50
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    ax2.legend()

    # 第三张图 price
    ax3 = plt.subplot2grid((4,2),(1,0),colspan=2)
    ax3.plot(x_axis, data_price['avg_price'], 'r-')
    ax3.plot(x_axis, data_price['close_price'], 'b-')
    ax3.set_title('avg & close price')
    #ax3.set_xlabel('date')
    ax3.set_ylabel('price')
    ax3.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    # 手动设置步长
    step = 50
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    ax3.legend()

    # 第四张图 macd
    ax3 = plt.subplot2grid((4,2),(2,0),colspan=2)
    ax3.plot(x_axis, data_macd['DIF'], 'r-') # 快白线
    ax3.plot(x_axis, data_macd['DEA'], 'y-') # 慢黄线
    ax3.set_title('macd')
    #ax3.set_xlabel('date')
    ax3.set_ylabel('')
    ax3.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    # 手动设置步长
    step = 50
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    ax3.legend()

    # 第五张图 j
    ax3 = plt.subplot2grid((4,2),(3,0),colspan=2)
    ax3.plot(x_axis, data_kdj['J'], 'r-') 
    ax3.set_title('J')
    #ax3.set_xlabel('date')
    ax3.set_ylabel('')
    ax3.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线

    print('=========',type(x_axis), type(data_kdj['J']))
    for x,y in zip(x_axis, data_kdj['J']):
        if y <= -5:
            plt.scatter(x,y,color='red',s=150,zorder=3)
            plt.annotate(f'{x}', xy=(x,y), xytext=(x+0.5,y+2), arrowprops=dict(facecolor='black', shrink=0.05), bbox=dict(boxstyle="round,pad=0.3", fc="yellow",alpha=0.5))

    # 手动设置步长
    step = 50
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    ax3.legend()

    # 手动设置步长
    step = 100
    # 选择x轴上的刻度
    selected_ticks = x_axis[::step]
    plt.xticks(selected_ticks, rotation=45)
    # 自动调整坐标、标题位置，避免重合
    plt.tight_layout()
    plt.show()

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