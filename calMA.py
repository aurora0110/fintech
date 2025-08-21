import pandas as pd
import numpy as np
import yfinance as yf  # 需要安装：pip install yfinance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import plotly.subplots as sp
import plotly.graph_objs as go
import pandas as pd

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
        # 将输入的日期也转为 datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)  # 示例结束日期

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
    plt.xticks(selected_ticks, rotation=105)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.text(2, 5, f'{type}', fontsize=10, color='gray') # 增加图片注释
    plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)  # 增加网格线
    plt.title(f'{ticker} {type}')
    plt.legend(loc='best')
    plt.show()

# 4.多张图汇集在一张画板上
def plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, ticker, windows):
    # x_axis 是形如 ['2020-01-01', ...] 的字符串列表
    x_axis = pd.to_datetime(data_ma['date'])

    fig = sp.make_subplots(
        rows=6, cols=2,
        specs=[[{}, {}],    # 第一行两列
            [{"colspan": 2}, None],  # 第二行一整行（跨2列）
            [{"colspan": 2}, None],  # 第三行一整行
            [{"colspan": 2}, None],  # 第四行一整行
            [{}, {}],
            [{"colspan": 2}, None]], # 第五行一整行（跨2列）
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        subplot_titles=[
            f'MA {windows[0]} {windows[1]} {windows[2]}', 'BBI',
            'Avg & Close Price',
            'KDJ-J Highlighted Points',
            'MACD',
            'KDJ-J Highlighted Points -10~90', 'KDJ-J Highlighted Points-15~100',
            'Shakeout Monitoring'
        ]
    )

    fig.update_layout(
    xaxis3=dict(matches='x'),
    xaxis4=dict(matches='x'),
    xaxis5=dict(matches='x'),
    xaxis6=dict(matches='x')
)

    # 第一行，左图 MA
    fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[0]}'], name=f'MA_{windows[0]}', line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[1]}'], name=f'MA_{windows[1]}', line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[2]}'], name=f'MA_{windows[2]}', line=dict(color='green')), row=1, col=1)

    # 第一行，右图 BBI
    fig.add_trace(go.Scatter(x=x_axis, y=data_bbi['bbi'], name='BBI', line=dict(color='orange')), row=1, col=2)

    # 第二行，整行 price
    fig.add_trace(go.Scatter(x=x_axis, y=data_price['avg_price'], name='avg_price', line=dict(color='yellow')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=data_price['close_price'], name='close_price', line=dict(color='green')), row=2, col=1)

    # 第三行，整行 KDJ-J + 高亮点
    fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=3, col=1)

    mask_low = data_kdj['J'] <= -5
    fig.add_trace(go.Scatter(
        x=x_axis[mask_low],
        y=data_kdj['J'][mask_low],
        mode='markers+text',
        name='J <= -5',
        marker=dict(color='red', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=3, col=1)

    mask_high = data_kdj['J'] > 80
    fig.add_trace(go.Scatter(
        x=x_axis[mask_high],
        y=data_kdj['J'][mask_high],
        mode='markers+text',
        name='J > 80',
        marker=dict(color='green', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=3, col=1)

    # 第四行，整行 MACD
    fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DIF'], name='DIF', line=dict(color='white')), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DEA'], name='DEA', line=dict(color='yellow')), row=4, col=1)

    # 第五行，左图 -10～90
    fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=5, col=1)

    mask_low = data_kdj['J'] <= -10
    fig.add_trace(go.Scatter(
        x=x_axis[mask_low],
        y=data_kdj['J'][mask_low],
        mode='markers+text',
        name='J <= -10',
        marker=dict(color='red', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=5, col=1)

    mask_high = data_kdj['J'] > 90
    fig.add_trace(go.Scatter(
        x=x_axis[mask_high],
        y=data_kdj['J'][mask_high],
        mode='markers+text',
        name='J > 90',
        marker=dict(color='green', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=5, col=1)

    # 第五行，右图 -15～100
    fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=5, col=2)

    mask_low = data_kdj['J'] <= -15
    fig.add_trace(go.Scatter(
        x=x_axis[mask_low],
        y=data_kdj['J'][mask_low],
        mode='markers+text',
        name='J <= -15',
        marker=dict(color='red', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=5, col=1)

    mask_high = data_kdj['J'] > 100
    fig.add_trace(go.Scatter(
        x=x_axis[mask_high],
        y=data_kdj['J'][mask_high],
        mode='markers+text',
        name='J > 100',
        marker=dict(color='green', size=8),
        text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
        textposition='top center',
        textfont=dict(color='white')
    ), row=5, col=2)

    # 第六行，整行 shakeout monitoring
    fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['短期'], name='短期', line=dict(color='white')), row=6, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['长期'], name='长期', line=dict(color='red')), row=6, col=1)

    # 添加掩码
    mask = (data_shakeout['短期'] < 20) & (data_shakeout['长期'] > 60)
    # 筛选符合条件的 x 和 y 值
    x_highlight = x_axis[mask]
    y_highlight = data_shakeout['短期'][mask]
    # 添加高亮点
    fig.add_trace(go.Scatter(
        x=x_highlight,
        y=y_highlight,
        mode='markers+text',
        name='短期<20 & 长期>60',
        marker=dict(color='cyan', size=10, symbol='circle'),
        text=[f'{v:.1f}' for v in y_highlight],
        textposition='top center',
        textfont=dict(color='white')
    ), row=6, col=1)

    # 在第六行子图（row=6, col=1）上绘制 y=20, 60, 80 三条横线 红线在60 80之间 白线在20以下
    for y_val in [60, 80]:
        fig.add_shape(
            type="line",
            x0=x_axis.min(),
            x1=x_axis.max(),
            y0=y_val,
            y1=y_val,
            line=dict(color="green", width=3, dash="solid"),
            xref="x8",  # row=6 col=1 的 subplot x 轴
            yref="y8"   # row=6 col=1 的 subplot y 轴
        )

    fig.add_shape(
            type="line",
            x0=x_axis.min(),
            x1=x_axis.max(),
            y0=20,
            y1=20,
            line=dict(color="yellow", width=3, dash="solid"),
            xref="x8",  # row=6 col=1 的 subplot x 轴
            yref="y8"   # row=6 col=1 的 subplot y 轴
        )
    # 更新布局
    fig.update_layout(
        height=1200,
        width=1400,
        title=f'{ticker}',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,1)',
        template='plotly_white'
    )

    fig.show()

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