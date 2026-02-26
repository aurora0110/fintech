"""
Streamlit应用：单针下35策略可视化工具

功能：
1. 加载指定的单针策略回测结果JSON文件
2. 输入股票代码，显示该股票的K线图
3. 在K线图上标记买卖点
4. 显示交易统计信息
5. K线颜色：阳线红色，阴线绿色
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置页面配置
st.set_page_config(
    page_title="单针下35策略可视化",
    page_icon="📈",
    layout="wide"
)

# 主标题
st.title("单针下35策略可视化工具")

# 侧边栏
st.sidebar.header("配置选项")

# 获取最新的单针策略JSON文件
results_dir = "/Users/lidongyang/Desktop/Qstrategy/results"
json_files = [f for f in os.listdir(results_dir) if f.startswith('backtest_pin_strategy_') and f.endswith('.json')]

if not json_files:
    st.error("未找到单针策略回测结果文件")
    st.stop()

# 选择最新的JSON文件
latest_file = sorted(json_files)[-1]
file_path = os.path.join(results_dir, latest_file)
st.sidebar.info(f"使用文件: {latest_file}")

# 加载JSON文件
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    st.sidebar.success("成功加载单针策略回测结果")
except Exception as e:
    st.error(f"加载文件失败: {str(e)}")
    st.stop()

# 显示回测概览
st.header("单针下35策略回测概览")
overview_cols = st.columns(4)
overview_cols[0].metric("总收益率", f"{data.get('total_return', 0) * 100:.2f}%")
overview_cols[1].metric("年化收益率", f"{data.get('annual_return', 0) * 100:.2f}%")
overview_cols[2].metric("交易成功率", f"{data.get('success_rate', 0) * 100:.2f}%")
overview_cols[3].metric("总交易次数", f"{data.get('total_trades', 0):,}")

# 显示收益率最高的前10只股票
st.header("收益率最高的前10只股票")
top_10_df = pd.DataFrame(data.get('top_10_stocks', []))
if not top_10_df.empty:
    top_10_df['total_return'] = top_10_df['total_return'] * 100
    top_10_df['annual_return'] = top_10_df['annual_return'] * 100
    st.dataframe(
        top_10_df.style.format({
            'total_return': '{:.2f}%',
            'annual_return': '{:.2f}%'
        })
    )

# 输入股票代码
st.header("股票交易详情")
stock_code = st.text_input("输入股票代码", "000831")

if stock_code not in data.get('stock_results', {}):
    st.warning(f"未找到股票 {stock_code} 的交易记录")
    st.stop()

stock_data = data['stock_results'][stock_code]

# 显示股票交易统计
st.subheader(f"{stock_code} 交易统计")
stats_cols = st.columns(3)
stats_cols[0].metric("交易次数", f"{stock_data.get('num_trades', 0)}")
stats_cols[1].metric("总收益", f"{stock_data.get('total_profit', 0):.4f}")
stats_cols[2].metric("总收益率", f"{stock_data.get('total_return', 0) * 100:.2f}%")

# 加载股票历史数据
data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/backtest_data"
target_stock_file = None
for file in os.listdir(data_dir):
    if file.endswith('.txt') and stock_code in file:
        target_stock_file = file
        break

if not target_stock_file:
    st.error(f"未找到股票 {stock_code} 的历史数据")
    st.stop()

# 计算KDJ指标
def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    n: 计算RSV的周期
    m1: K的平滑因子
    m2: D的平滑因子
    """
    try:
        df['最高n'] = df['最高'].rolling(window=n).max()
        df['最低n'] = df['最低'].rolling(window=n).min()
        df['RSV'] = (df['收盘'] - df['最低n']) / (df['最高n'] - df['最低n']).replace(0, 1) * 100
        df['K'] = df['RSV'].ewm(alpha=1/m1, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        df['最高n'] = df['最高n'].astype('float32')
        df['最低n'] = df['最低n'].astype('float32')
        df['RSV'] = df['RSV'].astype('float32')
        df['K'] = df['K'].astype('float32')
        df['D'] = df['D'].astype('float32')
        df['J'] = df['J'].astype('float32')
        df = df.fillna(0)
        return df
    except Exception as e:
        st.error(f"计算KDJ指标失败: {str(e)}")
        return df

# 计算趋势线指标

def calculate_trend(df):
    """
    计算知行多空线和知行短期趋势线
    与backtest_b1_strategy.py使用的方法一致
    """
    try:
        df['60日均线'] = df['收盘'].rolling(window=60).mean()
        
        # 计算知行短期趋势线：EMA(EMA(C,10),10)
        df['知行短期趋势线'] = df['收盘'].ewm(span=10, adjust=False).mean()  # 第一次EMA(C,10)
        df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()  # 第二次EMA(EMA(C,10),10)
        
        # 计算知行多空线：(MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4
        M1, M2, M3, M4 = 14, 28, 57, 114
        df['MA14'] = df['收盘'].rolling(window=M1).mean()
        df['MA28'] = df['收盘'].rolling(window=M2).mean()
        df['MA57'] = df['收盘'].rolling(window=M3).mean()
        df['MA114'] = df['收盘'].rolling(window=M4).mean()
        df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
        
        df['60日均线'] = df['60日均线'].astype('float32')
        df['知行多空线'] = df['知行多空线'].astype('float32')
        df['知行短期趋势线'] = df['知行短期趋势线'].astype('float32')
        df = df.fillna(0)
        return df
    except Exception as e:
        st.error(f"计算趋势线指标失败: {str(e)}")
        return df

# 加载股票数据
def load_stock_data(file_path):
    try:
        data = []
        # 尝试多种编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
        file_content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    file_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            raise Exception("无法识别文件编码")
        
        # 逐行处理
        for i, line in enumerate(file_content.split('\n')):
            line = line.strip()
            if not line:
                continue
            if i == 0 and '开盘' in line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    date_str = parts[0]
                    open_price = float(parts[1])
                    high_price = float(parts[2])
                    low_price = float(parts[3])
                    close_price = float(parts[4])
                    volume = float(parts[5])
                    data.append([date_str, open_price, high_price, low_price, close_price, volume])
                except ValueError:
                    continue
        
        df = pd.DataFrame(data, columns=['日期', '开盘', '最高', '最低', '收盘', '成交量'])
        df['日期'] = pd.to_datetime(df['日期'])
        df['开盘'] = df['开盘'].astype('float32')
        df['最高'] = df['最高'].astype('float32')
        df['最低'] = df['最低'].astype('float32')
        df['收盘'] = df['收盘'].astype('float32')
        df['成交量'] = df['成交量'].astype('float32')
        df = df.sort_values('日期').reset_index(drop=True)
        df = calculate_kdj(df)
        df = calculate_trend(df)
        return df
    except Exception as e:
        st.error(f"加载股票数据失败: {str(e)}")
        return None

stock_df = load_stock_data(os.path.join(data_dir, target_stock_file))
if stock_df is None:
    st.stop()

# 提取买卖点信号（包含类型信息）
buy_signals = []
sell_signals = []

for signal in stock_data.get('signals', []):
    signal_date = pd.to_datetime(signal['date'])
    price = signal['price']
    signal_type = signal.get('type', '未知')
    if signal['signal'] == 'buy':
        buy_signals.append({
            'date': signal_date,
            'price': price,
            'type': signal_type
        })
    elif signal['signal'] == 'sell':
        sell_signals.append({
            'date': signal_date,
            'price': price,
            'type': signal_type
        })

# 限制数据点数量
max_data_points = 2000
if len(stock_df) > max_data_points:
    step = len(stock_df) // max_data_points
    stock_df_sampled = stock_df.iloc[::step].copy()
    if stock_df_sampled.index[-1] != stock_df.index[-1]:
        stock_df_sampled = pd.concat([stock_df_sampled, stock_df.iloc[[-1]]])
else:
    stock_df_sampled = stock_df

# 创建K线图（阳线红色，阴线绿色）
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1, 
                   subplot_titles=(f'{stock_code} K线图 (单针下35策略)', 'KDJ指标', '成交量'),
                   row_heights=[0.5, 0.2, 0.3])

# 添加K线（阳线红色，阴线绿色）
fig.add_trace(
    go.Candlestick(
        x=stock_df_sampled['日期'],
        open=stock_df_sampled['开盘'],
        high=stock_df_sampled['最高'],
        low=stock_df_sampled['最低'],
        close=stock_df_sampled['收盘'],
        name='K线',
        increasing=dict(line=dict(color='red')),
        decreasing=dict(line=dict(color='green'))
    ), row=1, col=1)

# 添加60日均线
fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['60日均线'],
        mode='lines',
        name='60日均线',
        line=dict(color='orange', width=1, dash='dash')
    ), row=1, col=1)

# 添加知行多空线
fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['知行多空线'],
        mode='lines',
        name='知行多空线',
        line=dict(color='yellow', width=1.5)
    ), row=1, col=1)

# 添加知行短期趋势线
fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['知行短期趋势线'],
        mode='lines',
        name='知行短期趋势线',
        line=dict(color='white', width=1.5)
    ), row=1, col=1)

# 添加买入信号
if buy_signals:
    # 处理同一日期多个信号的情况，添加微小偏移
    buy_data = []
    date_offsets = {}
    for sig in buy_signals:
        date = sig['date']
        if date not in date_offsets:
            date_offsets[date] = 0
        else:
            date_offsets[date] += 1
        buy_data.append({
            'date': date,
            'price': sig['price'],
            'type': sig['type'],
            'offset': date_offsets[date]
        })
    
    buy_dates = [b['date'] for b in buy_data]
    buy_prices = [b['price'] * (1 - b['offset'] * 0.005) for b in buy_data]
    buy_texts = [f"买入 - {b['type']}<br>日期: {b['date'].strftime('%Y-%m-%d')}<br>价格: {b['price']:.4f}" for b in buy_data]
    
    fig.add_trace(
        go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            marker=dict(size=10, color='white', symbol='circle', line=dict(color='black', width=1)),
            text=buy_texts,
            hoverinfo='text',
            name='买入'
        ), row=1, col=1)

# 添加卖出信号
if sell_signals:
    # 处理同一日期多个信号的情况，添加微小偏移
    sell_data = []
    date_offsets = {}
    for sig in sell_signals:
        date = sig['date']
        if date not in date_offsets:
            date_offsets[date] = 0
        else:
            date_offsets[date] += 1
        sell_data.append({
            'date': date,
            'price': sig['price'],
            'type': sig['type'],
            'offset': date_offsets[date]
        })
    
    sell_dates = [s['date'] for s in sell_data]
    sell_prices = [s['price'] * (1 + s['offset'] * 0.005) for s in sell_data]
    sell_texts = [f"卖出 - {s['type']}<br>日期: {s['date'].strftime('%Y-%m-%d')}<br>价格: {s['price']:.4f}" for s in sell_data]
    
    fig.add_trace(
        go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            marker=dict(size=10, color='white', symbol='square', line=dict(color='black', width=1)),
            text=sell_texts,
            hoverinfo='text',
            name='卖出'
        ), row=1, col=1)

# 添加KDJ指标
fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['K'],
        mode='lines',
        name='K',
        line=dict(color='blue', width=1)
    ), row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['D'],
        mode='lines',
        name='D',
        line=dict(color='red', width=1)
    ), row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['J'],
        mode='lines',
        name='J',
        line=dict(color='green', width=1)
    ), row=2, col=1)

# 添加成交量 - 根据K线颜色区分
volume_colors = ['red' if stock_df_sampled['收盘'].iloc[i] >= stock_df_sampled['开盘'].iloc[i] else 'green' 
                 for i in range(len(stock_df_sampled))]
fig.add_trace(
    go.Bar(
        x=stock_df_sampled['日期'],
        y=stock_df_sampled['成交量'],
        name='成交量',
        marker_color=volume_colors
    ), row=3, col=1)

# 更新布局
fig.update_layout(
    title=f'{stock_code} 单针下35策略交易可视化',
    xaxis_title='日期',
    yaxis_title='价格',
    xaxis_rangeslider_visible=False,
    height=900,
    showlegend=True,
    autosize=False,
    width=1200
)

# 更新KDJ子图的布局
fig.update_yaxes(
    title_text='KDJ',
    row=2, col=1,
    range=[-20, 120]
)

# 显示图表
st.plotly_chart(fig, use_container_width=True)

# 显示交易记录
st.subheader("交易记录")
trades = stock_data.get('trades', [])
if trades:
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['profit_rate'] = trades_df['profit_rate'] * 100
        st.dataframe(
            trades_df.style.format({
                'profit_rate': '{:.2f}%'
            })
        )
else:
    st.info("无交易记录")
