"""
B1策略可视化回测面板
支持多种因子、止盈止损策略选择，回测并展示结果
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="B1策略回测面板", page_icon="📈", layout="wide")

st.title("📊 B1策略可视化回测面板")

# ======================== 侧边栏：参数选择 ========================
st.sidebar.header("🔧 回测参数配置")

# 回测时间
st.sidebar.subheader("📅 回测时间")
start_date = st.sidebar.date_input("开始日期", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("结束日期", datetime.today())

# 资金配置
st.sidebar.subheader("💰 资金配置")
initial_capital = st.sidebar.number_input("初始资金", value=1000000, step=100000)
max_positions = st.sidebar.slider("最大持仓数", 1, 10, 4)

# ======================== 因子选择 ========================
st.sidebar.subheader("🎯 买入因子选择")

buy_factors = {}
buy_factors['J值<13'] = st.sidebar.checkbox("J值<13", value=True)
buy_factors['MACD金叉'] = st.sidebar.checkbox("MACD金叉")
buy_factors['RSI超卖'] = st.sidebar.checkbox("RSI超卖 (RSI14<30)")
buy_factors['黄白线金叉'] = st.sidebar.checkbox("黄白线金叉")
buy_factors['缩量'] = st.sidebar.checkbox("缩量 (量能<20日均量)")
buy_factors['涨幅筛选'] = st.sidebar.checkbox("涨幅筛选 (-3%~2%)")
buy_factors['回踩均线'] = st.sidebar.checkbox("回踩均线黄白线")
buy_factors['长阴短柱'] = st.sidebar.checkbox("价格长阴短柱")
buy_factors['砖图短阳长柱'] = st.sidebar.checkbox("砖图短阳长柱")

# J值条件
j_condition = st.sidebar.selectbox("J值买入条件", ["J<13", "J<10%历史分位", "J<5%历史分位"], index=0)

# ======================== 止盈策略 ========================
st.sidebar.subheader("🎯 止盈策略")

take_profit_strategies = {}
take_profit_strategies['J>100'] = st.sidebar.checkbox("J>100 止盈")
take_profit_strategies['固定涨幅'] = st.sidebar.checkbox("固定涨幅止盈", value=True)
fixed_tp_pct = st.sidebar.slider("固定涨幅止盈 (%)", 5, 20, 7)
take_profit_strategies['固定止盈价'] = st.sidebar.checkbox("固定止盈价")
take_profit_strategies['偏离趋势线'] = st.sidebar.checkbox("偏离趋势线止盈")
take_profit_strategies['固定天数'] = st.sidebar.checkbox("固定交易天数止盈")
fixed_tp_days = st.sidebar.slider("止盈天数", 3, 30, 5)
take_profit_strategies['阳线放量滞涨'] = st.sidebar.checkbox("阳线放量滞涨止盈")

# ======================== 止损策略 ========================
st.sidebar.subheader("🛡️ 止损策略")

stop_loss_strategies = {}
stop_loss_strategies['连续2天收盘价新低'] = st.sidebar.checkbox("连续2天收盘价低于前一天最低价")
stop_loss_strategies['当根K线最低点'] = st.sidebar.checkbox("当根K线最低点止损")
stop_loss_strategies['前根K线最低点'] = st.sidebar.checkbox("前根K线最低点止损")
stop_loss_strategies['固定止损比例'] = st.sidebar.checkbox("固定止损比例", value=True)
fixed_sl_pct = st.sidebar.slider("固定止损比例 (%)", 2, 10, 5)
stop_loss_strategies['N型结构前低'] = st.sidebar.checkbox("N型结构前低止损")
stop_loss_strategies['趋势线多空线'] = st.sidebar.checkbox("趋势线多空线下死叉止损")
stop_loss_strategies['放量阴线'] = st.sidebar.checkbox("放量阴线止损")
stop_loss_strategies['固定天数'] = st.sidebar.checkbox("固定天数止损")
fixed_sl_days = st.sidebar.slider("止损天数", 3, 30, 10)
stop_loss_strategies['J位置止损保护'] = st.sidebar.checkbox("不在J<13位置止损")

# ======================== 风控策略 ========================
st.sidebar.subheader("⚠️ 风控策略")

fail_pause = st.sidebar.checkbox("连续失败暂停交易", value=True)
fail_pause_count = st.sidebar.slider("连续失败次数", 2, 10, 3)
fail_pause_days = st.sidebar.slider("暂停天数", 1, 10, 2)

# ======================== 加载数据 ========================

@st.cache_data
def load_data():
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    stock_data = {}
    all_dates_set = set()
    
    if not os.path.exists(data_dir):
        st.error(f"数据目录不存在: {data_dir}")
        return None
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for filename in tqdm(files, desc="加载股票数据"):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
            stock_data[filename.replace('.txt', '')] = df
            all_dates_set.update(df.index.tolist())
        except Exception as e:
            continue
    
    all_dates = sorted(list(all_dates_set))
    return {
        'stock_data': stock_data,
        'all_dates': [pd.Timestamp(d) for d in all_dates]
    }

from tqdm import tqdm

with st.spinner("加载数据中..."):
    cache_data = load_data()

if cache_data is None:
    st.error("❌ 数据加载失败")
    st.stop()

stock_data = cache_data['stock_data']
all_dates = cache_data['all_dates']

st.sidebar.success(f"✅ 数据加载成功: {len(stock_data)}只股票, {len(all_dates)}个交易日")

# ======================== 回测函数 ========================

def calculate_indicators(df):
    """计算技术指标"""
    df = df.copy()
    
    # MACD
    exp1 = df['CLOSE'].ewm(span=12, adjust=False).mean()
    exp2 = df['CLOSE'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_DIFF'] = df['MACD'] - df['MACD_SIGNAL']
    
    # RSI
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # KDJ
    low_n = df['LOW'].rolling(window=9).min()
    high_n = df['HIGH'].rolling(window=9).max()
    rsv = (df['CLOSE'] - low_n) / (high_n - low_n) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 均线
    df['MA5'] = df['CLOSE'].rolling(window=5).mean()
    df['MA10'] = df['CLOSE'].rolling(window=10).mean()
    df['MA20'] = df['CLOSE'].rolling(window=20).mean()
    
    # 量能
    df['VOL_MA20'] = df['VOLUME'].rolling(window=20).mean()
    
    return df

def check_buy_signals(df, idx, buy_factors, j_condition):
    """检查买入信号"""
    if idx < 30:
        return False
    
    row = df.iloc[idx]
    prev_row = df.iloc[idx-1] if idx > 0 else row
    
    # J值条件
    if buy_factors.get('J值<13', False):
        j_val = row.get('J', 50)
        if j_condition == "J<13":
            if j_val >= 13:
                return False
        elif j_condition == "J<10%历史分位":
            j_series = df.iloc[:idx]['J'].dropna()
            if len(j_series) > 50:
                p10 = j_series.quantile(0.10)
                if j_val >= p10:
                    return False
        elif j_condition == "J<5%历史分位":
            j_series = df.iloc[:idx]['J'].dropna()
            if len(j_series) > 50:
                p5 = j_series.quantile(0.05)
                if j_val >= p5:
                    return False
    
    # MACD金叉
    if buy_factors.get('MACD金叉', False):
        macd = row.get('MACD', 0)
        macd_signal = row.get('MACD_SIGNAL', 0)
        prev_macd = prev_row.get('MACD', 0)
        prev_macd_signal = prev_row.get('MACD_SIGNAL', 0)
        if not (prev_macd <= prev_macd_signal and macd > macd_signal):
            if buy_factors.get('MACD金叉', False):
                pass  # 需要金叉才买
    
    # RSI超卖
    if buy_factors.get('RSI超卖', False):
        rsi = row.get('RSI', 50)
        if rsi >= 30:
            return False
    
    # 黄白线金叉
    if buy_factors.get('黄白线金叉', False):
        ma5 = row.get('MA5', 0)
        ma10 = row.get('MA10', 0)
        prev_ma5 = prev_row.get('MA5', 0)
        prev_ma10 = prev_row.get('MA10', 0)
        if not (prev_ma5 <= prev_ma10 and ma5 > ma10):
            if buy_factors.get('黄白线金叉', False):
                pass
    
    # 缩量
    if buy_factors.get('缩量', False):
        vol = row.get('VOLUME', 0)
        vol_ma = row.get('VOL_MA20', 1)
        if vol >= vol_ma:
            return False
    
    # 涨幅筛选
    if buy_factors.get('涨幅筛选', False):
        change = (row['CLOSE'] - prev_row['CLOSE']) / prev_row['CLOSE'] * 100
        if change < -3 or change > 2:
            return False
    
    return True

def run_backtest():
    """执行回测"""
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    # 过滤日期
    valid_dates = [d for d in all_dates if start_date <= d.date() <= end_date]
    
    # 预处理：计算所有股票的技术指标
    progress_bar = tqdm(stock_data.items(), desc="计算指标")
    stock_indicators = {}
    for stock, df in progress_bar:
        df_calc = calculate_indicators(df)
        stock_indicators[stock] = df_calc
    
    cash = float(initial_capital)
    positions = []
    trades = []
    equity_curve = []
    
    total_trades = 0
    win_trades = 0
    loss_trades = 0
    consecutive_losses = 0
    pause_days = 0
    
    for current_date in tqdm(valid_dates, desc="回测中"):
        if pause_days > 0:
            pause_days -= 1
            equity_curve.append(cash + sum(p['current_value'] for p in positions))
            continue
        
        # 处理持仓
        new_positions = []
        for pos in positions:
            stock = pos['stock']
            df = stock_indicators.get(stock)
            if df is None or current_date not in df.index:
                new_positions.append(pos)
                continue
            
            row = df.loc[current_date]
            close = row['CLOSE']
            high = row['HIGH']
            low = row['LOW']
            
            if pd.isna(close) or close <= 0:
                new_positions.append(pos)
                continue
            
            # 止盈检查
            entry_price = pos['entry_price']
            holding_days = current_date - pos['entry_date']
            change_pct = (close - entry_price) / entry_price * 100
            
            tp_triggered = False
            if take_profit_strategies['固定涨幅'] and change_pct >= fixed_tp_pct:
                tp_triggered = True
            if take_profit_strategies['J>100'] and row.get('J', 0) > 100:
                tp_triggered = True
            if take_profit_strategies['固定天数'] and holding_days.days >= fixed_tp_days:
                tp_triggered = True
            
            # 止损检查
            sl_triggered = False
            if stop_loss_strategies['固定止损比例'] and change_pct <= -fixed_sl_pct:
                sl_triggered = True
            if stop_loss_strategies['当根K线最低点'] and low <= pos['entry_low']:
                sl_triggered = True
            if stop_loss_strategies['前根K线最低点'] and current_date in df.index:
                idx = df.index.get_loc(current_date)
                if idx > 0 and low <= df.iloc[idx-1]['LOW']:
                    sl_triggered = True
            
            if tp_triggered or sl_triggered:
                exit_price = close if not sl_triggered else min(low, entry_price * (1 - fixed_sl_pct/100))
                profit = (exit_price - entry_price) / entry_price
                cash += entry_price * (1 + profit)
                
                trades.append({
                    'stock': stock,
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit * 100,
                    'holding_days': holding_days.days
                })
                
                total_trades += 1
                if profit > 0:
                    win_trades += 1
                    consecutive_losses = 0
                else:
                    loss_trades += 1
                    consecutive_losses += 1
                    if fail_pause and consecutive_losses >= fail_pause_count:
                        pause_days = fail_pause_days
            else:
                pos['current_value'] = entry_price * (close / entry_price)
                pos['current_price'] = close
                new_positions.append(pos)
        
        positions = new_positions
        
        # 买入 - 生成当日信号
        if len(positions) < max_positions:
            candidates = []
            
            for stock, df in stock_indicators.items():
                if current_date not in df.index:
                    continue
                
                try:
                    idx = df.index.get_loc(current_date)
                except KeyError:
                    continue
                
                if idx + 1 >= len(df):
                    continue
                
                # 检查买入信号
                if check_buy_signals(df, idx, buy_factors, j_condition):
                    candidates.append(stock)
            
            # 选择候选股票
            available = [s for s in candidates if s not in [p['stock'] for p in positions]]
            
            if available and cash > 0:
                num_to_buy = min(len(available), max_positions - len(positions))
                per_position = cash / num_to_buy
                
                for stock in available[:num_to_buy]:
                    df = stock_indicators[stock]
                    idx = df.index.get_loc(current_date)
                    
                    entry_price = df.iloc[idx + 1]['OPEN']
                    entry_low = df.iloc[idx]['LOW']
                    
                    if entry_price <= 0:
                        continue
                    
                    cash -= per_position
                    
                    positions.append({
                        'stock': stock,
                        'entry_price': entry_price,
                        'entry_low': entry_low,
                        'entry_date': current_date,
                        'current_value': per_position,
                        'current_price': entry_price
                    })
        
        # 计算权益
        total_equity = cash + sum(p['current_value'] for p in positions)
        equity_curve.append(total_equity)
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'dates': valid_dates[:len(equity_curve)],
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'final_equity': equity_curve[-1] if equity_curve else initial_capital
    }

# ======================== 主界面 ========================

if st.button("🚀 开始回测", type="primary"):
    with st.spinner("回测运行中..."):
        result = run_backtest()
    
    # ======================== 显示结果 ========================
    
    # 基本统计
    st.header("📈 回测结果统计")
    
    if result['total_trades'] == 0:
        st.warning("⚠️ 没有产生任何交易，请检查因子设置")
    else:
        # 关键指标
        total_return = (result['final_equity'] - initial_capital) / initial_capital * 100
        years = len(result['dates']) / 252
        annual_return = ((result['final_equity'] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        success_rate = result['win_trades'] / result['total_trades'] * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 总收益率", f"{total_return:.2f}%")
        col2.metric("📊 年化收益率", f"{annual_return:.2f}%")
        col3.metric("🎯 成功率", f"{success_rate:.2f}%")
        col4.metric("📝 交易次数", f"{result['total_trades']}")
        
        # 更多统计
        st.subheader("📋 详细统计")
        
        if result['trades']:
            trades_df = pd.DataFrame(result['trades'])
            
            avg_profit = trades_df['profit_pct'].mean()
            max_profit = trades_df['profit_pct'].max()
            max_loss = trades_df['profit_pct'].min()
            
            # 计算连续失败
            trades_df['is_win'] = trades_df['profit_pct'] > 0
            consecutive_list = []
            current_consecutive = 0
            for is_win in trades_df['is_win']:
                if not is_win:
                    current_consecutive += 1
                else:
                    if current_consecutive > 0:
                        consecutive_list.append(current_consecutive)
                    current_consecutive = 0
            if current_consecutive > 0:
                consecutive_list.append(current_consecutive)
            
            max_consecutive_loss = max(consecutive_list) if consecutive_list else 0
            avg_consecutive_loss = np.mean(consecutive_list) if consecutive_list else 0
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("📈 平均收益率", f"{avg_profit:.2f}%")
            stat_col2.metric("🔥 最大盈利", f"{max_profit:.2f}%")
            stat_col3.metric("💔 最大亏损", f"{max_loss:.2f}%")
            stat_col4.metric("⚠️ 最大连续失败", f"{max_consecutive_loss}次")
            
            # 计算回撤
            equity = np.array(result['equity_curve'])
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max * 100
            max_drawdown = np.min(drawdowns)
            avg_drawdown = np.mean(drawdowns)
            
            # 计算夏普比率
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            dd_col1, dd_col2, dd_col3 = st.columns(3)
            dd_col1.metric("📉 最大回撤", f"{max_drawdown:.2f}%")
            dd_col2.metric("📉 平均回撤", f"{avg_drawdown:.2f}%")
            dd_col3.metric("📐 夏普比率", f"{sharpe:.2f}")
            
            # 显示交易记录
            st.subheader("📝 交易记录")
            st.dataframe(trades_df.head(50))
        
        # ======================== 资金曲线 ========================
        st.header("💰 资金曲线")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['dates'],
            y=result['equity_curve'],
            mode='lines',
            name='资金曲线',
            line=dict(color='#00CC96', width=2)
        ))
        
        fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="初始资金")
        
        fig.update_layout(
            title="账户权益曲线",
            xaxis_title="日期",
            yaxis_title="资金",
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ======================== 买卖记录图 ========================
        if result['trades']:
            st.header("🔔 买卖点标记")
            
            # 选择显示的股票
            stocks = list(set([t['stock'] for t in result['trades']]))
            selected_stock = st.selectbox("选择股票", stocks)
            
            # 获取股票数据
            df = stock_data.get(selected_stock)
            if df is not None:
                # 过滤日期范围
                df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
                
                # 找到买卖点
                buy_dates = [t['entry_date'] for t in result['trades'] if t['stock'] == selected_stock]
                sell_dates = [t['exit_date'] for t in result['trades'] if t['stock'] == selected_stock]
                
                fig2 = go.Figure()
                
                # K线
                fig2.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['OPEN'],
                    high=df['HIGH'],
                    low=df['LOW'],
                    close=df['CLOSE'],
                    name='K线'
                ))
                
                # 买入点
                if buy_dates:
                    buy_prices = [df.loc[d]['CLOSE'] for d in buy_dates if d in df.index]
                    fig2.add_trace(go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='买入',
                        marker=dict(symbol='triangle-up', size=12, color='green')
                    ))
                
                # 卖出点
                if sell_dates:
                    sell_prices = [df.loc[d]['CLOSE'] for d in sell_dates if d in df.index]
                    fig2.add_trace(go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name='卖出',
                        marker=dict(symbol='triangle-down', size=12, color='red')
                    ))
                
                fig2.update_layout(
                    title=f"{selected_stock} 买卖点",
                    template="plotly_white",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("💡 **使用说明**: 在左侧选择因子和止盈止损策略，点击开始回测按钮查看结果")
