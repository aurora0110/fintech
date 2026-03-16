import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import os
import yaml
from datetime import datetime
# 导入技术指标模块
from technical_indicators import calculate_trend


# ---------------------- 全局配置：看板样式+宽屏+中文支持 ----------------------
st.set_page_config(
    page_title='股票筛选策略看板',
    layout='wide',  # 宽屏显示，适配左导航右详情
    initial_sidebar_state='expanded'  # 左侧边栏默认展开
)
# 隐藏Streamlit默认的页眉页脚（优化样式）
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .sidebar .sidebar-content {padding-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

# ---------------------- 读取holding.yaml中的buy_date数据 ----------------------
def get_buy_dates():
    """读取holding.yaml文件，获取每个股票的buy_date数据"""
    holding_file = "/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml"
    try:
        with open(holding_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取hold_stocks中的buy_date数据
        buy_dates_dict = {}
        if 'hold_stocks' in config:
            for stock in config['hold_stocks']:
                stock_code = stock.get('stock_code')
                buy_date = stock.get('buy_date', [])
                # 确保buy_date是列表格式
                if not isinstance(buy_date, list):
                    buy_date = [buy_date]
                buy_dates_dict[stock_code] = buy_date
        return buy_dates_dict
    except Exception as e:
        print(f"读取holding.yaml文件失败: {str(e)}")
        return {}

# 全局变量：存储每个股票的buy_date数据
buy_dates_dict = get_buy_dates()

# ---------------------- 第一步：从JSON文件读取筛选结果（核心修改处） ----------------------

# 默认数据（当JSON文件不存在时使用）
default_b1_list = [['000559', '16.1', '16.65', '请人工判断盈亏比！'], ['000628', '46.8', '48.35', '3.3']]
default_b3_list = [['002572', '15.01', '15.08', '请人工判断盈亏比！'], ['002919', '25.78', '26.15', '请人工判断盈亏比！']]
default_sell_list = [['000799', '达到预设止盈位'], ['600362', '达到白线下两根止损位']]
default_hold_list = [['000799', '达到预设止盈位'], ['600362', '达到白线下两根止损位']]
default_pin_list = []  # 默认单针筛选结果

# 获取今天日期
today_str = datetime.today().strftime("%Y%m%d")

# 读取筛选结果文件
# 尝试读取今天的文件，如果不存在，读取20260206.json文件
result_file = os.path.join("/Users/lidongyang/Desktop/Qstrategy/results", f"{today_str}.json")
if not os.path.exists(result_file):
    # 使用20260206.json文件，因为它包含了大量的股票数据
    result_file = os.path.join("/Users/lidongyang/Desktop/Qstrategy/results", "20260206.json")
    print(f"今天的JSON文件不存在，使用20260206.json文件: {result_file}")
try:
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
        b1_list = results.get('b1_list', default_b1_list)
        b3_list = results.get('b3_list', default_b3_list)
        sell_list = results.get('sell_list', default_sell_list)
        hold_list = results.get('hold_list', default_hold_list)
        pin_list = results.get('pin_list', default_pin_list)
        print(f"已读取最新筛选结果：{result_file}")
except Exception as e:
    print(f"读取筛选结果失败：{str(e)}")
    print("使用默认数据")
    b1_list = default_b1_list
    b3_list = default_b3_list
    sell_list = default_sell_list
    hold_list = default_hold_list
    pin_list = default_pin_list

# 1. 定义列名，和小列表的元素一一对应
b_columns = ['股票代码', '止损价', '收盘价', '盈亏比', '筛选原因']
sell_columns = ['股票代码', '操作']

# 2. 处理hold_list的特殊结构
if hold_list and len(hold_list[0]) == 8:
        # hold_list的实际结构包含8个元素
        hold_columns = ['股票代码', '股票名称', '买入类型', '止损价格', '止盈价格', '持仓天数', '持仓占比', '备注信息']
else:
    # 默认结构（当hold_list结构不符合预期时）
    hold_columns = ['股票代码', '操作']

# 3. 嵌套列表直接转DataFrame
# 处理b1_list，确保包含筛选原因列
processed_b1_list = []
for item in b1_list:
    if len(item) < 5:
        # 如果缺少筛选原因列，添加空值
        item = item + ['']
    processed_b1_list.append(item)
b1_df = pd.DataFrame(processed_b1_list, columns=b_columns)

# 处理b3_list，确保包含筛选原因列
processed_b3_list = []
for item in b3_list:
    if len(item) < 5:
        # 如果缺少筛选原因列，添加空值
        item = item + ['']
    processed_b3_list.append(item)
b3_df = pd.DataFrame(processed_b3_list, columns=b_columns)

sell_df = pd.DataFrame(sell_list, columns=sell_columns)
hold_df = pd.DataFrame(hold_list, columns=hold_columns)

# 处理单针筛选结果
processed_pin_list = []
for item in pin_list:
    if isinstance(item, list):
        row = item[:2]
    else:
        row = [item]
    if len(row) < 2:
        row = row + ['']
    processed_pin_list.append(row)
pin_columns = ['股票代码', '单针类型']
pin_df = pd.DataFrame(processed_pin_list, columns=pin_columns)

# 读取holding.yaml文件，获取watch_stocks数据
import yaml
watch_list = []
try:
    with open('/Users/lidongyang/Desktop/Qstrategy/config/holding.yaml', 'r', encoding='utf-8') as f:
        holding_data = yaml.safe_load(f)
        if 'watch_stocks' in holding_data:
            watch_stocks = holding_data['watch_stocks']
            for stock in watch_stocks:
                watch_list.append([stock['stock_code'], stock['stock_name'], stock['type'], stock['note']])
            print(f"已读取watch_stocks数据，共{len(watch_list)}只股票")
except Exception as e:
    print(f"读取holding.yaml文件失败：{str(e)}")

# 处理watch_stocks数据
watch_columns = ['股票代码', '股票名称', '类型', '备注']
watch_df = pd.DataFrame(watch_list, columns=watch_columns)
# 确保股票代码列是字符串类型
watch_df['股票代码'] = watch_df['股票代码'].astype(str)
print(f"watch_df数据类型：{watch_df.dtypes}")

# 整合所有筛选规则的字典（方便左侧导航调用）
strategy_data = {
    'B1买入条件': b1_df,
    'B3买入条件': b3_df,
    '单针筛选结果': pin_df,
    '持有股票监控': hold_df,
    '卖出股票列表': sell_df,
    '重点关注股票': watch_df
}

# ---------------------- 第二步：定义核心函数（绘制K线+KDJ+成交量，复用） ----------------------
def load_data(file_path):
    """加载单个股票数据文件（逐行解析，兼容列数混乱/分隔符不一致的文件）"""
            # 定义必填字段（适配通达信导出格式）
    required_cols = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']
    # 价格字段（用于逻辑校验）
    price_cols = ['开盘', '最高', '最低', '收盘']
    # 数值字段（禁止负数）
    numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量', '成交额']
    try:
        # 步骤1：读取文件所有行（兼容编码）
        encodings = ['gbk', 'gb2312', 'utf-8', 'latin-1']
        lines = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            return None, "文件编码格式不支持，无法读取"
        
        # 步骤2：逐行过滤+清洗，只保留有效数据行
        data_rows = []
        for line in lines:
            # 跳过ST股票数据
            if any(keyword in line for keyword in ['ST', '*ST', '*']):
                break

            # 跳过标题行（含中文列名的行）
            if any(keyword in line for keyword in ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']):
                continue
            
            # 清洗行数据：把任意分隔符（空格/制表符）换成单个空格，再拆分
            clean_line = ' '.join(line.split())  # 合并多个空格/制表符为一个
            parts = clean_line.split(' ')  # 按单个空格拆分
            
            # 只保留7列的有效数据行（日期+4个价格+成交量+成交额）
            if len(parts) == 7:
                # 校验日期格式（YYYY/MM/DD），过滤无效行
                if '/' in parts[0] and len(parts[0].split('/')) == 3:
                    data_rows.append(parts)
        
        # 检查是否有有效数据
        if not data_rows:
            return None, "文件中未找到有效数据行（7列的日期+价格+成交量数据）"
        
        # 步骤3：转换为DataFrame并处理类型
        df = pd.DataFrame(data_rows, columns=required_cols)
        
        # 数值列转换（容错，失败设为NaN）
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 日期转换（容错）
        df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
        
        # 步骤4：删除无效行（日期或收盘价为空的行）
        df = df.dropna(subset=['日期', '收盘'])
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df, None
    except Exception as e:
        return None, f"文件读取失败：{str(e)}"

def calculate_kdj(df_kline, n=9, m1=3, m2=3):
    """计算KDJ指标（适配DataFrame格式的K线数据）"""
    low_list = df_kline['low'].rolling(window=n, min_periods=1).min()
    high_list = df_kline['high'].rolling(window=n, min_periods=1).max()
    rsv = (df_kline['close'] - low_list) / (high_list - low_list) * 100
    df_kline['K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    df_kline['D'] = df_kline['K'].ewm(alpha=1/m2, adjust=False).mean()
    df_kline['J'] = 3 * df_kline['K'] - 2 * df_kline['D']
    return df_kline

def get_stock_data(stock_code):
    """
    获取个股K线数据（含开盘/最高/最低/收盘/成交量）
    【你需替换】：真实使用时，改为Tushare/AKShare获取实时K线数据，保持返回列名一致（open/high/low/close/volume/date）
    """
    file_paths = list(Path("/Users/lidongyang/Desktop/Qstrategy/data/20260206/normal").glob('*.txt'))
    # 初始化变量
    df = None
    load_error = None
    stock_name = ""
    prefix = "SH"
    
    # 确保stock_code是字符串类型
    stock_code_str = str(stock_code)
    
    if stock_code_str.startswith('92'):
        prefix = "BJ"
    elif stock_code_str.startswith('60') or stock_code_str.startswith('68'):
        prefix = "SH"
    else:
        prefix = "SZ"

    # 修改文件路径，从20260206文件夹而不是normal文件夹中读取文件
    file_path = f"/Users/lidongyang/Desktop/Qstrategy/data/20260206/{prefix}#{stock_code_str}.txt"
    print(file_path)
    
    # 提取股票中文名称
    stock_name = ""
    try:
        # 尝试使用多种编码读取文件
        encodings = ['gbk', 'utf-8', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                    # 读取前20行，查找包含中文的行
                    for line in lines[:20]:
                        # 查找包含中文的行
                        chinese_chars = ''.join([c for c in line if '\u4e00' <= c <= '\u9fff'])
                        if chinese_chars:
                            stock_name = chinese_chars.strip()
                            print(f"提取到股票名称: {stock_name}")
                            break
                    if stock_name:
                        break
            except Exception as e:
                print(f"使用编码 {encoding} 读取文件失败: {str(e)}")
                continue
    except Exception as e:
        print(f"提取股票名称失败: {str(e)}")
    
    # 如果没有提取到股票名称，使用股票代码作为默认名称
    if not stock_name:
        stock_name = stock_code
        print(f"未提取到股票名称，使用股票代码作为默认名称: {stock_name}")
    
    df, load_error = load_data(file_path)
    
    # 处理文件不存在或读取失败的情况
    if df is None:
        print(f"文件读取失败: {load_error}")
        # 生成模拟数据，生成200天的数据以计算完整的MA60和知行多空线
        dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq='B')
        # 生成模拟收盘价（带趋势）
        base_price = 10.0
        trend = np.linspace(0, 4, 200)
        noise = np.random.normal(0, 0.2, 200)
        close = base_price + trend + noise
        # 生成开/高/低（围绕收盘价波动）
        open_ = close * (1 + np.random.normal(0, 0.01, 200))
        high = np.maximum(open_, close) * (1 + np.random.normal(0, 0.01, 200))
        low = np.minimum(open_, close) * (1 - np.random.normal(0, 0.01, 200))
        # 生成成交量（随机波动）
        volume = np.random.randint(100000, 1000000, 200)
    else:
        # 使用真实数据，取最近200天的数据以计算完整的MA60和知行多空线
        dates = df['日期'].tail(200).reset_index(drop=True)
        close = df['收盘'].tail(200).reset_index(drop=True)
        open_ = df['开盘'].tail(200).reset_index(drop=True)
        high = df['最高'].tail(200).reset_index(drop=True)
        # 修正列名错误，应该是'最低'而不是'最近'
        low = df['最低'].tail(200).reset_index(drop=True)
        volume = df['成交量'].tail(200).reset_index(drop=True)
    
    # 构造完整的DataFrame，使用连续的整数索引作为x轴
    df_full = pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        '开盘': open_,  # 为了兼容calculate_trend函数的列名
        '最高': high,
        '最低': low,
        '收盘': close
    })
    # 保留小数点后一位
    df_full['open'] = df_full['open'].round(1)
    df_full['high'] = df_full['high'].round(1)
    df_full['low'] = df_full['low'].round(1)
    df_full['close'] = df_full['close'].round(1)
    df_full['开盘'] = df_full['开盘'].round(1)
    df_full['最高'] = df_full['最高'].round(1)
    df_full['最低'] = df_full['最低'].round(1)
    df_full['收盘'] = df_full['收盘'].round(1)
    # 添加连续的整数索引列
    df_full['index'] = range(len(df_full))
    # 计算KDJ指标并保留一位小数
    df_full = calculate_kdj(df_full)
    df_full['K'] = df_full['K'].round(1)
    df_full['D'] = df_full['D'].round(1)
    df_full['J'] = df_full['J'].round(1)
    # 计算知行多空线和知行短期趋势线
    df_full = calculate_trend(df_full)
    
    # 确保指标列存在
    if 'MA60' not in df_full.columns:
        df_full['MA60'] = df_full['close'].rolling(window=60).mean()
    if '知行短期趋势线' not in df_full.columns:
        df_full['知行短期趋势线'] = df_full['收盘'].ewm(span=10, adjust=False).mean()
        df_full['知行短期趋势线'] = df_full['知行短期趋势线'].ewm(span=10, adjust=False).mean()
    if '知行多空线' not in df_full.columns:
        # 简化计算，使用更短的周期来确保有足够的数据点
        df_full['MA14'] = df_full['收盘'].rolling(window=14).mean()
        df_full['MA28'] = df_full['收盘'].rolling(window=28).mean()
        df_full['MA57'] = df_full['收盘'].rolling(window=57).mean()
        df_full['MA114'] = df_full['收盘'].rolling(window=114).mean()
        df_full['知行多空线'] = (df_full['MA14'] + df_full['MA28'] + df_full['MA57'] + df_full['MA114']) / 4
    
    # 移除四舍五入操作，保留原始精度，使均线和趋势线更加光滑
    # df_full['知行短期趋势线'] = df_full['知行短期趋势线'].round(1)
    # df_full['知行多空线'] = df_full['知行多空线'].round(1)
    # df_full['MA60'] = df_full['MA60'].round(1)
    # 添加涨跌颜色标记
    df_full['涨跌'] = np.where(df_full['close'] > df_full['open'], 1, -1)
    # 处理成交量单位，转换为汉字标注
    def format_volume(vol):
        if vol >= 100000000:
            return f"{vol/100000000:.1f}亿"
        elif vol >= 10000:
            return f"{vol/10000:.1f}万"
        else:
            return f"{vol:.0f}"
    df_full['volume_label'] = df_full['volume'].apply(format_volume)
    
    # 只保留最近三个月（约60天）的数据用于展示
    df_kline = df_full.tail(60).reset_index(drop=True)
    # 重新计算索引，确保x轴是连续的
    df_kline['index'] = range(len(df_kline))
    
    # 确保所有指标列都被保留
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', '开盘', '最高', '最低', '收盘', 'K', 'D', 'J', 'MA60', '知行短期趋势线', '知行多空线', '涨跌', 'volume_label', 'index']
    for col in required_columns:
        if col not in df_kline.columns:
            print(f"警告: 列 {col} 不存在于 df_kline 中")
    
    return df_kline, stock_name

def plot_stock_analysis(stock_code, current_strategy):
    """绘制个股分析图：K线（MA）+ KDJ + 成交量（三图联动）"""
    # 获取个股K线+KDJ数据
    df, stock_name = get_stock_data(stock_code)
    
    # 获取筛选原因
    filter_reason = ""
    try:
        # 优先获取当前选中策略中的筛选原因
        if current_strategy in strategy_data:
            current_strategy_df = strategy_data[current_strategy]
            if '股票代码' in current_strategy_df.columns:
                # 查找当前股票代码
                stock_row = current_strategy_df[current_strategy_df['股票代码'] == stock_code]
                if not stock_row.empty:
                    # 如果是卖出股票列表，使用操作作为筛选原因
                    if current_strategy == '卖出股票列表':
                        if '操作' in stock_row.columns:
                            filter_reason = stock_row['操作'].iloc[0]
                    # 否则使用筛选原因
                    else:
                        if '筛选原因' in stock_row.columns:
                            filter_reason = stock_row['筛选原因'].iloc[0]
        
        # 如果当前策略中没有找到，遍历所有策略数据
        if not filter_reason:
            for strategy_name, strategy_df in strategy_data.items():
                if '股票代码' in strategy_df.columns:
                    # 查找当前股票代码
                    stock_row = strategy_df[strategy_df['股票代码'] == stock_code]
                    if not stock_row.empty:
                        # 如果有筛选原因列，获取筛选原因
                        if '筛选原因' in stock_row.columns:
                            filter_reason = stock_row['筛选原因'].iloc[0]
                            break
        
        # 打印调试信息
        print(f"股票代码: {stock_code}, 筛选原因: {filter_reason}")
    except Exception as e:
        print(f"获取筛选原因失败: {str(e)}")
    
    # 创建子图：3行1列，共享x轴，分别放K线、KDJ、成交量
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,  # 共享x轴，拖动一个图所有图同步
        vertical_spacing=0.05,  # 子图之间的间距
        subplot_titles=(f'{stock_code} {stock_name} K线图', 'KDJ指标', '成交量'),
        row_heights=[0.5, 0.25, 0.25],  # 分配子图高度（K线占比最大）
        specs=[[{'secondary_y': False}], [{'secondary_y': False}], [{'secondary_y': False}]]
    )

    # 1. 绘制K线图（叠加MA60均线、知行短期趋势线、知行多空线）
    fig.add_trace(go.Candlestick(
        x=df['index'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', showlegend=False,
        increasing_line_color='red',  # 阳线用红色
        increasing_fillcolor='red',
        decreasing_line_color='green',  # 阴线用绿色
        decreasing_fillcolor='green'
    ), row=1, col=1)
    # 绘制MA60（使用在get_stock_data中计算好的MA60）
    # 使用与白色背景对比度更强的颜色
    fig.add_trace(go.Scatter(x=df['index'], y=df['MA60'], name='MA60', line=dict(color='#009900', width=1.5), showlegend=True), row=1, col=1)  # 绿色
    # 绘制知行短期趋势线
    fig.add_trace(go.Scatter(x=df['index'], y=df['知行短期趋势线'], name='知行短期趋势线', line=dict(color='#ffffff', width=2), showlegend=True), row=1, col=1)  # 白色实线
    # 绘制知行多空线
    fig.add_trace(go.Scatter(x=df['index'], y=df['知行多空线'], name='知行多空线', line=dict(color='#ff9900', width=2), showlegend=True), row=1, col=1)  # 橙色实线
    
    # 2. 在buy_date对应日期位置标注"b"点
    if stock_code in buy_dates_dict:
        buy_dates = buy_dates_dict[stock_code]
        for buy_date_str in buy_dates:
            try:
                # 将buy_date字符串转换为日期格式
                buy_date = pd.to_datetime(buy_date_str)
                # 在df中查找对应日期的行
                matching_rows = df[df['date'].dt.date == buy_date.date()]
                if not matching_rows.empty:
                    # 获取对应日期的索引、收盘价和最高价
                    idx = matching_rows.iloc[0]['index']
                    close_price = matching_rows.iloc[0]['close']
                    high_price = matching_rows.iloc[0]['high']
                    # 计算"b"点的位置，在最高价上方一定距离
                    b_point_y = high_price * 1.02  # 在最高价上方2%
                    # 在K线图上添加"b"点标注
                    # 去掉箭头，调大字号
                    fig.add_annotation(
                        x=idx,
                        y=b_point_y,
                        text="b",
                        showarrow=False,
                        arrowhead=1,
                        arrowcolor="yellow",
                        bgcolor="rgba(0,0,0,0.7)",
                        font=dict(color="yellow", size=18, weight="bold"),
                        row=1, col=1
                    )
            except Exception as e:
                print(f"处理buy_date {buy_date_str} 时出错: {str(e)}")
                continue

    # 2. 绘制KDJ指标（K/D/J三条线，添加0/100超买超卖线）
    fig.add_trace(go.Scatter(x=df['index'], y=df['K'], name='KDJ-K', line=dict(color='#2ca02c', width=1.2), showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['index'], y=df['D'], name='KDJ-D', line=dict(color='#d62728', width=1.2), showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['index'], y=df['J'], name='KDJ-J', line=dict(color='#ffffff', width=1.2), showlegend=True), row=2, col=1)
    # 添加KDJ超买超卖线（0和100）
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=2, col=1, annotation_text='超卖')
    fig.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1, row=2, col=1, annotation_text='超买')

    # 3. 绘制成交量柱形图（红色表示上涨，绿色表示下跌）
    # 创建颜色列表
    volume_colors = ['red' if change == 1 else 'green' for change in df['涨跌']]
    fig.add_trace(go.Bar(
        x=df['index'], 
        y=df['volume'], 
        name='成交量', 
        showlegend=False, 
        marker_color=volume_colors,
        # 添加自定义悬停文本，显示汉字单位的成交量
        hovertext=df['volume_label'],
        hovertemplate='成交量: %{hovertext}<extra></extra>'
    ), row=3, col=1)

    # 全局图表配置（优化样式，隐藏多余元素）
    fig.update_layout(
        height=800,  # 图表总高度
        plot_bgcolor='black',  # 背景色黑色
        paper_bgcolor='black',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.05, 
            xanchor='center', 
            x=0.5,  # 图例放顶部中央
            font=dict(
                color='#ffffff',  # 图例文字颜色设置为白色，与黑色背景对比度最强
                size=12  # 增大图例文字大小，提高可读性
            ),
            bgcolor='#333333',  # 为图例添加深灰色背景，进一步提高对比度
            bordercolor='#666666',  # 添加边框
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,  # 隐藏K线底部的滑动条
        # 设置标题颜色
        font=dict(
            color='#ffffff'  # 所有文字颜色设置为白色，与黑色背景对比度最强
        )
    )
    # 配置x轴（使用日期标签，隔一周展示一个日期）和y轴（价格/数值）样式
    # 计算隔一周的索引值
    weekly_indices = df['index'][::7]  # 每隔7天取一个索引
    weekly_dates = df['date'][::7].dt.strftime('%Y-%m-%d')  # 对应的日期
    
    # 设置x轴标签为日期，隔一周展示一个
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#333333', 
        tickvals=weekly_indices, 
        ticktext=weekly_dates,
        tickfont=dict(color='#ffffff'),
        row=3, col=1
    )
    # 为其他子图设置相同的x轴配置
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#333333', 
        tickvals=weekly_indices, 
        ticktext=weekly_dates,
        tickfont=dict(color='#ffffff'),
        row=1, col=1
    )
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#333333', 
        tickvals=weekly_indices, 
        ticktext=weekly_dates,
        tickfont=dict(color='#ffffff'),
        row=2, col=1
    )
    # 配置y轴
    fig.update_yaxes(showgrid=True, gridcolor='#333333', tickfont=dict(color='#ffffff'), row=1, col=1)
    # 配置KDJ轴，添加最新一天的J值标注
    latest_j_value = df['J'].iloc[-1]
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#333333', 
        tickfont=dict(color='#ffffff'), 
        row=2, col=1, 
        range=[-20, 120],  # KDJ轴范围限制（-20~120）
        title=dict(
            text=f'KDJ (J={latest_j_value:.1f})',  # 在y轴标题中添加最新的J值
            font=dict(color='#ffffff')
        )
    )
    # 配置成交量y轴，添加单位标注
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#333333', 
        tickfont=dict(color='#ffffff'),
        row=3, col=1,
        title=dict(
            text='成交量（万）',  # 在y轴标题中添加单位
            font=dict(color='#ffffff')
        )
    )

    # 添加筛选原因标注
    if filter_reason:
        print(f"添加筛选原因标注: {filter_reason}")
        fig.add_annotation(
            x=0.5,  # 位置：正中
            y=0.98,  # 位置：中间
            xref='paper',  # 使用纸张坐标系统
            yref='paper',  # 使用纸张坐标系统
            text=f'筛选原因: {filter_reason}',
            showarrow=False,
            bgcolor='rgba(0, 0, 0, 0.9)',  # 更深的背景色
            font=dict(color='yellow', size=16, weight='bold'),  # 更大的字体
            bordercolor='yellow',
            borderwidth=2,  # 更粗的边框
            # 移除row和col参数，使用整个图表的坐标系
        )
    else:
        print(f"未添加筛选原因标注，因为筛选原因为空")

    return fig

# ---------------------- 第三步：看板核心布局（左侧侧边栏+右侧主区域） ----------------------
# 左侧侧边栏：筛选规则导航（B1/持有/B3/卖出）
st.sidebar.title('📋 筛选规则')
# 选择筛选规则（可点击的选择框，核心导航）
selected_strategy = st.sidebar.selectbox(
    '选择要查看的规则',
    options=['B1买入条件', 'B3买入条件', '单针筛选结果', '持有股票监控', '卖出股票列表', '重点关注股票'],
    index=0,
    key='strategy_selector'
)

# 右侧主区域：根据左侧选择，联动展示数据
st.title(f'📈 {selected_strategy} 详情')
st.divider()

# 第一步：加载对应筛选规则的股票数据
df_selected = strategy_data[selected_strategy]
# 展示股票列表（自适应宽度，带搜索/排序）
st.subheader('🔍 筛选出的股票列表')
st.dataframe(df_selected, width='stretch', hide_index=True)

# 第二步：展示所有筛选结果的技术分析图，每行两张图，添加分页功能
if not df_selected.empty:
    st.subheader('📊 个股技术分析')
    # 获取所有股票代码
    stock_codes = df_selected['股票代码'].tolist()
    
    # 打印调试信息
    print(f"当前策略: {selected_strategy}")
    print(f"股票数量: {len(stock_codes)}")
    print(f"前5个股票代码: {stock_codes[:5]}")
    
    # 实现分页功能
    stocks_per_page = 50  # 每页显示50个股票（25行，每行2个）
    total_pages = (len(stock_codes) + stocks_per_page - 1) // stocks_per_page
    
    # 添加分页选择器
    page_number = st.selectbox(
        '选择页码',
        options=range(1, total_pages + 1),
        key='page_selector'
    )
    
    # 计算当前页的股票范围
    start_idx = (page_number - 1) * stocks_per_page
    end_idx = min(start_idx + stocks_per_page, len(stock_codes))
    current_page_stocks = stock_codes[start_idx:end_idx]
    
    # 显示当前页的股票
    st.write(f"显示第 {page_number} 页，共 {total_pages} 页（{start_idx + 1}-{end_idx} / {len(stock_codes)}）")
    
    # 每两行创建一个row
    for i in range(0, len(current_page_stocks), 2):
        # 创建两列布局
        cols = st.columns(2)
        # 第一列
        with cols[0]:
            stock_code = current_page_stocks[i]
            try:
                # 绘制并展示技术分析图
                fig = plot_stock_analysis(stock_code, selected_strategy)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{stock_code}_{i}")
            except Exception as e:
                print(f"绘制股票 {stock_code} 时出错: {str(e)}")
                st.error(f"绘制股票 {stock_code} 时出错: {str(e)}")
        # 第二列（如果还有股票）
        if i + 1 < len(current_page_stocks):
            with cols[1]:
                stock_code = current_page_stocks[i + 1]
                try:
                    # 绘制并展示技术分析图
                    fig = plot_stock_analysis(stock_code, selected_strategy)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{stock_code}_{i+1}")
                except Exception as e:
                    print(f"绘制股票 {stock_code} 时出错: {str(e)}")
                    st.error(f"绘制股票 {stock_code} 时出错: {str(e)}")
else:
    st.warning(f'⚠️ {selected_strategy} 暂无筛选出的股票')

# 底部备注
st.divider()
st.caption(f'本看板数据从 main.py 筛选结果读取 | 技术分析图包含60日K线（MA60）、KDJ指标、知行趋势线、成交量 | 图表支持滚轮缩放/拖动查看')
