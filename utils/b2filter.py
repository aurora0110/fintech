from pathlib import Path
from utils import stockDataValidator
from utils import stoploss
from utils import technical_indicators
import numpy as np

def volatility(price1, price2, percent1, file_name, type):
    change_pct = (price1 - price2) / price2 * 100
    change_pct_rounded = round(change_pct, 2)  # 保留2位小数
    
    # 判断涨幅是否在目标区间内（包含边界值）
    is_in_range = (change_pct >= percent1)

    if is_in_range:
        #print(f'{file_name}涨跌幅在目标区间内，涨跌幅为{change_pct_rounded}，达成{type}')
        return True

def check(file_path, hold_list):
    # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
    file_name_full = file_path.split('/')[-1]
    # 步骤2：去掉.txt后缀 → SZ#300319
    file_name_no_suffix = file_name_full.replace('.txt', '')
    # 步骤3：取#后的股票代码 → 300319
    file_name = file_name_no_suffix.split('#')[-1]

    # 计算技术指标
    df, load_error = stoploss.load_data(file_path)
    df_trend = technical_indicators.calculate_trend(df)
    df_kdj = technical_indicators.calculate_kdj(df)

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    b2_label = -1
    b2_label_lastday = -1

    j_label = False
    volatility_label = False
    volume_label = False
    shadow_label = False
    j_label_lastday = False
    volatility_label_lastday = False
    volume_label_lastday = False
    shadow_label_lastday = False

    stop_loss_price = 0
    take_profit_price = 0 
    ratio = 0


    if df_trend['知行多空线'].iloc[-1] > df_trend['知行短期趋势线'].iloc[-1]:
        return [[b2_label], [b2_label_lastday]]
    else:
        # 计算当日是否是B2，按照J<30筛选，重要的不是J值的大小，而是下杀的动作
        if df_kdj['J'].iloc[-2] <= 50 and df_kdj['J'].iloc[-1] <= 80:
            j_label = True
        # 计算涨跌幅
        if volatility(float(today['收盘']), float(yesterday['收盘']), 4, file_name, 'B2目标涨幅'):
            volatility_label = True
        # 计算成交量
        if float(today['成交量']) > float(yesterday['成交量']):
            volume_label = True
        # 计算上影线比例
        # 计算价格实体长度（绝对值，保证为正）
        length = float(abs(df['最高'].iloc[-1] - df['开盘'].iloc[-1]))
        # 计算上影线长度（最高价 - 实体顶端）
        shadow_length = float(df['最高'].iloc[-1] - df['收盘'].iloc[-1].max())
        # 计算上影线占实体长度的比例（处理实体长度为0的情况）where：实体长度≠0时计算占比，否则设为NaN
        shadow_ratio = shadow_length / length if length != 0 else 0
        if shadow_ratio < 0.3:
            shadow_label = True

        # 计算前一日是否是B2
        if df_kdj['J'].iloc[-3] <= 50 and df_kdj['J'].iloc[-2] <= 80:
            j_label_lastday = True
        # 计算涨跌幅
        if volatility(float(yesterday['收盘']), float(df['收盘'].iloc[-3]), 4, file_name, 'B2目标涨幅'):
            volatility_label_lastday = True
        # 计算成交量
        if float(yesterday['成交量']) > float(df['成交量'].iloc[-3]):
            volume_label_lastday = True
        # 计算价格实体长度（绝对值，保证为正）
        length = float(abs(df['最高'].iloc[-2] - df['开盘'].iloc[-2]))
        # 计算上影线长度（最高价 - 实体顶端）
        shadow_length = float(df['最高'].iloc[-2] - df['收盘'].iloc[-2].max())
        # 计算上影线占实体长度的比例（处理实体长度为0的情况）where：实体长度≠0时计算占比，否则设为NaN
        shadow_ratio = shadow_length / length if length != 0 else 0
        if shadow_ratio < 0.3:
            shadow_label_lastday = True

        if j_label and volatility_label and volume_label and shadow_label:
            b2_label = 1
            #print(f"{file_name}B2筛选通过")
            #若是，给出买入信号、止损价（或止损类型）、盈亏比、买入价格
            stop_loss_price = ((today['收盘'] - today['开盘'])/2).round(2)
            take_profit_price =  df['最高'].rolling(window=60, min_periods=1).max().round(2)
            ratio = (take_profit_price - today['收盘']) / (today['收盘'] - stop_loss_price)
        if j_label_lastday and volatility_label_lastday and volume_label_lastday and shadow_label_lastday:
            b2_label_lastday = 1

        return [[b2_label, str(stop_loss_price), today['收盘']], b2_label_lastday]
        



            