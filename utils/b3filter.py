from pathlib import Path
from utils import stockDataValidator
from utils import stoploss
from utils import technical_indicators
from scipy.signal import argrelextrema
import numpy as np

def get_last_n_high(df):
    """
    核心功能：获取股票最近一个正N型结构的高点（C点）价格
    正N型结构定义：上涨(A高点)→下跌(B低点)→上涨(C高点)，满足时间/幅度约束
    参数：df - 股票日线数据DataFrame，必须包含['日期','最高','最低']列
    返回值：float - 最近N型结构C点的最高价；None - 无符合条件的N型结构
    """
    # 1. 数据时间排序：按日期列升序排列数据
    # 关键作用：保证后续识别的高低点是按时间先后顺序，避免因数据乱序导致N型结构判断错误
    df = df.sort_values('日期')
    
    # 2. 识别所有局部高点的索引
    # argrelextrema：scipy的局部极值识别函数，返回极值点在数组中的索引
    # df['最高'].values：将最高价列转为numpy数组（函数要求数组输入）
    # np.greater：指定找局部最大值（即高点，当前值>前后order个值）
    # order=3：极值窗口为3，即当前点是前后3天内的最高价（控制高低点灵敏度，值越大越筛选关键高点）
    # [0]：argrelextrema返回的是元组，取第一个元素（索引数组）
    h_idx = argrelextrema(df['最高'].values, np.greater, order=3)[0]
    
    # 3. 识别所有局部低点的索引
    # np.less：指定找局部最小值（即低点，当前值<前后order个值）
    # 其余参数逻辑同高点识别，仅把"最高"换为"最低"，把"最大值"换为"最小值"
    l_idx = argrelextrema(df['最低'].values, np.less, order=3)[0]
    
    # 4. 整理高点数据为列表：[(位置索引, 最高价)]
    # 遍历所有高点索引h_idx，取出每个索引i对应的行的"最高"列值
    # 结果格式示例：[(10, 18.5), (25, 20.2)]，10是行索引（对应日期），18.5是该日最高价
    highs = [(i, df.loc[i, '最高']) for i in h_idx]
    
    # 5. 整理低点数据为列表：[(位置索引, 最低价)]
    # 逻辑同高点整理，仅把"最高"换为"最低"
    # 结果格式示例：[(15, 16.3), (30, 18.1)]，15是行索引，16.3是该日最低价
    lows = [(i, df.loc[i, '最低']) for i in l_idx]
    
    # 6. 倒序遍历所有高点（从最新的高点开始）
    # reversed(highs)：将highs列表反转，优先检查最近的高点（A点），找到符合条件的N型就立即返回，提升效率
    # a_pos：A点在df中的行索引（用于计算时间周期：行索引差=交易日数）；a_p：A点的最高价
    for a_pos,a_p in reversed(highs):
        # 7. 筛选A点之后符合条件的B点（N型的回调低点），用列表推导式过滤低点列表
        # 筛选条件拆解：
        # 1) l_pos > a_pos：B点在A点之后（时间上A→B是回调）
        # 2) 3<=l_pos-a_pos<=15：A到B的时间周期在3~15个交易日（回调周期约束）
        # 3) 0.03<=(a_p-l_p)/a_p<=0.2：A到B的回调幅度在3%~20%（幅度约束，避免回调太小/太大）
        # bs最终格式：[(15, 16.3)]，即符合条件的B点（索引+价格）
        bs = [(l_pos,l_p) for l_pos,l_p in lows if l_pos>a_pos and 3<=l_pos-a_pos<=15 and 0.03<=(a_p-l_p)/a_p<=0.2]
        
        # 8. 遍历所有符合条件的B点
        # b_pos：B点的行索引；b_p：B点的最低价
        for b_pos,b_p in bs:
            # 9. 筛选B点之后符合条件的C点（N型的上涨高点），列表推导式过滤高点列表
            # 筛选条件拆解：
            # 1) h_pos > b_pos：C点在B点之后（时间上B→C是上涨）
            # 2) 3<=h_pos-b_pos<=15：B到C的时间周期在3~15个交易日（上涨周期约束）
            # 3) h_p > b_p：C点价格>B点价格（确保是上涨）
            # 4) abs(h_p-a_p)/a_p<=0.05：C点与A点的价格偏差≤5%（C接近A，符合N型形态）
            # cs最终格式：[(20, 18.3)]，即符合条件的C点（索引+价格）
            cs = [(h_pos,h_p) for h_pos,h_p in highs if h_pos>b_pos and 3<=h_pos-b_pos<=15 and h_p>b_p and abs(h_p-a_p)/a_p<=0.05]
            
            # 10. 若找到符合条件的C点，返回最近的C点价格
            if cs:
                # max(cs, key=lambda x:x[0])：在多个C点中，按行索引x[0]取最大的（即时间最新的C点）
                # [1]：取该C点元组的第二个元素（价格），作为函数返回值
                return max(cs,key=lambda x:x[0])[1]
    
    # 11. 若遍历完所有高低点都未找到符合条件的N型结构，返回None
    return None

def check(file_path, hold_list, b2_result):
    if b2_result == 1:
        # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
        file_name_full = file_path.split('/')[-1]
        # 步骤2：去掉.txt后缀 → SZ#300319
        file_name_no_suffix = file_name_full.replace('.txt', '')
        # 步骤3：取#后的股票代码 → 300319
        file_name = file_name_no_suffix.split('#')[-1]

        # 计算技术指标
        df, load_error = stoploss.load_data(file_path)

        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        lastyesterday = df.iloc[-3]

        today_close = today['收盘']
        yesterday_close = yesterday['收盘']
        yesterday_volume = yesterday['成交量']
        lastyesterday_volume = lastyesterday['成交量']

        if (today_close > yesterday_close) and (yesterday_volume > (lastyesterday_volume * 2)) and (today['成交量'] < yesterday['成交量']):
            #print(f'{file_name}B3筛选通过')
            #若是，给出买入信号、止损价（或止损类型）、盈亏比、买入价格
            stop_loss_price = (today['最低']).round(2)
            take_profit_price =  df['最高'].rolling(window=60, min_periods=1).max().round(2)
            # 寻找N型结构的最高价
            near_high_price = get_last_n_high(df)
            if near_high_price > stop_loss_price:
                ratio = ((near_high_price - today_close) / (today_close - stop_loss_price)).round(1)
            else:
                ratio = "请人工判断盈亏比！"
            return [1, stop_loss_price, today_close, ratio]
        else:
            return [-1]
    else:
        return [-1]

