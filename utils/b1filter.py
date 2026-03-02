from pathlib import Path
from utils import stockDataValidator
from utils import stoploss
from utils import technical_indicators
from scipy.signal import argrelextrema
import numpy as np

def volatility(price1, price2, percent1, percent2, file_name, type):
    change_pct = (price1 - price2) / price2 * 100
    change_pct_rounded = round(change_pct, 2)  # 保留2位小数
    
    # 判断是否在目标区间内（包含边界值）
    is_in_range = (change_pct >= -percent1) and (change_pct <= percent2)

    if is_in_range:
        #print(f'{file_name}涨跌幅在目标区间内，涨跌幅为{change_pct_rounded}，达成{type}')
        return True

# 寻找N型结构的最高价
def get_last_n_high(df, order=5, ab_period=(3, 15), ab_retracement=(0.03, 0.2), bc_period=(3, 15), ac_deviation=0.05):
    """
    核心功能：获取股票最近一个正N型结构的高点（C点）价格
    正N型结构定义：上涨(A高点)→下跌(B低点)→上涨(C高点)，满足时间/幅度约束
    
    参数：
        df - 股票日线数据DataFrame，必须包含['日期','最高','最低']列
        order - 局部极值窗口大小，默认5（当前点是前后5天内的极值）
        ab_period - A到B的时间周期范围，默认(3, 15)（3-15个交易日）
        ab_retracement - A到B的回调幅度范围，默认(0.03, 0.2)（3%-20%）
        bc_period - B到C的时间周期范围，默认(3, 15)（3-15个交易日）
        ac_deviation - C与A的价格偏差，默认0.05（≤5%）
        
    返回值：
        float - 最近N型结构C点的最高价
        None - 无符合条件的N型结构或输入数据无效
    """
    # 检查必要的列
    required_cols = ['日期', '最高', '最低']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：DataFrame缺少必要的列 '{col}'")
            return None
    
    # 检查数据量
    if len(df) < 2 * order + 1:
        print("警告：数据量不足，无法识别局部极值")
        return None
    
    # 1. 数据时间排序：按日期列升序排列数据
    df = df.sort_values('日期')
    
    try:
        # 2. 识别所有局部高点的索引
        h_idx = argrelextrema(df['最高'].values, np.greater, order=order)[0]
        
        # 3. 识别所有局部低点的索引
        l_idx = argrelextrema(df['最低'].values, np.less, order=order)[0]
        
        # 4. 整理高点数据为列表：[(位置索引, 最高价)]
        highs = [(i, df.loc[i, '最高']) for i in h_idx]
        
        # 5. 整理低点数据为列表：[(位置索引, 最低价)]
        lows = [(i, df.loc[i, '最低']) for i in l_idx]
        
        # 检查是否有高低点数据
        if not highs:
            print("警告：未找到局部高点")
            return None
        
        if not lows:
            print("警告：未找到局部低点")
            return None
        
        # 6. 倒序遍历所有高点（从最新的高点开始）
        for a_pos, a_p in reversed(highs):
            # 7. 筛选A点之后符合条件的B点
            bs = [(l_pos, l_p) for l_pos, l_p in lows if l_pos > a_pos and 
                  ab_period[0] <= l_pos - a_pos <= ab_period[1] and 
                  ab_retracement[0] <= (a_p - l_p) / a_p <= ab_retracement[1]]
            
            # 8. 遍历所有符合条件的B点
            for b_pos, b_p in bs:
                # 9. 筛选B点之后符合条件的C点
                cs = [(h_pos, h_p) for h_pos, h_p in highs if h_pos > b_pos and 
                      bc_period[0] <= h_pos - b_pos <= bc_period[1] and 
                      h_p > b_p and abs(h_p - a_p) / a_p <= ac_deviation]
                
                # 10. 若找到符合条件的C点，返回最近的C点价格
                if cs:
                    return max(cs, key=lambda x: x[0])[1]
        
        # 11. 若遍历完所有高低点都未找到符合条件的N型结构，返回None
        return None
        
    except Exception as e:
        print(f"处理过程中出错：{str(e)}")
        return None

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
    df_rsi = technical_indicators.calculate_rsi(df)
    df_ma = technical_indicators.calculate_daily_ma(df)

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    volume_label = False
    declining_volume_label = False
    consistent_label = False
    low_volume_price_label = False
    pullback_ma_label = False
    pullback_trend_label = False
    SB1_label = False
    shrink_volume_label = False
    df_dk = df_trend['知行多空线'].iloc[-1]
    df_qs = df_trend['知行短期趋势线'].iloc[-1]

    # 标记阳线/阴线：阳线=收盘>开盘，阴线=收盘<开盘（平盘忽略）
    df['是否阳线'] = df['收盘'] > df['开盘']
    df['是否阴线'] = df['收盘'] < df['开盘']

    if df_dk > df_qs:
        #print(f"{file_name}当前J值为：{df_kdj['J'].iloc[-1].round(2)}, 多空线{df_dk}，趋势线{df_qs}")
        return [-1]
    else:
        if float(df_kdj['J'].iloc[-1]) < 13:
            # 1、判断是否缩量
            if float(df['成交量'].iloc[-1]) < float(df['成交量'].iloc[-2]):
                #print(f'{file_name}且缩量')
                declining_volume_label = True

            # 2、判断涨幅是否达成一致
            today_close = today['收盘']    # 今日收盘价（最新）
            today_open = today['开盘']
            yesterday_close = yesterday['收盘']# 今日开盘价
            consistent_label = volatility(today_close, yesterday_close, 3, 2.5, file_name, '一致')
            
            # 3、判断是否地量低价
            result = {}
            # 遍历5/10/20/30周期，判断是否大于对应周期最大值
            for period in [5, 10, 20, 30]:
                # 截取最近period行（排除最后一行），不足则取所有历史行
                recent_volume_data = df["成交量"].iloc[-(period+1):-1] if len(df)>=period+1 else df["成交量"].iloc[:-1]
                recent_close_data = df["收盘"].iloc[-(period+1):-1] if len(df)>=period+1 else df["收盘"].iloc[:-1]

                result[f'{file_name}今日是否最近{period}天的地量低价'] = (df["成交量"].iloc[-1] < recent_volume_data.min() if not recent_volume_data.empty else False) and (df["收盘"].iloc[-1] < recent_close_data.min() if not recent_close_data.empty else False)

            for k, v in result.items():
                if v:
                    #print(f"{k}：{'是' if v else '否'}")
                    low_volume_price_label = True
            
            # 4、判断是否回踩均线、黄白线
            today_low = df['最低'].iloc[-1]
            today_ma20 = df_ma['MA20'].iloc[-1]
            today_ma60 = df_ma['MA60'].iloc[-1]
            today_yellow = df_trend['知行多空线'].iloc[-1]
            today_white = df_trend['知行短期趋势线'].iloc[-1]

            volatility_ma20_label = volatility(today_low, today_ma20, 2, 2, file_name, '20日线回踩')
            volatility_ma60_label = volatility(today_low, today_ma60, 2, 2, file_name, '60日线回踩')
            volatility_yellow_label = volatility(today_low, today_yellow, 2, 2, file_name, '黄线回踩')
            volatility_white_label = volatility(today_low, today_white, 2, 2, file_name, '白线回踩')
            
            if volatility_ma20_label or volatility_ma60_label:
                pullback_ma_label = True
            if volatility_yellow_label or volatility_white_label:
                pullback_trend_label = True

            # 5、判断是否SB1
            SB1_label = (float(today_close) < float(today_open) < float(yesterday['收盘'])) and (df_kdj['J'].iloc[-1] < 0 and df_kdj['J'].iloc[-2] < 0)
            if SB1_label:
                #print(f"发现SB1: {file_name} :{today_close} {today_open} {yesterday['收盘']} {df_kdj['J'].iloc[-1]}")
                SB1_label = True   

            #  找出近2个月最高的n条成交量记录
            # 按成交量降序排列，取前n行
            df_60 = df.tail(60)
            top5_volume_df = df_60.sort_values('成交量', ascending=False).head(1)
            # 添加"是否阳线"列（阳线：收盘 > 开盘）
            top5_volume_df['是否阳线'] = top5_volume_df['收盘'] > top5_volume_df['开盘']
            # 判断前5大成交量是否全部为阳线
            all_are_bullish = top5_volume_df['是否阳线'].all()  

            # 6、判断近30天阴线、阳线成交量是否符合黄金分割比例
            df_60['K线类型'] = np.where(df_60['收盘'] > df_60['开盘'], 1,  # 阳线
                                    np.where(df_60['收盘'] < df_60['开盘'], -1, 0))  # 阴线/平盘

            # 步骤2：分别生成阳线、阴线的成交量列（非对应K线则成交量为0）
            df_60['阳线成交量'] = np.where(df_60['K线类型'] == 1, df_60['成交量'], 0)
            df_60['阴线成交量'] = np.where(df_60['K线类型'] == -1, df_60['成交量'], 0)

            # 步骤3：滚动计算最近30条数据的阳/阴线累计成交量（min_periods=1：不足30条也计算，可改为30强制满30条）
            df_60['30日阳线累计成交量'] = df_60['阳线成交量'].rolling(window=30, min_periods=1).sum()
            df_60['30日阴线累计成交量'] = df_60['阴线成交量'].rolling(window=30, min_periods=1).sum()

            # 步骤4：核心判断：阳线累计成交量 > 阴线累计成交量×1.3（处理阴线累计为0的情况，避免除0/误判）
            # 阴线累计为0时，若有阳线则直接判定为True（无阴线，阳线自然满足倍数）
            df_60['量能满足条件'] = np.where(
                df_60['30日阴线累计成交量'] == 0,
                df_60['30日阳线累计成交量'] > 0,  # 无阴线时，有阳线则True
                df_60['30日阳线累计成交量'] > df_60['30日阴线累计成交量'] * 1.382  # 有阴线时，判断1.3倍
            ) 


            # 7、判断知行短期趋势线第一次上穿知行多空线后，当前价格是否第一次回踩
            first_pullback_after_cross_label = False
            try:
                # 获取完整的趋势线数据
                qs_line = df_trend['知行短期趋势线']
                dk_line = df_trend['知行多空线']
                
                # 找到最近一次知行短期趋势线上穿知行多空线的日期
                cross_date = None
                for i in range(1, len(df_trend)):
                    # 前一天短期趋势线低于多空线，当天短期趋势线高于多空线
                    if qs_line.iloc[i-1] < dk_line.iloc[i-1] and qs_line.iloc[i] > dk_line.iloc[i]:
                        cross_date = i  # 记录上穿的索引
                
                if cross_date is not None:
                    # 检查上穿后是否第一次回踩
                    # 遍历上穿后到当前的所有数据点
                    has_pullback_before = False
                    for j in range(cross_date + 1, len(df)):
                        # 检查该点是否回踩了多空线或短期趋势线
                        low_j = df['最低'].iloc[j]
                        dk_j = dk_line.iloc[j]
                        qs_j = qs_line.iloc[j]
                        
                        # 使用volatility函数判断是否回踩（2%范围内）
                        pullback_dk = volatility(low_j, dk_j, 2, 2, file_name, '回踩多空线')
                        pullback_qs = volatility(low_j, qs_j, 2, 2, file_name, '回踩短期趋势线')
                        
                        if pullback_dk or pullback_qs:
                            # 如果当前是最后一个点（今天），且之前没有回踩过，则标记为第一次回踩
                            if j == len(df) - 1 and not has_pullback_before:
                                first_pullback_after_cross_label = True
                            else:
                                # 之前已经有回踩过，不是第一次
                                has_pullback_before = True
            except Exception as e:
                print(f"计算第一次回踩标签时出错：{str(e)}")
                first_pullback_after_cross_label = False

            # 8、判断最近一个月内是否存在跳空K线，且后续K线收盘价从未跌破这根跳空K线的收盘价，且跳空K线之后有放量大阳线
            gap_up_label = False
            try:
                # 取最近一个月的数据（约20个交易日）
                recent_df = df.tail(20)
                # 获取对应的趋势线数据
                recent_trend_df = df_trend.tail(20)
                
                # 遍历最近一个月的数据，寻找跳空K线
                for i in range(1, len(recent_df)):
                    # 跳空条件：当天收盘价 > 前一天最高价
                    if recent_df['收盘'].iloc[i] > recent_df['最高'].iloc[i-1]:
                        # 跳空K线的位置是知行多空线大于知行短期趋势线
                        if recent_trend_df['知行多空线'].iloc[i] > recent_trend_df['知行短期趋势线'].iloc[i]:
                            # 记录跳空K线的收盘价
                            gap_close = recent_df['收盘'].iloc[i]
                            
                            # 检查后续K线是否从未跌破这根跳空K线的收盘价
                            subsequent_close = recent_df['收盘'].iloc[i:]
                            if all(subsequent_close >= gap_close):
                                # 检查跳空K线之后是否有放量大阳线
                                # 遍历从跳空K线之后到最近的数据点
                                has_big_bullish = False
                                for j in range(i+1, len(recent_df)):
                                    # 计算当日涨幅（昨日收盘价和今日收盘价的涨幅）
                                    prev_close = recent_df['收盘'].iloc[j-1]
                                    current_close = recent_df['收盘'].iloc[j]
                                    change_pct = (current_close - prev_close) / prev_close * 100
                                    
                                    # 计算成交量是否是前一天的3倍以上
                                    current_volume = recent_df['成交量'].iloc[j]
                                    prev_volume = recent_df['成交量'].iloc[j-1]
                                    volume_ratio = current_volume / prev_volume
                                    
                                    # 大阳线条件：涨幅大于等于8%
                                    # 放量条件：成交量是前一天的3倍以上
                                    if change_pct >= 8 and volume_ratio >= 3:
                                        has_big_bullish = True
                                        #print(f"发现放量大阳线: {file_name}，日期: {recent_df['日期'].iloc[j]}，涨幅: {change_pct:.2f}%，成交量比: {volume_ratio:.2f}")
                                        break
                                
                                # 如果跳空K线之后有放量大阳线，则标记为符合条件
                                if has_big_bullish:
                                    gap_up_label = True
                                    #print(f"发现符合条件的跳空K线: {file_name}，日期: {recent_df['日期'].iloc[i]}，收盘价: {gap_close}")
                                    break
            except Exception as e:
                print(f"检查最新一日趋势线关系时出错：{str(e)}")
                gap_up_label = False
            
            # 10、增加长阴短柱筛选条件
            long_negative_label = False
            try:
                # 获取当日和前一日的数据
                current_day = df.iloc[-1]
                prev_day = df.iloc[-2]
                
                # 计算前一日的收盘价和开盘价的差幅
                prev_open = prev_day['开盘']
                prev_close = prev_day['收盘']
                prev_diff_pct = abs((prev_close - prev_open) / prev_open) * 100
                
                # 计算当日的收盘价和开盘价的差幅
                current_open = current_day['开盘']
                current_close = current_day['收盘']
                current_diff_pct = abs((current_close - current_open) / current_open) * 100
                
                # 计算当日和前一日的成交量
                current_volume = current_day['成交量']
                prev_volume = prev_day['成交量']
                
                # 计算30日内的最高成交量
                recent_30_volume = df['成交量'].tail(30)
                max_30_volume = recent_30_volume.max()
                
                # 长阴短柱条件：
                # 1. 前一日的收盘价和开盘价的差幅小于2%
                # 2. 当日的收盘价和开盘价的差幅大于4%
                # 3. 当日的成交量小于前一日的成交量
                # 4. 当日的成交量小于30日内最高成交量的一半
                if (prev_diff_pct < 2 and 
                    current_diff_pct > 4 and 
                    current_volume < prev_volume and 
                    current_volume < max_30_volume / 2):
                    long_negative_label = True
                    #print(f"发现长阴短柱: {file_name}，当日差幅: {current_diff_pct:.2f}%，前一日差幅: {prev_diff_pct:.2f}%")
            except Exception as e:
                print(f"计算长阴短柱标签时出错：{str(e)}")
                long_negative_label = False

            if (SB1_label and all_are_bullish and df_60['量能满足条件'].iloc[-1]) or ((pullback_ma_label or pullback_trend_label or first_pullback_after_cross_label or low_volume_price_label or first_pullback_after_cross_label or gap_up_label or long_negative_label) and consistent_label and declining_volume_label and all_are_bullish and df_60['量能满足条件'].iloc[-1]): #   
                # 确定筛选原因
                filter_reason = ""
                if SB1_label and consistent_label:
                    filter_reason = "SB1条件"
                elif pullback_ma_label:
                    filter_reason = "回踩均线"
                elif pullback_trend_label:
                    filter_reason = "回踩趋势线"
                elif first_pullback_after_cross_label:
                    filter_reason = "第一次回踩"
                elif low_volume_price_label:
                    filter_reason = "地量低价"
                elif gap_up_label:
                    filter_reason = "跳空K线"
                elif long_negative_label:
                    filter_reason = "长阴短柱"
                elif declining_volume_label and consistent_label:
                    filter_reason = "缩量且涨幅一致"
                else:
                    filter_reason = "其他原因"
                
                #若是，给出买入信号、止损价（或止损类型）、盈亏比、买入价格、筛选原因
                stop_loss_price = (min(yesterday['最低'], today_low) * 0.97).round(1)
                # 寻找N型结构的最高价
                near_high_price = get_last_n_high(df)
                #print(f"买入信号：{file_name}，买入价格：{today_close}，止损价：{stop_loss_price}，最高价：{near_high_price}，",min(yesterday['最低'], today_low),min(yesterday['最低'], today_low).round(1))
                if near_high_price and near_high_price > stop_loss_price:
                    ratio = ((near_high_price - today_close) / (today_close - stop_loss_price)).round(1)
                else:
                    ratio = "请人工判断盈亏比！"
                #print(f"{file_name}B1筛选通过，最近的N型高价为{near_high_price}，止损价格为{stop_loss_price}")
                return [1, stop_loss_price, today_close, ratio, filter_reason]
            else:
                return [-1]
        else:
            return [-1]



