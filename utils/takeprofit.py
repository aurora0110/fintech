# 止盈校验
# 1、连续上涨
# 2、连续两根中长阳线

from pathlib import Path
from utils import stockDataValidator, technical_indicators
from utils import stoploss

def check(file_path, hold_list):
    # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
    file_name_full = file_path.split('/')[-1]
    # 步骤2：去掉.txt后缀 → SZ#300319
    file_name_no_suffix = file_name_full.replace('.txt', '')
    # 步骤3：取#后的股票代码 → 300319
    file_name = file_name_no_suffix.split('#')[-1]
    #print(f"正在校验止盈：{file_name}")

    # 定义中长阳的涨幅
    long_line_pct = 4

    df, load_error = stoploss.load_data(file_path)
    df_trend = technical_indicators.calculate_trend(df)
    # 提取所有股票代码到新列表
    code_list = [str(item[0]) for item in hold_list]
    # 直接判断目标代码是否在新列表中
    is_exist = file_name in code_list

    if is_exist:
        #print(f"---------止盈校验 当前股票{file_name}正在持有中---------")
        # ------------- 1. 判断最近5天是否连续上涨 -------------
        is_5d_cont_rise = False
        # 计算每日涨跌幅（后一天 - 前一天）
        daily_change = df['收盘'].iloc[-6:].diff().dropna()
        # 连续上涨：近5天每天涨幅>0（排除平盘）
        is_5d_cont_rise = (daily_change > 0).all()
        if is_5d_cont_rise:
            #print(f"{file_name}最近5天连续上涨")
            #print(f"---------当前持有股票{file_name}止盈校验已完成---------\n")
            return [1, "达到连续上涨止盈位"]
        
        # ------------- 2. 判断近一天或者两天是否为中长阳线 -------------
        is_1d_limit_up = False
        is_2d_long_line = False
        if len(df['收盘']) >= 2:
            # 近两天的涨幅（%）：(当日收盘价-前日收盘价)/前日收盘价 *100
            rise_pct_1d = (df['收盘'].iloc[-1] - df['收盘'].iloc[-2]) / df['收盘'].iloc[-2] * 100  # 最近1天涨幅
            rise_pct_2d = (df['收盘'].iloc[-2] - df['收盘'].iloc[-3]) / df['收盘'].iloc[-3] * 100 if len(df['收盘'])>=3 else 0  # 倒数第2天涨幅
            
            # 中长阳线：单日涨幅≥设定阈值（默认4%）
            is_1d_long = rise_pct_1d >= long_line_pct
            is_2d_long = rise_pct_2d >= long_line_pct
            is_1d_limit_up = (rise_pct_1d >= long_line_pct * 2) # 单日涨幅达到阈值
            is_2d_long_line = is_1d_long and is_2d_long  # 近两天至少1天是中长阳线

        if  is_1d_limit_up or is_2d_long_line:
            return [1, "达到中长阳线止盈位"]
        
        # ------------- 3. 判断是否达到放量止损一段时间内的巨量 -------------
        result = {}
        if df.iloc[-1]['收盘'] > df.iloc[-1]['开盘']:
            # 遍历5/10/30周期，判断是否大于对应周期最大值
            for period in [30, 60, 120]:
                # 截取最近period行（排除最后一行），不足则取所有历史行
                recent_data = df["成交量"].iloc[-(period+1):-1] if len(df)>=period+1 else df["成交量"].iloc[:-1]
                result[f'{file_name}今日是否放量阳线且大于最近{period}天最大值'] = df["成交量"].iloc[-1] > recent_data.max() if not recent_data.empty else False

            for k, v in result.items():
                if v:
                    #print(f"{k}：{'是' if v else '否'}")
                    return [1, "达到周期放量阳线止盈位"]
        
        # ------------- 4. 判断是否偏离白线过多 -------------
        today_close = df.iloc[-1]['收盘']
        df_trend = technical_indicators.calculate_trend(df)
        df_qs = df_trend['知行短期趋势线'].iloc[-1]
        if (today_close - df_qs) / df_qs >= 0.25:
            return [1, "偏离白线第三止盈位"] # 第三止盈位 25%
        elif (today_close - df_qs) / df_qs >= 0.15:
            return [1, "达到白线第二止盈位"] # 第二止盈位 15%
        elif (today_close - df_qs) / df_qs >= 0.1:
            return [1, "达到白线第一止盈位"] # 第一止盈位 10%
        
        # ------------- 5. 判断是否达到止盈价 -------------
        # 找预设的止损位
        target_price = [float(item[7]) for item in hold_list if str(item[0]) == str(file_name)][0]
        today_high = df.iloc[-1]['最高']
        if today_close >= target_price or today_high >= target_price:
            #print(f"股票代码{file_name}今日收盘价为{today_close}，达到对应的止损价：{target_value}")
            #print(f"---------当前持有股票{file_name}止损校验已完成---------\n")
            return [1, "达到预设止盈位"]

        return [-1, "未达到止盈条件"]
    return [-1, "未达到止盈条件"]

        
        #print(f"未达到止盈条件")
        #print(f"---------当前持有股票{file_name}止盈校验已完成---------\n")

        
