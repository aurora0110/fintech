# 止损校验
# 1、达到预设止损位
# 2、触发滴滴
# 3、放量阴线
# 4、破N型结构前低

from pathlib import Path
from typing import Any
from utils import stockDataValidator, technical_indicators
import pandas as pd

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

def check(file_path, hold_list):
    # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
    file_name_full = file_path.split('/')[-1]
    # 步骤2：去掉.txt后缀 → SZ#300319
    file_name_no_suffix = file_name_full.replace('.txt', '')
    # 步骤3：取#后的股票代码 → 300319
    file_name = file_name_no_suffix.split('#')[-1]
    #print(f"正在校验止损：{file_name}")

    # 加载数据
    df, load_error = load_data(file_path)
    df_trend = technical_indicators.calculate_trend(df)
    # 提取所有股票代码到新列表
    code_list = [str(item[0]) for item in hold_list]
    # 直接判断目标代码是否在新列表中
    is_exist = file_name in code_list

    # 只有找到止损价才说明该股票正在持有中
    if is_exist:
        #print(f"---------止损校验 当前股票{file_name}正在持有中---------")
        # 取今天、昨天、前天、大前天的数据
        today_close = float(df.iloc[-1]['收盘'])
        today_open = float(df.iloc[-1]['开盘'])
        today_volume = float(df.iloc[-1]['成交量'])
        last1day_close = float(df.iloc[-2]['收盘'])
        last1day_low = float(df.iloc[-2]['最低'])
        last1day_volume = float(df.iloc[-2]['成交量'])
        last2day_close = float(df.iloc[-3]['收盘'])
        last2day_low = float(df.iloc[-3]['最低'])
        last2day_volume = float(df.iloc[-3]['成交量'])
        last3day_close = float(df.iloc[-4]['收盘'])
        last3day_low = float(df.iloc[-4]['最低'])

        # 找预设的止损位
        target_value = [float(item[3]) for item in hold_list if str(item[0]) == str(file_name)][0]

        # 1、判断是否达到预设止损位
        if today_close < target_value:
            #print(f"股票代码{file_name}今日收盘价为{today_close}，达到对应的止损价：{target_value}")
            #print(f"---------当前持有股票{file_name}止损校验已完成---------\n")
            return [1, "达到预设止损位"]

        # 连续两天滴滴
        didi_sell_2days = today_close < last1day_low and last1day_close < last2day_low
        # 连续三天滴滴 
        didi_sell_3days = today_close < last1day_low and last1day_close < last2day_low and last2day_close < last3day_low
        # 滴滴买回
        didi_buy = today_close > today_open and today_volume > last1day_volume and last1day_close < last2day_low and last2day_close < last3day_low 

        # 2、判断是否达到滴滴止损
        if didi_sell_2days or didi_sell_3days:
            #print(f"股票代码{file_name}连续两天或三天滴滴，达到止损位！")
            #print(f"---------当前持有股票{file_name}止损校验已完成---------\n")
            return [1, "达到滴滴止损位"]
        if didi_buy:
            print(f"股票代码{file_name}满足滴滴买回条件！")
   
        # 3、判断是否达到放量止损一段时间内的巨量
        result = {}
        if df.iloc[-1]['收盘'] < df.iloc[-1]['开盘']:
            # 遍历5/10/30周期，判断是否大于对应周期最大值
            for period in [30, 60, 120]:
                # 截取最近period行（排除最后一行），不足则取所有历史行
                recent_data = df["成交量"].iloc[-(period+1):-1] if len(df)>=period+1 else df["成交量"].iloc[:-1]
                result[f'{file_name}今日是否放量阴线且大于最近{period}天最大值'] = df["成交量"].iloc[-1] > recent_data.max() if not recent_data.empty else False

            for k, v in result.items():
                if v:
                    #print(f"{k}：{'是' if v else '否'}")
                    return [1, "达到周期放量阴线止损位"]
        
        df_dk = df_trend['知行多空线'].iloc[-1]
        df_qs = df_trend['知行短期趋势线'].iloc[-1]
        df_dk_yesterday = df_trend['知行多空线'].iloc[-2]
        df_qs_yesterday = df_trend['知行短期趋势线'].iloc[-2]
        # 4、判断是否黄线下两根止损
        if df.iloc[-1]['收盘'] < df_dk and df.iloc[-2]['收盘'] < df_dk_yesterday:
            return [1, "达到黄线下两根止损位"]
        # 5、判断是否白线下两根止损
        if df.iloc[-1]['收盘'] < df_qs and df.iloc[-2]['收盘'] < df_qs_yesterday:
            return [1, "达到白线下两根止损位"]

        return [-1, "未达到止损条件"]
    return [-1, "未达到止损条件"]
        #print("未达到止损条件")
        #print(f"---------当前持有股票{file_name}止损校验已完成---------\n")




    
    
    
    

