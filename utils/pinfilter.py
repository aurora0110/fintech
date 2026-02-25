from pathlib import Path
from utils import stockDataValidator
from utils import stoploss
from utils import technical_indicators

def check(file_path):
    '''
    单针下35
    '''
    # 步骤1：取最后一个/后的文件名 → SZ#300319.txt
    file_name_full = file_path.split('/')[-1]
    # 步骤2：去掉.txt后缀 → SZ#300319
    file_name_no_suffix = file_name_full.replace('.txt', '')
    # 步骤3：取#后的股票代码 → 300319
    file_name = file_name_no_suffix.split('#')[-1]

    # 计算技术指标
    df, load_error = stoploss.load_data(file_path)
    df_trend = technical_indicators.calculate_trend(df)
    df_rsi = technical_indicators.calculate_rsi(df)

    if df_trend['知行多空线'].iloc[-1] > df_trend['知行短期趋势线'].iloc[-1]:
        return False

    # 找出近2个月最高的n条成交量记录
    # 按成交量降序排列，取前n行
    df_60 = df.tail(60)
    top5_volume_df = df_60.sort_values('成交量', ascending=False).head(2)
    # 添加"是否阳线"列（阳线：收盘 > 开盘）
    top5_volume_df['是否阳线'] = top5_volume_df['收盘'] > top5_volume_df['开盘']
    # 判断前n大成交量是否全部为阳线
    all_are_bullish = top5_volume_df['是否阳线'].all()
    
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    today_close = today['收盘']
    today_open = today['开盘']
    yesterday_close = yesterday['收盘']
    yesterday_open = yesterday['开盘']
    today_volume = today['成交量']
    yesterday_volume = yesterday['成交量']
    
    # 计算30日内的最高成交量
    recent_30_volume = df['成交量'].tail(30)
    max_30_volume = recent_30_volume.max()

    if df_rsi['RSI_14'].iloc[-1] > df_rsi['RSI_28'].iloc[-1] > df_rsi['RSI_57'].iloc[-1]:
        today_pin_label = technical_indicators.caculate_pin(df)

        if today_pin_label and today_volume <= max_30_volume / 2 and today_close < today_open and yesterday_close > yesterday_open and all_are_bullish:
            return True


