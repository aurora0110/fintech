from re import M
import pandas as pd
import numpy as np

PRICE_COLUMNS = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']


def _empty_price_df():
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _load_price_data(file_path):
    """加载通达信导出的日线 txt 文件，返回标准价格列 DataFrame。"""
    numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量', '成交额']
    encodings = ['gbk', 'gb2312', 'utf-8', 'latin-1']
    lines = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            break
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            continue

    if lines is None:
        return _empty_price_df()

    data_rows = []
    for line in lines:
        if any(keyword in line for keyword in ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']):
            continue

        clean_line = ' '.join(line.split())
        parts = clean_line.split(' ')
        if len(parts) != 7:
            continue
        if '/' not in parts[0] or len(parts[0].split('/')) != 3:
            continue
        data_rows.append(parts)

    if not data_rows:
        return _empty_price_df()

    df = pd.DataFrame(data_rows, columns=PRICE_COLUMNS)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')
    df = df.dropna(subset=['日期', '开盘', '最高', '最低', '收盘'])
    if df.empty:
        return _empty_price_df()

    df = df[PRICE_COLUMNS].sort_values('日期').drop_duplicates(subset=['日期'], keep='last').reset_index(drop=True)
    return df


def calculate_week_price(file_path):
    """
    将通达信日线 txt 数据聚合为周线数据。
    :param file_path: 日线 txt 文件路径
    :return: 周线 DataFrame，列格式与原始数据一致
    """
    df = _load_price_data(file_path)
    if df.empty:
        return _empty_price_df()

    weekly_df = (
        df.groupby(pd.Grouper(key='日期', freq='W-FRI'))
        .agg({
            '日期': 'max',
            '开盘': 'first',
            '最高': 'max',
            '最低': 'min',
            '收盘': 'last',
            '成交量': 'sum',
            '成交额': 'sum',
        })
        .dropna(subset=['日期', '开盘', '收盘'])
        .reset_index(drop=True)
    )

    weekly_df = weekly_df[['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额']]
    return weekly_df.sort_values('日期').reset_index(drop=True)


def calculate_trend(df, N=9, M1=14, M2=28, M3=57, M4=114):
    """
    计算短期趋势线和多空线
    :param df: 股票日线数据DataFrame，需包含'收盘'（C）、'最高'（H）、'最低'（L）列
    :param N: RSI指标周期（默认9）
    :param M1-M4: 多空线移动平均周期（默认14/28/57/114）
    :return: 新增指标列和选股信号的DataFrame
    """
    # 1. 知行短期趋势线：EMA(EMA(C,10),10)
    df['知行短期趋势线'] = df['收盘'].ewm(span=10, adjust=False).mean()  # 第一次EMA(C,10)
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()  # 第二次EMA(EMA(C,10),10)
    
    # 2. 60日均线（MA1）和13日EMA（MA2）（原公式未用于选股，保留用于扩展）
    df['MA60'] = df['收盘'].rolling(window=60).mean()
    df['EMA13'] = df['收盘'].ewm(span=13, adjust=False).mean()
    
    # 3. 知行多空线：(MA(C,M1)+MA(C,M2)+MA(C,M3)+MA(C,M4))/4
    df['MA14'] = df['收盘'].rolling(window=M1).mean()
    df['MA28'] = df['收盘'].rolling(window=M2).mean()
    df['MA57'] = df['收盘'].rolling(window=M3).mean()
    df['MA114'] = df['收盘'].rolling(window=M4).mean()
    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
    
    return df

def calculate_kdj(df, N=9, M1=3, M2=3):
    """
    计算KDJ指标
    :param df: 股票日线数据DataFrame，需包含'最高'（H）、'最低'（L）、'收盘'（C）列
    :param N: KDJ指标周期（默认9）
    :param M1: K值平滑系数（默认3）
    :param M2: D值平滑系数（默认3）
    :return: 新增KDJ指标列的DataFrame
    """
    # RNG:=HHV(H,N)-LLV(L,N)
    df['HHV9'] = df['最高'].rolling(window=N, min_periods=1).max()
    df['LLV9'] = df['最低'].rolling(window=N, min_periods=1).min()
    df['RNG'] = df['HHV9'] - df['LLV9']
    
    df['RSV'] = (df['收盘'] - df['LLV9']) / (df['HHV9'] - df['LLV9']) * 100
    
    # 指数移动平均非简单移动平均
    df['K'] = df['RSV'].ewm(alpha=1/M1, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/M2, adjust=False).mean()
    
    # J:=3*K-2*D
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

def calculate_rsi(df, periods=[14, 28, 57], price_col='收盘'):
    """
    仅计算RSI指标（适配通达信算法）
    :param df: 股票数据DataFrame，必须包含 price_col 指定的列（默认'收盘'）
    :param periods: RSI计算周期列表（默认6/12/24，通达信常用周期）
    :param price_col: 计算基准列（默认'收盘'价）
    :return: 新增RSI列的DataFrame
    """
    # 1. 计算每日价格变动
    df['价格变动'] = df[price_col].diff()
    
    # 2. 分离上涨和下跌幅度（下跌取绝对值，无变动记为0）
    df['上涨幅度'] = np.where(df['价格变动'] > 0, df['价格变动'], 0)
    df['下跌幅度'] = np.where(df['价格变动'] < 0, -df['价格变动'], 0)
    
    # 3. 计算各周期RSI（通达信EMA平滑算法）
    for period in periods:
        # EMA平滑计算平均上涨/下跌幅度
        df[f'RSI_{period}'] = df['上涨幅度'].ewm(span=period, adjust=False).mean()
        df[f'平均下跌_{period}'] = df['下跌幅度'].ewm(span=period, adjust=False).mean()
        
        # 计算相对强度RS，避免除零
        df[f'RS_{period}'] = np.where(
            df[f'平均下跌_{period}'] == 0,
            100,
            df[f'RSI_{period}'] / df[f'平均下跌_{period}']
        )
        
        # 最终RSI公式
        df[f'RSI_{period}'] = 100 - (100 / (1 + df[f'RS_{period}']))
    
    # 4. 清理中间列，仅保留原始列和RSI列
    drop_cols = ['价格变动', '上涨幅度', '下跌幅度'] + \
                [col for col in df.columns if '平均下跌_' in col or 'RS_' in col]
    df = df.drop(columns=drop_cols)
    
    # 5. 填充周期内NaN值（通达信默认用50填充）
    for period in periods:
        df[f'RSI_{period}'] = df[f'RSI_{period}'].fillna(50)
    
    return df

def calculate_daily_ma(df, ma_periods=[5, 10, 20, 30, 60], price_col='收盘'):
    """
    仅计算指定周期的移动平均线（日线MA），与通达信算法一致
    :param df: 股票数据DataFrame，必须包含 '日期' 列和 price_col 指定的价格列（默认'收盘'）
    :param ma_periods: 要计算的日线周期列表（默认5/10/20/30/60）
    :param price_col: 计算均线的价格基准列（默认用收盘价计算）
    :return: 新增均线列的DataFrame
    """
    # 确保数据按日期升序排列（避免计算顺序错误）
    df = df.sort_values('日期').reset_index(drop=True)
    
    # 计算每个周期的简单移动平均（通达信MA算法）
    for period in ma_periods:
        # min_periods=1：不足周期时也计算（如前4天MA5=当日收盘价）
        df[f'MA{period}'] = df[price_col].rolling(window=period, min_periods=1).mean().round(2)
    
    return df

def caculate_pin(df, N1=3, N2=21):
    '''
    计算单针下20、单针下35
    '''
    # 短期指标
    llv_l_n1 = df['最低'].rolling(window=N1).min()
    hhv_c_n1 = df['收盘'].rolling(window=N1).max()
    df['短期'] = (df['收盘'] - llv_l_n1) / (hhv_c_n1 - llv_l_n1) * 100

    # 长期指标
    llv_l_n2 = df['最低'].rolling(window=N2).min()
    hhv_l_n2 = df['收盘'].rolling(window=N2).max()
    df['长期'] = (df['收盘'] - llv_l_n2) / (hhv_l_n2 - llv_l_n2) * 100

    # 单针下30
    pin_label = df['短期'].iloc[-1] <= 30 and df['长期'].iloc[-1] >= 85
    return pin_label


def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    :param df: 股票日线数据DataFrame，需包含'收盘'列
    :param fast_period: 快速EMA周期（默认12）
    :param slow_period: 慢速EMA周期（默认26）
    :param signal_period: 信号线EMA周期（默认9）
    :return: 新增MACD指标列的DataFrame
    """
    # 计算快速EMA和慢速EMA
    df['EMA12'] = df['收盘'].ewm(span=fast_period, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=slow_period, adjust=False).mean()
    
    # 计算DIFF线
    df['MACD_DIFF'] = df['EMA12'] - df['EMA26']
    
    # 计算DEA线（信号线）
    df['MACD_DEA'] = df['MACD_DIFF'].ewm(span=signal_period, adjust=False).mean()
    
    # 计算MACD柱状图
    df['MACD_HIST'] = 2 * (df['MACD_DIFF'] - df['MACD_DEA'])
    
    return df
