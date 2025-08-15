import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.subplots as sp
import plotly.graph_objs as go
from pathlib import Path
import getData

def int_to_chinese_num(num):
    '''
    转换整数为中文数字
    '''
    if not isinstance(num, int):
        return "请输入整数"

    # 处理负数情况
    sign = ""
    if num < 0:
        sign = "-"
        num = abs(num)

    if num == 0:
        return "零"

    digit_map = ["", "十", "百", "千"]
    unit_map = ["", "万", "亿", "兆"]  # 可扩展更高单位
    num_str = str(num)
    length = len(num_str)
    result = []

    # 每4位一组（中文数字以万为单位）
    for i in range(0, length, 4):
        segment = num_str[max(0, length - i - 4): length - i]
        segment_len = len(segment)
        segment_str = ""

        # 处理每一段（千、百、十、个位）
        for j in range(segment_len):
            digit = int(segment[j])
            if digit == 0:
                continue  # 零不单独显示，除非在中间（如 1001 → 一千零一）
            # 添加数字和单位（如 "3" + "百" → "3百"）
            segment_str += str(digit) + digit_map[segment_len - j - 1]

        # 添加段单位（万、亿等）
        if segment_str:  # 如果该段不为空
            segment_str += unit_map[i // 4]
        result.append(segment_str)

    # 拼接所有段（从高到低）
    chinese_num = "".join(reversed(result))

    # 处理连续的零（如 "1001" → "一千零一"）
    chinese_num = chinese_num.replace("零零", "零").strip("零")
    
    # 加上符号（如果是负数）
    return sign + chinese_num if chinese_num else "零"

class StockAnalyzer:
    def __init__(self, ticker, file_path, start_date=None, end_date=None, kdj_days = 9, kdj_m1=3, kdj_m2=3, windows = [20, 30, 60, 120]):
        self.ticker = ticker
        self.file_path = file_path
        self.end_date = pd.to_datetime(end_date or datetime.now().strftime('%Y-%m-%d'))
        self.start_date = pd.to_datetime(start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.stock_data = self.get_data()
        self.windows = windows # 均线窗口

        self.data_ma = {}
        self.data_bbi = {}
        self.data_price = {}
        self.data_macd = {}
        self.data_kdj = {}
        self.data_shakeout = {}

        self.kdj_days = kdj_days
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2
    
    def get_data(self):
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件{path}不存在")
        try:
            data = pd.read_csv(self.file_path, engine=None)
            data['日期'] = pd.to_datetime(data['日期'])
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
            return data
        except Exception as e:
            print(f"读取文件{path}失败，错误信息：{e}")
        return pd.read_csv(path, encoding="utf-8")

    def calculate_all_indicators(self):
        self.calculate_moving_averages()
        self.calculate_bbi()
        self.calculate_price()
        self.calculate_macd()
        self.calculate_kdj()
        self.calculate_shakeout()

    def calculate_kdj(self):
        df = self.stock_data.copy()
        # 使用 datetime 比较
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)]
        df['high_kdj'] = df['最高'].rolling(window = self.kdj_days, min_periods = 1).max()
        df['low_kdj'] = df['最低'].rolling(window = self.kdj_days, min_periods = 1).min()
        df['RSV'] = (df['收盘'] - df['low_kdj']) / (df['high_kdj'] - df['low_kdj']) * 100
        df['K'] = df['RSV'].ewm(alpha=1/self.kdj_m1, adjust=False).mean()
        df['D'] = df['K'].ewm(alpha=1/self.kdj_m2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        data_list = df[['日期','J', '收盘', '最低', '最高']].to_numpy().tolist()
        # 定义区间阈值和筛选函数
        thresholds = {
            'low_0': lambda j: j <= 0,
            'low_5': lambda j: j <= -5,
            'low_10': lambda j: j <= -10,
            'low_15': lambda j: j <= -15,
            'low_20': lambda j: j <= -20,
            'low_25': lambda j: j <= -25,
            'high_80': lambda j: j >= 80,
            'high_90': lambda j: j >= 90,
            'high_100': lambda j: j >= 100,
            'high_110': lambda j: j >= 110,
            'high_120': lambda j: j >= 120,
        }
        fast_down_j_label = False
        # 初始化结果字典
        ret_kdj_dict = {key: [] for key in thresholds}

        # 遍历每一行数据
        for row in data_list:
            date, j_val, close, high, low = row[0], float(row[1]), row[2], row[3], row[4]
            row_data = [date, round(j_val, 3), close, high, low]

            for key, condition in thresholds.items():
                if condition(j_val):
                    ret_kdj_dict[key].append(row_data)

        # 筛选三天内J快速下降
        if data_list[-1][1] < 10 and data_list[-3][1] > 80:
            fast_down_j_label = True
        # 整理为列表（按 thresholds 的顺序）
        ret_kdj = [ret_kdj_dict[key] for key in thresholds]

        # 返回结构化结果
        return {
            'K': df['K'],
            'D': df['D'],
            'J': df['J'],
            'ret_kdj': ret_kdj,
            'fast_down_j_label': fast_down_j_label
        }

    def calculate_rsi(self):
        df = self.stock_data.copy()
        # 使用 datetime 比较
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)]

        delta = df['收盘'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi
        return df

    def calculate_moving_averages(self):
        windows = self.windows
        result = {}
        for window in windows:
            result[f'MA_{window}'] = self.stock_data['收盘'].rolling(window=window).mean()
        result['date'] = self.stock_data['日期']
        self.data_ma = pd.DataFrame(result)
        return self.data_ma

    def calculate_bbi(self):
        data = self.stock_data.copy()
        data['avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
        data['ma3'] = data['avg_price'].rolling(3).mean()
        data['ma6'] = data['avg_price'].rolling(6).mean()
        data['ma12'] = data['avg_price'].rolling(12).mean()
        data['ma24'] = data['avg_price'].rolling(24).mean()
        data['bbi'] = (data['ma3'] + data['ma6'] + data['ma12'] + data['ma24']) / 4
        return {'date': data['日期'], 'bbi': data['bbi']}

    def calculate_price(self):
        data = self.stock_data.copy()
        data['avg_price'] = (data['收盘'] + data['最高'] + data['最低']) / 3
        data['close_price'] = data['收盘']
        data['open_price'] = data['开盘']
        return {'date': data['日期'], 'avg_price': data['avg_price'], 'close_price': data['close_price'], 'open_price':data['open_price']}

    def calculate_macd(self, fast=12, slow=26, signal=9):
        df = self.stock_data.copy()
        ema_fast = df['收盘'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['收盘'].ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return {'date': df['日期'], 'DIF': dif, 'DEA': dea, 'MACD': macd}

    def calculate_shakeout(self):
        '''
        | 指标名称       | 条件                  | 含义解释              |
        | ---------- | ------------------- | ----------------- |
        | **四线归零买**  | 短期、中期、中长期、长期都 <= 6  | 四个指标都超卖，极端低点，可能反弹 |
        | **白线下20买** | 短期 <= 20 且 长期 >= 60 | 短期超卖，长期仍强，可能是回调买点 |
        | **白穿红线买**  | 短期上穿长期 且 长期 < 20    | 动量金叉且低位，反转可能性大    |
        | **白穿黄线买**  | 短期上穿中期 且 中期 < 30    | 动量拐头，初步反弹信号       |
        '''
        N1 = 3 # 短期指标
        N2 = 21 # 长期指标
        df = self.stock_data.copy()
        #df = df[(df['日期'] >= str(self.start_date)) & (df['日期'] <= str(self.end_date))]


        # 计算函数
        def momentum_indicator(C, L, n):
            return 100 * (C - L.rolling(n).min()) / (C.rolling(n).max() - L.rolling(n).min())

        # 计算短中长期指标
        df['短期'] = momentum_indicator(df['收盘'], df['最低'], N1)
        df['中期'] = momentum_indicator(df['收盘'], df['最低'], 10)
        df['中长期'] = momentum_indicator(df['收盘'], df['最低'], 20)
        df['长期'] = momentum_indicator(df['收盘'], df['最低'], N2)

        # 买点条件
        df['四线归零买'] = np.where(
            (df['短期'] <= 6) & (df['中期'] <= 6) & (df['中长期'] <= 6) & (df['长期'] <= 6),
            1, 0)

        df['白线下20买'] = np.where(
            (df['短期'] <= 25) & (df['长期'] >= 55),
            1, 0)

        df['白线下20买_小V'] = np.where(
            (df['长期'] - df['短期'] >= 40) & (df['长期'] >= 60),
            1, 0)

        # 白穿红线买（金叉）
        df['白穿红线买'] = np.where(
            (df['短期'] > df['长期']) & (df['短期'].shift(1) <= df['长期'].shift(1)) & (df['长期'] < 20),
            1, 0)

        # 白穿黄线买（金叉）
        df['白穿黄线买'] = np.where(
            (df['短期'] > df['中期']) & (df['短期'].shift(1) <= df['中期'].shift(1)) & (df['中期'] < 30),
            1, 0)

        # 输出最后几行查看
        #print(df[['短期', '中期', '中长期', '长期', '四线归零买', '白线下20买', '白穿红线买', '白穿黄线买']].tail())

        return df

    def plot_moving_averages(self, colors=['red', 'blue', 'green']):
        if not hasattr(self, 'ma_data'):
            raise ValueError("请先调用 calculate_moving_averages 方法！")
        x_axis = self.ma_data['date']
        plt.figure(figsize=(14, 8))
        for i, column in enumerate(self.ma_data.columns):
            if 'MA_' in column:
                plt.plot(x_axis, self.ma_data[column], label=column, color=colors[i % len(colors)], linewidth=1.5)
        step = 50
        selected_ticks = x_axis[::step]
        plt.xticks(selected_ticks, rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.ticker} Moving Averages')
        plt.grid(True, linestyle='--', linewidth=0.2, alpha=1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_all(self, data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, ticker, windows):
        # x_axis 是形如 ['2020-01-01', ...] 的字符串列表
        x_axis = pd.to_datetime(data_ma['date'])

        fig = sp.make_subplots(
            rows=6, cols=2,
            specs=[[{}, {}],    # 第一行两列
                [{"colspan": 2}, None],  # 第二行一整行（跨2列）
                [{"colspan": 2}, None],  # 第三行一整行
                [{"colspan": 2}, None],  # 第四行一整行
                [{}, {}],
                [{"colspan": 2}, None]], # 第五行一整行（跨2列）
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            subplot_titles=[
                f'MA {windows[0]} {windows[1]} {windows[2]}', 'BBI',
                'Avg & Close Price',
                'KDJ-J Highlighted Points',
                'MACD',
                'KDJ-J Highlighted Points -10~90', 'KDJ-J Highlighted Points-15~100',
                'Shakeout Monitoring'
            ]
        )

        fig.update_layout(
        xaxis3=dict(matches='x'),
        xaxis4=dict(matches='x'),
        xaxis5=dict(matches='x'),
        xaxis6=dict(matches='x')
    )

        # 第一行，左图 MA
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[0]}'], name=f'MA_{windows[0]}', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[1]}'], name=f'MA_{windows[1]}', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[2]}'], name=f'MA_{windows[2]}', line=dict(color='green')), row=1, col=1)

        # 第一行，右图 BBI
        fig.add_trace(go.Scatter(x=x_axis, y=data_bbi['bbi'], name='BBI', line=dict(color='orange')), row=1, col=2)

        # 第二行，整行 price
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['avg_price'], name='avg_price', line=dict(color='gray')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['close_price'], name='close_price', line=dict(color='green')), row=2, col=1)

        # 第三行，整行 KDJ-J + 高亮点
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=3, col=1)

        mask_low = data_kdj['J'] <= -5
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -5',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=3, col=1)

        mask_high = data_kdj['J'] > 80
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 80',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=3, col=1)

        # 第四行，整行 MACD
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DIF'], name='DIF', line=dict(color='green')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DEA'], name='DEA', line=dict(color='gray')), row=4, col=1)

        # 第五行，左图 -10～90
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=5, col=1)

        mask_low = data_kdj['J'] <= -10
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -10',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        mask_high = data_kdj['J'] > 90
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 90',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        # 第五行，右图 -15～100
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='gray')), row=5, col=2)

        mask_low = data_kdj['J'] <= -15
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -15',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=1)

        mask_high = data_kdj['J'] > 100
        fig.add_trace(go.Scatter(
            x=x_axis[mask_high],
            y=data_kdj['J'][mask_high],
            mode='markers+text',
            name='J > 100',
            marker=dict(color='green', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_high]],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=5, col=2)

        # 第六行，整行 shakeout monitoring
        fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['短期'], name='短期', line=dict(color='green')), row=6, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['长期'], name='长期', line=dict(color='red')), row=6, col=1)

        # 添加掩码
        mask = (data_shakeout['短期'] < 20) & (data_shakeout['长期'] > 60)

        # 筛选符合条件的 x 和 y 值
        x_highlight = x_axis[mask]
        y_highlight = data_shakeout['短期'][mask]
        # 添加高亮点
        fig.add_trace(go.Scatter(
            x=x_highlight,
            y=y_highlight,
            mode='markers+text',
            name='短期<20 & 长期>60',
            marker=dict(color='cyan', size=10, symbol='circle'),
            text=[f'{v:.1f}' for v in y_highlight],
            textposition='top center',
            textfont=dict(color='blue')
        ), row=6, col=1)

        # 在第六行子图（row=6, col=1）上绘制 y=20, 60, 80 三条横线 红线在60 80之间 白线在20以下
        for y_val in [60, 80]:
            fig.add_shape(
                type="line",
                x0=x_axis.min(),
                x1=x_axis.max(),
                y0=y_val,
                y1=y_val,
                line=dict(color="green", width=3, dash="solid"),
                xref="x8",  # row=6 col=1 的 subplot x 轴
                yref="y8"   # row=6 col=1 的 subplot y 轴
            )

        fig.add_shape(
                type="line",
                x0=x_axis.min(),
                x1=x_axis.max(),
                y0=20,
                y1=20,
                line=dict(color="yellow", width=3, dash="solid"),
                xref="x8",  # row=6 col=1 的 subplot x 轴
                yref="y8"   # row=6 col=1 的 subplot y 轴
            )
        # 更新布局
        fig.update_layout(
            height=1200,
            width=1400,
            title=f'{ticker}',
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,1)',
            template='plotly_white'
        )

        fig.show()
    
    def save_moving_averages(self, filename=None):
        if not hasattr(self, 'ma_data'):
            raise ValueError("请先调用 calculate_moving_averages 方法！")
        filename = filename or f'{self.ticker}_moving_averages.csv'
        self.ma_data.to_csv(filename, index=False)
        print(f"数据已保存至：{filename}")
    
class StockMonitor:
    def __init__(self, ticker, file_path,  file_volume_path, start_date=None, end_date=None, lookback_period=10, min_signal_count=3):
        self.ticker = ticker
        self.file_path = file_path
        self.file_volume_path = file_volume_path
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_period = lookback_period # 连续n天内出现单针下20的信号
        self.min_signal_count = min_signal_count # 出现n次单针下20的信号
    
    def fastdown_J(self):
        '''
        买入逻辑
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_kdj = analyzer.calculate_kdj()
        label = False
        if (data_kdj['J'].iloc[-3] - data_kdj['J'].iloc[-1]) >= 60 and data_kdj['J'].iloc[-1] < 10:
            label = True
        '''
        label = all(
        any([
            (data_kdj['J'].iloc[-1] < 10 and data_kdj['J'].iloc[period] > 70),  
            (data_kdj['J'].iloc[-1] < 5 and data_kdj['J'].iloc[period] > 65),  
            (data_kdj['J'].iloc[-1] < 0 and data_kdj['J'].iloc[period] > 60) 
        ])
        for period in [-1, -2, -3] )
        '''
        print(f"{self.ticker}:J值是否快速下降：{'true✅' if label else 'false❌'}，最近3️⃣天的J值：{round(data_kdj['J'].iloc[-3],1)}，{round(data_kdj['J'].iloc[-2],1)}，{round(data_kdj['J'].iloc[-1],1)}")

        return label

    def continuous_shakeout(self):
        '''
        买入逻辑
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_shakeout = analyzer.calculate_shakeout()
        label = False
        label = all(
        any([
            data_shakeout.iloc[period].get("四线归零买", False) == 1,
            data_shakeout.iloc[period].get("白线下20买", False) == 1,
            data_shakeout.iloc[period].get("白穿红线买", False) == 1,
            data_shakeout.iloc[period].get("白穿黄线买", False) == 1,
            data_shakeout.iloc[period].get("白线下20买_小V", False) == 1
        ])
        for period in [-1, -2] )

        return label

    def check_signal_frequency(self):
        '''
        买入逻辑，检查最近10天内是否至少有3个周期满足任意买入信号
        '''
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        data_shakeout = analyzer.calculate_shakeout()
        signal_count = 0
        for period in range(-1, -self.lookback_period-1, -1): 
            if any([
                data_shakeout.iloc[period].get("四线归零买", 0) == 1,
                data_shakeout.iloc[period].get("白线下20买", 0) == 1,
                data_shakeout.iloc[period].get("白穿红线买", 0) == 1,
                data_shakeout.iloc[period].get("白穿黄线买", 0) == 1,
                data_shakeout.iloc[period].get("白线下20买_小V", 0) == 1
            ]):
                signal_count += 1
                if signal_count >= self.min_signal_count:  # 达到最小信号数就提前返回
                    return True
        return signal_count >= self.min_signal_count

    def bs_abnormal_monitor(self):
        '''
        * 买入、卖出逻辑
        * 监控异常交易量、价格、买卖笔数，比如当日绿线，但是买入笔数大于卖出笔数，可能是有人在低位收筹码
        * 开盘收盘价格是从000001.csv（历史价格）文件中获取的，开盘收盘总价和总量是从000001_volume.csv（只有每天最新的价格）文件中获取的，如果想看历史数据可以去通达信导出
        '''
        # 获取的是当天最新的数据
        df = getData.read_from_csv(self.file_volume_path)
        sell_list = []
        buy_list = []
        sellprice_amount = 0
        buyprice_amount = 0
        sellvolume_amount = 0
        buyvolume_amount = 0
        label = False
        abnormal_type = 'none'
        for _, row in df.iterrows():
            record = {
                '成交金额': row['成交金额'],
                '成交量': row['成交量'],
                '性质': row['性质']
            }
            if row['性质'] == '卖盘':
                sell_list.append(record)
                sellprice_amount += int(row['成交金额'])
                sellvolume_amount += int(row['成交量'])
            elif row['性质'] == '买盘':
                buy_list.append(record)
                buyprice_amount += int(row['成交金额'])
                buyvolume_amount += int(row['成交量'])
        # 获取历史上最新的数据
        analyzer = StockAnalyzer(self.ticker, self.file_path)
        price_dict = analyzer.calculate_price()
        open_price = price_dict['open_price'].iloc[-1]
        close_price = price_dict['close_price'].iloc[-1]

        if (close_price < open_price) and (buyvolume_amount > sellvolume_amount):
            print(f"❗️当日绿线📉，但是买入量大于卖出量，可能是有人偷偷在低位收筹码❗️")
            label = True
            abnormal_type = 'buy'
        elif(close_price > open_price) and (buyvolume_amount < sellvolume_amount):
            print(f"❗️当日红线📈，但是买入量小于卖出量，可能是有人偷偷在高位卖筹码❗️")
            label = True
            abnormal_type = 'sell'
        else:
            print(f"成交量无异常")

        if (close_price < open_price) and (buyprice_amount > sellprice_amount):
            print(f"❗️当日绿线📉，但是买入总额大于卖出总额，可能是有人偷偷在低位收筹码❗️")
            label = True
            abnormal_type = 'buy'
        elif(close_price > open_price) and (buyprice_amount < sellprice_amount):
            print(f"❗️当日红线📈，但是买入总额小于卖出总额，可能是有人偷偷在高位卖筹码❗️")
            label = True
            abnormal_type = 'sell'
        else:
            print(f"成交总额无异常")

        # 获取总市值和总股本
        market_cap, share_cap = getData.download_total_cap(self.ticker)

        print(f"当日开盘价：{open_price}，收盘价：{close_price}， {'📈' if close_price > open_price else '📉'}， 卖出总额：{sellprice_amount}={int_to_chinese_num(sellprice_amount)}，买入总额：{buyprice_amount}={int_to_chinese_num(buyprice_amount)}，净买入总额：{buyprice_amount-sellprice_amount}={int_to_chinese_num(buyprice_amount-sellprice_amount)}，卖出总量：{sellvolume_amount}={int_to_chinese_num(sellvolume_amount)}，买入总量：{buyvolume_amount}={int_to_chinese_num(buyvolume_amount)}，净买入总量：{buyvolume_amount-sellvolume_amount}={int_to_chinese_num(buyvolume_amount-sellvolume_amount)}")
        print(f"当前交易占总股本比重:{round(abs(buyprice_amount-sellprice_amount) / int(share_cap),3)}，占总市值比重为:{round(abs(buyprice_amount-sellprice_amount) / int(market_cap),3)}")
        return {'open_price': open_price, 'close_price': close_price, 'sellprice_amount': sellprice_amount, 'buyprice_amount': buyprice_amount, 'sellvolume_amount': sellvolume_amount
                , 'buyvolume_amount': buyvolume_amount, 'label':label, 'abnormal_type':abnormal_type, 'market_cap_percentage':round(abs(buyprice_amount-sellprice_amount) / int(share_cap),3)
                , 'share_cap_percentage':round(abs(buyprice_amount-sellprice_amount) / int(market_cap),3)}

    def below_bbi_monitor(self):
        '''
        卖出信号，跌破BBI两根卖出信号，即收盘价低于BBI
        '''
        # 获取的是当天最新的价格数据
        df = getData.read_from_csv(self.file_path)

        analyzer = StockAnalyzer(self.ticker, self.file_path)
        bbi = analyzer.calculate_bbi()

        bbi_label = False

        if (df['开盘'].iloc[-1] > bbi['bbi'].iloc[-1] > df['收盘'].iloc[-1]) and (df['开盘'].iloc[-2] > bbi['bbi'].iloc[-2] > df['收盘'].iloc[-2]):
            bbi_label = True
            print(f"❗️跌破BBI两根卖出信号❗️")
        else:
            print(f"未跌破BBI两根卖出信号")

        return bbi_label

    def twin_tails_monitor(self):
        '''
        双马尾，根据价格判断----看一下这个逻辑能不能筛选出来娜娜的图形

        检查DataFrame中最近period个交易日是否符合以下模式：
        1. 只有两天达到最高价，且差值在1%以内；
        2. 这两天之间相差15天以上；
        3. 周期内最高价和最低价落差超过30%。

        参数:
            df (pd.DataFrame): 包含 '日期', '最高', '最低' 三列
            period (int): 检查周期（默认60天）

        返回:
            bool: 是否满足条件
        '''
        period_days_60 = 60 # 设置计算周期为60天

        df = getData.read_from_csv(self.file_path)
        required_cols = {'日期','最高价','最低价'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"输入DataFrame必须包含列：{required_cols}")
        
        recent_df_60 = df.tail(period_days_60).copy()

        if recent_df_60.shape[0] < period_days_60:
            print("数据不足指定周期，跳过检查。")
            return False

        # 最高价和最低价
        highest_price_60 = recent_df_60['最高价'].max()
        lowest_price_60 = recent_df_60['最低价'].min()

        # 落差 > 15%
        if (highest_price_60 - lowest_price_60) / lowest_price_60 < 0.15:
            return False
        
        # 取第二高，判断与最高价的差距
        second_high = recent_df_60['最高价'].nlargest(2).iloc[1]
        diff_pct = abs(highest_price_60 - second_high) / highest_price_60
        if diff_pct > 0.05:
            return False
        
        # 找出达到最高价的行
        highest_rows = recent_df_60[recent_df_60['最高价'] == highest_price_60]
        # 两个最高价的日期间隔要超过15天
        day_diff = abs((highest_rows['日期'].iloc[1] - highest_rows['日期'].iloc[0]).days)
        if day_diff <= 5:
            return False
        
        return True
    
    def position_building_monitor(self):
        '''
        建仓波，准备撅小土包
        '''
        return
    
    def heavy_cannos_monitor(self):
        '''
        两门重炮，取7天为1个周期
        '''
        df = getData.read_from_csv(self.file_path)
        df_volume = getData.read_from_csv(self.file_volume_path)
        period_df = df.iloc[-7:].copy()
        period_df_volume = df_volume.iloc[-7:].copy()

        today = df.iloc[-1]
        today_volume = df_volume.iloc[-1]

        today_high = today['最高价']
        today_low = today['最低价']
        today_volume_amount = today_volume['成交金额']
    
        for i in range(len(period_df) - 1):  # 最后一行是今天，不比较
            day = period_df.iloc[i]
            day_volume = period_df_volume.iloc[i]
            
            # 判断价格与成交量的相似性
            high_similar = abs(day['最高价'] - today_high) / today_high <= 0.05
            low_similar = abs(day['最低价'] - today_low) / today_low <= 0.05
            volume_similar = abs(day_volume['成交金额'] - today_volume_amount) / today_volume_amount <= 0.15
            
            if high_similar and low_similar and volume_similar:
                # 找出这两天中较低的成交量
                vol_min = min(day_volume['成交金额'], today_volume_amount)
                
                # 这两天之间的所有天的成交量必须都低于这两天的最小成交量
                intermediate = period_df.iloc[i+1:-1]
                if all(intermediate['成交金额'] < vol_min):
                    return True  # 满足条件
    
        return False  # 所有天都不满足

# 示例调用
if __name__ == "__main__":
    
    ticker = '600036.SS'
    file_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/000001.csv'  # 替换为你的路径
    file_volume_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/000001_volume.csv'


    analyzer = StockAnalyzer(ticker, file_path)
    ma = analyzer.calculate_moving_averages()
    bbi = analyzer.calculate_bbi()
    kdj = analyzer.calculate_kdj()
    macd = analyzer.calculate_macd()
    price = analyzer.calculate_price()
    shakeout = analyzer.calculate_shakeout()

    StockMonitor(ticker, file_path, file_volume_path).below_bbi_monitor()
    
        #analyzer.plot_all(ma, bbi, price, macd, kdj, shakeout, '000001', windows=[20, 30, 60, 120])
    
