import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.subplots as sp
import plotly.graph_objs as go
from pathlib import Path


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
            data = pd.read_excel(self.file_path, engine=None)
            data = data[(data['日期'] >= self.start_date) & (data['日期'] <= self.end_date)]
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

        # 初始化结果字典
        ret_kdj_dict = {key: [] for key in thresholds}

        # 遍历每一行数据
        for row in data_list:
            date, j_val, close, high, low = row[0], float(row[1]), row[2], row[3], row[4]
            row_data = [date, round(j_val, 3), close, high, low]

            for key, condition in thresholds.items():
                if condition(j_val):
                    ret_kdj_dict[key].append(row_data)

        # 整理为列表（按 thresholds 的顺序）
        ret_kdj = [ret_kdj_dict[key] for key in thresholds]

        # 返回结构化结果
        return {
            'K': df['K'],
            'D': df['D'],
            'J': df['J'],
            'ret_kdj': ret_kdj
        }

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
        return {'date': data['日期'], 'avg_price': data['avg_price'], 'close_price': data['close_price']}

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
        df = df[(df['日期'] >= str(self.start_date)) & (df['日期'] <= str(self.end_date))]


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
            -30, 0)

        df['白线下20买'] = np.where(
            (df['短期'] <= 20) & (df['长期'] >= 60),
            -30, 0)

        # 白穿红线买（金叉）
        df['白穿红线买'] = np.where(
            (df['短期'] > df['长期']) & (df['短期'].shift(1) <= df['长期'].shift(1)) & (df['长期'] < 20),
            -30, 0)

        # 白穿黄线买（金叉）
        df['白穿黄线买'] = np.where(
            (df['短期'] > df['中期']) & (df['短期'].shift(1) <= df['中期'].shift(1)) & (df['中期'] < 30),
            -30, 0)

        # 输出最后几行查看
        df[['短期', '中期', '中长期', '长期', '四线归零买', '白线下20买', '白穿红线买', '白穿黄线买']].tail()

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
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[0]}'], name=f'MA_{windows[0]}', line=dict(color='white')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[1]}'], name=f'MA_{windows[1]}', line=dict(color='yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_ma[f'MA_{windows[2]}'], name=f'MA_{windows[2]}', line=dict(color='green')), row=1, col=1)

        # 第一行，右图 BBI
        fig.add_trace(go.Scatter(x=x_axis, y=data_bbi['bbi'], name='BBI', line=dict(color='orange')), row=1, col=2)

        # 第二行，整行 price
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['avg_price'], name='avg_price', line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_price['close_price'], name='close_price', line=dict(color='green')), row=2, col=1)

        # 第三行，整行 KDJ-J + 高亮点
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=3, col=1)

        mask_low = data_kdj['J'] <= -5
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -5',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='white')
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
            textfont=dict(color='white')
        ), row=3, col=1)

        # 第四行，整行 MACD
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DIF'], name='DIF', line=dict(color='white')), row=4, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=data_macd['DEA'], name='DEA', line=dict(color='yellow')), row=4, col=1)

        # 第五行，左图 -10～90
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=5, col=1)

        mask_low = data_kdj['J'] <= -10
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -10',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='white')
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
            textfont=dict(color='white')
        ), row=5, col=1)

        # 第五行，右图 -15～100
        fig.add_trace(go.Scatter(x=x_axis, y=data_kdj['J'], name='KDJ-J', line=dict(color='blue')), row=5, col=2)

        mask_low = data_kdj['J'] <= -15
        fig.add_trace(go.Scatter(
            x=x_axis[mask_low],
            y=data_kdj['J'][mask_low],
            mode='markers+text',
            name='J <= -15',
            marker=dict(color='red', size=8),
            text=[f'{v:.1f}' for v in data_kdj['J'][mask_low]],
            textposition='top center',
            textfont=dict(color='white')
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
            textfont=dict(color='white')
        ), row=5, col=2)

        # 第六行，整行 shakeout monitoring
        fig.add_trace(go.Scatter(x=x_axis, y=data_shakeout['短期'], name='短期', line=dict(color='white')), row=6, col=1)
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
            textfont=dict(color='white')
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
    
    

# 示例调用
if __name__ == "__main__":
    ticker = '600036.SS'
    file_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/000001.csv'  # 替换为你的路径

    analyzer = StockAnalyzer(ticker, file_path)
    ma = analyzer.calculate_moving_averages()
    bbi = analyzer.calculate_bbi()
    kdj = analyzer.calculate_kdj()
    macd = analyzer.calculate_macd()
    price = analyzer.calculate_price()
    shakeout = analyzer.calculate_shakeout()
    analyzer.plot_all(ma, bbi, price, macd, kdj, shakeout, '000001', windows=[20, 30, 60, 120])
