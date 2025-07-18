import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.subplots as sp
import plotly.graph_objs as go
from pathlib import Path


class StockAnalyzer:
    def __init__(self, ticker, file_path, start_date=None, end_date=None, kdj_days = 9, kdj_m1=3, kdj_m2=3):
        self.ticker = ticker
        self.file_path = file_path
        self.end_date = pd.to_datetime(end_date or datetime.now().strftime('%Y-%m-%d'))
        self.start_date = pd.to_datetime(start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.stock_data = self.get_data()

        self.data_ma = {}
        self.data_bbi = {}
        self.data_price = {}
        self.data_macd = {}
        self.data_kdj = {}
        self.data_shakeout = {}

        self.kdj_days = kdj_days
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2
    
    def calculate_all_indicators(self):
        self.calculate_moving_averages()
        self.calculate_bbi()
        self.calculate_price()
        self.calculate_macd()
        self.calculate_kdj()
        self.calculate_shakeout()


    def get_data(self):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件{path}不存在")
    
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"读取文件{path}失败，错误信息：{e}")
        return pd.read_csv(path, encoding="utf-8")
        data = pd.read_excel(self.file_path, engine=None)
        data = data[(data['日期'] >= self.start_date) & (data['日期'] <= self.end_date)]
        return data

    def calculate_kdj(self):
        # 确保日期列是 datetime 类型
        self.stock_data['日期'] = pd.to_datetime(self.stock_data['日期'])

        # 将输入的日期也转为 datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)  # 示例结束日期

        # 使用 datetime 比较
        self.stock_data = self.stock_data[(self.stock_data['日期'] >= start_date) & (self.stock_data['日期'] <= end_date)]
        self.stock_data['high_kdj'] = self.stock_data['最高'].rolling(window = self.kdj_days, min_periods = 1).max()
        self.stock_data['low_kdj'] = self.stock_data['最低'].rolling(window = self.kdj_days, min_periods = 1).min()
        self.stock_data['RSV'] = (self.stock_data['收盘'] - self.stock_data['low_kdj']) / (self.stock_data['high_kdj'] - self.stock_data['low_kdj']) * 100
        self.stock_data['K'] = self.stock_data['RSV'].ewm(alpha=1/self.kdj_m1, adjust=False).mean()
        self.stock_data['D'] = self.stock_data['K'].ewm(alpha=1/self.kdj_m2, adjust=False).mean()
        self.stock_data['J'] = 3 * self.stock_data['K'] - 2 * self.stock_data['D']

        data_list = self.stock_data[['日期','J', '收盘', '最低', '最高']].to_numpy().tolist()
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
            'K': self.stock_data['K'],
            'D': self.stock_data['D'],
            'J': self.stock_data['J'],
            'ret_kdj': ret_kdj
        }
        return result_kv

    def calculate_moving_averages(self, windows=[20, 60, 120]):
        result = {}
        for window in windows:
            result[f'MA_{window}'] = self.stock_data['收盘'].rolling(window=window).mean()
        result['date'] = self.stock_data['日期']
        self.ma_data = pd.DataFrame(result)
        return self.ma_data

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
    analyzer.calculate_moving_averages()
    analyzer.plot_moving_averages()
    analyzer.save_moving_averages()
