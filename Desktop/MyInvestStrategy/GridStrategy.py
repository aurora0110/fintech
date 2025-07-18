import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# 设置参数
TICKER = "BTC-USD"  # 示例为比特币，可替换为股票代码如"TSLA"
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
WINDOW_SIZE = 20     # 布林带周期
MA_PERIOD = 50       # 移动平均线周期

# 获取历史数据
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

# 计算技术指标
def calculate_indicators(df):
    # 布林带
    bb = BollingerBands(df["Close"], window=WINDOW_SIZE, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()  # 将二维数组转换为一维
    df["bb_middle"] = bb.bollinger_mavg()  # 将二维数组转换为一维
    df["bb_lower"] = bb.bollinger_lband()  # 将二维数组转换为一维
    
    # 移动平均线
    df["ma"] = SMAIndicator(df["Close"], window=MA_PERIOD).sma_indicator()
    return df

# 识别水平支撑/阻力位（简化方法）
def find_key_levels(df, lookback=30):
    # 支撑位：过去lookback天内的最低点
    df["support"] = df["Low"].rolling(lookback).min()
    
    # 阻力位：过去lookback天内的最高点
    df["resistance"] = df["High"].rolling(lookback).max()
    return df

# 可视化
def plot_chart(df):
    plt.figure(figsize=(16, 8))
    
    # 绘制K线图
    plt.plot(df.index, df["Close"], label="Close Price", color="black", alpha=0.7)
    
    # 布林带
    plt.fill_between(df.index, df["bb_upper"], df["bb_lower"], color="orange", alpha=0.2, label="Bollinger Bands")
    plt.plot(df.index, df["bb_middle"], linestyle="--", color="orange", alpha=0.5, label="BB Middle (20MA)")
    
    # 移动平均线
    plt.plot(df.index, df["ma"], color="blue", linestyle="--", label=f"{MA_PERIOD}MA")
    
    # 支撑/阻力位
    plt.plot(df.index, df["support"], color="green", linestyle="-.", label="Support (30-day low)")
    plt.plot(df.index, df["resistance"], color="red", linestyle="-.", label="Resistance (30-day high)")
    
    # 标注关键点
    plt.title(f"{TICKER} Price with Support/Resistance Levels")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 主流程
if __name__ == "__main__":
    # 获取数据
    df = fetch_data(TICKER, START_DATE, END_DATE)
    
    # 计算指标
    df = calculate_indicators(df)
    df = find_key_levels(df)
    
    # 可视化
    plot_chart(df)