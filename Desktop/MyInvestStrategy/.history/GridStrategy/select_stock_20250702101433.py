import config
import getData
import calMA as cm
import calExtremaPoints as cep
import calAvgPoints as cap
import calKDJ
import calRSV
import calMACD
import calTradingRange as ctr
from datetime import datetime
import logging
import pandas as pd
import sys
import backTest as bt
import shakeoutMonitoring as som

'''
sfzf ➕ 单针下20 ➕ bbi
'''

if __name__ == '__main__':
    # 读取文件路径
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    symbol_list = config.symbol_list
    start_date = config.start_date
    end_date = config.end_date

    # 获取当前时间
    now = datetime.now()
    now_date = now.strftime("%Y%m%d")

    for symbol in symbol_list:
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        print(f"读取文件：{file_path}")

        # 读取数据
        data = getData.read_from_csv(file_path)
        print(f'今日：{data.iloc[-1]["日期"]}，{symbol}，收盘价为：{data.iloc[-1]["收盘"]}，最高价为：{data.iloc[-1]["最高"]}，最低价为：{data.iloc[-1]["最低"]}')
