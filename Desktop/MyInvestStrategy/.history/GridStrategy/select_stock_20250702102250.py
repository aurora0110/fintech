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

    J_boolean = True
    SHAKEOUT_boolean = True
    BBI_boolean = True

    for symbol in symbol_list:
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        print(f"读取文件：{file_path}")

        # 读取数据
        data = getData.read_from_csv(file_path)
        print(f'今日：{data.iloc[-1]["日期"]}，{symbol}，收盘价为：{data.iloc[-1]["收盘"]}，最高价为：{data.iloc[-1]["最高"]}，最低价为：{data.iloc[-1]["最低"]}')

        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3, start_date, end_date)
        # 监控洗盘
        data_shakeout= som.monitor(data, start_date, end_date)
        # 计算bbi
        data_bbi = calBBI.cal_BBI(data, 3, 6, 12, 24, start_date, end_date)

        if data_kdj['J'][-1] <= -5:
            J_boolean = True
        
        if data_shakeout['短期'][-1] < 20 and data_shakeout['长期'][-1] > 60:
            SHAKEOUT_boolean = True
        
