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

只做了解的票，找到了合适的就重复抡

变量：J值；单针下20的短长期值；BBI维持上涨的天数
'''

if __name__ == '__main__':
    # 读取文件路径
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    stock_symbol_list = config.stock_symbol_list
    stock_start_date = config.stock_start_date
    end_date = config.end_date

    J_boolean = True
    SHAKEOUT_boolean = True
    BBI_boolean = True

    for symbol in stock_symbol_list:
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        print(f"读取文件：{file_path}")

        # 读取数据
        data = getData.read_from_csv(file_path)
        print(f'今日：{data.iloc[-1]["日期"]}，{symbol}，收盘价为：{data.iloc[-1]["收盘"]}，最高价为：{data.iloc[-1]["最高"]}，最低价为：{data.iloc[-1]["最低"]}')

        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3, stock_start_date, end_date)
        # 监控洗盘
        data_shakeout= som.monitor(data, stock_start_date, end_date)
        # 计算bbi
        data_bbi = cm.calculate_bbi(data, stock_start_date, end_date)

        if data_kdj['J'].iloc[-1] <= -5:
            J_boolean = True
        
        if data_shakeout['短期'].iloc[-1] < 20 and data_shakeout['长期'].iloc[-1] > 60:
            SHAKEOUT_boolean = True
        
        # 比较价格满足大于bbi，并且最近10天的收盘价格大于bbi
        days = 10
        bbi_last = data_bbi.tail(days).reset_index(drop=True)
        price_last = data['收盘'].tail(days).reset_index(drop=True)

        condition = (price_last > bbi_last).sum()
        if condition == days:
            BBI_boolean = True
            print("满足：价格大于bbi，并且最近10天的收盘价格大于bbi")
        else:
            print("不满足：价格大于bbi，并且最近10天的收盘价格大于bbi")


        
