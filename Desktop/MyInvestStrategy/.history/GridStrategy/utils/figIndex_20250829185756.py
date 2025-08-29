# -*- coding: utf-8 -*-
"""
功能清单
1) 用 stock_zh_index_spot_em 下载目录（上证/深证/中证/国证），只输出你关心的指数
2) 下载“指定指数”的历史数据（日线，东财接口）
3) 绘制：中证全指(000985) PE × 深综指(399106)（双轴）
4) 绘制：全A PB × 深综指（PB用申万“市场表征-全A”近似，双轴）
5) 绘制：中证全指（PE或PB就近）× 深综指（双轴）
6) 绘制：你关心的“行业指数” PE、PB（优先中证估值，缺失则用申万月报）
7) 绘制：你关心的“宽基指数” PE、PB（中证估值为主；海外/港股若无估值则跳过估值，仅保留点位对比提示）

沪深300（上证000300）、上证180（上证000010）、深证100（深证399330）、科创50（上证000688）、创业板指（深证399006）、上证50（上证000016）、中证500（上证000905）、中证1000（上证000852）、恒生科技、恒生医疗、标普500、恒生、纳斯达克；
行业我只关心：全指医药（上证000991）、全指金融（上证000992）、全指消费（上证000990）、中证环保（上证000827）、全指信息（上证000993）、中证医疗（深证399989）、食品饮料（上证000807）、中证红利（上证000922）、中证军工（深证399967）、中证传媒（深证399971）、中国互联

输出：./output/ 下若干 CSV 与 PNG

1、下载数据
"""
# utils/figIndex.py 顶部
from pathlib import Path
import sys
# 把“GridStrategy 的父目录（MyInvestStrategy）”加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GridStrategy import getData, config
import warnings, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
from GridStrategy import getData, config
import ssl, certifi, urllib.request

_ctx = ssl.create_default_context(cafile=certifi.where())  # 用 certifi 的 CA 列表
_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
urllib.request.install_opener(_opener)  # 让 pandas/urllib 用这个带 CA 的 opener
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial Unicode MS','Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    index_list = config.index_list
    downloadIndexNewDataSwitch = config.downloadIndexNewDataSwitch
    stock_start_date = config.stock_start_date
    end_date = config.end_date
    index_save_path = '/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/indexdata/'

     # 下载最新数据并保存成csv文件
    if downloadIndexNewDataSwitch:
        for x in index_list:
            index_data = getData.download_csindex_data(x, start_date=stock_start_date, end_date=end_date)
            for key, value in index_data.items():
                getData.save_2_csv(value, key, index_save_path)
    