import config
import getData
from datetime import datetime
import logging
import pandas as pd
import sys
#import backTest as bt
import shakeoutMonitoring as som
import os
from StockAnalyzer import StockAnalyzer
from StockAnalyzer import StockMonitor
from utils import holdingConfig
from tqdm import tqdm

if __name__ == '__main__':
    # 读取参数
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    etf_symbol_list = config.etf_symbol_list
    stock_symbol_list = config.stock_symbol_list
    all_stock_symbol_list = config.all_stock_symbol_list
    etf_start_date = config.etf_start_date
    stock_start_date = config.stock_start_date
    end_date = config.end_date
    amount = config.amount
    ineterval_days = config.ineterval_days
    total_shares = config.total_shares
    each_buy_shares = config.each_buy_shares
    backtest_log_path = config.backtest_log_path
    downloadNewDataSwitch = config.downloadNewDataSwitch
    figSwitch = config.figSwitch
    bbi_days = config.bbi_days
    J_threshold = config.J_threshold
    holding_stock_codes = holdingConfig.stock_codes
    volatility = config.volatilitySwitch
    categorySwitch = config.categorySwitch
    category_name = config.category_name
    all_etf_codes = config.all_etf_codes

    logger = logging.getLogger("myLogger")
    # 关闭特定的警告
    pd.options.mode.chained_assignment = None  # 默认='warn'
    # 获取当前时间
    now = datetime.now()
    now_date = now.strftime("%Y%m%d")

    # 下载全市场股票目录
    if categorySwitch:
        df = getData.download_stock_category()
        getData.save_2_csv(df, category_name, file_path)
    # 下载最新数据并保存成csv文件
    if downloadNewDataSwitch:
        data_new = getData.batch_download_etf_data(all_etf_codes, "all", etf_start_date, end_date, 5)
        for key, value in data_new.items():
            getData.save_2_csv(value, key, file_path)

    # 回测策略年化收益大于5%
    well_list = []
    ordinary_list = []
    select_list_J = []
    select_list_J_sell = []
    select_list_JS = []
    select_list_JSBBI = []
    select_list_JM = []
    select_list_S = [] 
    fast_down_j_list = []
    etf_2days_shakeout_list = []
    etf_5days_shakeout_list = []
    bs_vol_price_list = []
    below_bbi_list = []
    holding_codes = []
    # 计算ETF
    for symbol in tqdm(all_etf_codes):
        J_boolean = False
        SHAKEOUT_boolean = False
        BBI_boolean = False
        MACD_boolean = False
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        backtest_log_path_new = backtest_log_path + symbol + ".txt"
        everyday_abnormal_volume_path = backtest_log_path + "abnormal_volume.txt"
        # 读取数据
        print(f"📃读取文件：{file_path}\n回测结果保存路径：{backtest_log_path_new}")

        # 读取数据
        try:
            data = getData.read_from_csv(file_path)
        except FileNotFoundError:
            print(f"[skip] 文件不存在：{file_path}")
            continue

        analyzer = StockAnalyzer(symbol, file_path)
        data_ma = analyzer.calculate_moving_averages()
        data_bbi = analyzer.calculate_bbi()
        data_kdj = analyzer.calculate_kdj()
        data_macd = analyzer.calculate_macd()
        # 画图
        if figSwitch:
            analyzer.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, symbol, windows=[20, 30, 60, 120])

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_input = []
        data_input.extend(data_kdj.get('ret_kdj')[1]) # low 5
        data_input.extend(data_kdj.get('ret_kdj')[-4]) # high 80


        if data_kdj['J'].iloc[-1] <= J_threshold:
            J_boolean = True
                
        # 筛选MACD在水上的
        if data_macd['DIF'].iloc[-1] > 0:
            MACD_boolean = True
        # 比较价格满足大于bbi，并且最近10天的收盘价格大于bbi
        bbi_last = data_bbi['bbi'].tail(bbi_days).reset_index(drop=True)
        price_last = data['收盘'].tail(bbi_days).reset_index(drop=True)
        condition = (price_last > bbi_last).sum()
        if condition == bbi_days:
            BBI_boolean = True
        
        # 读取全市场股票代码和对应名字
        pd_file = getData.read_from_csv("/Users/lidongyang/Desktop/MyInvestStrategy/GridStrategy/data/全市场etf目录20250612.csv")
        # 读取的代码左侧缺0，补0
        #pd["code"] = pd["code"].fillna(0).astype(int).astype(str).str.zfill(6)
        #pd_dict = pd.set_index("code")["name"].to_dict()
        pd_dict = pd_file.set_index("代码")["名称"].to_dict()
        
        if J_boolean:
            select_list_J.append([symbol, pd_dict[symbol]])
        
        if J_boolean and MACD_boolean:
            select_list_JM.append(symbol)
        
        print(f"⏰今日：{data.iloc[-1]['日期']}，{symbol}，收盘价为：{data.iloc[-1]['收盘']}，最高价为：{data.iloc[-1]['最高']}，最低价为：{data.iloc[-1]['最低']}，J值为：{round(data_kdj['J'].iloc[-1],3)}，MACD值为：{round(data_macd['DIF'].iloc[-1],3)}，单针下20短期指标为：{round(data_shakeout['短期'].iloc[-1],3)}，单针下20长期指标为：{round(data_shakeout['长期'].iloc[-1],3)}")
        print(f"💹技术指标：J值小于{J_threshold}：{'true✅' if J_boolean else 'false❌'}，MACD指标：DIF水上：{'true✅' if MACD_boolean else 'false❌'}，单针下20短期指标小于20且长期指标大于60：{'true✅' if SHAKEOUT_boolean else 'false❌'}，最近连续{bbi_days}天的收盘价格大于bbi：{'true✅' if BBI_boolean else 'false❌'}")
        print(f"J值小于阈值{J_threshold}的ETF有：{select_list_J}")
        print("🐤" * 90)
 