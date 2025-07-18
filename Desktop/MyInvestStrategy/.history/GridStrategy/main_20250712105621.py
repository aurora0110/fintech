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
import os
from utils import holdingConfig

if __name__ == '__main__':
    # 读取参数
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    etf_symbol_list = config.etf_symbol_list
    stock_symbol_list = config.stock_symbol_list
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
    holding_stock_codes = holdingConfig.stock_codes

    if not os.path.exists(backtest_log_path):
        os.makedirs(backtest_log_path)

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
    logger = logging.getLogger("myLogger")

    # 关闭特定的警告
    pd.options.mode.chained_assignment = None  # 默认='warn'
    # 获取当前时间
    now = datetime.now()
    now_date = now.strftime("%Y%m%d")

    # 下载最新数据并保存成csv文件

    if downloadNewDataSwitch:
        category_name = "全市场etf目录" + now_date
        data_new = getData.batch_download_etf_data(etf_symbol_list, "all", etf_start_date, end_date, 5)
        for key, value in data_new.items():
            getData.save_2_csv(value, key, file_path)

        stock_data = getData.batch_download_stock_data(stock_symbol_list, days="all", start_date=stock_start_date, end_date=end_date, year_interval=1)
        for key, value in stock_data.items():
            getData.save_2_csv(value, key, file_path)


    # 回测策略年化收益大于5%
    well_list = []
    ordinary_list = []
    select_list_J = []
    select_list_J_sell = []
    select_list_JS = []
    select_list_JSBBI = []
    select_list_JM = []

    # 计算ETF
    for symbol in etf_symbol_list:
        J_boolean = False
        SHAKEOUT_boolean = False
        BBI_boolean = False
        MACD_boolean = False
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        backtest_log_path_new = backtest_log_path + symbol + ".txt"
        # 读取数据
        print(f"📃读取文件：{file_path}\n回测结果保存路径：{backtest_log_path_new}")

        # 读取数据
        data = getData.read_from_csv(file_path)

        # 计算极值 1 2 3 4 5
        extremaPoints = cep.one_years_extrema_points(file_path)
        # 计算平均值 1 2 3 4 5
        extremaPoints = cap.x_years_avg_points(file_path,3)

        # 计算rsv
        data_rsv = calRSV.calrsv(data, 9)

        # 计算时间窗口内的价格波动幅度
        day_window = 100
        range = ctr.cal_range(data, day_window)
        ctr.cal_volatility(data, day_window)
        atr = ctr.cal_ATR(data, day_window)

        # 计算MA bbi均线并画图
        data_ma = cm.calculate_moving_averages(data, etf_start_date, end_date, windows)
        # cm.plot_moving_averages(data_ma, symbol, colors, 'MA')
        data_bbi = cm.calculate_bbi(data, etf_start_date, end_date)
        # cm.plot_moving_averages(data_bbi, symbol, [colors[0]], 'BBI')
        data_price = cm.calculate_price(data, etf_start_date, end_date)
        # 计算MACD
        data_macd = calMACD.cal_macd(data, etf_start_date, end_date, 12, 26, '收盘')
        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3, etf_start_date, end_date)
        # 监控洗盘
        data_shakeout= som.monitor(data, etf_start_date, end_date)
        # 画图
        if figSwitch:
            cm.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, symbol, windows)

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_input = []
        data_input.extend(data_kdj.get('ret_kdj')[1]) # low 5
        data_input.extend(data_kdj.get('ret_kdj')[-5]) # high 80
        data_back = bt.backTest(data_input, amount, ineterval_days, total_shares, each_buy_shares, etf_start_date, end_date, backtest_log_path_new)
        if data_back['avg_profit'] > 0.1:
            well_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])
        else:
            ordinary_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])

        if data_kdj['J'].iloc[-1] <= -5:
            J_boolean = True
        
        if data_kdj['J'].iloc[-1] >= 90 and symbol in holding_stock_codes:
            select_list_J_sell.append(symbol)
        
        if data_shakeout['短期'].iloc[-1] < 20 and data_shakeout['长期'].iloc[-1] > 60:
            SHAKEOUT_boolean = True
        
                
        # 筛选MACD在水上的
        if data_macd['DIF'].iloc[-1] > 0:
            MACD_boolean = True
        
        # 比较价格满足大于bbi，并且最近10天的收盘价格大于bbi
        bbi_last = data_bbi['bbi'].tail(bbi_days).reset_index(drop=True)
        price_last = data['收盘'].tail(bbi_days).reset_index(drop=True)
        condition = (price_last > bbi_last).sum()
        if condition == bbi_days:
            BBI_boolean = True
        
        if J_boolean:
            select_list_J.append(symbol)
        
        if J_boolean and MACD_boolean:
            select_list_JM.append(symbol)

        if J_boolean and SHAKEOUT_boolean and BBI_boolean:
            select_list_JSBBI.append(symbol)
    
        if J_boolean and SHAKEOUT_boolean:
            select_list_JS.append(symbol)

        with open(backtest_log_path_new, 'a') as f:
            f.write(f'*************当前回测策略为：可投入金额为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手*************')    
        print(f"⏰今日：{data.iloc[-1]['日期']}，{symbol}，收盘价为：{data.iloc[-1]['收盘']}，最高价为：{data.iloc[-1]['最高']}，最低价为：{data.iloc[-1]['最低']}，J值为：{round(data_kdj['J'].iloc[-1],3)}，MACD值为：{round(data_macd['DIF'].iloc[-1],3)}，单针下20短期指标为：{round(data_shakeout['短期'].iloc[-1],3)}，单针下20长期指标为：{round(data_shakeout['长期'].iloc[-1],3)}")
        print(f"💹技术指标：J值小于-5：{'true✅' if J_boolean else 'false❌'}，MACD指标：DIF水上：{'true✅' if MACD_boolean else 'false❌'}，单针下20短期指标小于20且单针下20长期指标大于60：{'true✅' if SHAKEOUT_boolean else 'false❌'}，最近连续{bbi_days}天的收盘价格大于bbi：{'true✅' if BBI_boolean else 'false❌'}")
        print("🐤" * 90)

    # 回测策略年化收益大于5%
    stock_well_list = []
    stock_ordinary_list = []
    stock_select_list_J = []
    stock_select_list_J_sell = []
    stock_select_list_JS = []
    stock_select_list_JSBBI = []
    stock_select_list_JM = []

    # 计算stock
    for symbol in stock_symbol_list:
        J_boolean = False
        SHAKEOUT_boolean = False
        BBI_boolean = False
        MACD_boolean = False
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        backtest_log_path_new = backtest_log_path + symbol + ".txt"
        # 读取数据
        print(f"读取文件：{file_path}，回测结果保存路径：{backtest_log_path_new}")

        # 读取数据
        data = getData.read_from_csv(file_path)

        # 计算极值 1 2 3 4 5
        extremaPoints = cep.one_years_extrema_points(file_path)
        # 计算平均值 1 2 3 4 5
        extremaPoints = cap.x_years_avg_points(file_path,3)

        # 计算rsv
        data_rsv = calRSV.calrsv(data, 9)

        # 计算时间窗口内的价格波动幅度
        day_window = 100
        range = ctr.cal_range(data, day_window)
        ctr.cal_volatility(data, day_window)
        atr = ctr.cal_ATR(data, day_window)

        # 计算MA bbi均线并画图
        data_ma = cm.calculate_moving_averages(data, stock_start_date, end_date, windows)
        # cm.plot_moving_averages(data_ma, symbol, colors, 'MA')
        data_bbi = cm.calculate_bbi(data, stock_start_date, end_date)
        # cm.plot_moving_averages(data_bbi, symbol, [colors[0]], 'BBI')
        data_price = cm.calculate_price(data, stock_start_date, end_date)
        # 计算MACD
        data_macd = calMACD.cal_macd(data, stock_start_date, end_date, 12, 26, '收盘')
        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3, stock_start_date, end_date)
        # 监控洗盘
        data_shakeout= som.monitor(data, stock_start_date, end_date)
        # 画图
        if figSwitch:
            cm.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, symbol, windows)

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_input = []
        data_input.extend(data_kdj.get('ret_kdj')[1])
        data_input.extend(data_kdj.get('ret_kdj')[-5])
        data_back = bt.backTest(data_input, amount, ineterval_days, total_shares, each_buy_shares, stock_start_date, end_date, backtest_log_path_new)
        if data_back['avg_profit'] > 0.1:
            stock_well_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])
        else:
            stock_ordinary_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])

        if data_kdj['J'].iloc[-1] <= -5:
            J_boolean = True

        if data_kdj['J'].iloc[-1] >= 90 and symbol in holding_stock_codes:
            stock_select_list_J_sell.append(symbol)
        
        if data_shakeout['短期'].iloc[-1] < 20 and data_shakeout['长期'].iloc[-1] > 60:
            SHAKEOUT_boolean = True
        
        # 筛选MACD在水上的
        if data_macd['DIF'].iloc[-1] > 0:
            MACD_boolean = True
        
        # 比较价格满足大于bbi，并且最近10天的收盘价格大于bbi
        bbi_last = data_bbi['bbi'].tail(bbi_days).reset_index(drop=True)
        price_last = data['收盘'].tail(bbi_days).reset_index(drop=True)
        condition = (price_last > bbi_last).sum()
        if condition == bbi_days:
            BBI_boolean = True
        
        if J_boolean:
            stock_select_list_J.append(symbol)
        
        if J_boolean and MACD_boolean:
            stock_select_list_JM.append(symbol)
        
        if J_boolean and SHAKEOUT_boolean and BBI_boolean:
            stock_select_list_JSBBI.append(symbol)
    
        if J_boolean and SHAKEOUT_boolean:
            stock_select_list_JS.append(symbol)

        with open(backtest_log_path_new, 'a') as f:
            f.write(f'*************当前回测策略为：可投入金额为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手*************')    
        print(f"⏰今日：{data.iloc[-1]['日期']}，{symbol}，收盘价为：{data.iloc[-1]['收盘']}，最高价为：{data.iloc[-1]['最高']}，最低价为：{data.iloc[-1]['最低']}，J值为：{round(data_kdj['J'].iloc[-1],3)}，MACD值为：{round(data_macd['DIF'].iloc[-1],3)}，单针下20短期指标为：{round(data_shakeout['短期'].iloc[-1],3)}，单针下20长期指标为：{round(data_shakeout['长期'].iloc[-1],3)}")
        print(f"💹技术指标：J值小于-5：{'true✅' if J_boolean else 'false❌'}，MACD指标：DIF水上：{'true✅' if MACD_boolean else 'false❌'}，单针下20短期指标小于20且单针下20长期指标大于60：{'true✅' if SHAKEOUT_boolean else 'false❌'}，最近连续{bbi_days}天的收盘价格大于bbi：{'true✅' if BBI_boolean else 'false❌'}")
        print("🐤" * 95)

    print("💗" * 40, "ETF 今日数据如下", "💗" * 40)
    print(f"ETF当前回测策略为：可投入金额💰为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手")
    print(f"✅ETF回测策略年化收益大于1️⃣0️⃣%有{len(well_list)}个：{well_list}，分别为：{well_list}")
    print(f"ETF回测策略年化收益小于1️⃣0️⃣%有{len(ordinary_list)}个：{ordinary_list}，分别为：{ordinary_list}")
    print(f"✅ETF当日满足J值小于-5️⃣的ETF有{len(select_list_J)}个：{select_list_J}，❗️持有且大于9️⃣0️⃣的有{len(select_list_J_sell)}个：{select_list_J_sell}")
    print(f"ETF当日满足J值小于-5️⃣的ETF,且MACD水上💦的有{len(select_list_JM)}个：{select_list_JM}")
    print(f"✅ETF当日满足J值小于-5️⃣，单针下20短期指标小于20且单针下20长期指标大于60的ETF有{len(select_list_JS)}个：{select_list_JS}")
    print(f"ETF当日满足J值小于-5️⃣，单针下20短期指标小于20且单针下20长期指标大于60，最近连续{bbi_days}天的收盘价格大于bbi的ETF有{len(select_list_JSBBI)}个：{select_list_JSBBI}")
    print("💗" * 40, "STOCK 今日数据如下", "💗" * 40)
    print(f"STOCK当前回测策略为：可投入金额💰为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手")
    print(f"✅STOCK回测策略年化收益大于1️⃣0️⃣%有{len(stock_well_list)}个：{stock_well_list}，分别为：{stock_well_list}")
    print(f"STOCK回测策略年化收益小于1️⃣0️⃣%有{len(stock_ordinary_list)}个：{stock_ordinary_list}，分别为：{stock_ordinary_list}")   
    print(f"✅STOCK当日满足J值小于-5️⃣的有{len(stock_select_list_J)}个：{stock_select_list_J}，❗️持有且大于9️⃣0️⃣的有{len(stock_select_list_J_sell)}个：{stock_select_list_J_sell}")
    print(f"STOCK当日满足J值小于-5️⃣的,且MACD水上💦的有{len(stock_select_list_JM)}个：{stock_select_list_JM}")
    print(f"✅STOCK当日满足J值小于-5️⃣，单针下20短期指标小于20且单针下20长期指标大于60的有{len(stock_select_list_JS)}个：{stock_select_list_JS}")
    print(f"STOCK当日满足J值小于-5️⃣，单针下20短期指标小于20且单针下20长期指标大于60，最近连续{bbi_days}天的收盘价格大于bbi的有{len(stock_select_list_JSBBI)}个：{stock_select_list_JSBBI}")

    '''

    j <-5 买，j >= 100卖：回测策略年化收益大于5%有18个：[['sh515450', '33.15%'], ['sh563300', '8.727%'], ['sh512580', '11.72%'], ['sh588000', '43.853%'], ['sz159985', '34.575%'], ['sh520990', '22.023%'], ['sh510300', '8.248%'], ['sh510050', '5.664%'], ['sh518880', '15.299%'], ['sh512660', '54.898%'], ['sh512100', '31.273%'], ['sh512170', '6.721%'], ['sh513180', '29.988%'], ['sz159920', '19.895%'], ['sh512980', '58.359%'], ['sh515180', '23.133%'], ['sh512880', '21.619%'], ['sh512070', '43.714%']]
    j <-5 买，j >= 90卖：回测策略年化收益大于5%有19个：[['sh515450', '23.494%'], ['sh563300', '8.368%'], ['sh512580', '21.007%'], ['sh588000', '46.655%'], ['sz159985', '24.288%'], ['sh520990', '18.806%'], ['sh510300', '10.157%'], ['sh510050', '10.294%'], ['sh518880', '12.595%'], ['sh512660', '52.128%'], ['sh512100', '17.886%'], ['sh512170', '27.299%'], ['sh513180', '36.293%'], ['sz159920', '28.681%'], ['sh512980', '33.322%'], ['sh515180', '19.142%'], ['sz159938', '16.125%'], ['sh512880', '33.652%'], ['sh512070', '33.672%']]
    j <-5 买，j >= 80卖：回测策略年化收益大于5%有19个：[['sh515450', '20.666%'], ['sh563300', '8.368%'], ['sh512580', '23.326%'], ['sh588000', '44.329%'], ['sz159985', '19.201%'], ['sh520990', '18.573%'], ['sh510300', '10.578%'], ['sh510050', '8.167%'], ['sh518880', '10.802%'], ['sh512660', '42.412%'], ['sh512100', '16.21%'], ['sh512170', '35.676%'], ['sh513180', '31.666%'], ['sz159920', '23.266%'], ['sh512980', '33.456%'], ['sh515180', '15.083%'], ['sz159938', '20.701%'], ['sh512880', '28.0%'], ['sh512070', '19.654%']]

    '''




