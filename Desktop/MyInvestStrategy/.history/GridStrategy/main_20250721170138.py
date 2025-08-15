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
from StockAnalyzer import StockAnalyzer
from StockAnalyzer import StockMonitor
from utils import holdingConfig

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

        analyzer = StockAnalyzer(symbol, file_path)
        data_ma = analyzer.calculate_moving_averages()
        data_bbi = analyzer.calculate_bbi()
        data_kdj = analyzer.calculate_kdj()
        data_macd = analyzer.calculate_macd()
        data_price = analyzer.calculate_price()
        data_shakeout = analyzer.calculate_shakeout()
        # 画图
        if symbol == 'sh512980':
            figSwitch = True
        else:
            figSwitch = False
        if figSwitch:
            analyzer.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, symbol, windows=[20, 30, 60, 120])

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_input = []
        data_input.extend(data_kdj.get('ret_kdj')[1]) # low 5
        data_input.extend(data_kdj.get('ret_kdj')[-4]) # high 80
        data_back = bt.backTest(data_input, amount, ineterval_days, total_shares, each_buy_shares, etf_start_date, end_date, backtest_log_path_new)
        if data_back['avg_profit'] > 0.1:
            well_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])
        else:
            ordinary_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])

        if data_kdj['J'].iloc[-1] <= J_threshold:
            J_boolean = True
        
        if data_kdj['J'].iloc[-1] >= 90 and symbol in holding_stock_codes:
            select_list_J_sell.append(symbol)
        
        if data_shakeout['白线下20买'].iloc[-1] == 1:
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
        print(f"💹技术指标：J值小于{J_threshold}：{'true✅' if J_boolean else 'false❌'}，MACD指标：DIF水上：{'true✅' if MACD_boolean else 'false❌'}，单针下20短期指标小于20且长期指标大于60：{'true✅' if SHAKEOUT_boolean else 'false❌'}，最近连续{bbi_days}天的收盘价格大于bbi：{'true✅' if BBI_boolean else 'false❌'}")
        print("🐤" * 90)

    # 回测策略年化收益大于5%
    stock_well_list = []
    stock_ordinary_list = []
    stock_select_list_J = []
    stock_select_list_J_sell = []
    stock_select_list_JS = []
    stock_select_list_JSBBI = []
    stock_select_list_JM = []
    stock_select_list_S = [] 
    stock_fast_down_j_list = []
    stock_2days_shakeout_list = []
    stock_5days_shakeout_list = []

    # 计算stock
    for symbol in stock_symbol_list:
        J_boolean = False
        SHAKEOUT_boolean = False
        BBI_boolean = False
        MACD_boolean = False
        FALLBBI_signal = False
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

        analyzer = StockAnalyzer(symbol, file_path)
        data_ma = analyzer.calculate_moving_averages()
        data_bbi = analyzer.calculate_bbi()
        data_kdj = analyzer.calculate_kdj()
        data_macd = analyzer.calculate_macd()
        data_price = analyzer.calculate_price()
        data_shakeout = analyzer.calculate_shakeout()
        # 画图
        if figSwitch:
            analyzer.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, data_shakeout, symbol, windows=[20, 30, 60, 120])

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_input = []
        data_input.extend(data_kdj.get('ret_kdj')[1])
        data_input.extend(data_kdj.get('ret_kdj')[-4])
        data_back = bt.backTest(data_input, amount, ineterval_days, total_shares, each_buy_shares, stock_start_date, end_date, backtest_log_path_new)
        if data_back['avg_profit'] > 0.1:
            stock_well_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])
        else:
            stock_ordinary_list.append([symbol, f"{round(data_back['avg_profit'] * 100, 3)}%"])

        if data_kdj['J'].iloc[-1] <= J_threshold:
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
        
        if SHAKEOUT_boolean:
            stock_select_list_S.append(symbol)

        if J_boolean and MACD_boolean:
            stock_select_list_JM.append(symbol)
        
        if J_boolean and SHAKEOUT_boolean and BBI_boolean:
            stock_select_list_JSBBI.append(symbol)
    
        if J_boolean and SHAKEOUT_boolean:
            stock_select_list_JS.append(symbol)
        
        Jmonitor = StockMonitor(symbol, file_path).fastdown_J()
        ShakeOut_Monitor = StockMonitor(symbol, file_path).continuous_shakeout()
        Frequency_Monitor = StockMonitor(symbol, file_path).check_signal_frequency()

        if Jmonitor:
            stock_fast_down_j_list.append(symbol)
        if ShakeOut_Monitor:
            stock_2days_shakeout_list.append(symbol)
        if Frequency_Monitor:
            stock_5days_shakeout_list.append(symbol)

        with open(backtest_log_path_new, 'a') as f:
            f.write(f'*************当前回测策略为：可投入金额为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手*************')    
        print(f"⏰今日：{data.iloc[-1]['日期']}，{symbol}，收盘价为：{data.iloc[-1]['收盘']}，最高价为：{data.iloc[-1]['最高']}，最低价为：{data.iloc[-1]['最低']}，J值为：{round(data_kdj['J'].iloc[-1],3)}，MACD值为：{round(data_macd['DIF'].iloc[-1],3)}，单针下20短期指标为：{round(data_shakeout['短期'].iloc[-1],3)}，单针下20长期指标为：{round(data_shakeout['长期'].iloc[-1],3)}")
        print(f"💹技术指标：J值小于{J_threshold}：{'true✅' if J_boolean else 'false❌'}，MACD指标：DIF水上：{'true✅' if MACD_boolean else 'false❌'}，单针下20短期指标小于20且长期指标大于60：{'true✅' if SHAKEOUT_boolean else 'false❌'}，最近连续{bbi_days}天的收盘价格大于bbi：{'true✅' if BBI_boolean else 'false❌'}，3天内J快速下降：{'true✅' if fast_down_j_label else 'false❌'}，J值快速下降监控：{'true✅' if Jmonitor else 'false❌'}，连续洗盘信号：{'true✅' if ShakeOut_Monitor else 'false❌'}")
        print("🐤" * 95)

    print("💗" * 40, "ETF 今日数据如下", "💗" * 40)
    #print(f"ETF当前回测策略为：可投入金额💰为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手")
    #print(f"✅ETF回测策略年化收益大于1️⃣0️⃣%有{len(well_list)}个：{well_list}，分别为：{well_list}")
    #print(f"ETF回测策略年化收益小于1️⃣0️⃣%有{len(ordinary_list)}个：{ordinary_list}，分别为：{ordinary_list}")
    print(f"✅ETF当日满足J值小于{J_threshold}的ETF有{len(select_list_J)}个：{select_list_J}，❗️持有且大于9️⃣0️⃣的有{len(select_list_J_sell)}个：{select_list_J_sell}")
    print(f"ETF当日满足J值小于{J_threshold}的ETF,且MACD水上💦的有{len(select_list_JM)}个：{select_list_JM}")
    print(f"✅ETF当日满足J值小于{J_threshold}，单针下20短期指标小于20且长期指标大于60的ETF有{len(select_list_JS)}个：{select_list_JS}")
    print(f"ETF当日满足J值小于{J_threshold}，单针下20短期指标小于20且长期指标大于60，最近连续{bbi_days}天的收盘价格大于bbi的ETF有{len(select_list_JSBBI)}个：{select_list_JSBBI}")
    print("💗" * 40, "STOCK 今日数据如下", "💗" * 40)
    #print(f"STOCK当前回测策略为：可投入金额💰为{amount}元，最小操作间隔为{ineterval_days}天，计划操作手数为{total_shares}手")
    #print(f"✅STOCK回测策略年化收益大于1️⃣0️⃣%有{len(stock_well_list)}个：{stock_well_list}，分别为：{stock_well_list}")
    #print(f"STOCK回测策略年化收益小于1️⃣0️⃣%有{len(stock_ordinary_list)}个：{stock_ordinary_list}，分别为：{stock_ordinary_list}")   
    print(f"✅STOCK当日满足J值小于{J_threshold}的有{len(stock_select_list_J)}个：{stock_select_list_J}，❗️持有且大于9️⃣0️⃣的有{len(stock_select_list_J_sell)}个：{stock_select_list_J_sell}，⬇️单针下20信号的有{len(stock_select_list_S)}个:{stock_select_list_S}，J值快速下降的有{len(stock_fast_down_j_list)}个：{stock_fast_down_j_list}",)
    print(f"STOCK当日满足J值小于{J_threshold}的,且MACD水上💦的有{len(stock_select_list_JM)}个：{stock_select_list_JM}")
    print(f"✅STOCK当日满足J值小于{J_threshold}，单针下20短期指标小于20且长期指标大于60的有{len(stock_select_list_JS)}个：{stock_select_list_JS}")
    print(f"STOCK当日满足J值小于{J_threshold}，单针下20短期指标小于20且长期指标大于60，最近连续{bbi_days}天的收盘价格大于bbi的有{len(stock_select_list_JSBBI)}个：{stock_select_list_JSBBI}")
