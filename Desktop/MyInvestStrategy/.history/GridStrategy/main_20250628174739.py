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

if __name__ == '__main__':
    # 读取文件路径
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    symbol_list = config.symbol_list
    start_date = config.start_date
    end_date = config.end_date

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
    category_name = "全市场etf目录" + now_date
    #data_new = getData.batch_download_etf_data(symbol_list, "all")
    #for key, value in data_new.items():
        #getData.save_2_csv(value, key)

    for symbol in symbol_list:
        file_path = config.file_path 
        file_path = file_path + symbol + ".csv"
        print(f"读取文件：{file_path}")

        # 读取数据
        data = getData.read_from_csv(file_path)
        print(f'今日：{data.iloc[-1]["日期"]}，{symbol}，收盘价为：{data.iloc[-1]["收盘"]}，最高价为：{data.iloc[-1]["最高"]}，最低价为：{data.iloc[-1]["最低"]}')

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
        data_ma = cm.calculate_moving_averages(data, "2024-06-12", "2025-06-12", windows)
        # cm.plot_moving_averages(data_ma, symbol, colors, 'MA')
        data_bbi = cm.calculate_bbi(data, "2024-06-12", "2025-06-12")
        # cm.plot_moving_averages(data_bbi, symbol, [colors[0]], 'BBI')
        data_price = cm.calculate_price(data, "2024-06-12", "2025-06-12")
        # 计算MACD
        data_macd = calMACD.cal_macd(data, "2024-06-12", "2025-06-12", 12, 26, '收盘')
        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3, "2024-06-12", "2025-06-12")
        cm.plot_all(data_ma, data_bbi, data_price, data_macd, data_kdj, symbol, windows)

        # 计算回测收益，策略：每到j值满足条件就买入或者卖出
        data_back_list1 = []
        data_back_list2 = []
        data_input = []
        data_back_list1.append(data_kdj.get('ret_kdj')[1]) # 拿到j < -5的值
        data_back_list2.append(data_kdj.get('ret_kdj')[-5]) # 拿到j > 80的值
        data_input.extend(data_back_list1)
        data_input.extend(data_back_list2)
        
        data_back = bt.backTest(data_input, 200000, 30, 10, '2020-06-12', '2025-06-12')

        print('***********' * 10)


