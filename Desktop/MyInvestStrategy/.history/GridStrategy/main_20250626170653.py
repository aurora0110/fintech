import config
import getData
import calMA as cm
import calExtremaPoints as cep
import calAvgPoints as cap
import calKDJ
import calRSV
import calTradingRange as ctr
from datetime import datetime

if __name__ == '__main__':
    # 读取文件路径
    file_path = config.file_path 
    windows = config.windows
    colors = config.colors
    symbol_list = config.symbol_list

    import pandas as pd

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

        # 计算kdj
        data_kdj = calKDJ.cal_KDJ(data, 9, 3, 3)
        print(f'{data.iloc[-1]["日期"]}至{data.iloc[0]["日期"]}\nj值小于等于0有{len(data_kdj[0])}个\nj值小于等于-5有{len(data_kdj[1])}个\nj值小于等于-8有{len(data_kdj[2])}个\nj值小于等于-10有{len(data_kdj[3])}个\nj值小于等于-15有{len(data_kdj[4])}个\nj值小于等于-20有{len(data_kdj[5])}个\nj值小于等于-25有{len(data_kdj[6])}个\nj值大于等于80有{len(data_kdj[7])}个\nj值大于等于90有{len(data_kdj[8])}个\nj值大于等于100有{len(data_kdj[9])}个\nj值大于等于110有{len(data_kdj[10])}个\nj值大于等于120有{len(data_kdj[11])}个')

        # 计算rsv
        data_rsv = calRSV.calrsv(data, 9)

        # 计算时间窗口内的价格波动幅度
        day_window = 100
        range = ctr.cal_range(data, day_window)
        std_roll_values, std_values = ctr.cal_volatility(data, day_window)
        atr = ctr.cal_ATR
        (data, day_window)

        # 计算MA bbi均线并画图
        data_ma = cm.calculate_moving_averages(data, "2022-06-12", "2025-06-12", windows)
        # cm.plot_moving_averages(data_ma, symbol, colors, 'MA')
        data_bbi = cm.calculate_bbi(data, "2022-06-12", "2025-06-12")
        # cm.plot_moving_averages(data_bbi, symbol, [colors[0]], 'BBI')

        print('*********' * 10)


