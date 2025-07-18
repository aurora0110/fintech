import config
import getData
import calMA as cm
import calExtremaPoints as cep
import calAvgPoints as cap
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
        print(data.tail())

        # 计算极值 1 2 3 4 5
        extremaPoints = cep.one_years_extrema_points(file_path)

        # 计算平均值 1 2 3 4 5
        extremaPoints = cap.x_years_avg_points(file_path,3)

        # 计算MA bbi均线并画图
        data_ma = cm.calculate_moving_averages(data, "2022-06-12", "2025-06-12", windows)
        # cm.plot_moving_averages(data_ma, symbol, colors, 'MA')
        data_bbi = cm.calculate_bbi(data, "2022-06-12", "2025-06-12")
        # cm.plot_moving_averages(data_bbi, symbol, [colors[0]], 'BBI')


