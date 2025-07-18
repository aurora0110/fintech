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
    symbol = file_path[-18:-12] # symbol 调用em东方财富接口不用加前缀，调用sina新浪接口要加上市场前缀 sh sz
    symbol_list = config.symbol_list

    # 获取当前时间
    now = datetime.now()
    now_date = now.strftime("%Y%m%d")

    # 下载数据
    category_name = "全市场etf目录" + now_date

    # 读取数据
    data = getData.read_from_csv(file_path)

    # 计算极值 1 2 3 4 5
    extremaPoints = cep.one_years_extrema_points(file_path)

    # 计算平均值 1 2 3 4 5
    extremaPoints = cap.x_years_avg_points(file_path,3)

    # 计算MA bbi均线并画图
    data_ma = cm.calculate_moving_averages(data, "2022-06-12", "2025-06-12", windows)
    cm.plot_moving_averages(data_ma, symbol, colors)
    data_bbi = cm.calculate_bbi(data, "2022-06-12", "2025-06-12")
    cm.plot_moving_averages(data_bbi, symbol, [colors[0]])


