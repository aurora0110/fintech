import pandas as pd

def cal_KDJ(data, days, m1, m2):
    """
    计算KDJ指标
    :param df: pandas DataFrame，需包含列 '最高', '最低', '收盘'
    :param n: 计算RSV的周期（默认9）
    :param m1: K值的平滑周期（默认3）
    :param m2: D值的平滑周期（默认3）
    :return: 原始DataFrame添加 'K', 'D', 'J' 列
    """

    data['high_kdj'] = data['最高'].rolling(window = days, min_periods = 1).max()
    data['low_kdj'] = data['最低'].rolling(window = days, min_periods = 1).min()
    data['RSV'] = (data['收盘'] - data['low_kdj']) / (data['high_kdj'] - data['low_kdj']) * 100
    data['K'] = data['RSV'].ewm(alpha=1/m1, adjust=False).mean()
    data['D'] = data['K'].ewm(alpha=1/m2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    data_list = data[['日期','J']].to_numpy().tolist()
    print(data_list)
    for x in data_list:
        if float(x) < 0:
            print(x)

    return data   

if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/vscodePython/sh51030020250612.csv'
    data = pd.read_csv(file_path)

    cal_KDJ(data, 9, 3, 3)