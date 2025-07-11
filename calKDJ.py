import pandas as pd

def cal_KDJ(data, days, m1, m2, start_date, end_date):
    """
    计算KDJ指标
    :param df: pandas DataFrame，需包含列 '最高', '最低', '收盘'
    :param n: 计算RSV的周期（默认9）
    :param m1: K值的平滑周期（默认3）
    :param m2: D值的平滑周期（默认3）
    :return: 原始DataFrame添加 'K', 'D', 'J' 列
    """
    import pandas as pd

    # 确保日期列是 datetime 类型
    data['日期'] = pd.to_datetime(data['日期'])

    # 将输入的日期也转为 datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)  # 示例结束日期

    # 使用 datetime 比较
    data = data[(data['日期'] >= start_date) & (data['日期'] <= end_date)]
    data['high_kdj'] = data['最高'].rolling(window = days, min_periods = 1).max()
    data['low_kdj'] = data['最低'].rolling(window = days, min_periods = 1).min()
    data['RSV'] = (data['收盘'] - data['low_kdj']) / (data['high_kdj'] - data['low_kdj']) * 100
    data['K'] = data['RSV'].ewm(alpha=1/m1, adjust=False).mean()
    data['D'] = data['K'].ewm(alpha=1/m2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    data_list = data[['日期','J', '收盘', '最低', '最高']].to_numpy().tolist()

    ret_kdj = []
    j_listlow0 = []
    j_listlow5 = []
    j_listlow8 = []
    j_listlow10 = []
    j_listlow15 = []
    j_listlow20 = []
    j_listlow25 = []
    j_listhigh80 = []
    j_listhigh90 = []
    j_listhigh100 = []
    j_listhigh110 = []
    j_listhigh120 = []

    for x in data_list:
        if float(x[1]) <= 0:
            j_listlow0.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -5:
            j_listlow5.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -8:
            j_listlow8.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -10:
            j_listlow10.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -15:
            j_listlow15.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -20:
            j_listlow20.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) <= -25:
            j_listlow25.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) >= 80:
            j_listhigh80.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) >= 90:
            j_listhigh90.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) >= 100:
            j_listhigh100.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) >= 110:
            j_listhigh110.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
        if float(x[1]) >= 120:
            j_listhigh120.append([x[0], round(float(x[1]), 3), x[2], x[3], x[4]])
    
    ret_kdj.append(j_listlow0)
    ret_kdj.append(j_listlow5)
    ret_kdj.append(j_listlow8)
    ret_kdj.append(j_listlow10)
    ret_kdj.append(j_listlow15)
    ret_kdj.append(j_listlow20)
    ret_kdj.append(j_listlow25)
    ret_kdj.append(j_listhigh80)
    ret_kdj.append(j_listhigh90)
    ret_kdj.append(j_listhigh100)
    ret_kdj.append(j_listhigh110)
    ret_kdj.append(j_listhigh120)
    ##print(f'j值小于0:{j_listlow0}\nj值小于-5:{j_listlow5}\nj值小于-8:{j_listlow8}\nj值小于-10:{j_listlow10}\nj值小于-15:{j_listlow15}\nj值小于-20:{j_listlow20}\nj值小于-25:{j_listlow25}\nj值大于80:{j_listhigh80}\nj值大于90:{j_listhigh90}\nj值大于100:{j_listhigh100}\nj值大于110:{j_listhigh110}\nj值大于120:{j_listhigh120}')

    result_kv = {}
    result_kv['K'] = data['K']
    result_kv['D'] = data['D']
    result_kv['J'] = data['J']
    result_kv['ret_kdj'] = ret_kdj
    
    return result_kv


if __name__ == '__main__':
    file_path = '/Users/lidongyang/Desktop/MYINVESTSTRATEGY/sh51298020250612.csv'
    data = pd.read_csv(file_path)

    cal_KDJ(data, 9, 3, 3)