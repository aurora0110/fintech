import pandas as pd
import numpy as np

def cal_PEG(PE, EGR):
    '''
    计算彼得林奇公式，即PEG比率
    :param data: dataframe数据，包含市盈率和盈利增长率
    :param PE: 市盈率
    :param EGR: 盈利增长率/未来几年利润复合增长率（未来3 5年年报里净利润同比增长率的数据）
    :return: PEG比率
    '''
    PEG = PE / EGR

    if PEG == 1:
        print("PEG为1，表示市盈率与盈利增长率匹配，公司估值合理")
    elif PEG < 1:
        print(f"PEG值为{PEG}，小于1，表示市盈率低于盈利增长率，公司估值较低")
    else:
        print(f"PEG值为{PEG}，大于1，表示市盈率高于盈利增长率，公司估值较高")

    return PE/EGR