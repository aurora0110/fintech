import pandas as pd
import numpy as np

def cal_PEG(stock, price, PE, EGR):
    '''
    计算彼得林奇公式，即PEG比率
    :param stock: 股票代码
    :param price: 每股股票价格
    :param PE: 市盈率，每股价格/每股利润
    :param EGR: 盈利增长率/未来几年利润复合增长率（未来3 5年年报里净利润同比增长率的数据，由机构预测得出，取多个机构预测平均值）
    :return: PEG比率
    '''
    PEG = round(PE / EGR, 1)

    if PEG == 1:
        print(f"PEG为1，表示市盈率与盈利增长率匹配，{stock}公司估值合理")
        predict_stock_price = PE / price * EGR
        print(f"预测未来3-5年每股股票合理价格为{predict_stock_price}")
    elif PEG < 1:
        print(f"PEG值为{PEG}，小于1，表示市盈率低于盈利增长率，{stock}公司估值较低")
    else:
        print(f"PEG值为{PEG}，大于1，表示市盈率高于盈利增长率，{stock}公司估值较高")

    
    return PEG