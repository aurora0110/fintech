import numpy as np
from datetime import datetime

buy_price_512980 = ["0.679","0.679","0.740","0.740","0.800","0.576","0.679","0.740","0.800","0.860","0.698","0.586","0.698","0.698","0.775","0.775"]
buy_date_512980 = ["2021-07-28","2021-10-27","2021-07-26","2021-11-29","2021-03-15","2022-04-25","2022-03-08","2022-01-27","2022-01-24","2020-12-16","2023-10-20","2024-07-08","2024-09-30","2024-10-11","2023-09-08","2024-12-31"]
sell_price_512980 = ["0.740","0.740","0.800","0.800","0.860","0.679","0.749","0.800","0.860","0.920","0.775","0.698","0.812","0.775","0.853","0.853"]
sell_date_512980 = ["2021-09-08","2021-11-05","2021-11-16","2021-12-13","2021-12-31","2023-02-07","2023-03-20","2023-03-24","2023-04-03","2023-04-12","2024-03-19","2024-09-30","2024-10-08","2024-10-22","2024-12-05","2025-02-10"]

buy_price_159920 = ['1.46','1.241', '1.314', '1.314', '1.387', '1.314', '1.387', '1.46', '1.46', '1.314', '1.314', '0.978', '1.095', '0.978', '1.085', '1.085', '0.978', '1.085', '1.066', '1.168', '1.241', '1.314', '1.245']
buy_date_159920 = ['2020-02-03','2020-03-19', '2020-03-13', '2020-05-22', '2020-03-09', '2020-09-21', '2020-07-25', '2020-02-28', '2021-03-24', '2021-07-28', '2021-08-20', '2022-03-15', '2022-03-07', '2022-10-12', '2022-08-03', '2023-03-14', '2023-12-05', '2023-08-17', '2024-07-24', '2022-02-25', '2021-11-29', '2021-09-15', '2024-10-10']
sell_price_159920 = ['1.533', '1.314', '1.387', '1.387', '1.462', '1.387', '1.465', '1.533', '1.533', '1.387', '1.387', '1.095', '1.169', '1.096', '1.190', '1.168', '1.097', '1.168', '1.172', '1.269', '1.314', '1.416', '1.387']
sell_date_159920 = ['2020-02-12', '2020-03-26', '2020-04-29', '2020-06-04', '2020-07-06', '2020-11-09', '2021-01-11', '2021-01-19', '2021-04-19', '2021-08-04', '2021-09-07', '2022-03-17', '2022-06-08', '2022-12-01', '2023-01-05', '2023-06-16', '2024-05-06', '2024-05-17', '2024-09-25', '2024-09-30', '2024-09-30', '2024-10-08', '2025-02-13']

buy_price_513180 = ['0.665', '0.494', '0.494', '0.562', '0.494', '0.494', '0.570', '0.430', '0.494', '0.494', '0.548', '0.618', '0.665', '0.618']
buy_date_513180 = ['2022-01-05', '2022-03-14', '2022-04-21', '2022-03-07', '2022-09-22', '2023-05-25', '2022-07-18', '2024-01-17', '2023-12-05', '2024-06-27', '2023-08-21', '2022-02-24', '2022-01-27', '2022-02-24']
sell_price_513180 = ['0.713', '0.575', '0.570', '0.626', '0.570', '0.570', '0.623', '0.494', '0.552', '0.551', '0.618', '0.707', '0.713', '0.730']
sell_date_513180 = ['2022-01-12', '2022-03-23', '2022-05-31', '2022-06-09', '2022-12-07', '2023-06-16', '2023-07-31', '2024-03-13', '2024-05-14', '2024-09-26', '2024-09-30', '2024-10-08', '2025-02-27', '2025-02-10']


# 统计价格区间
bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

def backStockPoint(stock):
    buy_price_key = f"buy_price_{stock}"
    buy_date_key = f"buy_date_{stock}"
    sell_price_key = f"sell_price_{stock}"
    sell_date_key = f"sell_date_{stock}"

    buy_price = globals()[buy_price_key]
    buy_date = globals()[buy_date_key]
    sell_price = globals()[sell_price_key]
    sell_date = globals()[sell_date_key]
    print("buy_price: ", buy_price, "\n", "buy_date: ", buy_date, "\n", "sell_price: ", sell_price, "\n", "sell_date: ",sell_date)

    return buy_price, buy_date, sell_price, sell_date

def backBuy(input_buy_price, input_buy_date):
    print("Buy price: ", input_buy_price)
    print("Buy date: ", input_buy_date)
    return input_buy_price, input_buy_date

def backSell(input_sell_price, input_sell_date):
    print("Sell price: ", input_sell_price)
    print("Sell date: ", input_sell_date)
    return input_sell_price, input_sell_date

def cal_percentage(input_buy_price, input_buy_date, input_sell_price, input_sell_date, stock_code):
    # 统计每个区间数字出现的次数
    buy_price_hist, bin_edges = np.histogram(input_buy_price, bins=bins)
    buy_date_hist, bin_edges = np.histogram(input_buy_date, bins=bins)
    sell_price_hist, bin_edges = np.histogram(input_sell_price, bins=bins)
    sell_date_hist, bin_edges = np.histogram(input_sell_date, bins=bins)

    # 计算每个区间数字出现的百分比
    buy_price_percentage = buy_price_hist / len(input_buy_price)
    buy_date_percentage = buy_date_hist / len(input_buy_date)
    sell_price_percentage = sell_price_hist / len(input_sell_price)
    sell_date_percentage = sell_date_hist / len(input_sell_date)

    holding_days_list = []

    # 打印结果
    for i in range(len(bins) - 1):
        if buy_price_percentage[i] > 0.0:
            print('*' * 20)
            print(f"buy Price between {bins[i]} and {bins[i+1]}: {buy_price_percentage[i]}")
        # print(f"buy Date between {bins[i]} and {bins[i+1]}: {buy_date_percentage[i]}")
        if sell_price_percentage[i] > 0.0:
            print(f"sell Price between {bins[i]} and {bins[i+1]}: {sell_price_percentage[i]}")
        # print(f"sell Date between {bins[i]} and {bins[i+1]}: {sell_date_percentage[i]}")
    print('*' * 20)
    
    for number in np.unique(input_buy_price):
        count = buy_price.count(number)
        percentage = count / len(input_buy_price)
        print(f"买入数字 {number} 出现了 {count} 次，占比 {percentage*100:.2f}%")

        for i in range(len(input_buy_price)):
            if input_buy_price[i] == number:
                less_date = cal_holgingdates(input_buy_date[i], input_sell_date[i])
                holding_days_list.append(less_date)
                print(f"第 {i+1} 次出现于 {buy_date[i]}，卖出于 {input_sell_date[i]}, 卖出价格为 {input_sell_price[i]},持有天数 {less_date},利润率 {(float(input_sell_price[i]) - float(input_buy_price[i])) / float(input_buy_price[i]) * 100:.2f}%")
        
    avg_holding_days = sum(holding_days_list) / len(holding_days_list)
    print(f"买入数字 {stock_code} 的平均持有天数: {avg_holding_days}")

def cal_holgingdates(date1, date2):
    ndate1 = datetime.strptime(date1, "%Y-%m-%d")
    ndate2 = datetime.strptime(date2, "%Y-%m-%d")
    return (ndate2 - ndate1).days

if __name__ == "__main__":
    stock_code = "512980" # 512980 513180 159920

    buy_price, buy_date, sell_price, sell_date = backStockPoint(stock_code)

    print('*' * 50)

    length = len(buy_price)
    for i in range(length):
        print("Buy price: ", buy_price[i], "Buy date: ", buy_date[i], "Sell price: ", sell_price[i], "Sell date: ", sell_date[i])
    
    cal_percentage(buy_price, buy_date, sell_price, sell_date, stock_code)