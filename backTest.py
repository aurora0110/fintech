import numpy as np
import pandas as pd
import calKDJ as kdj
from datetime import datetime

def backTest(file, amount, windows, total_shares, each_buy_shares, start_date, end_date, backtest_log_path):
    '''
    回测
    :param file: 数据文件路径
    :param amount: 初始资金
    :param windows: 窗口大小，多长时间操作一次
    :param total_shares: 投入总手数
    :param each_buy_shares: 每次买入份数
    :param backtest_log_path: 回测日志路径
    :return:回测结果
    '''
    #data = pd.read_csv(file_path)
    #data_list = kdj.cal_KDJ(data, 9, 3, 3)
    # 时间戳
    log_content = f"[{datetime.now()}] 开始回测\n"
    df = pd.DataFrame(file, columns=['date', 'j_value', 'price', 'high_price', 'low_price']) # 以收盘价用作后续价格
    df = df[(df['date'] >= str(start_date)) & (df['date'] <= str(end_date))]

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    start_year = df['date'].iloc[0].year # 计算年化收益率用
    end_year = df['date'].iloc[-1].year

    # 初始化变量
    cash = amount # 手上还有的现金
    shares_held = 0 # 手上持有的份数
    portfolio_value = cash # 投资组合的价值
    max_investment = 0 # 最大投入金额
    peak_value = cash
    max_drawdown = 0 # 最大回撤

    trade_history = [] # 记录交易历史
    buy_prices = [] # 记录买入价格
    operate_counts = 0

    # 创建时间间隔，记录买入/卖出窗口
    buy_window = pd.Timedelta(days=windows)
    sell_window = pd.Timedelta(days=windows)

    for i in range(len(df)):
        current_date = df.at[i, 'date']
        j_value = df.at[i, 'j_value']
        price = df.at[i, 'price']

        # 检查窗口内是否有交易
        recent_buy = False
        recent_sell = False

        for trade in trade_history:
            if (current_date - trade['date']) <= buy_window and trade['type'] == 'buy':
                    recent_buy = True
            if (current_date - trade['date']) <= sell_window and trade['type'] == 'sell':
                    recent_sell = True
        
        # 买入逻辑 ---------------------------------》这里可以添加上其他的策略作为限制条件，看回测效果如何
        if not recent_buy and shares_held < total_shares:
            buy_shares = 0
            if j_value <= -15:
                buy_shares = 1

            elif j_value <= -10:
                buy_shares = 1

            elif j_value <= -8:
                buy_shares = 1

            elif j_value <= -5:
                buy_shares = 1

            if buy_shares > 0 and cash >= price * buy_shares:
                cost = price * buy_shares * each_buy_shares # buy_shares--当前买入手数，each_buy_shares--每次买入份数
                cash -= cost
                operate_counts += 1
                shares_held += buy_shares
                buy_prices.append(price) # 每次满足条件的买入都放到list里
                trade_history.append({'date': current_date, 'type': 'buy', 'shares': buy_shares, 'price': price}) # 'cost':cost
                #print(f"进行一次买入操作，当前持有份额为：{shares_held}，还可操作份额为：{total_shares - shares_held}，当前价格为：{price}，当前现金为：{cash}")
                with open(backtest_log_path, 'a') as f:
                    f.write(f"进行一次买入操作，当前持有份额为：{shares_held}，还可操作份额为：{total_shares - shares_held}，当前价格为：{price}，当前现金为：{cash}\n")

        # 卖出逻辑 ---------------------------------》这里可以添加上其他的策略作为限制条件，看回测效果如何？比如20250702是否加一个当盈利率达到一定的标准再卖出？
        if not recent_sell and shares_held >= 0 and len(buy_prices) > 0:
            sell_shares = 0
            if j_value >= 110:
                sell_shares = 1

            elif j_value >= 100:
                sell_shares = 1

            elif j_value >= 90:
                sell_shares = 1

            #elif j_value >= 80:
                #sell_shares = 1
            
            # 检查当前价格是否大于 最近买入价格？未卖出价格？ price > buy_prices.max() price > buy_prices.min() price > buy_prices[-1]?
            if sell_shares > 0 and shares_held >= sell_shares and price > min(buy_prices):
                revenue = price * sell_shares * each_buy_shares
                cash += revenue
                operate_counts += 1
                shares_held -= sell_shares
                min_price = min(buy_prices)
                # 移除已买入的价格中最小的买入价格记录
                buy_prices.remove(min_price) if len(buy_prices) > 0 else None
                trade_history.append({'date': current_date, 'type': 'sell', 'shares': sell_shares, 'price': price, 'buy_prices':buy_prices}) # , 'revenue':revenue     
                #print(f"进行一次卖出操作，当前持有份额为：{shares_held}，还可操作份额为：{total_shares - shares_held}，当前价格为：{price}，当前现金为：{cash}")
                with open(backtest_log_path, 'a') as f:
                    f.write(f"进行一次卖出操作，当前持有份额为：{shares_held}，还可操作份额为：{total_shares - shares_held}，当前价格为：{price}，当前现金为：{cash}\n")

        # 更新投资组合价值
        portfolio_value = cash + shares_held * price

        # 更新最大投入金额
        current_investment = amount - cash
        if current_investment > max_investment:
            max_investment = current_investment
        
        # 更新最大回撤
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        else:
            drawdown = (peak_value - portfolio_value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
    # 计算最终结果
    final_value = cash + shares_held * df.iloc[-1]['price']
    total_return = (final_value - amount) / amount

    # 整理交易历史
    trade_history_df = pd.DataFrame(trade_history)

    result = {
        'amount':amount,
        'final_value': final_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'max_investment': max_investment,
        'trade_history': trade_history_df,
        'portfolio_value_over_time': portfolio_value}

    buy_list = []
    sell_list = []
    for key, value in result.items():
        if key == 'trade_history':
            for index, row in value.iterrows():
                if row['type'] == 'buy':
                    buy_list.append([row['date'], row['price']])
                elif row['type'] == 'sell':
                    sell_list.append([row['date'], row['price']])

    earning_results = []
    i = 0 # sell_list索引
    j = 0 # buy_list索引

    holddays_list = []
    while i < len(sell_list) and j < len(buy_list):
        sell_date, sell_price = sell_list[i]
        buy_date, buy_price = buy_list[j]

        if sell_date > buy_date:
            if sell_price > buy_price:
                profit = (sell_price - buy_price) / buy_price
                earning_results.append(profit)
                #print(f'sell_price:{sell_price}, sell_date:{sell_date}, buy_price:{buy_price}, buy_date:{buy_date}, profit:{round(profit * 100, 3)}%, holddays:{(sell_date - buy_date).days}')
                with open(backtest_log_path, 'a') as f:
                    f.write(f'sell_price:{sell_price}, sell_date:{sell_date}, buy_price:{buy_price}, buy_date:{buy_date}, profit:{round(profit * 100, 3)}%, holddays:{(sell_date - buy_date).days}\n')

                holddays_list.append(int((sell_date - buy_date).days))
                buy_list.pop(j)
                sell_list.pop(i)

                # 重置索引
                i = 0
                j = 0
            else:
                # 价格条件不满足，这个买入价格不卖出，匹配下一个便宜的买入价格
                j += 1
        else:
            # 日期条件不满足，这个卖出价格不买入，匹配下一个合适的卖出价格
            i += 1
            j = 0
        
        # 如果buy超出索引范围，移动到下一个sell
        if j >= len(buy_list):
            i += 1
            j = 0
    avg_profit = sum(earning_results) / int(end_year - start_year) if earning_results else 1
    result['avg_profit'] = avg_profit
    #print(f'{start_date}到{end_date}平均年化收益: {round(avg_profit * 100, 3)}%，平均持有时长为：{round(sum(holddays_list) / len(holddays_list),1)}天')
    with open(backtest_log_path, 'a') as f:
        f.write(f'{start_date}到{end_date}平均年化收益: {round(avg_profit * 100, 3)}%，平均持有时长为：{round(sum(holddays_list) / len(holddays_list),1)}天，年平均买卖操作次数为：{operate_counts / int(end_year - start_year)}次\n\n')
    return result

sample_data = [
    ['2020-02-03', -10.211, 0.79, 0.79, 0.83], ['2020-05-22', -10.439, 0.821, 0.816, 0.84], ['2020-05-25', -14.261, 0.816, 0.812, 0.821], ['2020-09-09', -9.828, 0.978, 0.976, 1.008], ['2020-09-10', -16.037, 0.952, 0.948, 0.988], ['2021-02-05', -6.162, 0.83, 0.828, 0.853], ['2021-05-07', -9.508, 0.791, 0.791, 0.802], ['2021-06-03', -6.734, 0.815, 0.815, 0.827], ['2021-06-17', -5.047, 0.804, 0.801, 0.81], ['2021-09-22', -8.058, 0.709, 0.706, 0.715], ['2021-09-29', -7.952, 0.689, 0.687, 0.701], ['2021-11-26', -6.328, 0.755, 0.754, 0.765], ['2021-11-29', -12.969, 0.743, 0.738, 0.751], ['2021-11-30', -13.515, 0.741, 0.736, 0.75], ['2022-01-27', -5.632, 0.723, 0.721, 0.751], ['2022-04-11', -6.082, 0.635, 0.632, 0.656], ['2022-04-13', -10.938, 0.63, 0.628, 0.645], ['2022-04-14', -10.47, 0.635, 0.629, 0.641], ['2022-04-15', -12.735, 0.621, 0.618, 0.631], ['2022-07-07', -7.509, 0.609, 0.607, 0.613], ['2022-07-12', -9.501, 0.601, 0.6, 0.607], ['2022-07-13', -5.373, 0.604, 0.601, 0.61], ['2022-08-03', -5.605, 0.577, 0.575, 0.59], ['2022-11-25', -12.145, 0.572, 0.568, 0.576], ['2022-12-20', -9.938, 0.572, 0.57, 0.586], ['2023-02-24', -7.202, 0.674, 0.668, 0.678], ['2023-02-27', -14.409, 0.663, 0.661, 0.675], ['2023-02-28', -5.089, 0.671, 0.662, 0.672], ['2023-05-16', -7.34, 0.866, 0.863, 0.911], ['2023-05-17', -8.432, 0.872, 0.853, 0.889], ['2023-05-23', -5.891, 0.843, 0.842, 0.866], ['2023-06-27', -10.246, 0.9, 0.883, 0.914], ['2023-06-28', -8.282, 0.888, 0.856, 0.899], ['2023-08-21', -9.325, 0.78, 0.779, 0.799], ['2023-09-08', -7.097, 0.765, 0.759, 0.782], ['2023-10-18', -12.161, 0.719, 0.717, 0.735], ['2023-10-19', -15.744, 0.716, 0.715, 0.725], ['2023-10-20', -14.709, 0.7, 0.698, 0.716], ['2023-10-23', -11.24, 0.68, 0.676, 0.699], ['2023-11-29', -6.877, 0.757, 0.754, 0.765], ['2023-11-30', -6.66, 0.754, 0.747, 0.76], ['2023-12-22', -9.456, 0.71, 0.705, 0.781], ['2023-12-25', -6.725, 0.707, 0.69, 0.713], ['2023-12-26', -7.77, 0.699, 0.693, 0.714], ['2023-12-27', -10.029, 0.691, 0.688, 0.708], ['2024-03-27', -6.182, 0.725, 0.724, 0.757], ['2024-06-25', -5.693, 0.593, 0.589, 0.605], ['2024-12-24', -5.387, 0.812, 0.803, 0.816], ['2024-12-25', -6.386, 0.8, 0.791, 0.815], ['2025-02-26', -8.818, 0.859, 0.851, 0.866], ['2025-02-28', -10.742, 0.815, 0.813, 0.846],['2020-01-02', 97.917, 0.883, 0.836, 0.884], ['2020-01-03', 86.574, 0.886, 0.877, 0.896], ['2020-01-06', 84.291, 0.905, 0.869, 0.915], ['2020-01-07', 91.889, 0.933, 0.9, 0.935], ['2020-01-13', 86.829, 0.929, 0.9, 0.93], ['2020-01-14', 86.403, 0.926, 0.925, 0.94], ['2020-02-07', 102.179, 0.931, 0.9, 0.936], ['2020-02-10', 108.703, 0.921, 0.903, 0.936], ['2020-02-11', 104.819, 0.912, 0.902, 0.925], ['2020-02-12', 112.252, 0.934, 0.909, 0.935], ['2020-02-13', 104.649, 0.916, 0.908, 0.933], ['2020-02-14', 100.306, 0.918, 0.904, 0.926], ['2020-02-17', 106.395, 0.947, 0.918, 0.947], ['2020-02-18', 107.726, 0.959, 0.934, 0.96], ['2020-02-21', 80.326, 0.971, 0.943, 0.981], ['2020-02-24', 93.283, 0.982, 0.956, 0.984], ['2020-02-25', 92.705, 0.975, 0.92, 0.978], ['2020-04-09', 91.717, 0.832, 0.823, 0.837], ['2020-04-17', 84.743, 0.827, 0.823, 0.838], ['2020-04-20', 95.606, 0.832, 0.822, 0.833], ['2020-04-21', 99.688, 0.832, 0.818, 0.833], ['2020-04-22', 107.53, 0.838, 0.823, 0.839], ['2020-04-23', 88.194, 0.824, 0.821, 0.841], ['2020-05-06', 98.636, 0.855, 0.828, 0.857], ['2020-05-07', 100.164, 0.851, 0.849, 0.86], ['2020-05-08', 105.803, 0.861, 0.853, 0.864], ['2020-05-11', 100.33, 0.86, 0.854, 0.871], ['2020-05-12', 96.457, 0.86, 0.85, 0.862], ['2020-05-13', 95.512, 0.862, 0.851, 0.864], ['2020-05-14', 91.049, 0.862, 0.857, 0.872], ['2020-06-01', 93.042, 0.87, 0.844, 0.872], ['2020-06-02', 102.683, 0.867, 0.863, 0.877], ['2020-06-03', 103.932, 0.866, 0.865, 0.873], ['2020-06-04', 98.897, 0.863, 0.86, 0.869], ['2020-06-05', 109.198, 0.877, 0.862, 0.878], ['2020-06-08', 102.298, 0.872, 0.87, 0.883], ['2020-06-09', 107.794, 0.888, 0.868, 0.889], ['2020-06-10', 104.638, 0.884, 0.877, 0.888], ['2020-06-11', 91.108, 0.889, 0.881, 0.901], ['2020-06-12', 96.582, 0.902, 0.867, 0.904], ['2020-06-15', 83.608, 0.907, 0.901, 0.922], ['2020-06-16', 87.011, 0.915, 0.903, 0.918], ['2020-06-17', 92.308, 0.918, 0.904, 0.919], ['2020-06-18', 83.386, 0.915, 0.902, 0.928], ['2020-06-19', 80.174, 0.916, 0.906, 0.923], ['2020-06-23', 85.591, 0.94, 0.913, 0.947], ['2020-06-24', 90.711, 0.944, 0.94, 0.951], ['2020-06-30', 81.043, 0.948, 0.93, 0.952], ['2020-07-01', 83.716, 0.947, 0.933, 0.955], ['2020-07-02', 85.665, 0.95, 0.938, 0.958], ['2020-07-03', 94.657, 0.956, 0.936, 0.957], ['2020-07-06', 97.125, 0.987, 0.945, 0.992], ['2020-07-07', 98.237, 1.015, 0.973, 1.021], ['2020-07-08', 97.917, 1.016, 0.985, 1.022], ['2020-07-09', 98.713, 1.057, 1.016, 1.063], ['2020-07-10', 95.304, 1.056, 1.038, 1.068], ['2020-07-13', 99.575, 1.078, 1.043, 1.079], ['2020-07-14', 90.583, 1.07, 1.047, 1.093], ['2020-08-05', 88.832, 1.007, 0.993, 1.014], ['2020-08-18', 87.205, 0.998, 0.986, 1.0], ['2020-08-21', 90.16, 1.006, 0.981, 1.011], ['2020-08-24', 103.459, 1.027, 1.002, 1.031], ['2020-08-25', 103.971, 1.028, 1.022, 1.036], ['2020-08-26', 96.077, 1.022, 1.016, 1.036], ['2020-08-27', 95.078, 1.026, 1.019, 1.034], ['2020-08-28', 104.07, 1.042, 1.018, 1.042], ['2020-08-31', 85.274, 1.028, 1.027, 1.055], ['2020-09-02', 81.81, 1.043, 1.025, 1.046], ['2020-10-12', 89.5, 0.992, 0.974, 0.993], ['2020-10-13', 107.77, 0.991, 0.973, 0.994], ['2020-10-14', 91.628, 0.989, 0.985, 1.01], ['2020-11-09', 96.079, 0.945, 0.933, 0.95], ['2020-12-01', 97.021, 0.931, 0.903, 0.934], ['2020-12-02', 106.124, 0.931, 0.925, 0.939], ['2020-12-03', 109.297, 0.932, 0.925, 0.936], ['2020-12-04', 105.652, 0.93, 0.92, 0.932], ['2020-12-07', 82.905, 0.916, 0.915, 0.928], ['2021-01-04', 105.43, 0.874, 0.861, 0.876], ['2021-01-05', 109.061, 0.868, 0.86, 0.877], ['2021-01-06', 100.814, 0.862, 0.851, 0.866], ['2021-01-08', 90.929, 0.874, 0.834, 0.878], ['2021-01-11', 96.482, 0.88, 0.867, 0.889], ['2021-01-12', 88.238, 0.875, 0.862, 0.879], ['2021-01-21', 89.525, 0.895, 0.878, 0.898], ['2021-02-19', 91.535, 0.885, 0.855, 0.886], ['2021-02-22', 85.36, 0.865, 0.864, 0.889], ['2021-04-06', 92.726, 0.808, 0.799, 0.809], ['2021-04-07', 103.843, 0.806, 0.799, 0.807], ['2021-04-08', 93.381, 0.802, 0.8, 0.807], ['2021-04-09', 86.085, 0.802, 0.799, 0.804], ['2021-04-19', 89.208, 0.81, 0.794, 0.811], ['2021-04-20', 90.557, 0.814, 0.805, 0.824], ['2021-04-21', 84.195, 0.811, 0.808, 0.813], ['2021-04-22', 88.809, 0.816, 0.809, 0.819], ['2021-04-23', 84.609, 0.813, 0.809, 0.823], ['2021-05-17', 86.19, 0.81, 0.802, 0.818], ['2021-05-18', 102.857, 0.815, 0.808, 0.818], ['2021-05-19', 107.237, 0.814, 0.808, 0.817], ['2021-05-20', 107.106, 0.822, 0.807, 0.827], ['2021-05-21', 93.48, 0.816, 0.814, 0.829], ['2021-05-24', 98.593, 0.825, 0.812, 0.825], ['2021-05-25', 102.974, 0.837, 0.822, 0.839], ['2021-05-26', 91.21, 0.831, 0.829, 0.84], ['2021-05-27', 94.929, 0.838, 0.83, 0.841], ['2021-06-10', 80.929, 0.837, 0.831, 0.842], ['2021-08-11', 87.0, 0.689, 0.684, 0.698], ['2021-08-12', 89.853, 0.688, 0.686, 0.697], ['2021-08-13', 87.126, 0.685, 0.681, 0.69], ['2021-08-16', 93.456, 0.69, 0.683, 0.691], ['2021-08-25', 92.043, 0.696, 0.682, 0.696], ['2021-08-26', 88.839, 0.689, 0.687, 0.697], ['2021-09-02', 83.786, 0.702, 0.699, 0.71], ['2021-09-03', 90.026, 0.703, 0.698, 0.708], ['2021-09-06', 99.946, 0.711, 0.701, 0.714], ['2021-09-07', 106.159, 0.727, 0.708, 0.729], ['2021-09-08', 109.603, 0.762, 0.727, 0.763], ['2021-09-09', 91.938, 0.74, 0.736, 0.75], ['2021-11-01', 93.993, 0.708, 0.696, 0.716], ['2021-11-02', 90.776, 0.701, 0.694, 0.714], ['2021-11-03', 98.44, 0.713, 0.697, 0.722], ['2021-11-04', 113.341, 0.724, 0.71, 0.724], ['2021-11-05', 108.879, 0.74, 0.722, 0.75], ['2021-11-08', 108.448, 0.745, 0.732, 0.751], ['2021-11-09', 101.763, 0.74, 0.736, 0.747], ['2021-11-10', 98.789, 0.746, 0.739, 0.755], ['2021-11-11', 97.433, 0.758, 0.735, 0.765], ['2021-11-12', 96.021, 0.758, 0.753, 0.765], ['2021-11-15', 98.226, 0.785, 0.757, 0.79], ['2021-11-16', 92.952, 0.791, 0.78, 0.803], ['2021-11-17', 91.351, 0.794, 0.788, 0.802], ['2021-12-09', 88.22, 0.765, 0.754, 0.77], ['2021-12-10', 104.665, 0.776, 0.76, 0.782], ['2021-12-13', 118.446, 0.808, 0.78, 0.809], ['2021-12-14', 115.432, 0.811, 0.799, 0.82], ['2021-12-15', 110.554, 0.812, 0.804, 0.822], ['2021-12-16', 108.521, 0.826, 0.805, 0.834], ['2021-12-17', 102.471, 0.822, 0.814, 0.833], ['2021-12-22', 85.914, 0.838, 0.81, 0.845], ['2021-12-31', 88.437, 0.858, 0.839, 0.86], ['2022-01-04', 100.199, 0.891, 0.858, 0.896], ['2022-01-05', 100.48, 0.892, 0.878, 0.903], ['2022-01-06', 89.331, 0.879, 0.865, 0.887], ['2022-02-15', 85.526, 0.751, 0.745, 0.758], ['2022-02-16', 96.468, 0.754, 0.749, 0.758], ['2022-03-21', 98.625, 0.663, 0.653, 0.67], ['2022-03-22', 110.211, 0.667, 0.653, 0.672], ['2022-03-23', 113.72, 0.67, 0.664, 0.675], ['2022-03-24', 98.639, 0.654, 0.653, 0.667], ['2022-03-25', 90.279, 0.655, 0.654, 0.665], ['2022-03-28', 92.913, 0.665, 0.651, 0.669], ['2022-04-01', 96.476, 0.691, 0.657, 0.691], ['2022-04-06', 93.91, 0.693, 0.686, 0.7], ['2022-05-09', 82.636, 0.58, 0.569, 0.584], ['2022-05-10', 93.444, 0.583, 0.567, 0.584], ['2022-05-11', 95.643, 0.586, 0.582, 0.601], ['2022-05-12', 94.723, 0.586, 0.577, 0.588], ['2022-05-13', 92.334, 0.588, 0.581, 0.59], ['2022-05-16', 89.458, 0.59, 0.586, 0.593], ['2022-05-30', 83.51, 0.603, 0.596, 0.604], ['2022-05-31', 96.071, 0.614, 0.598, 0.617], ['2022-06-01', 95.645, 0.615, 0.61, 0.622], ['2022-06-02', 94.385, 0.615, 0.606, 0.616], ['2022-06-06', 102.367, 0.626, 0.612, 0.627], ['2022-06-07', 97.637, 0.624, 0.62, 0.631], ['2022-06-08', 92.853, 0.629, 0.614, 0.635], ['2022-06-16', 82.157, 0.645, 0.624, 0.649], ['2022-06-20', 81.908, 0.641, 0.631, 0.642], ['2022-06-21', 92.79, 0.646, 0.636, 0.649], ['2022-07-20', 88.06, 0.615, 0.612, 0.62], ['2022-07-21', 103.69, 0.621, 0.613, 0.626], ['2022-07-22', 88.859, 0.611, 0.607, 0.622], ['2022-08-11', 87.459, 0.609, 0.593, 0.609], ['2022-08-12', 106.296, 0.615, 0.605, 0.617], ['2022-08-15', 100.476, 0.608, 0.606, 0.613], ['2022-08-16', 88.048, 0.604, 0.602, 0.611], ['2022-08-17', 95.599, 0.613, 0.602, 0.615], ['2022-08-22', 91.107, 0.618, 0.598, 0.618], ['2022-10-17', 97.408, 0.541, 0.534, 0.543], ['2022-10-18', 105.517, 0.539, 0.537, 0.544], ['2022-10-19', 92.156, 0.531, 0.531, 0.539], ['2022-10-20', 91.476, 0.535, 0.528, 0.54], ['2022-11-02', 85.289, 0.54, 0.536, 0.545], ['2022-11-03', 91.291, 0.535, 0.531, 0.537], ['2022-11-04', 106.501, 0.546, 0.533, 0.548], ['2022-11-07', 108.725, 0.555, 0.54, 0.56], ['2022-11-08', 107.802, 0.555, 0.55, 0.558], ['2022-11-09', 101.629, 0.552, 0.55, 0.558], ['2022-11-10', 106.583, 0.567, 0.546, 0.568], ['2022-11-11', 96.998, 0.569, 0.567, 0.58], ['2022-11-14', 89.115, 0.571, 0.562, 0.576], ['2022-11-15', 97.478, 0.579, 0.568, 0.58], ['2022-11-16', 96.244, 0.587, 0.58, 0.592], ['2022-11-17', 100.548, 0.603, 0.59, 0.604], ['2022-11-18', 81.219, 0.601, 0.6, 0.623], ['2022-12-02', 89.546, 0.6, 0.595, 0.608], ['2022-12-05', 106.955, 0.614, 0.603, 0.616], ['2022-12-06', 101.508, 0.606, 0.604, 0.617], ['2022-12-07', 101.016, 0.61, 0.601, 0.618], ['2022-12-08', 97.86, 0.609, 0.599, 0.611], ['2022-12-09', 86.23, 0.605, 0.595, 0.609], ['2022-12-28', 90.748, 0.593, 0.583, 0.595], ['2022-12-29', 82.497, 0.588, 0.588, 0.599], ['2022-12-30', 101.044, 0.599, 0.588, 0.601], ['2023-01-03', 106.991, 0.615, 0.598, 0.619], ['2023-01-04', 106.788, 0.622, 0.613, 0.627], ['2023-01-05', 104.952, 0.632, 0.622, 0.638], ['2023-01-06', 80.341, 0.617, 0.615, 0.633], ['2023-01-09', 82.9, 0.628, 0.618, 0.63], ['2023-01-10', 93.892, 0.636, 0.623, 0.639], ['2023-01-13', 80.442, 0.633, 0.622, 0.634], ['2023-01-19', 84.457, 0.64, 0.626, 0.641], ['2023-01-20', 96.908, 0.644, 0.638, 0.645], ['2023-02-02', 81.604, 0.66, 0.653, 0.669], ['2023-02-03', 94.5, 0.668, 0.656, 0.67], ['2023-02-06', 96.03, 0.672, 0.658, 0.677], ['2023-02-07', 99.858, 0.686, 0.667, 0.689], ['2023-02-09', 90.584, 0.687, 0.663, 0.688], ['2023-02-10', 88.198, 0.688, 0.682, 0.697], ['2023-02-13', 94.425, 0.694, 0.681, 0.696], ['2023-02-14', 83.977, 0.688, 0.686, 0.698], ['2023-02-15', 94.238, 0.701, 0.685, 0.702], ['2023-03-02', 87.581, 0.709, 0.696, 0.714], ['2023-03-03', 99.204, 0.706, 0.697, 0.713], ['2023-03-06', 95.443, 0.701, 0.698, 0.708], ['2023-03-17', 87.365, 0.73, 0.699, 0.733], ['2023-03-20', 92.466, 0.75, 0.73, 0.765], ['2023-03-21', 101.87, 0.766, 0.745, 0.772], ['2023-03-22', 109.966, 0.789, 0.761, 0.789], ['2023-03-23', 110.428, 0.787, 0.775, 0.79], ['2023-03-24', 111.213, 0.82, 0.784, 0.82], ['2023-03-27', 104.969, 0.828, 0.804, 0.839], ['2023-03-28', 94.073, 0.814, 0.808, 0.833], ['2023-03-29', 81.972, 0.806, 0.795, 0.823], ['2023-03-31', 86.073, 0.846, 0.791, 0.847], ['2023-04-03', 93.763, 0.887, 0.835, 0.893], ['2023-04-04', 87.143, 0.882, 0.875, 0.905], ['2023-04-12', 90.622, 0.939, 0.88, 0.941], ['2023-04-13', 85.06, 0.92, 0.916, 0.954], ['2023-04-26', 80.012, 0.922, 0.887, 0.952], ['2023-04-28', 84.205, 0.961, 0.871, 0.968], ['2023-05-04', 99.252, 1.019, 0.968, 1.028], ['2023-05-05', 92.501, 0.992, 0.98, 1.016], ['2023-05-08', 84.1, 0.984, 0.955, 1.001], ['2023-05-30', 91.762, 0.9, 0.835, 0.904], ['2023-05-31', 109.295, 0.913, 0.895, 0.922], ['2023-06-01', 110.236, 0.934, 0.902, 0.955], ['2023-06-02', 107.94, 0.934, 0.928, 0.949], ['2023-06-05', 112.939, 0.954, 0.925, 0.959], ['2023-06-06', 96.109, 0.927, 0.92, 0.968], ['2023-06-07', 89.335, 0.936, 0.912, 0.946], ['2023-06-09', 85.371, 0.955, 0.908, 0.957], ['2023-06-13', 90.137, 0.979, 0.945, 0.985], ['2023-06-14', 87.955, 0.992, 0.973, 1.011], ['2023-06-15', 80.122, 0.986, 0.981, 1.015], ['2023-06-16', 88.897, 1.005, 0.967, 1.014], ['2023-06-19', 97.146, 1.012, 0.995, 1.016], ['2023-06-20', 94.21, 1.019, 1.0, 1.035], ['2023-08-07', 93.819, 0.851, 0.84, 0.862], ['2023-08-08', 97.102, 0.852, 0.846, 0.865], ['2023-08-30', 101.095, 0.845, 0.826, 0.849], ['2023-08-31', 99.107, 0.832, 0.827, 0.844], ['2023-09-01', 84.707, 0.822, 0.817, 0.833], ['2023-09-26', 100.199, 0.77, 0.757, 0.777], ['2023-09-27', 112.84, 0.775, 0.766, 0.779], ['2023-09-28', 109.92, 0.771, 0.769, 0.778], ['2023-10-09', 88.632, 0.76, 0.752, 0.769], ['2023-10-11', 83.852, 0.769, 0.757, 0.779], ['2023-10-31', 88.743, 0.717, 0.711, 0.728], ['2023-11-01', 90.222, 0.72, 0.711, 0.738], ['2023-11-02', 98.922, 0.734, 0.72, 0.745], ['2023-11-03', 100.991, 0.735, 0.722, 0.741], ['2023-11-06', 110.805, 0.771, 0.74, 0.773], ['2023-11-07', 112.155, 0.774, 0.764, 0.778], ['2023-11-08', 108.878, 0.798, 0.767, 0.806], ['2023-11-09', 94.736, 0.784, 0.78, 0.798], ['2023-12-07', 95.856, 0.803, 0.79, 0.808], ['2023-12-08', 98.592, 0.798, 0.788, 0.809], ['2023-12-11', 109.772, 0.822, 0.791, 0.824], ['2023-12-12', 99.831, 0.819, 0.816, 0.837], ['2023-12-13', 86.155, 0.812, 0.811, 0.829], ['2024-01-25', 98.54, 0.69, 0.657, 0.691], ['2024-01-26', 94.478, 0.682, 0.678, 0.704], ['2024-02-19', 102.578, 0.694, 0.671, 0.697], ['2024-02-20', 112.111, 0.692, 0.678, 0.701], ['2024-02-21', 109.261, 0.691, 0.678, 0.71], ['2024-02-22', 111.911, 0.703, 0.69, 0.707], ['2024-02-23', 114.467, 0.72, 0.694, 0.721], ['2024-02-26', 108.971, 0.71, 0.703, 0.721], ['2024-02-27', 109.575, 0.739, 0.705, 0.74], ['2024-03-04', 91.825, 0.755, 0.731, 0.758], ['2024-03-05', 84.938, 0.745, 0.738, 0.759], ['2024-03-13', 86.418, 0.768, 0.743, 0.773], ['2024-03-18', 86.616, 0.771, 0.746, 0.771], ['2024-03-19', 81.092, 0.762, 0.761, 0.78], ['2024-03-20', 96.416, 0.789, 0.757, 0.791], ['2024-03-21', 91.786, 0.796, 0.795, 0.812], ['2024-03-22', 93.191, 0.81, 0.78, 0.82], ['2024-04-26', 89.065, 0.707, 0.685, 0.708], ['2024-04-29', 109.091, 0.725, 0.71, 0.728], ['2024-04-30', 98.657, 0.712, 0.705, 0.724], ['2024-05-06', 97.873, 0.717, 0.716, 0.728], ['2024-05-07', 92.478, 0.717, 0.715, 0.731], ['2024-07-22', 80.066, 0.591, 0.583, 0.593], ['2024-07-31', 91.262, 0.599, 0.575, 0.6], ['2024-08-01', 94.669, 0.594, 0.592, 0.604], ['2024-08-02', 80.833, 0.586, 0.586, 0.599], ['2024-08-06', 87.485, 0.599, 0.589, 0.601], ['2024-08-07', 84.509, 0.596, 0.591, 0.602], ['2024-09-05', 92.529, 0.591, 0.574, 0.595], ['2024-09-06', 92.39, 0.585, 0.584, 0.595], ['2024-09-09', 81.545, 0.58, 0.579, 0.589], ['2024-09-10', 90.248, 0.59, 0.573, 0.591], ['2024-09-11', 83.522, 0.588, 0.583, 0.592], ['2024-09-24', 88.74, 0.602, 0.576, 0.602], ['2024-09-25', 85.021, 0.611, 0.605, 0.63], ['2024-09-26', 103.938, 0.638, 0.607, 0.638], ['2024-09-27', 109.806, 0.671, 0.644, 0.674], ['2024-09-30', 113.126, 0.738, 0.69, 0.738], ['2024-10-08', 110.93, 0.805, 0.746, 0.812], ['2024-10-09', 82.085, 0.725, 0.725, 0.775], ['2024-10-28', 91.159, 0.772, 0.754, 0.779], ['2024-11-07', 95.772, 0.803, 0.782, 0.806], ['2024-11-08', 91.743, 0.793, 0.788, 0.811], ['2024-11-11', 106.255, 0.828, 0.781, 0.828], ['2024-11-12', 88.266, 0.799, 0.791, 0.827], ['2024-11-13', 96.195, 0.821, 0.792, 0.828], ['2024-11-29', 89.744, 0.82, 0.797, 0.827], ['2024-12-02', 101.773, 0.837, 0.82, 0.842], ['2024-12-03', 102.45, 0.833, 0.825, 0.84], ['2024-12-04', 87.98, 0.819, 0.807, 0.836], ['2024-12-05', 98.563, 0.859, 0.816, 0.863], ['2024-12-06', 101.762, 0.884, 0.858, 0.891], ['2024-12-09', 97.519, 0.876, 0.862, 0.883], ['2024-12-10', 80.313, 0.869, 0.866, 0.9], ['2025-01-15', 89.632, 0.759, 0.755, 0.767], ['2025-01-16', 98.284, 0.762, 0.75, 0.779], ['2025-01-17', 92.671, 0.755, 0.747, 0.761], ['2025-01-20', 89.529, 0.757, 0.753, 0.767], ['2025-01-21', 89.149, 0.76, 0.746, 0.763], ['2025-01-24', 87.449, 0.77, 0.747, 0.771], ['2025-02-05', 94.675, 0.805, 0.784, 0.805], ['2025-02-06', 101.238, 0.822, 0.797, 0.827], ['2025-02-07', 99.344, 0.831, 0.817, 0.842], ['2025-02-10', 104.664, 0.859, 0.829, 0.861], ['2025-02-11', 103.452, 0.866, 0.85, 0.873], ['2025-02-12', 105.032, 0.888, 0.861, 0.89], ['2025-02-13', 98.19, 0.897, 0.878, 0.914], ['2025-02-14', 100.538, 0.914, 0.88, 0.917], ['2025-02-17', 84.983, 0.901, 0.893, 0.936], ['2025-03-07', 84.609, 0.872, 0.863, 0.891], ['2025-03-12', 85.164, 0.876, 0.868, 0.888], ['2025-03-17', 84.499, 0.874, 0.788, 0.886], ['2025-03-18', 89.346, 0.877, 0.869, 0.883], ['2025-03-19', 84.024, 0.864, 0.86, 0.876], ['2025-04-17', 89.932, 0.775, 0.769, 0.782], ['2025-04-18', 103.873, 0.783, 0.77, 0.783], ['2025-04-21', 112.451, 0.799, 0.774, 0.8], ['2025-04-22', 94.642, 0.79, 0.787, 0.801], ['2025-05-06', 95.766, 0.819, 0.794, 0.82], ['2025-05-07', 83.356, 0.813, 0.807, 0.835], ['2025-05-08', 80.732, 0.817, 0.809, 0.821], ['2025-05-29', 91.471, 0.808, 0.791, 0.808], ['2025-06-04', 80.011, 0.814, 0.806, 0.817], ['2025-06-05', 96.743, 0.825, 0.813, 0.827], ['2025-06-06', 92.005, 0.819, 0.816, 0.827], ['2025-06-09', 97.056, 0.826, 0.818, 0.83], ['2025-06-10', 93.014, 0.823, 0.808, 0.828], ['2025-06-11', 97.802, 0.832, 0.82, 0.835], ['2025-06-12', 96.235, 0.842, 0.827, 0.848]
    ]
#results = backTest(sample_data, 50000, 30, 20, '2020-02-03', '2025-06-12')
