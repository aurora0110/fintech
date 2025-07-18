import numpy as np
import pandas as pd
import calKDJ as kdj

def backTest(file_path, amount, windows, total_shares):
    '''
    回测
    :param file_path: 数据文件路径
    :param amount: 初始资金
    :param windows: 窗口大小
    :param total_shares: 投入总份数
    :return:回测结果
    '''
    #data = pd.read_csv(file_path)
    #data_list = kdj.cal_KDJ(data, 9, 3, 3)

    df = pd.DataFrame(file_path, columns=['date', 'j_value', 'price', 'high_price', 'low_price'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 初始化变量
    cash = amount # 手上还有的现金
    shares_held = 0 # 手上持有的份数
    portfolio_value = cash # 投资组合的价值
    max_investment = 0 # 最大投入金额
    peak_value = cash
    max_drawdown = 0 # 最大回撤


    trade_history = [] # 记录交易历史
    buy_prices = [] # 记录买入价格

    # 记录买入/卖出窗口
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
                cost = price * buy_shares
                cash -= cost
                shares_held += buy_shares
                buy_prices.append(price) # 每次满足条件的买入都放到list里
                trade_history.append({'date': current_date, 'type': 'buy', 'shares': buy_shares, 'price': price}) # 'cost':cost

        # 卖出逻辑 ---------------------------------》这里可以添加上其他的策略作为限制条件，看回测效果如何
        if not recent_sell and shares_held >= 0 and len(buy_prices) > 0:
            sell_shares = 0
            if j_value >= 110:
                sell_shares = 1

            elif j_value >= 100:
                sell_shares = 1

            elif j_value >= 90:
                sell_shares = 1

            elif j_value >= 80:
                sell_shares = 1
            
            # 检查当前价格是否大于 最近买入价格？未卖出价格？ price > buy_prices.max() price > buy_prices.min() price > buy_prices[-1]?
            if sell_shares > 0 and shares_held >= sell_shares and price > min(buy_prices):
                revenue = price * sell_shares
                cash += revenue
                shares_held -= sell_shares
                print(price, buy_prices)
                min_price = min(buy_prices)
                # 移除已买入的价格中最小的买入价格记录
                buy_prices.remove(min_price) if len(buy_prices) > 0 else None
                print(price, buy_prices)
                trade_history.append({'date': current_date, 'type': 'sell', 'shares': sell_shares, 'price': price, 'buy_prices':buy_prices}) # , 'revenue':revenue

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
    print(result)
    return result

sample_data = [
    ['2020-02-17', 114.405, 3.784, 3.706, 3.79], ['2020-10-14', 112.306, 4.599, 4.583, 4.617], ['2020-11-06', 111.732, 4.671, 4.628, 4.682], ['2021-02-10', 112.106, 5.599, 5.489, 5.625], ['2021-04-02', 115.371, 4.953, 4.901, 4.958], ['2021-04-23', 111.797, 4.932, 4.873, 4.942], ['2021-08-10', 112.661, 4.897, 4.794, 4.9], ['2022-06-06', 110.228, 4.043, 3.932, 4.044], ['2022-11-07', 110.679, 3.701, 3.676, 3.722], ['2023-01-05', 113.744, 3.896, 3.839, 3.902], ['2023-01-06', 111.025, 3.912, 3.893, 3.929], ['2023-03-24', 115.537, 3.957, 3.945, 3.965], ['2023-06-15', 112.484, 3.873, 3.811, 3.874], ['2023-06-16', 113.54, 3.91, 3.88, 3.919], ['2023-12-29', 111.292, 3.43, 3.413, 3.436], ['2024-02-19', 115.31, 3.389, 3.355, 3.392], ['2024-02-20', 115.749, 3.401, 3.372, 3.405], ['2024-07-15', 114.198, 3.526, 3.508, 3.531], ['2024-07-16', 116.832, 3.551, 3.517, 3.555], ['2024-07-17', 110.915, 3.556, 3.54, 3.57], ['2024-07-18', 112.246, 3.577, 3.532, 3.579], ['2024-09-24', 111.957, 3.427, 3.293, 3.429], ['2024-09-26', 118.737, 3.64, 3.467, 3.643], ['2024-09-27', 115.047, 3.869, 3.67, 3.933], ['2024-09-30', 117.146, 4.233, 3.876, 4.24], ['2025-04-18', 112.317, 3.873, 3.847, 3.879], ['2025-04-21', 113.088, 3.875, 3.857, 3.886]
    ,['2020-05-22', -11.94, 3.535, 3.533, 3.618], ['2021-02-26', -13.08, 5.121, 5.113, 5.212], ['2021-04-13', -10.036, 4.739, 4.721, 4.786], ['2022-07-12', -13.358, 4.219, 4.205, 4.274], ['2022-07-13', -10.123, 4.226, 4.195, 4.246], ['2022-07-15', -11.636, 4.162, 4.16, 4.267], ['2023-02-08', -11.704, 4.004, 3.996, 4.04], ['2024-01-10', -10.406, 3.272, 3.263, 3.312], ['2024-03-27', -12.819, 3.493, 3.492, 3.532], ['2024-05-31', -11.774, 3.582, 3.58, 3.616], ['2024-07-29', -11.208, 3.454, 3.453, 3.472], ['2024-07-30', -11.625, 3.431, 3.417, 3.446], ['2025-05-28', -14.005, 3.942, 3.938, 3.956]
    ]
results = backTest(sample_data, 20000, 30, 10)