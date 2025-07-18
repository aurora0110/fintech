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

    df = pd.DataFrame(file_path, columns=['date', 'j_value', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values.reset_index('date')

    # 初始化变量
    cash = amount
    shares_held = 0
    portfolio_value = cash
    max_investment = 0
    peak_value = cash
    max_drawdown = 0

    # 记录交易历史
    trade_history = []

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
                if trade['type'] == 'buy':
                    recent_buy = True
                elif trade['type'] == 'sell':
                    recent_sell = True
        
        # 买入逻辑
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

                trade_history.append({'date': current_date, 'type': 'buy', 'shares': buy_shares, 'price': price, 'cost':cost})

        # 卖出逻辑
        if not recent_sell and shares_held > 0:
            sell_shares = 0
            if j_value >= 110:
                sell_shares = 1

            elif j_value >= 100:
                sell_shares = 1

            elif j_value >= 90:
                sell_shares = 1

            elif j_value >= 80:
                sell_shares = 1
            
            if sell_shares > 0 and shares_held >= sell_shares:
                revenue = price * sell_shares
                cash += revenue
                shares_held -= sell_shares

                trade_history.append({'date': current_date, 'type': 'sell', 'shares': sell_shares, 'price': price, 'revenue':revenue})

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

    return {
        'amount':amount,
        'final_value': final_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'max_investment': max_investment,
        'trade_history': trade_history_df,
        'portfolio_value_over_time': portfolio_value    }

sample_data = [
    ['2020-05-22', -11.94], ['2021-02-26', -13.08], ['2021-04-13', -10.036], ['2022-07-12', -13.358], ['2022-07-13', -10.123], ['2022-07-15', -11.636], ['2023-02-08', -11.704], ['2024-01-10', -10.406], ['2024-03-27', -12.819], ['2024-05-31', -11.774], ['2024-07-29', -11.208], ['2024-07-30', -11.625], ['2025-05-28', -14.005],
['2020-02-17', 114.405], ['2020-10-14', 112.306], ['2020-11-06', 111.732], ['2021-02-10', 112.106], ['2021-04-02', 115.371], ['2021-04-23', 111.797], ['2021-08-10', 112.661], ['2022-06-06', 110.228], ['2022-11-07', 110.679], ['2023-01-05', 113.744], ['2023-01-06', 111.025], ['2023-03-24', 115.537], ['2023-06-15', 112.484], ['2023-06-16', 113.54], ['2023-12-29', 111.292], ['2024-02-19', 115.31], ['2024-02-20', 115.749], ['2024-07-15', 114.198], ['2024-07-16', 116.832], ['2024-07-17', 110.915], ['2024-07-18', 112.246], ['2024-09-24', 111.957], ['2024-09-26', 118.737], ['2024-09-27', 115.047], ['2024-09-30', 117.146], ['2025-04-18', 112.317], ['2025-04-21', 113.088]
]

results = backTest(sample_data, 20000, 30, 10)