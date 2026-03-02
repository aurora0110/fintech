import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ===========================
# 参数设置
# ===========================

STRATEGY_TYPE = "G"  # "A"=持有1天次日开盘卖, "B"=持有1天次日收盘卖, "C"=出现阴线次日开盘卖, "D"=出现阴线当天收盘卖, "E"=出现阴线次日收盘卖, "F"=出现阴线后次日次日开盘卖, "G"=出现阴线后次日次日收盘卖

# ===========================
# 涨跌幅限制
# ===========================

def is_chi_next_or_star(stock_code):
    """判断是否为创业板或科创板股票"""
    code = stock_code.upper()
    if code.startswith("300") or code.startswith("688"):
        return True
    return False

def limit_price_change(price, prev_price, stock_code, direction="both"):
    """
    限制涨跌幅
    direction: "up"=涨停, "down"=跌停, "both"=双向限制
    """
    if prev_price <= 0 or price <= 0:
        return price
    
    change_pct = (price - prev_price) / prev_price
    
    if is_chi_next_or_star(stock_code):
        max_change = 0.20  # 创业板/科创板 20%
    else:
        max_change = 0.10  # 主板 10%
    
    if direction == "up":
        max_change = max_change
    elif direction == "down":
        max_change = -max_change
    else:
        max_change = max_change
    
    if change_pct > max_change:
        return prev_price * (1 + max_change)
    elif change_pct < -max_change:
        return prev_price * (1 - max_change)
    
    return price

# ===========================
# 股票过滤
# ===========================

def is_valid_stock(stock_code):
    code = stock_code.upper()

    if "ST" in code:
        return False
    if code.startswith("688"):  # 科创板
        return False
    if code.startswith("92"):    # 北交所
        return False

    return True

# ===========================
# 指标计算
# ===========================

def calculate_pin_conditions(df):
    N1, N2 = 3, 21

    llv_l_n1 = df['最低'].rolling(N1).min()
    hhv_c_n1 = df['收盘'].rolling(N1).max()
    df['短期'] = (df['收盘'] - llv_l_n1) / (hhv_c_n1 - llv_l_n1 + 1e-6) * 100

    llv_l_n2 = df['最低'].rolling(N2).min()
    hhv_l_n2 = df['收盘'].rolling(N2).max()
    df['长期'] = (df['收盘'] - llv_l_n2) / (hhv_l_n2 - llv_l_n2 + 1e-6) * 100

    df['PIN信号'] = (df['短期'] <= 30) & (df['长期'] >= 85)

    return df


def calculate_trend(df):
    df['知行短期趋势线'] = df['收盘'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()

    df['MA14'] = df['收盘'].rolling(14).mean()
    df['MA28'] = df['收盘'].rolling(28).mean()
    df['MA57'] = df['收盘'].rolling(57).mean()
    df['MA114'] = df['收盘'].rolling(114).mean()

    df['知行多空线'] = (
        df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']
    ) / 4

    return df

# ===========================
# 加载数据
# ===========================

def load_all_data(data_dir):
    data_dict = {}

    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue

        stock_code = file.replace(".txt", "")

        if not is_valid_stock(stock_code):
            continue

        path = os.path.join(data_dir, file)

        try:
            df = pd.read_csv(path, sep='\s+', encoding='utf-8')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)

            if len(df) < 120:
                continue

            df = calculate_trend(df)
            df = calculate_pin_conditions(df)

            data_dict[stock_code] = df

        except:
            continue

    return data_dict

# ===========================
# 回测逻辑（单股串行）
# ===========================

def run_backtest(data_dict):

    trade_returns = []
    trade_details = []  # 记录每笔交易的详细信息
    position = None  # 当前持仓（单股）

    # 获取全市场交易日
    all_dates = sorted(
        list(
            set(
                date
                for df in data_dict.values()
                for date in df['日期']
            )
        )
    )

    for current_date in tqdm(all_dates):

        # ======================
        # 1️⃣ 处理卖出
        # ======================

        if position:

            stock = position['stock']
            df = data_dict[stock]

            if current_date not in df['日期'].values:
                continue

            idx = df.index[df['日期'] == current_date][0]
            row = df.iloc[idx]

            # 获取前一日收盘价
            if idx > 0:
                prev_row = df.iloc[idx - 1]
                prev_close = prev_row['收盘']
            else:
                prev_close = row['收盘']

            # 策略A：持有1天，次日开盘卖出
            if STRATEGY_TYPE == "A":
                if position['holding_days'] == 1:
                    if idx + 1 < len(df):
                        next_row = df.iloc[idx + 1]
                        exit_price = next_row['开盘']
                        exit_price = limit_price_change(exit_price, prev_close, stock)
                        ret = (exit_price - position['entry_price']) / position['entry_price']
                        trade_returns.append(ret)
                        trade_details.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': next_row['日期'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return': ret,
                            'return_pct': ret * 100
                        })
                        position = None

            # 策略B：持有1天，次日收盘卖出
            elif STRATEGY_TYPE == "B":
                if position['holding_days'] == 1:
                    exit_price = row['收盘']
                    exit_price = limit_price_change(exit_price, prev_close, stock)
                    ret = (exit_price - position['entry_price']) / position['entry_price']
                    trade_returns.append(ret)
                    trade_details.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': ret,
                        'return_pct': ret * 100
                    })
                    position = None

            # 策略C：出现阴线，次日开盘卖出
            elif STRATEGY_TYPE == "C":
                if row['收盘'] < row['开盘']:
                    if idx + 1 < len(df):
                        next_row = df.iloc[idx + 1]
                        exit_prev_close = row['收盘']
                        exit_price = next_row['开盘']
                        exit_price = limit_price_change(exit_price, exit_prev_close, stock)
                        ret = (exit_price - position['entry_price']) / position['entry_price']
                        trade_returns.append(ret)
                        trade_details.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': next_row['日期'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return': ret,
                            'return_pct': ret * 100
                        })
                        position = None
            
            # 策略D：出现阴线，当天收盘卖出
            elif STRATEGY_TYPE == "D":
                if row['收盘'] < row['开盘']:
                    exit_price = row['收盘']
                    exit_price = limit_price_change(exit_price, prev_close, stock)
                    ret = (exit_price - position['entry_price']) / position['entry_price']
                    trade_returns.append(ret)
                    trade_details.append({
                        'stock': stock,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': ret,
                        'return_pct': ret * 100
                    })
                    position = None
            
            # 策略E：出现阴线，次日收盘卖出
            elif STRATEGY_TYPE == "E":
                if row['收盘'] < row['开盘']:
                    if idx + 1 < len(df):
                        next_row = df.iloc[idx + 1]
                        exit_prev_close = row['收盘']
                        exit_price = next_row['收盘']
                        exit_price = limit_price_change(exit_price, exit_prev_close, stock)
                        ret = (exit_price - position['entry_price']) / position['entry_price']
                        trade_returns.append(ret)
                        trade_details.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': next_row['日期'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return': ret,
                            'return_pct': ret * 100
                        })
                        position = None
            
            # 策略F：出现阴线后，次日次日开盘卖出
            elif STRATEGY_TYPE == "F":
                if position['holding_days'] == 2:
                    if idx + 1 < len(df):
                        next_row = df.iloc[idx + 1]
                        exit_price = next_row['开盘']
                        exit_price = limit_price_change(exit_price, prev_close, stock)
                        ret = (exit_price - position['entry_price']) / position['entry_price']
                        trade_returns.append(ret)
                        trade_details.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': next_row['日期'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'return': ret,
                            'return_pct': ret * 100
                        })
                        position = None
            
            # 策略G：出现阴线后，次日次日收盘卖出
            elif STRATEGY_TYPE == "G":
                if position['holding_days'] == 2:
                    exit_price = row['收盘']
                    exit_price = limit_price_change(exit_price, prev_close, stock)
                    ret = (exit_price - position['entry_price']) / position['entry_price']
                    trade_returns.append(ret)
                    trade_details.append({
                        'stock': stock,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'return': ret,
                        'return_pct': ret * 100
                    })
                    position = None

            if position:
                position['holding_days'] += 1

        # ======================
        # 2️⃣ 处理买入
        # ======================

        if not position:

            for stock, df in data_dict.items():

                if current_date not in df['日期'].values:
                    continue

                idx = df.index[df['日期'] == current_date][0]

                if idx + 1 >= len(df):
                    continue

                row = df.iloc[idx]

                buy_condition = (
                    row['PIN信号'] and
                    row['知行短期趋势线'] > row['知行多空线'] and
                    row['收盘'] > row['知行多空线']
                )

                if buy_condition:
                    next_row = df.iloc[idx + 1]
                    entry_price = next_row['开盘']
                    
                    # 应用涨跌幅限制（买入价格基于当天收盘价）
                    entry_price = limit_price_change(entry_price, row['收盘'], stock)

                    if entry_price <= 0:
                        continue

                    position = {
                        "stock": stock,
                        "entry_price": entry_price,
                        "entry_date": next_row['日期'],
                        "holding_days": 0
                    }

                    break  # 单股串行

    return trade_returns, trade_details

# ===========================
# 统计分析
# ===========================

def analyze_returns(trade_returns):

    returns = np.array(trade_returns)

    if len(returns) == 0:
        print("无交易")
        return

    win_trades = returns[returns > 0]
    loss_trades = returns[returns <= 0]

    win_rate = len(win_trades) / len(returns)
    avg_return = returns.mean()
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    expectation = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    print("\n====== 单笔统计结果 ======")
    print(f"总交易次数: {len(returns)}")
    print(f"胜率: {win_rate * 100:.2f}%")
    print(f"平均单笔收益: {avg_return * 100:.2f}%")
    print(f"平均盈利: {avg_win * 100:.2f}%")
    print(f"平均亏损: {avg_loss * 100:.2f}%")
    print(f"盈亏比: {profit_loss_ratio:.2f}")
    print(f"单笔期望值: {expectation * 100:.2f}%")
    print(f"最大盈利: {returns.max() * 100:.2f}%")
    print(f"最大亏损: {returns.min() * 100:.2f}%")
    print(f"收益标准差: {returns.std() * 100:.2f}%")

    print("\n====== 持有期间收益率分布 ======")
    returns_pct = returns * 100
    
    bins = [(-200, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 200)]
    print(f"{'收益率区间':<15} {'交易次数':<10} {'占比':<10}")
    print("-" * 35)
    for low, high in bins:
        count = np.sum((returns_pct >= low) & (returns_pct < high))
        pct = count / len(returns) * 100 if len(returns) > 0 else 0
        print(f"[{low:>5}%, {high:<5}%)    {count:<10} {pct:.2f}%")
    
    median_return = np.median(returns) * 100
    print(f"\n收益率中位数: {median_return:.2f}%")
    
    if len(returns) >= 2:
        returns_sorted = np.sort(returns)
        print(f"25分位数: {returns_sorted[len(returns)//4] * 100:.2f}%")
        print(f"75分位数: {returns_sorted[len(returns)*3//4] * 100:.2f}%")
    
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    print(f"VaR(95%): {var_95:.2f}%")
    print(f"CVaR(95%): {cvar_95:.2f}%")

# ===========================
# 主程序
# ===========================

if __name__ == "__main__":

    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

    data_dict = load_all_data(data_dir)

    trade_returns, trade_details = run_backtest(data_dict)

    analyze_returns(trade_returns)