import os
import pandas as pd
import numpy as np
import datetime
import pickle


# ==================== 增强功能配置参数 ====================

REBALANCE_CONFIG = {
    'enabled': False,
    'period': 'monthly',
    'top_n': 10,
    'force_rebalance': True,
}

MULTI_START_CONFIG = {
    'enabled': False,
    'start_dates': ['2022-01-01', '2023-01-01', '2024-01-01'],
}

ROLLING_WINDOW_CONFIG = {
    'enabled': False,
    'window_years': 2,
    'frequency': 'monthly',
}

TURNOVER_CONFIG = {
    'enabled': True,
}


# ==================== 增强功能函数 ====================

def get_rebalance_dates(all_dates, period='monthly'):
    """生成调仓日期列表"""
    dates = pd.to_datetime(all_dates)
    rebalance_dates = []
    
    if period == 'monthly':
        df_dates = pd.DataFrame({'date': dates})
        df_dates['month'] = df_dates['date'].dt.to_period('M')
        rebalance_dates = df_dates.groupby('month').first()['date'].tolist()
    elif period == '20d':
        for i in range(0, len(dates), 20):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    elif period == '5d':
        for i in range(0, len(dates), 5):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    elif period.endswith('d'):
        interval = int(period.replace('d', ''))
        for i in range(0, len(dates), interval):
            if i < len(dates):
                rebalance_dates.append(dates[i])
    
    return [d for d in rebalance_dates if d in all_dates]


def calculate_b1_factor_scores(stock_data, current_date, use_bullish_ma=False):
    """计算B1策略因子得分"""
    scores = {}
    
    for stock_code, df in stock_data.items():
        if current_date not in df.index:
            continue
        
        try:
            idx = df.index.get_loc(current_date)
        except KeyError:
            continue
        
        if idx < 2:
            continue
        
        row = df.iloc[idx]
        
        if pd.isna(row['J']):
            continue
        
        if row['J'] >= -5:
            continue
        
        if use_bullish_ma:
            ma5 = row.get('MA5')
            ma10 = row.get('MA10')
            ma30 = row.get('MA30')
            close = row.get('CLOSE')
            if any(pd.isna(x) for x in [ma5, ma10, ma30, close]):
                continue
            if not (ma5 > ma10 > ma30 and close > ma30):
                continue
        
        score = max(1, (-row['J'] - 5) / 10)
        scores[stock_code] = score
    
    return scores


def run_rebalance_backtest_b1(stock_data, all_dates, use_bullish_ma, rebalance_config, turnover_config):
    """带强制调仓的B1策略回测"""
    max_positions = rebalance_config.get('top_n', 10)
    rebalance_dates = get_rebalance_dates(all_dates, rebalance_config.get('period', 'monthly'))
    
    initial_capital = 1000000
    fee_rate = 0.0003
    slippage = 0.001
    
    cash = float(initial_capital)
    positions = {}
    equity_curve = []
    
    turnover_records = []
    rebalance_count = 0
    
    pending_signals = {}
    
    for i, current_date in enumerate(all_dates):
        if rebalance_config.get('enabled') and current_date in rebalance_dates:
            scores = calculate_b1_factor_scores(stock_data, current_date, use_bullish_ma)
            pending_signals[current_date] = scores
        
        if i > 0 and all_dates[i-1] in pending_signals:
            exec_date = all_dates[i-1]
            scores = pending_signals[exec_date]
            
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_stocks = [s[0] for s in sorted_stocks[:max_positions]]
            
            current_value = cash + sum([pos['shares'] * stock_data[pos['stock']].loc[exec_date, 'CLOSE'] 
                                      if pos['stock'] in stock_data and exec_date in stock_data[pos['stock']].index
                                      and not pd.isna(stock_data[pos['stock']].loc[exec_date, 'CLOSE'])
                                      else 0 for pos in positions.values()])
            
            sell_value = 0
            buy_value = 0
            stocks_to_sell = [s for s in positions.keys() if s not in target_stocks]
            for stock in stocks_to_sell:
                if stock in stock_data and exec_date in stock_data[stock].index:
                    price = stock_data[stock].loc[exec_date, 'CLOSE']
                    if price > 0 and stock in positions:
                        shares = positions[stock]['shares']
                        proceeds = shares * price * (1 - fee_rate - slippage)
                        sell_value += proceeds
                        cash += proceeds
                        del positions[stock]
            
            total_after_sell = cash + sum([pos['shares'] * stock_data[pos['stock']].loc[exec_date, 'CLOSE'] 
                                         if pos['stock'] in stock_data and exec_date in stock_data[pos['stock']].index
                                         and not pd.isna(stock_data[pos['stock']].loc[exec_date, 'CLOSE'])
                                         else 0 for pos in positions.values()])
            
            stocks_to_buy = [s for s in target_stocks if s not in positions]
            target_prices = {}
            for stock in stocks_to_buy:
                if stock in stock_data and current_date in stock_data[stock].index:
                    price = stock_data[stock].loc[current_date, 'OPEN']
                    if price > 0:
                        target_prices[stock] = price
            
            scale_factor = 1.0
            if stocks_to_buy:
                needed_cash = sum([target_prices.get(s, 0) * int(total_after_sell / len(stocks_to_buy) / max(target_prices.get(s, 1), 1) / 100) * 100 
                                 for s in stocks_to_buy if s in target_prices])
                if needed_cash > cash and needed_cash > 0:
                    scale_factor = cash / needed_cash * 0.95
            
            target_value_per_stock = (total_after_sell * scale_factor) / max_positions
            
            for stock in stocks_to_buy:
                if stock not in target_prices:
                    continue
                price = target_prices[stock]
                
                shares = int(target_value_per_stock / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + fee_rate + slippage)
                    if cost <= cash:
                        cash -= cost
                        buy_value += cost
                        positions[stock] = {
                            'stock': stock,
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': current_date
                        }
            
            while len(positions) < max_positions and stocks_to_buy:
                remaining_stocks = [s for s in target_stocks if s not in positions]
                if not remaining_stocks:
                    break
                next_stock = remaining_stocks[0]
                if next_stock in target_prices:
                    price = target_prices[next_stock]
                    shares = int(target_value_per_stock / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + fee_rate + slippage)
                        if cost <= cash:
                            cash -= cost
                            buy_value += cost
                            positions[next_stock] = {
                                'stock': next_stock,
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': current_date
                            }
                stocks_to_buy.remove(next_stock)
            
            if turnover_config.get('enabled') and current_value > 0:
                turnover = (sell_value + buy_value) / current_value
                turnover_records.append(turnover)
            
            if sell_value + buy_value > 0:
                rebalance_count += 1
        
        total_value = cash
        for stock, pos in positions.items():
            if stock in stock_data and current_date in stock_data[stock].index:
                price = stock_data[stock].loc[current_date, 'CLOSE']
                if not pd.isna(price) and price > 0:
                    total_value += pos['shares'] * price
        
        equity_curve.append(total_value)
    
    if len(equity_curve) < 2:
        return None
    
    final_multiple = equity_curve[-1] / initial_capital
    returns_arr = np.array([(equity_curve[i+1] - equity_curve[i]) / equity_curve[i] 
                           for i in range(len(equity_curve)-1) if equity_curve[i] > 0])
    
    sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252) if returns_arr.std() > 0 else 0
    
    years = len(all_dates) / 252
    CAGR = (final_multiple ** (1 / years)) - 1 if years > 0 and final_multiple > 0 else -1
    
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_dd = drawdown.min()
    
    avg_turnover = np.mean(turnover_records) if turnover_records else 0
    
    return {
        'desc': f'强制调仓: {rebalance_config.get("period")}',
        'trade_count': rebalance_count,
        'CAGR': CAGR * 100,
        'final_multiple': final_multiple,
        'max_dd': max_dd * 100,
        'sharpe': sharpe,
        'turnover': avg_turnover,
        'rebalance_count': rebalance_count,
    }


def run_multi_start_backtest_b1(stock_data, all_dates, config, use_bullish_ma):
    """B1策略多起点回测"""
    results = []
    
    for start_date_str in config.get('start_dates', []):
        start_date = pd.to_datetime(start_date_str)
        valid_dates = [d for d in all_dates if d >= start_date]
        if not valid_dates:
            continue
        
        start_date = valid_dates[0]
        
        if len(valid_dates) < 60:
            continue
        
        result = run_rebalance_backtest_b1(
            stock_data, valid_dates, use_bullish_ma,
            {'enabled': True, 'period': config.get('period', 'monthly'), 'top_n': config.get('top_n', 10)},
            {'enabled': True}
        )
        
        if result:
            result['start_date'] = start_date
            results.append(result)
    
    if not results:
        return None
    
    returns = [r['CAGR'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    max_dds = [r['max_dd'] for r in results]
    
    return {
        'results': results,
        'summary': {
            'avg_annual_return': np.mean(returns),
            'std_annual_return': np.std(returns),
            'min_annual_return': np.min(returns),
            'max_annual_return': np.max(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_max_dd': np.mean(max_dds),
            'count': len(results)
        }
    }


def run_rolling_window_test_b1(stock_data, all_dates, config, use_bullish_ma):
    """B1策略滚动窗口测试"""
    window_years = config.get('window_years', 2)
    frequency = config.get('frequency', 'monthly')
    window_days = window_years * 252
    results = []
    
    if frequency == 'monthly':
        df_dates = pd.DataFrame({'date': all_dates})
        df_dates['month'] = df_dates['date'].dt.to_period('M')
        start_indices = df_dates.groupby('month').first().index
        start_dates = [all_dates[i] for i in range(len(all_dates)) 
                     if i in start_indices or (i > 0 and df_dates.iloc[i]['month'] != df_dates.iloc[i-1]['month'])]
    else:
        start_dates = all_dates[::20]
    
    for start_date in start_dates:
        start_idx = all_dates.index(start_date)
        end_idx = start_idx + window_days
        
        if end_idx >= len(all_dates):
            continue
        
        window_dates = all_dates[start_idx:end_idx]
        
        result = run_rebalance_backtest_b1(
            stock_data, window_dates, use_bullish_ma,
            {'enabled': True, 'period': config.get('period', 'monthly'), 'top_n': config.get('top_n', 10)},
            {'enabled': True}
        )
        
        if result:
            result['start_date'] = start_date
            result['end_date'] = window_dates[-1]
            results.append(result)
    
    if not results:
        return None
    
    returns = [r['CAGR'] for r in results]
    
    return {
        'results': results,
        'distribution': {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns),
            'count': len(results)
        }
    }


def write_log(content):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    log_file = os.path.join(script_dir, script_name + ".txt")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_record = f"\n========== {timestamp } ==========\n"
    new_record += content + "\n\n"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            old = f.read()
    else:
        old = ""

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(new_record + old)


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


def check_data_anomaly(df):
    anomaly_dates = set()
    
    if len(df) < 2:
        return anomaly_dates
    
    for i in range(len(df)):
        row = df.iloc[i]
        open_p = row['OPEN']
        high = row['HIGH']
        low = row['LOW']
        close = row['CLOSE']
        volume = row['VOLUME']
        
        if pd.isna(open_p) or pd.isna(high) or pd.isna(low) or pd.isna(close):
            anomaly_dates.add(df.index[i])
            continue
        
        if high == low and low == close:
            anomaly_dates.add(df.index[i])
            continue
        
        if i > 0:
            prev_close = df.iloc[i-1]['CLOSE']
            if not pd.isna(prev_close) and prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 20 or change_pct < -20:
                    anomaly_dates.add(df.index[i])
                    continue
        
        if volume <= 0:
            anomaly_dates.add(df.index[i])
            continue
        
        if close > 0 and open_p > 0:
            if close > high or close < low or open_p > high or open_p < low:
                anomaly_dates.add(df.index[i])
                continue
        
        if i > 0:
            prev_open = df.iloc[i-1]['OPEN']
            prev_close = df.iloc[i-1]['CLOSE']
            if pd.isna(prev_open) or pd.isna(prev_close):
                anomaly_dates.add(df.index[i])
                continue
    
    return anomaly_dates


def calculate_indicators(df):
    close = df['CLOSE']
    open_p = df['OPEN']
    high = df['HIGH']
    low = df['LOW']
    volume = df['VOLUME']
    
    df['涨跌幅'] = close.pct_change() * 100
    
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA30'] = close.rolling(30).mean()
    df['MA60'] = close.rolling(60).mean()
    
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df['MACD_DIF'] = exp12 - exp26
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_MACD'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    df['RSI28'] = df['RSI14'].rolling(28).mean()
    df['RSI57'] = df['RSI14'].rolling(57).mean()
    
    df['是否阳线'] = close > open_p
    df['是否阴线'] = close < open_p
    
    df['VOLUME_MA5'] = volume.rolling(5).mean()
    df['VOLUME_MA20'] = volume.rolling(20).mean()
    df['VOLUME_MA60'] = volume.rolling(60).mean()
    
    df['知行短期趋势线'] = close.ewm(span=5, adjust=False).mean()
    df['知行多空线'] = close.ewm(span=10, adjust=False).mean()
    
    high = df['HIGH']
    low = df['LOW']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(window=14).mean()
    
    return df


def is_bullish_ma(df, idx):
    """
    判断是否满足日线多头条件:
    1. 5日均线 > 10日均线
    2. 10日均线 > 30日均线
    3. 收盘价位于30日均线上方
    """
    if idx < 1:
        return False
    
    row = df.iloc[idx]
    
    ma5 = row.get('MA5')
    ma10 = row.get('MA10')
    ma30 = row.get('MA30')
    close = row.get('CLOSE')
    
    if any(pd.isna(x) for x in [ma5, ma10, ma30, close]):
        return False
    
    cond1 = ma5 > ma10
    cond2 = ma10 > ma30
    cond3 = close > ma30
    
    return cond1 and cond2 and cond3


def calculate_b1_score(df, idx, use_bullish_ma=False):
    if idx < 2:
        return 0, {}
    
    row = df.iloc[idx]
    
    if pd.isna(row['J']):
        return 0, {}
    
    if row['J'] >= -5:
        return 0, {}
    
    if use_bullish_ma:
        if not is_bullish_ma(df, idx):
            return 0, {}
    
    # 根据J值的大小计算分数，J值越小，分数越高
    score = max(1, (-row['J'] - 5) / 10)
    details = {'J': row['J'], 'score': score}
    
    return score, details


def run_backtest(data_dir, min_score=0.5, use_bullish_ma=False, desc=""):
    initial_capital = 1000000
    fee_rate = 0.0003
    slippage = 0.001
    
    stock_data = {}
    daily_signals = {}
    daily_scores = {}
    all_dates = []
    
    print("加载数据...")
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    loaded_count = 0
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        # 排除北交所股票（BJ开头或代码中包含北交所标识）
        if stock_code.startswith('BJ'):
            continue
        
        # 获取股票代码部分（去掉市场前缀）
        if '#' in stock_code:
            code_part = stock_code.split('#')[1]
        else:
            code_part = stock_code
        
        # 排除北交所股票（8开头且第二位是3，或83、87开头）
        if code_part.startswith('8') and len(code_part) >= 3:
            if code_part[1] == '3' or code_part.startswith('83') or code_part.startswith('87'):
                continue
        
        # 确定涨跌幅限制
        limit_pct = 0.10  # 默认主板10%
        if code_part.startswith('300'):
            limit_pct = 0.20  # 创业板20%
        elif code_part.startswith('688'):
            limit_pct = 0.30  # 科创板30%
        
        path = os.path.join(data_dir, file)
        
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            df['limit_pct'] = limit_pct
            
            anomaly_dates = check_data_anomaly(df)
            if len(anomaly_dates) > len(df) * 0.1:
                continue
            
            df = calculate_indicators(df)
            stock_data[stock_code] = df
            loaded_count += 1
            
            for i in range(2, len(df)):
                score, details = calculate_b1_score(df, i, use_bullish_ma)
                if score >= min_score:
                    date = df.index[i]
                    if date not in daily_signals:
                        daily_signals[date] = []
                        daily_scores[date] = []
                    daily_signals[date].append(stock_code)
                    daily_scores[date].append((stock_code, score, details))
            
            # 加载全部数据，不限制数量
        
        except Exception as e:
            print(f"加载失败 {file}: {e}")
            continue
    
    print(f"成功加载 {loaded_count} 只股票")
    
    # 收集所有日期
    all_dates = []
    for df in stock_data.values():
        all_dates.extend(df.index.tolist())
    all_dates = sorted(set(all_dates))
    
    print(f"总交易日: {len(all_dates)}")
    print(f"有信号的天数: {len(daily_signals)}")
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    
    trade_count = 0
    win_count = 0
    loss_count = 0
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    pause_trading_days = 0
    
    yearly_returns = {}
    holding_returns = []  # 存储所有已完成交易的收益率
    exit_records = []  # 存储卖出记录
    
    # 生成待执行的买入信号（当日收盘后生成，次日执行）
    pending_buy_signals = {}
    for date in daily_signals:
        date_idx = all_dates.index(date)
        if date_idx + 1 < len(all_dates):
            next_date = all_dates[date_idx + 1]
            if next_date not in pending_buy_signals:
                pending_buy_signals[next_date] = []
            pending_buy_signals[next_date].extend(daily_scores[date])
    
    pending_exit_signals = {}
    first_trade_date = None
    last_trade_date = None
    
    for current_date in all_dates:
        if stopped:
            break
        
        if pause_trading_days > 0:
            pause_trading_days -= 1
        
        current_idx = date_to_idx[current_date]
        new_positions = []
        
        for pos in positions:
            stock = pos["stock"]
            df = stock_data[stock]
            
            if current_date not in df.index:
                new_positions.append(pos)
                continue
            
            row = df.loc[current_date]
            
            if pd.isna(row["CLOSE"]) or row["CLOSE"] <= 0:
                new_positions.append(pos)
                continue
            
            open_p = row["OPEN"]
            high = row["HIGH"]
            low = row["LOW"]
            close = row["CLOSE"]
            
            entry_price = pos["entry_price"]
            entry_idx = pos["entry_idx"]
            holding_days = current_idx - entry_idx
            
            if current_idx < entry_idx + 1:
                new_positions.append(pos)
                continue
            
            entry_low = pos["entry_low"]
            stop_price = entry_low * 0.95
            
            exit_flag = False
            exit_price = None
            exit_reason = ""
            
            if current_date in pending_exit_signals and stock in pending_exit_signals[current_date]:
                exit_flag = True
                exit_info = pending_exit_signals[current_date][stock]
                exit_price = exit_info['exit_price']
                exit_reason = exit_info['exit_reason']
            elif not pos.get('exit_marked', False):
                if close < stop_price:
                    if current_idx + 1 < len(all_dates):
                        next_day = all_dates[current_idx + 1]
                        if next_day not in pending_exit_signals:
                            pending_exit_signals[next_day] = {}
                        pending_exit_signals[next_day][stock] = {
                            'exit_price': row['OPEN'],
                            'exit_reason': '止损'
                        }
                        pos['exit_marked'] = True
                
                exit_days = [3, 5, 10, 20, 30]
                if holding_days in exit_days:
                    if current_idx + 1 < len(all_dates):
                        next_day = all_dates[current_idx + 1]
                        if next_day not in pending_exit_signals:
                            pending_exit_signals[next_day] = {}
                        pending_exit_signals[next_day][stock] = {
                            'exit_price': row['OPEN'],
                            'exit_reason': f'到期{holding_days}日卖出'
                        }
                        pos['exit_marked'] = True
            
            if exit_flag:
                sell_price = exit_price
                
                prev_row = df.iloc[current_idx - 1] if current_idx > 0 else None
                if prev_row is not None and 'limit_pct' in df.columns:
                    prev_close = prev_row['CLOSE']
                    if prev_close > 0:
                        limit_down = prev_close * (1 - df.iloc[current_idx]['limit_pct'])
                        if sell_price <= limit_down:
                            sell_price = limit_down
                
                sell_value = pos['shares'] * sell_price * (1 - fee_rate - slippage)
                cash += sell_value
                
                pnl = (sell_price - entry_price) / entry_price
                trade_count += 1
                holding_returns.append(pnl)
                
                if pnl > 0:
                    win_count += 1
                    current_consecutive_losses = 0
                else:
                    loss_count += 1
                    current_consecutive_losses += 1
                    if current_consecutive_losses > max_consecutive_losses:
                        max_consecutive_losses = current_consecutive_losses
                
                if current_consecutive_losses >= 3:
                    pause_trading_days = 2
                
                # 记录卖出事件，用于后续统计
                exit_record = {
                    'stock': pos['stock'],
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'holding_days': holding_days
                }
                exit_records.append(exit_record)
            else:
                new_positions.append(pos)
        
        positions = new_positions
        
        # 执行前一天发出的买入信号（次日以开盘价买入）
        if current_date in pending_buy_signals and pause_trading_days == 0:
            candidates = pending_buy_signals[current_date]
            
            if candidates:
                candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
                top_stock = candidates_sorted[0][0]
                
                df = stock_data[top_stock]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                open_price = row['OPEN']
                
                if open_price <= 0:
                    continue
                
                # 检查当日趋势线>多空线
                trend_line = row.get('知行短期趋势线')
                dk_line = row.get('知行多空线')
                if pd.isna(trend_line) or pd.isna(dk_line):
                    continue
                if trend_line <= dk_line:
                    continue
                
                # 如果使用日线多头条件，检查日线多头排列
                if use_bullish_ma:
                    ma5 = row.get('MA5')
                    ma10 = row.get('MA10')
                    ma30 = row.get('MA30')
                    close = row.get('CLOSE')
                    if any(pd.isna(x) for x in [ma5, ma10, ma30, close]):
                        continue
                    if not (ma5 > ma10 > ma30 and close > ma30):
                        continue
                
                # 全仓交易，使用最低价的0.95作为止损
                stop_loss = row['LOW'] * 0.95
                
                prev_row = df.iloc[current_idx - 1] if current_idx > 0 else None
                limit_up = None
                limit_down = None
                if prev_row is not None and 'limit_pct' in df.columns:
                    prev_close = prev_row['CLOSE']
                    if prev_close > 0:
                        limit_up = prev_close * (1 + df.iloc[current_idx]['limit_pct'])
                        limit_down = prev_close * (1 - df.iloc[current_idx]['limit_pct'])
                        if open_price >= limit_up:
                            continue
                
                shares = int(cash / open_price / 100) * 100
                if shares > 0:
                    cost = shares * open_price * (1 + fee_rate + slippage)
                    if cost <= cash:
                        cash -= cost
                        
                        positions.append({
                            'stock': top_stock,
                            'shares': shares,
                            'entry_price': open_price,
                            'entry_low': row['LOW'],
                            'entry_date': current_date,
                            'entry_idx': current_idx
                        })
        
        position_value = 0
        for pos in positions:
            stock = pos['stock']
            if stock in stock_data and current_date in stock_data[stock].index:
                price = stock_data[stock].loc[current_date, 'CLOSE']
                if pd.isna(price) or price <= 0:
                    df_s = stock_data[stock]
                    valid_prices = df_s.loc[:current_date]['CLOSE']
                    valid_prices = valid_prices[valid_prices > 0]
                    if len(valid_prices) > 0:
                        price = valid_prices.iloc[-1]
                if price > 0:
                    position_value += pos['shares'] * price
        
        total_value = cash + position_value
        equity_curve.append(total_value)
        
        year = current_date.year
        if year not in yearly_returns:
            yearly_returns[year] = []
        if len(equity_curve) > 1:
            daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            yearly_returns[year].append(daily_return)
        
        if positions and first_trade_date is None:
            first_trade_date = current_date
        last_trade_date = current_date
    
    final_multiple = equity_curve[-1] / initial_capital if equity_curve else 1
    returns_arr = np.array([(equity_curve[i+1] - equity_curve[i]) / equity_curve[i] 
                           for i in range(len(equity_curve)-1) if equity_curve[i] > 0])
    
    if len(returns_arr) > 0:
        sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(252) if returns_arr.std() > 0 else 0
    else:
        sharpe = 0
    
    if first_trade_date and last_trade_date:
        years = (last_trade_date - first_trade_date).days / 365.25
        years = max(years, 0.01)
    else:
        years = len(all_dates) / 252
    
    if years > 0 and final_multiple > 0:
        CAGR = (final_multiple ** (1 / years)) - 1
    else:
        CAGR = -1
    
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_dd = drawdown.min()
    
    success_rate = win_count / trade_count * 100 if trade_count > 0 else 0
    
    # 持有期间收益率已经在交易完成时记录
    avg_holding_return = np.mean(holding_returns) * 100 if holding_returns else 0
    max_holding_return = np.max(holding_returns) * 100 if holding_returns else 0
    
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"初始资金: {initial_capital:,.0f}")
    print(f"最终资金: {equity_curve[-1]:,.2f}")
    print(f"最终倍数: {final_multiple:.2f}")
    print(f"年化收益率(CAGR): {CAGR*100:.2f}%")
    print(f"最大回撤: {max_dd*100:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"总交易次数: {trade_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均持有期间收益率: {avg_holding_return:.2f}%")
    print(f"最大持有期间收益率: {max_holding_return:.2f}%")
    print(f"最大连续失败次数: {max_consecutive_losses}")
    
    return {
        'desc': desc,
        'initial_capital': initial_capital,
        'final_capital': equity_curve[-1],
        'final_multiple': final_multiple,
        'CAGR': CAGR * 100,
        'max_dd': max_dd * 100,
        'sharpe': sharpe,
        'trade_count': trade_count,
        'success_rate': success_rate,
        'avg_holding_return': avg_holding_return,
        'max_holding_return': max_holding_return,
        'max_consecutive_losses': max_consecutive_losses,
        'exit_records': exit_records,
        'stock_data': stock_data
    }


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    
    print("=" * 70)
    print("对比测试：J<-5+趋势线>多空线 vs J<-5+趋势线>多空线+日线多头排列")
    print("=" * 70)
    
    result1 = run_backtest(data_dir, min_score=0.5, use_bullish_ma=False, desc="测试1: J<-5+趋势线>多空线")
    
    result2 = run_backtest(data_dir, min_score=0.5, use_bullish_ma=True, desc="测试2: J<-5+趋势线>多空线+日线多头排列")
    
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    print(f"{'指标':<25} {'J<-5+趋势线>多空线':<18} {'J<-5+趋势线>多空线+日线多头':<18}")
    print("-" * 70)
    print(f"{'平均年化收益率':<20} {result1['CAGR']:.2f}%{'':<8} {result2['CAGR']:.2f}%")
    print(f"{'最大年化收益率':<20} {result1['final_multiple']:.2f}x{'':<8} {result2['final_multiple']:.2f}x")
    print(f"{'平均持有期间收益率':<18} {result1['avg_holding_return']:.2f}%{'':<8} {result2['avg_holding_return']:.2f}%")
    print(f"{'最大持有期间收益率':<16} {result1['max_holding_return']:.2f}%{'':<8} {result2['max_holding_return']:.2f}%")
    print(f"{'最大回撤率':<22} {result1['max_dd']:.2f}%{'':<8} {result2['max_dd']:.2f}%")
    print(f"{'最大连续失败次数':<17} {result1['max_consecutive_losses']}{'':<12} {result2['max_consecutive_losses']}")
    print(f"{'夏普比率':<22} {result1['sharpe']:.2f}{'':<12} {result2['sharpe']:.2f}")
    
    print("\n" + "=" * 70)
    print("卖出后N日涨跌概率统计")
    print("=" * 70)
    
    stock_data = result1.get('stock_data', {})
    
    for days in [3, 5, 10, 30]:
        print(f"\n--- 卖出后{days}日 ---")
        for idx, result in enumerate([result1, result2], 1):
            exit_records = result.get('exit_records', [])
            up_count = 0
            down_count = 0
            total = 0
            
            for record in exit_records:
                stock = record['stock']
                exit_date = record['exit_date']
                exit_price = record['exit_price']
                
                df = stock_data.get(stock)
                if df is None:
                    continue
                
                try:
                    exit_idx = df.index.get_loc(exit_date)
                    future_idx = exit_idx + days
                    if future_idx < len(df):
                        future_price = df.iloc[future_idx]['CLOSE']
                        if pd.notna(future_price) and exit_price > 0:
                            if future_price > exit_price:
                                up_count += 1
                            else:
                                down_count += 1
                            total += 1
                except:
                    continue
            
            if total > 0:
                up_pct = up_count / total * 100
                down_pct = down_count / total * 100
                print(f"测试{idx}: 上涨概率 {up_pct:.1f}% ({up_count}/{total}), 下跌概率 {down_pct:.1f}% ({down_count}/{total})")
            else:
                print(f"测试{idx}: 无数据")
