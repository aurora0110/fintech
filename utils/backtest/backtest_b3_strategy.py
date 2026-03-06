import os
import pandas as pd
import numpy as np
from datetime import datetime


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


def calculate_b3_factor_scores(stock_data, current_date, min_score=0.5):
    """计算B3策略因子得分"""
    scores = {}
    
    for stock_code, df in stock_data.items():
        if current_date not in df.index:
            continue
        
        try:
            idx = df.index.get_loc(current_date)
        except KeyError:
            continue
        
        if idx < 10:
            continue
        
        row = df.iloc[idx]
        
        if pd.isna(row.get('知行多空线')) or pd.isna(row.get('知行短期趋势线')):
            continue
        
        trend_ok = row['知行多空线'] <= row['知行短期趋势线']
        j_ok = row['J'] <= -10
        
        if trend_ok and j_ok:
            score = max(1, min_score)
            scores[stock_code] = score
    
    return scores


def run_rebalance_backtest_b3(stock_data, all_dates, min_score, rebalance_config, turnover_config):
    """带强制调仓的B3策略回测"""
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
            scores = calculate_b3_factor_scores(stock_data, current_date, min_score)
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


def run_multi_start_backtest_b3(stock_data, all_dates, config, min_score):
    """B3策略多起点回测"""
    results = []
    
    for start_date_str in config.get('start_dates', []):
        start_date = pd.to_datetime(start_date_str)
        valid_dates = [d for d in all_dates if d >= start_date]
        if not valid_dates:
            continue
        
        start_date = valid_dates[0]
        
        if len(valid_dates) < 60:
            continue
        
        result = run_rebalance_backtest_b3(
            stock_data, valid_dates, min_score,
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


def run_rolling_window_test_b3(stock_data, all_dates, config, min_score):
    """B3策略滚动窗口测试"""
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
        
        result = run_rebalance_backtest_b3(
            stock_data, window_dates, min_score,
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
    
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df


def check_b3_signal(df, idx):
    """
    检查B3买入信号
    a日: idx
    a-1日: idx-1
    a-2日: idx-2
    a-3日: idx-3
    a-4日: idx-4
    """
    if idx < 4:
        return False, {}
    
    a = df.iloc[idx]
    a_1 = df.iloc[idx-1]
    a_2 = df.iloc[idx-2]
    a_3 = df.iloc[idx-3]
    a_4 = df.iloc[idx-4]
    
    if pd.isna(a['J']) or pd.isna(a_1['J']) or pd.isna(a_2['J']):
        return False, {}
    
    cond_a1 = a['J'] < 80
    
    cond_a2 = a['CLOSE'] > a['OPEN']
    
    if a_1['CLOSE'] > 0:
        change_pct = (a['CLOSE'] - a_1['CLOSE']) / a_1['CLOSE'] * 100
    else:
        change_pct = 0
    cond_a3 = change_pct < 2
    
    if a['LOW'] > 0:
        amplitude = (a['HIGH'] - a['LOW']) / a['LOW'] * 100
    else:
        amplitude = 0
    cond_a4 = amplitude < 4
    
    cond_a5 = a['VOLUME'] < a_1['VOLUME']
    
    cond_a6 = a_1['CLOSE'] > a_1['OPEN']
    
    cond_a7 = a_1['VOLUME'] >= a_2['VOLUME'] * 1.8
    
    cond_a8 = a_2['J'] < 30
    
    cond_a9 = a_3['CLOSE'] < a_3['OPEN']
    
    cond_a10 = a_4['CLOSE'] < a_4['OPEN']
    
    if cond_a1 and cond_a2 and cond_a3 and cond_a4 and cond_a5 and cond_a6 and cond_a7 and cond_a8 and cond_a9 and cond_a10:
        return True, {
            'J': a['J'],
            'change_pct': change_pct,
            'amplitude': amplitude,
            'volume_ratio': a['VOLUME'] / a_1['VOLUME']
        }
    
    return False, {}


def run_backtest(data_dir, hold_days_list, stop_loss_pct=0.05, take_profit_list=None, desc=""):
    initial_capital = 100000000
    fee_rate = 0.0003
    slippage = 0.001
    
    stock_data = {}
    daily_signals = {}
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    loaded_count = 0
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
            continue
        
        code_part = stock_code.split('#')[-1]
        if code_part.startswith('8') and len(code_part) >= 3:
            if code_part[1] == '3' or code_part.startswith('83') or code_part.startswith('87'):
                continue
        
        path = os.path.join(data_dir, file)
        
        try:
            for encoding in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(path, sep='\s+', encoding=encoding, skiprows=1)
                    df = df[df['日期'].astype(str).str.match(r'^\d{4}')]
                    if len(df.columns) >= 6:
                        break
                except:
                    continue
            
            if len(df.columns) < 6:
                continue
            
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'] + list(df.columns[6:])
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            df = calculate_indicators(df)
            stock_data[stock_code] = df
            loaded_count += 1
            
            for i in range(4, len(df)):
                is_signal, details = check_b3_signal(df, i)
                if is_signal:
                    date = df.index[i]
                    signal_low = df.iloc[i]['LOW']
                    if date not in daily_signals:
                        daily_signals[date] = []
                    daily_signals[date].append((stock_code, details, signal_low))
        
        except Exception as e:
            continue
    
    print(f"成功加载 {loaded_count} 只股票")
    
    total_signals = sum(len(v) for v in daily_signals.values())
    print(f"共生成 {total_signals} 个买入信号")
    
    all_dates = []
    for df in stock_data.values():
        all_dates.extend(df.index.tolist())
    all_dates = sorted(set(all_dates))
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    results = []
    
    for hold_days in hold_days_list:
        cash = float(initial_capital)
        total_value = float(initial_capital)
        positions = []
        equity_curve = []
        
        trade_count = 0
        win_count = 0
        holding_returns = []
        holding_days_list = []
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        pending_buy_signals = {}
        pending_exit_signals = {}
        first_trade_date = None
        last_trade_date = None
        
        for date in daily_signals:
            date_idx = all_dates.index(date)
            if date_idx + 1 < len(all_dates):
                next_date = all_dates[date_idx + 1]
                if next_date not in pending_buy_signals:
                    pending_buy_signals[next_date] = []
                pending_buy_signals[next_date].extend(daily_signals[date])
        
        for current_date in all_dates:
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
                
                stop_price = pos["stop_price"]
                target_price = pos["target_price"]
                
                exit_flag = False
                exit_price = None
                exit_reason = ""
                
                if current_date in pending_exit_signals and pos['stock'] in pending_exit_signals[current_date]:
                    exit_flag = True
                    exit_info = pending_exit_signals[current_date][pos['stock']]
                    exit_price = exit_info['exit_price']
                    exit_reason = exit_info['exit_reason']
                elif not pos.get('exit_marked', False):
                    if close <= stop_price:
                        if current_idx + 1 < len(all_dates):
                            next_day = all_dates[current_idx + 1]
                            if next_day not in pending_exit_signals:
                                pending_exit_signals[next_day] = {}
                            pending_exit_signals[next_day][pos['stock']] = {
                                'exit_price': row['OPEN'],
                                'exit_reason': '止损'
                            }
                            pos['exit_marked'] = True
                    elif close >= target_price:
                        if current_idx + 1 < len(all_dates):
                            next_day = all_dates[current_idx + 1]
                            if next_day not in pending_exit_signals:
                                pending_exit_signals[next_day] = {}
                            pending_exit_signals[next_day][pos['stock']] = {
                                'exit_price': row['OPEN'],
                                'exit_reason': '止盈'
                            }
                            pos['exit_marked'] = True
                    elif holding_days >= hold_days:
                        if current_idx + 1 < len(all_dates):
                            next_day = all_dates[current_idx + 1]
                            if next_day not in pending_exit_signals:
                                pending_exit_signals[next_day] = {}
                            pending_exit_signals[next_day][pos['stock']] = {
                                'exit_price': row['OPEN'],
                                'exit_reason': f'持有{hold_days}天'
                            }
                            pos['exit_marked'] = True
                
                if exit_flag:
                    sell_price = exit_price
                    
                    prev_close = row.get('PREV_CLOSE', row.get('CLOSE'))
                    limit_pct = 0.10
                    if 'limit_pct' in row:
                        limit_pct = row['limit_pct']
                    
                    if pd.notna(prev_close) and prev_close > 0:
                        limit_down = prev_close * (1 - limit_pct)
                        if sell_price <= limit_down:
                            sell_price = limit_down
                    
                    sell_value = pos['shares'] * sell_price * (1 - fee_rate - slippage)
                    cash += sell_value
                    
                    pnl = (sell_price - entry_price) / entry_price
                    trade_count += 1
                    holding_returns.append(pnl)
                    holding_days_list.append(holding_days)
                    
                    if pnl > 0:
                        win_count += 1
                        current_consecutive_losses = 0
                    else:
                        current_consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                else:
                    new_positions.append(pos)
            
            positions = new_positions
            
            if current_date in pending_buy_signals:
                candidates = pending_buy_signals[current_date]
                
                for candidate in candidates:
                    stock = candidate[0]
                    details = candidate[1]
                    signal_low = candidate[2]
                    df = stock_data[stock]
                    
                    if current_date not in df.index:
                        continue
                    
                    row = df.loc[current_date]
                    open_price = row['OPEN']
                    
                    if open_price <= 0:
                        continue
                    
                    entry_low = row['LOW']
                    stop_loss_price = min(entry_low, signal_low) * (1 - stop_loss_pct)
                    
                    prev_close = row.get('PREV_CLOSE', row.get('CLOSE'))
                    limit_pct = 0.10
                    if 'limit_pct' in row:
                        limit_pct = row['limit_pct']
                    elif 'code' in df.columns:
                        code = df.iloc[current_idx].get('code', '')
                        if code.startswith('300'):
                            limit_pct = 0.20
                        elif code.startswith('688'):
                            limit_pct = 0.20
                    
                    if pd.notna(prev_close) and prev_close > 0:
                        limit_up = prev_close * (1 + limit_pct)
                        if open_price >= limit_up:
                            continue
                    
                    available_slots = 10 - len(positions)
                    if available_slots <= 0:
                        continue
                    
                    allocation = cash / available_slots
                    shares = int(allocation / open_price / 100) * 100
                    if shares > 0:
                        cost = shares * open_price * (1 + fee_rate + slippage)
                        if cost <= cash:
                            cash -= cost
                            
                            positions.append({
                                'stock': stock,
                                'shares': shares,
                                'entry_price': open_price,
                                'entry_idx': current_idx,
                                'entry_low': entry_low,
                                'stop_price': stop_loss_price,
                                'target_price': open_price * (1 + stop_loss_pct)
                            })
            
            current_total = cash
            for pos in positions:
                stock = pos['stock']
                df = stock_data[stock]
                if current_date in df.index:
                    current_price = df.loc[current_date]['CLOSE']
                    if pd.isna(current_price) or current_price <= 0:
                        valid_prices = df.loc[:current_date]['CLOSE']
                        valid_prices = valid_prices[valid_prices > 0]
                        if len(valid_prices) > 0:
                            current_price = valid_prices.iloc[-1]
                else:
                    valid_prices = df.loc[:current_date]['CLOSE']
                    valid_prices = valid_prices[valid_prices > 0]
                    if len(valid_prices) > 0:
                        current_price = valid_prices.iloc[-1]
                    else:
                        current_price = 0
                current_total += pos['shares'] * current_price
            
            equity_curve.append(current_total)
            
            if positions and first_trade_date is None:
                first_trade_date = current_date
            last_trade_date = current_date
        
        if len(equity_curve) < 2:
            continue
        
        final_capital = equity_curve[-1]
        final_multiple = final_capital / initial_capital
        
        if first_trade_date and last_trade_date:
            years = (last_trade_date - first_trade_date).days / 365.25
            years = max(years, 0.01)
        else:
            years = len(all_dates) / 252
        
        CAGR = (final_multiple ** (1 / years)) - 1 if years > 0 else 0
        
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdowns)
        avg_dd = np.mean(drawdowns)
        
        success_rate = win_count / trade_count if trade_count > 0 else 0
        avg_holding_return = np.mean(holding_returns) * 100 if holding_returns else 0
        max_holding_return = np.max(holding_returns) * 100 if holding_returns else 0
        
        results.append({
            'hold_days': hold_days,
            'final_multiple': final_multiple,
            'CAGR': CAGR * 100,
            'trade_count': trade_count,
            'success_rate': success_rate,
            'max_dd': max_dd * 100,
            'avg_dd': avg_dd * 100,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_holding_return': avg_holding_return,
            'max_holding_return': max_holding_return,
            'avg_holding_days': np.mean(holding_days_list) if holding_days_list else 0
        })
    
    return results


def run_backtest_take_profit(data_dir, take_profit_pct, stop_loss_pct=0.05, desc=""):
    initial_capital = 100000000
    fee_rate = 0.0003
    slippage = 0.001
    
    stock_data = {}
    daily_signals = {}
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
            continue
        
        code_part = stock_code.split('#')[-1]
        if code_part.startswith('8') and len(code_part) >= 3:
            if code_part[1] == '3' or code_part.startswith('83') or code_part.startswith('87'):
                continue
        
        path = os.path.join(data_dir, file)
        
        try:
            for encoding in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(path, sep='\s+', encoding=encoding, skiprows=1)
                    df = df[df['日期'].astype(str).str.match(r'^\d{4}')]
                    if len(df.columns) >= 6:
                        break
                except:
                    continue
            
            if len(df.columns) < 6:
                continue
            
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'] + list(df.columns[6:])
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            df = calculate_indicators(df)
            stock_data[stock_code] = df
            
            for i in range(4, len(df)):
                is_signal, details = check_b3_signal(df, i)
                if is_signal:
                    date = df.index[i]
                    signal_low = df.iloc[i]['LOW']
                    if date not in daily_signals:
                        daily_signals[date] = []
                    daily_signals[date].append((stock_code, details, signal_low))
        
        except:
            continue
    
    all_dates = []
    for df in stock_data.values():
        all_dates.extend(df.index.tolist())
    all_dates = sorted(set(all_dates))
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    cash = float(initial_capital)
    total_value = float(initial_capital)
    positions = []
    equity_curve = []
    
    trade_count = 0
    win_count = 0
    holding_returns = []
    holding_days_list = []
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    
    pending_buy_signals = {}
    for date in daily_signals:
        date_idx = all_dates.index(date)
        if date_idx + 1 < len(all_dates):
            next_date = all_dates[date_idx + 1]
            if next_date not in pending_buy_signals:
                pending_buy_signals[next_date] = []
            pending_buy_signals[next_date].extend(daily_signals[date])
    
    for current_date in all_dates:
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
            
            if current_idx < entry_idx + 1:
                new_positions.append(pos)
                continue
            
            stop_price = pos["stop_price"]
            target_price = pos["target_price"]
            
            exit_flag = False
            exit_price = None
            exit_reason = ""
            
            if close <= stop_price:
                exit_flag = True
                exit_price = open_p
                exit_reason = "止损"
            elif close >= target_price:
                exit_flag = True
                exit_price = open_p
                exit_reason = f"{take_profit_pct*100}%止盈"
            
            if exit_flag:
                sell_value = pos['shares'] * exit_price * (1 - fee_rate - slippage)
                cash += sell_value
                
                pnl = (exit_price - entry_price) / entry_price
                trade_count += 1
                holding_returns.append(pnl)
                holding_days_list.append(current_idx - entry_idx)
                
                if pnl > 0:
                    win_count += 1
                    current_consecutive_losses = 0
                else:
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                new_positions.append(pos)
        
        positions = new_positions
        
        if current_date in pending_buy_signals:
            candidates = pending_buy_signals[current_date]
            
            for candidate in candidates:
                stock = candidate[0]
                details = candidate[1]
                signal_low = candidate[2]
                df = stock_data[stock]
                
                if current_date not in df.index:
                    continue
                
                row = df.loc[current_date]
                open_price = row['OPEN']
                
                if open_price <= 0:
                    continue
                
                entry_low = row['LOW']
                stop_loss_price = min(entry_low, signal_low) * (1 - stop_loss_pct)
                target_price = open_price * (1 + take_profit_pct)
                
                available_slots = 10 - len(positions)
                if available_slots <= 0:
                    continue
                
                allocation = cash / available_slots
                shares = int(allocation / open_price / 100) * 100
                if shares > 0:
                    cost = shares * open_price * (1 + fee_rate + slippage)
                    if cost <= cash:
                        cash -= cost
                        
                        positions.append({
                            'stock': stock,
                            'shares': shares,
                            'entry_price': open_price,
                            'entry_idx': current_idx,
                            'entry_low': entry_low,
                            'stop_price': stop_loss_price,
                            'target_price': target_price
                        })
        
        current_total = cash
        for pos in positions:
            stock = pos['stock']
            df = stock_data[stock]
            if current_date in df.index:
                current_price = df.loc[current_date]['CLOSE']
                current_total += pos['shares'] * current_price
        
        equity_curve.append(current_total)
    
    if len(equity_curve) < 2:
        return None
    
    final_capital = equity_curve[-1]
    final_multiple = final_capital / initial_capital
    
    years = len(all_dates) / 252
    CAGR = (final_multiple ** (1 / years)) - 1 if years > 0 else 0
    
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    avg_dd = np.mean(drawdowns)
    
    success_rate = win_count / trade_count if trade_count > 0 else 0
    avg_holding_return = np.mean(holding_returns) * 100 if holding_returns else 0
    max_holding_return = np.max(holding_returns) * 100 if holding_returns else 0
    
    return {
        'take_profit_pct': take_profit_pct,
        'final_multiple': final_multiple,
        'CAGR': CAGR * 100,
        'trade_count': trade_count,
        'success_rate': success_rate,
        'max_dd': max_dd * 100,
        'avg_dd': avg_dd * 100,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_holding_return': avg_holding_return,
        'max_holding_return': max_holding_return,
        'avg_holding_days': np.mean(holding_days_list) if holding_days_list else 0
    }


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    
    print("=" * 90)
    print("B3策略回测")
    print("=" * 90)
    
    print("""
================================================================================
                           B3策略详细方案
================================================================================

【买入条件】(a日满足以下所有条件):
  1. a日J值 < 80
  2. a日是阳线(收盘价 > 开盘价)
  3. a日涨幅 < 2%  (收盘价-前一日收盘价)/前一日收盘价
  4. a日振幅 < 4%  (最高价-最低价)/最低价
  5. a日成交量 < a-1日成交量(缩量)
  6. a-1日是阳线
  7. a-1日成交量 >= a-2日成交量 × 1.8倍(倍量)
  8. a-2日J值 < 30
  9. a-3日是阴线
  10. a-4日是阴线

【信号发出】a日收盘后发出买入信号

【买入执行】a+1日以开盘价买入

【止损价】min(买入日最低价, 信号日最低价) × 0.95

【策略1: 固定持有天数】
  卖出条件: 持有到期后次日开盘卖出
  持有天数: 2、3、4、5、10、15、20、30、60天

【策略2: 止盈止损】
  止盈条件: 收盘价 >= 买入价 × (1+止盈比例)
  止损条件: 收盘价 < 止损价
  止盈比例: 7%、8%、9%、10%、11%、12%
  卖出执行: 达到条件后次日开盘卖出
================================================================================
""")
    
    hold_days_list = [2, 3, 4, 5, 10, 15, 20, 30, 60]
    
    print("\n" + "=" * 80)
    print("策略1：固定持有天数")
    print("=" * 80)
    
    results1 = run_backtest(data_dir, hold_days_list, stop_loss_pct=0.05, desc="固定持有")
    
    print(f"\n{'持有天数':<10} {'最终倍数':<12} {'年化收益率':<12} {'交易次数':<10} {'成功率':<10} {'最大回撤':<12} {'平均回撤':<12} {'最大连亏':<10} {'平均持有收益':<14} {'最大持有收益':<14} {'平均持有天数':<14}")
    print("-" * 140)
    
    for r in results1:
        print(f"{r['hold_days']:<10} {r['final_multiple']:<12.2f} {r['CAGR']:<12.2f}% {r['trade_count']:<10} {r['success_rate']:<10.2%} {r['max_dd']:<12.2f}% {r['avg_dd']:<12.2f}% {r['max_consecutive_losses']:<10} {r['avg_holding_return']:<14.2f}% {r['max_holding_return']:<14.2f}% {r['avg_holding_days']:<14.1f}")
    
    print("\n" + "=" * 80)
    print("策略2：止盈止损")
    print("=" * 80)
    
    take_profit_list = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    
    results2 = []
    for tp in take_profit_list:
        r = run_backtest_take_profit(data_dir, tp, stop_loss_pct=0.05)
        if r:
            results2.append(r)
    
    print(f"\n{'止盈比例':<10} {'最终倍数':<12} {'年化收益率':<12} {'交易次数':<10} {'成功率':<10} {'最大回撤':<12} {'平均回撤':<12} {'最大连亏':<10} {'平均持有收益':<14} {'最大持有收益':<14} {'平均持有天数':<14}")
    print("-" * 140)
    
    for r in results2:
        print(f"{r['take_profit_pct']*100:<10.0f}% {r['final_multiple']:<12.2f} {r['CAGR']:<12.2f}% {r['trade_count']:<10} {r['success_rate']:<10.2%} {r['max_dd']:<12.2f}% {r['avg_dd']:<12.2f}% {r['max_consecutive_losses']:<10} {r['avg_holding_return']:<14.2f}% {r['max_holding_return']:<14.2f}% {r['avg_holding_days']:<14.1f}")
    
    print("\n" + "=" * 80)
    print("对比汇总")
    print("=" * 80)
    
    best_hold = max(results1, key=lambda x: x['CAGR'])
    best_tp = max(results2, key=lambda x: x['CAGR']) if results2 else None
    
    print(f"\n最佳固定持有: {best_hold['hold_days']}天, 年化{best_hold['CAGR']:.2f}%, 倍数{best_hold['final_multiple']:.2f}x")
    if best_tp:
        print(f"最佳止盈: {best_tp['take_profit_pct']*100:.0f}%, 年化{best_tp['CAGR']:.2f}%, 倍数{best_tp['final_multiple']:.2f}x")
