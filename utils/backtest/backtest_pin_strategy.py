import os
import pandas as pd
import numpy as np


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


def calculate_pin_factor_scores(stock_data, current_date):
    """计算Pinbar策略因子得分"""
    scores = {}
    
    for stock_code, df in stock_data.items():
        if current_date not in df.index:
            continue
        
        try:
            idx = df.index.get_loc(current_date)
        except KeyError:
            continue
        
        if idx < 5:
            continue
        
        row = df.iloc[idx]
        
        if pd.isna(row.get('PINBAR')) or row.get('PINBAR') != 1:
            continue
        
        score = 1.0
        scores[stock_code] = score
    
    return scores


def run_rebalance_backtest_pin(stock_data, all_dates, rebalance_config, turnover_config):
    """带强制调仓的Pinbar策略回测"""
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
            scores = calculate_pin_factor_scores(stock_data, current_date)
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


def run_multi_start_backtest_pin(stock_data, all_dates, config):
    """Pinbar策略多起点回测"""
    results = []
    
    for start_date_str in config.get('start_dates', []):
        start_date = pd.to_datetime(start_date_str)
        valid_dates = [d for d in all_dates if d >= start_date]
        if not valid_dates:
            continue
        
        start_date = valid_dates[0]
        
        if len(valid_dates) < 60:
            continue
        
        result = run_rebalance_backtest_pin(
            stock_data, valid_dates,
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


def run_rolling_window_test_pin(stock_data, all_dates, config):
    """Pinbar策略滚动窗口测试"""
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
        
        result = run_rebalance_backtest_pin(
            stock_data, window_dates,
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


def check_data_anomaly(df):
    anomaly_reasons = []
    
    if len(df) < 2:
        return True, anomaly_reasons
    
    for i in range(len(df)):
        row = df.iloc[i]
        open_p = row['OPEN']
        high = row['HIGH']
        low = row['LOW']
        close = row['CLOSE']
        volume = row['VOLUME']
        
        if pd.isna(open_p) or pd.isna(high) or pd.isna(low) or pd.isna(close):
            anomaly_reasons.append(f"{df.index[i]} 数据缺失")
            continue
        
        if high == low == close:
            anomaly_reasons.append(f"{df.index[i]} 一字板")
            continue
        
        if i > 0:
            prev_close = df.iloc[i-1]['CLOSE']
            if prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 20 or change_pct < -20:
                    anomaly_reasons.append(f"{df.index[i]} 涨跌幅异常: {change_pct:.2f}%")
                    continue
        
        if volume <= 0:
            anomaly_reasons.append(f"{df.index[i]} 成交量为0")
            continue
        
        if i >= 60:
            rolling_vol = df.iloc[i-60:i]['VOLUME']
            avg_vol = rolling_vol.mean()
            if avg_vol > 0 and volume > avg_vol * 5:
                anomaly_reasons.append(f"{df.index[i]} 成交量异常放大: {volume/avg_vol:.2f}倍")
                continue
        
        if open_p < low or open_p > high:
            anomaly_reasons.append(f"{df.index[i]} 开盘价不在高低区间")
            continue
        
        if high > 0 and low > 0:
            amplitude = (high - low) / low * 100
            if i > 0:
                prev_vol = df.iloc[i-1]['VOLUME']
                vol_change = volume / prev_vol if prev_vol > 0 else 1
                if amplitude > 15 and vol_change < 1.2:
                    anomaly_reasons.append(f"{df.index[i]} 振幅异常但无放量")
                    continue
        
        if i >= 1:
            for j in range(1, min(6, i+1)):
                if i - j >= 0:
                    prev_change = (df.iloc[i-j]['CLOSE'] - df.iloc[i-j-1]['CLOSE']) / df.iloc[i-j-1]['CLOSE'] * 100
                    if prev_change < -15:
                        anomaly_reasons.append(f"{df.index[i]} 连续异常大跌")
                        break
    
    if len(df) >= 2:
        first_price = df.iloc[0]['CLOSE']
        last_price = df.iloc[-1]['CLOSE']
        if first_price > 0 and last_price > 0:
            total_change = (last_price - first_price) / first_price
            if abs(total_change) > 100:
                anomaly_reasons.append("全历史价格比例变化异常")
        
        if last_price < 0.5:
            anomaly_reasons.append("股价极低")
        
        if i >= 1:
            avg_vol = df.iloc[:-1]['VOLUME'].mean()
            if avg_vol > 0:
                turnover = df.iloc[-1]['VOLUME'] * close
                avg_turnover = avg_vol * first_price
                if avg_turnover > 0 and turnover < avg_turnover * 0.001:
                    anomaly_reasons.append("成交额过低")
    
    return len(anomaly_reasons) > 0, anomaly_reasons


def tongdaxin_sma(series, n, m=1):
    result = np.zeros(len(series))
    prev_sma = 0
    for i in range(len(series)):
        val = series.iloc[i]
        if i < n - 1:
            sma = series.iloc[:i+1].sum() / (i + 1)
        else:
            sma = (val * m + prev_sma * (n - m)) / n
        result[i] = sma
        prev_sma = sma
    return pd.Series(result, index=series.index)


def brick_chart_indicator(df):
    df = df.copy()
    
    df['HHV_H4'] = df['HIGH'].rolling(window=4).max()
    df['LLV_L4'] = df['LOW'].rolling(window=4).min()
    
    df['VAR1A'] = (df['HHV_H4'] - df['CLOSE']) / (df['HHV_H4'] - df['LLV_L4']) * 100 - 90
    df['VAR1A'] = df['VAR1A'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['VAR2A'] = tongdaxin_sma(df['VAR1A'], 4, 1) + 100
    
    df['VAR3A'] = (df['CLOSE'] - df['LLV_L4']) / (df['HHV_H4'] - df['LLV_L4']) * 100
    df['VAR3A'] = df['VAR3A'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['VAR4A'] = tongdaxin_sma(df['VAR3A'], 6, 1)
    
    df['VAR5A'] = tongdaxin_sma(df['VAR4A'], 6, 1) + 100
    
    df['VAR6A'] = df['VAR5A'] - df['VAR2A']
    
    df['砖型图数值'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)
    
    df['砖型图变化量'] = df['砖型图数值'] - df['砖型图数值'].shift(1)
    df['当日柱体长度'] = df['砖型图变化量'].abs()
    df['前日柱体长度'] = df['当日柱体长度'].shift(1)
    df['当日红柱'] = df['砖型图变化量'] > 0
    df['前日绿柱'] = df['砖型图变化量'].shift(1) < 0
    
    df['买入信号'] = np.where(
        (df['前日绿柱'] == True) &
        (df['当日红柱'] == True) &
        (df['当日柱体长度'] >= df['前日柱体长度'] * 0.66),
        1,
        0
    )
    
    df = df.dropna()
    
    return df


def calculate_trend(df, N=9, M1=14, M2=28, M3=57, M4=114):
    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()
    
    df['MA14'] = df['CLOSE'].rolling(window=M1).mean()
    df['MA28'] = df['CLOSE'].rolling(window=M2).mean()
    df['MA57'] = df['CLOSE'].rolling(window=M3).mean()
    df['MA114'] = df['CLOSE'].rolling(window=M4).mean()
    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
    
    return df


def calculate_kdj(df, N=9, M1=3, M2=3):
    df['HHV9'] = df['HIGH'].rolling(window=N, min_periods=1).max()
    df['LLV9'] = df['LOW'].rolling(window=N, min_periods=1).min()
    df['RNG'] = df['HHV9'] - df['LLV9']
    
    df['RSV'] = (df['CLOSE'] - df['LLV9']) / (df['HHV9'] - df['LLV9']) * 100
    
    df['K'] = df['RSV'].ewm(alpha=1/M1, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/M2, adjust=False).mean()
    
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df


def check_buy_conditions(df, i, next_data):
    if i < 60 or i < 30 or i < 60 or i < 3:
        return False
    
    current_data = df.iloc[i]
    current_close = current_data['CLOSE']
    current_open = current_data['OPEN']
    duokongxian = current_data['知行多空线']
    short_trend = current_data['知行短期趋势线']
    
    brick_value = current_data.get('砖型图数值', 0)
    brick_value_yesterday = df.iloc[i-1].get('砖型图数值', 0)
    brick_change = current_data.get('砖型图变化量', 0)
    brick_change_yesterday = df.iloc[i-1].get('砖型图变化量', 0)
    
    prev_close = df.iloc[i-1]['CLOSE']
    prev_high = df.iloc[i-1]['HIGH']
    prev_open = df.iloc[i-1]['OPEN']
    
    condition1 = next_data['OPEN'] <= prev_high
    
    condition2 = next_data['OPEN'] < short_trend * 1.02
    
    closes_60 = df['CLOSE'].iloc[i-59:i+1].values
    opens_60 = df['OPEN'].iloc[i-59:i+1].values
    
    min_close_idx = np.argmin(closes_60)
    min_close_second = closes_60[min_close_idx]
    
    condition3 = True
    bottoms = []
    for j in range(len(closes_60)):
        if j == min_close_idx:
            bottoms.append(closes_60[j])
        elif closes_60[j] < min_close_second * 1.01:
            min_close_second = closes_60[j]
            min_close_idx = j
            bottoms.append(closes_60[j])
    
    for k in range(1, len(bottoms)):
        if bottoms[k] <= bottoms[k-1]:
            condition3 = False
            break
    
    today_pct = (current_close - prev_close) / prev_close * 100
    condition4 = True
    
    condition5 = True
    green_count = 0
    for j in range(1, 4):
        if df.iloc[i-j].get('砖型图变化量', 0) < 0:
            green_count += 1
        else:
            break
    if green_count < 3:
        condition5 = False
    
    condition6 = brick_change > 0 and brick_change_yesterday < 0 and brick_value > brick_value_yesterday
    
    condition8 = True
    for j in range(30):
        if i - j - 3 < 0:
            break
        pct_j = (df.iloc[i-j]['CLOSE'] - df.iloc[i-j-1]['CLOSE']) / df.iloc[i-j-1]['CLOSE'] * 100
        pct_j1 = (df.iloc[i-j-1]['CLOSE'] - df.iloc[i-j-2]['CLOSE']) / df.iloc[i-j-2]['CLOSE'] * 100
        pct_j2 = (df.iloc[i-j-2]['CLOSE'] - df.iloc[i-j-3]['CLOSE']) / df.iloc[i-j-3]['CLOSE'] * 100
        if pct_j > 8 and pct_j1 > 8 and pct_j2 > 8:
            if i - j - 1 >= 0:
                current_k = df.iloc[i-j-1]
                next_k = df.iloc[i-j]
                if current_k['CLOSE'] > current_k['OPEN'] and next_k['CLOSE'] < next_k['OPEN']:
                    if next_k['VOLUME'] > current_k['VOLUME'] / 2:
                        condition8 = False
                        break
    
    condition9 = True
    if i >= 89:
        volumes_90 = df['VOLUME'].iloc[i-89:i+1].values
        max_vol_idx_global = np.argmax(volumes_90)
        if volumes_90[max_vol_idx_global] > 0:
            max_vol_day = df.iloc[i-89+max_vol_idx_global]
            if max_vol_day['CLOSE'] < max_vol_day['OPEN']:
                condition9 = False
    
    return condition1 and condition2 and condition3 and condition4 and condition5 and condition6 and condition8 and condition9


def check_b2_filter(df):
    if len(df) < 4:
        return False, False
    
    df_simple = df[['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].copy()
    
    df_simple = calculate_trend(df_simple)
    df_simple = calculate_kdj(df_simple)
    
    today = df_simple.iloc[-1]
    yesterday = df_simple.iloc[-2]
    day_before_yesterday = df_simple.iloc[-3]
    
    if today['知行多空线'] > today['知行短期趋势线']:
        return False, False
    
    j_label = False
    volatility_label = False
    volume_label = False
    shadow_label = False
    
    if yesterday['J'] <= 50 and today['J'] <= 80:
        j_label = True
    
    change_pct = (today['CLOSE'] - yesterday['CLOSE']) / yesterday['CLOSE'] * 100
    if change_pct >= 4:
        volatility_label = True
    
    if today['VOLUME'] > yesterday['VOLUME']:
        volume_label = True
    
    length = abs(df_simple['HIGH'].iloc[-1] - df_simple['OPEN'].iloc[-1])
    if length > 0:
        shadow_length = df_simple['HIGH'].iloc[-1] - df_simple['CLOSE'].iloc[-1]
        shadow_ratio = shadow_length / length
        if shadow_ratio < 0.3:
            shadow_label = True
    else:
        shadow_label = True
    
    b2_today = j_label and volatility_label and volume_label and shadow_label
    
    j_label_lastday = False
    volatility_label_lastday = False
    volume_label_lastday = False
    shadow_label_lastday = False
    
    if day_before_yesterday['J'] <= 50 and yesterday['J'] <= 80:
        j_label_lastday = True
    
    change_pct_lastday = (yesterday['CLOSE'] - day_before_yesterday['CLOSE']) / day_before_yesterday['CLOSE'] * 100
    if change_pct_lastday >= 4:
        volatility_label_lastday = True
    
    if yesterday['VOLUME'] > day_before_yesterday['VOLUME']:
        volume_label_lastday = True
    
    length_lastday = abs(df_simple['HIGH'].iloc[-2] - df_simple['OPEN'].iloc[-2])
    if length_lastday > 0:
        shadow_length_lastday = df_simple['HIGH'].iloc[-2] - df_simple['CLOSE'].iloc[-2]
        shadow_ratio_lastday = shadow_length_lastday / length_lastday
        if shadow_ratio_lastday < 0.3:
            shadow_label_lastday = True
    else:
        shadow_label_lastday = True
    
    b2_lastday = j_label_lastday and volatility_label_lastday and volume_label_lastday and shadow_label_lastday
    
    return b2_today, b2_lastday


class BrickStrategyBacktest:
    def __init__(self, data_dir, results_file=None):
        self.data_dir = data_dir
        self.results_file = results_file
        self.results = []
        self.initial_capital = 1000000
        self.data_cache = {}
        self.sell_at_open = True
        self.yearly_returns = {}  # 🔥 每年收益率统计
    
    def load_stock_data(self, file_path):
        encodings = ['gbk', 'utf-8', 'latin-1', 'gb18030']
        df = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                
                data = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    if i == 0 and '开盘' in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            date_str = parts[0]
                            open_price = float(parts[1])
                            high_price = float(parts[2])
                            low_price = float(parts[3])
                            close_price = float(parts[4])
                            volume = float(parts[5])
                            
                            data.append([date_str, open_price, high_price, low_price, close_price, volume])
                        except ValueError:
                            continue
                
                df = pd.DataFrame(data, columns=['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
                df['日期'] = pd.to_datetime(df['日期'])
                break
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {str(e)}")
                return None
        
        if df is None:
            return None
        
        df = df.sort_values('日期').reset_index(drop=True)
        
        if len(df) < 10:
            return None
        
        is_anomaly, reasons = check_data_anomaly(df)
        if is_anomaly:
            return None
        
        return df
    
    def run_strategy(self, df, stock_code, strategy_type):
        df = calculate_trend(df)
        df = brick_chart_indicator(df)
        
        position = 0
        entry_price = 0
        entry_date = None
        entry_low = 0
        entry_is_bearish = False
        half_sold = False
        entry_index = 0
        max_high_since_entry = 0
        min_low_since_entry = float('inf')
        trades = []
        completed_trades = []
        capital = self.initial_capital
        
        pending_buy = None
        pending_exit = None
        invested_amount = 0
        
        for i in range(len(df)):
            current_data = df.iloc[i]
            current_date = current_data['日期']
            current_close = current_data['CLOSE']
            current_open = current_data['OPEN']
            current_low = current_data['LOW']
            current_high = current_data['HIGH']
            brick_value = current_data.get('砖型图数值', 0)
            xg_signal = current_data.get('买入信号', 0)
            duokongxian = current_data.get('知行多空线', 0)
            
            if pending_exit is not None:
                exit_price = current_open
                exit_date = current_date
                profit = invested_amount * (exit_price - entry_price) / entry_price
                capital += profit
                
                drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'profit_pct': (exit_price - entry_price) / entry_price * 100,
                    'position': position,
                    'type': pending_exit,
                    'drawdown': drawdown
                })
                completed_trades.append(trades[-1])
                
                position = 0
                entry_price = 0
                entry_date = None
                entry_low = 0
                entry_is_bearish = False
                half_sold = False
                max_high_since_entry = 0
                min_low_since_entry = float('inf')
                pending_exit = None
                invested_amount = 0
                continue
            
            if pending_buy is not None:
                buy_signal = pending_buy['signal']
                price_change = pending_buy['price_change']
                
                if price_change <= 3:
                    next_close = current_close
                    next_high = current_high
                    next_low = current_low
                    
                    if next_close < current_open:
                        is_bearish = True
                    else:
                        is_bearish = False
                    
                    if next_close >= duokongxian:
                        if check_buy_conditions(df, pending_buy['idx'], current_data):
                            entry_price = current_open
                            entry_date = current_date
                            entry_low = pending_buy['low']
                            entry_is_bearish = is_bearish
                            position = 1
                            entry_index = i
                            half_sold = False
                            max_high_since_entry = next_high
                            min_low_since_entry = next_low
                            invested_amount = position * entry_price * capital / entry_price
                            
                            trades.append({
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'position': position,
                                'type': '砖型图信号买入'
                            })
                
                pending_buy = None
            
            if position > 0:
                holding_days = i - entry_index
                
                if current_high > max_high_since_entry:
                    max_high_since_entry = current_high
                if current_low < min_low_since_entry:
                    min_low_since_entry = current_low
                
                if holding_days >= 1:
                    period_max_profit_pct = (max_high_since_entry - entry_price) / entry_price * 100
                    current_profit_pct = (current_close - entry_price) / entry_price * 100
                    
                    if strategy_type == '2%止盈':
                        if period_max_profit_pct >= 2 and holding_days <= 2:
                            exit_price = current_close
                            exit_date = current_date
                            profit = (exit_price - entry_price) * position * capital / entry_price
                            capital += profit
                            
                            drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': profit,
                                'profit_pct': 2.0,
                                'position': position,
                                'type': '2%止盈卖出',
                                'drawdown': drawdown
                            })
                            completed_trades.append(trades[-1])
                            position = 0
                            entry_price = 0
                            entry_date = None
                            entry_low = 0
                            entry_is_bearish = False
                            half_sold = False
                            max_high_since_entry = 0
                            min_low_since_entry = float('inf')
                        
                        elif holding_days >= 3:
                            exit_price = current_close
                            exit_date = current_date
                            profit = (exit_price - entry_price) * position * capital / entry_price
                            capital += profit
                            
                            drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': profit,
                                'profit_pct': (exit_price - entry_price) / entry_price * 100,
                                'position': position,
                                'type': '第3天收盘卖出',
                                'drawdown': drawdown
                            })
                            completed_trades.append(trades[-1])
                            position = 0
                            entry_price = 0
                            entry_date = None
                            entry_low = 0
                            entry_is_bearish = False
                            half_sold = False
                            max_high_since_entry = 0
                            min_low_since_entry = float('inf')
                    
                    elif strategy_type == '3%止盈':
                        if period_max_profit_pct >= 3 and holding_days <= 2:
                            exit_price = current_close
                            exit_date = current_date
                            profit = (exit_price - entry_price) * position * capital / entry_price
                            capital += profit
                            
                            drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': profit,
                                'profit_pct': 3.0,
                                'position': position,
                                'type': '3%止盈卖出',
                                'drawdown': drawdown
                            })
                            completed_trades.append(trades[-1])
                            position = 0
                            entry_price = 0
                            entry_date = None
                            entry_low = 0
                            entry_is_bearish = False
                            half_sold = False
                            max_high_since_entry = 0
                            min_low_since_entry = float('inf')
                    
                    elif strategy_type == '3天收盘卖':
                        if holding_days >= 3:
                            pending_exit = '第3天收盘卖出'
                    
                    elif strategy_type == '5%卖一半':
                        if holding_days == 2:
                            if current_profit_pct >= 5 and not half_sold:
                                exit_price = current_close
                                exit_date = current_date
                                sell_ratio = 0.5
                                profit = (exit_price - entry_price) * sell_ratio * position * capital / entry_price
                                capital += profit
                                
                                trades.append({
                                    'entry_date': entry_date,
                                    'exit_date': exit_date,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'profit': profit,
                                    'profit_pct': (exit_price - entry_price) / entry_price * 100,
                                    'position': sell_ratio * position,
                                    'type': '第3天卖出一半'
                                })
                                
                                position = position * 0.5
                                half_sold = True
                                max_high_since_entry = 0
                                min_low_since_entry = float('inf')
                            
                            elif not half_sold:
                                exit_price = current_close
                                exit_date = current_date
                                profit = (exit_price - entry_price) * position * capital / entry_price
                                capital += profit
                                
                                drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                                
                                trades.append({
                                    'entry_date': entry_date,
                                    'exit_date': exit_date,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'profit': profit,
                                    'profit_pct': (exit_price - entry_price) / entry_price * 100,
                                    'position': position,
                                    'type': '第3天全仓卖出',
                                    'drawdown': drawdown
                                })
                                completed_trades.append(trades[-1])
                                position = 0
                                entry_price = 0
                                entry_date = None
                                entry_low = 0
                                entry_is_bearish = False
                                half_sold = False
                                max_high_since_entry = 0
                                min_low_since_entry = float('inf')
                        
                        elif half_sold and holding_days >= 5:
                            if i < len(df) - 1:
                                next_open = df.iloc[i+1]['OPEN']
                                exit_price = next_open
                                exit_date = df.iloc[i+1]['日期']
                            else:
                                exit_price = current_close
                                exit_date = current_date
                            profit = (exit_price - entry_price) * position * capital / entry_price
                            capital += profit
                            
                            drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': profit,
                                'profit_pct': (exit_price - entry_price) / entry_price * 100,
                                'position': position,
                                'type': '剩余一半卖出',
                                'drawdown': drawdown
                            })
                            completed_trades.append(trades[-1])
                            position = 0
                            entry_price = 0
                            entry_date = None
                            entry_low = 0
                            entry_is_bearish = False
                            half_sold = False
                            max_high_since_entry = 0
                            min_low_since_entry = float('inf')
                    
                    elif strategy_type == '2天收盘卖':
                        if holding_days >= 2:
                            pending_exit = '第2天收盘卖出'
                    
                    elif strategy_type == '3天开盘卖':
                        if holding_days >= 2:
                            pending_exit = '第3天开盘卖出'
                    
                    elif strategy_type == '3天收盘卖':
                        if holding_days >= 3:
                            exit_price = current_close
                            exit_date = current_date
                            profit = (exit_price - entry_price) * position * capital / entry_price
                            capital += profit
                            
                            drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'profit': profit,
                                'profit_pct': (exit_price - entry_price) / entry_price * 100,
                                'position': position,
                                'type': '第3天收盘卖出',
                                'drawdown': drawdown
                            })
                            completed_trades.append(trades[-1])
                            position = 0
                            entry_price = 0
                            entry_date = None
                            entry_low = 0
                            entry_is_bearish = False
                            half_sold = False
                            max_high_since_entry = 0
                            min_low_since_entry = float('inf')
                    
                    elif entry_is_bearish and holding_days >= 1:
                        if i < len(df) - 1:
                            next_open = df.iloc[i+1]['OPEN']
                            exit_price = next_open
                            exit_date = df.iloc[i+1]['日期']
                        else:
                            exit_price = current_close
                            exit_date = current_date
                        profit = (exit_price - entry_price) * position * capital / entry_price
                        capital += profit
                        
                        drawdown = (entry_price - min_low_since_entry) / entry_price * 100 if min_low_since_entry < entry_price else 0
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'profit_pct': (exit_price - entry_price) / entry_price * 100,
                            'position': position,
                            'type': '阴线次日止损',
                            'drawdown': drawdown
                        })
                        completed_trades.append(trades[-1])
                        position = 0
                        entry_price = 0
                        entry_date = None
                        entry_low = 0
                        entry_is_bearish = False
                        half_sold = False
                        max_high_since_entry = 0
                        min_low_since_entry = float('inf')
            
            if position == 0:
                buy_signal = current_data.get('买入信号', 0)
                if buy_signal == 1:
                    if i < len(df) - 1:
                        next_data = df.iloc[i+1]
                        next_open = next_data['OPEN']
                        price_change = (next_open - current_close) / current_close * 100
                        
                        pending_buy = {
                            'signal': buy_signal,
                            'price_change': price_change,
                            'idx': i,
                            'low': current_low
                        }
        
        num_trades = len(completed_trades)
        total_profit = sum([trade.get('profit', 0) for trade in completed_trades])
        avg_profit = total_profit / num_trades if num_trades > 0 else 0
        
        win_trades = [trade for trade in completed_trades if trade.get('profit', 0) > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        
        all_profit_pcts = [trade.get('profit_pct', 0) for trade in completed_trades if 'profit_pct' in trade]
        avg_profit_pct = sum(all_profit_pcts) / len(all_profit_pcts) if all_profit_pcts else 0
        max_profit_pct = max(all_profit_pcts) if all_profit_pcts else 0
        
        # 🔥 每年收益率统计
        yearly_profits = {}
        for trade in completed_trades:
            exit_date = trade.get('exit_date')
            profit_pct = trade.get('profit_pct', 0)
            if exit_date is not None and profit_pct is not None:
                if isinstance(exit_date, str):
                    year = int(exit_date[:4])
                else:
                    year = exit_date.year
                if year not in yearly_profits:
                    yearly_profits[year] = []
                yearly_profits[year].append(profit_pct / 100)
        
        cumulative_profit = capital / self.initial_capital
        
        first_date = df['日期'].iloc[0]
        last_date = df['日期'].iloc[-1]
        total_days = (last_date - first_date).days
        years = total_days / 365.25
        years = max(years, 0.01)
        
        if cumulative_profit > 0:
            annual_return = cumulative_profit ** (1 / years) - 1
        else:
            annual_return = -1
        
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        consecutive_losses = []
        
        for trade in completed_trades:
            if trade.get('profit', 0) <= 0:
                current_consecutive_losses += 1
                if current_consecutive_losses > max_consecutive_losses:
                    max_consecutive_losses = current_consecutive_losses
            else:
                if current_consecutive_losses > 0:
                    consecutive_losses.append(current_consecutive_losses)
                    current_consecutive_losses = 0
        
        if current_consecutive_losses > 0:
            consecutive_losses.append(current_consecutive_losses)
        
        avg_consecutive_losses = sum(consecutive_losses) / len(consecutive_losses) if consecutive_losses else 0
        
        all_drawdowns = [trade.get('drawdown', 0) for trade in completed_trades if 'drawdown' in trade]
        max_drawdown = max(all_drawdowns) if all_drawdowns else 0
        avg_drawdown = sum(all_drawdowns) / len(all_drawdowns) if all_drawdowns else 0
        
        trading_days = years * 252
        avg_daily_trades = num_trades / trading_days if trading_days > 0 else 0
        
        return {
            'stock_code': stock_code,
            'strategy_type': strategy_type,
            'num_trades': num_trades,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit,
            'success_rate': win_rate,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'cumulative_return': cumulative_profit - 1,
            'annual_return': annual_return,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_consecutive_losses': avg_consecutive_losses,
            'avg_profit_pct': avg_profit_pct,
            'max_profit_pct': max_profit_pct,
            'final_capital': capital,
            'avg_daily_trades': avg_daily_trades,
            'trades': trades,
            'completed_trades': completed_trades,
            'yearly_returns': yearly_profits  # 🔥 每年收益率
        }
    
    def run(self, test_mode=False):
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        print(f"找到 {len(stock_files)} 个股票文件")
        
        if test_mode:
            print("测试模式：只运行前10个股票")
            stock_files = stock_files[:10]
        
        print(f"预计总计算量: {len(stock_files)} × 3 种策略")
        print("开始运行回测...")
        
        strategy_types = ['2%止盈', '3%止盈', '2天收盘卖', '3天开盘卖', '3天收盘卖']
        
        for strategy_type in strategy_types:
            print(f"\n{'='*60}")
            print(f"运行策略: {strategy_type}")
            print(f"{'='*60}")
            
            self.results = []
            
            for i, file_name in enumerate(stock_files):
                if i % 100 == 0:
                    print(f"处理进度: {i}/{len(stock_files)}")
                
                stock_code = file_name.split('#')[-1].replace('.txt', '')
                
                file_path = os.path.join(self.data_dir, file_name)
                if file_name in self.data_cache:
                    df = self.data_cache[file_name]
                else:
                    df = self.load_stock_data(file_path)
                    if df is not None:
                        self.data_cache[file_name] = df
                
                if df is not None and len(df) > 120:
                    result = self.run_strategy(df, stock_code, strategy_type)
                    self.results.append(result)
            
            self.print_summary(strategy_type)
            
            if self.results_file and self.results:
                results_df = pd.DataFrame(self.results)
                strategy_file = self.results_file.replace('.csv', f'_{strategy_type}.csv')
                results_df.to_csv(strategy_file, index=False, encoding='utf-8-sig')
                print(f"回测结果已保存到 {strategy_file}")
        
        self.print_strategy_comparison()
        
        return self.results
    
    def print_summary(self, strategy_type):
        if not self.results:
            print("没有回测结果")
            return
        
        results_df = pd.DataFrame(self.results)
        
        total_stocks = len(results_df)
        traded_stocks = len(results_df[results_df['num_trades'] > 0])
        total_trades = results_df['num_trades'].sum()
        total_profit = results_df['total_profit'].sum()
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        profitable_stocks = len(results_df[results_df['total_profit'] > 0])
        profitable_stocks_ratio = profitable_stocks / total_stocks if total_stocks > 0 else 0
        avg_success_rate = results_df['success_rate'].mean()
        avg_annual_return = results_df['annual_return'].mean()
        max_annual_return = results_df['annual_return'].max() if not results_df.empty else 0
        max_drawdown = results_df['max_drawdown'].max() if not results_df.empty else 0
        avg_drawdown = results_df['avg_drawdown'].mean() if not results_df.empty else 0
        max_consecutive_losses = results_df['max_consecutive_losses'].max() if not results_df.empty else 0
        avg_consecutive_losses = results_df['avg_consecutive_losses'].mean() if not results_df.empty else 0
        
        print(f"\n回测总结 ({strategy_type}):")
        print(f"总股票数: {total_stocks}")
        print(f"有交易的股票数: {traded_stocks}")
        print(f"总交易次数: {total_trades}")
        print(f"总盈利: {total_profit:.4f}")
        print(f"平均每笔交易盈利: {avg_profit_per_trade:.4f}")
        print(f"盈利股票比例: {profitable_stocks_ratio:.2%}")
        print(f"平均成功率: {avg_success_rate:.2%}")
        print(f"平均年化收益率: {avg_annual_return:.4f}")
        print(f"最大年化收益率: {max_annual_return:.4f}")
        print(f"最大回撤: {max_drawdown:.4f}%")
        print(f"平均回撤: {avg_drawdown:.4f}%")
        print(f"最大连续失败次数: {max_consecutive_losses}")
        print(f"平均连续失败次数: {avg_consecutive_losses:.2f}")
        print(f"平均每天交易次数: {results_df['avg_daily_trades'].mean():.4f}")
        
        # 🔥 打印每年收益率分布
        print("\n" + "=" * 50)
        print("每年收益率分布")
        print("=" * 50)
        
        # 汇总所有股票的 yearly_returns
        all_yearly_returns = {}
        for _, row in results_df.iterrows():
            if 'yearly_returns' in row and isinstance(row['yearly_returns'], dict):
                for year, returns in row['yearly_returns'].items():
                    if year not in all_yearly_returns:
                        all_yearly_returns[year] = []
                    all_yearly_returns[year].extend(returns)
        
        for year in sorted(all_yearly_returns.keys()):
            returns = all_yearly_returns[year]
            if returns:
                avg_return = np.mean(returns) * 100
                win_rate = np.mean([r > 0 for r in returns]) * 100
                print(f"{year}年: 交易次数={len(returns)}, 平均收益率={avg_return:.2f}%, 胜率={win_rate:.1f}%")
    
    def print_strategy_comparison(self):
        print(f"\n{'='*90}")
        print("策略对比总结")
        print(f"{'='*90}")
        
        all_results = {}
        strategy_types = ['2%止盈', '3%止盈', '2天收盘卖', '3天开盘卖', '3天收盘卖']
        
        for strategy_type in strategy_types:
            strategy_file = self.results_file.replace('.csv', f'_{strategy_type}.csv')
            if os.path.exists(strategy_file):
                df = pd.DataFrame(pd.read_csv(strategy_file))
                all_results[strategy_type] = df
        
        print(f"{'策略名称':<15} {'成功率':<10} {'平均年化收益率':<16} {'最大年化收益率':<16} {'最大回撤':<12} {'平均回撤':<12} {'最大连续失败':<12} {'日均交易次数':<12}")
        print("-" * 105)
        
        for strategy_type in strategy_types:
            if strategy_type in all_results:
                df = all_results[strategy_type]
                avg_success_rate = df['success_rate'].mean()
                avg_annual_return = df['annual_return'].mean()
                max_annual_return = df['annual_return'].max()
                max_drawdown = df['max_drawdown'].max()
                avg_drawdown = df['avg_drawdown'].mean()
                max_consecutive_losses = df['max_consecutive_losses'].max()
                avg_daily_trades = df['avg_daily_trades'].mean()
                
                print(f"{strategy_type:<15} {avg_success_rate:>6.2%}    {avg_annual_return:>12.4f}    {max_annual_return:>12.4f}    {max_drawdown:>10.2f}%    {avg_drawdown:>10.2f}%    {max_consecutive_losses:>10}    {avg_daily_trades:>10.4f}")


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    results_file = "/Users/lidongyang/Desktop/Qstrategy/backtest_brick_new_results.csv"
    
    backtest = BrickStrategyBacktest(data_dir, results_file)
    backtest.run(test_mode=False)
