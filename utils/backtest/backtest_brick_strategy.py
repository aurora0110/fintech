import os
import pandas as pd
import numpy as np


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
            
            if position > 0:
                holding_days = (current_date - entry_date).days
                
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
                                'type': '第2天收盘卖出',
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
                    
                    elif strategy_type == '3天开盘卖':
                        if holding_days >= 2:
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
                                'type': '第3天开盘卖出',
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
                        
                        if price_change <= 3:
                            next_close = next_data['CLOSE']
                            next_high = next_data['HIGH']
                            next_low = next_data['LOW']
                            
                            if next_close < next_open:
                                is_bearish = True
                            else:
                                is_bearish = False
                            
                            if next_close >= duokongxian:
                                if check_buy_conditions(df, i, next_data):
                                    entry_price = next_open
                                    entry_date = next_data['日期']
                                    entry_low = current_low
                                    entry_is_bearish = is_bearish
                                    position = 1
                                    entry_index = i + 1
                                    half_sold = False
                                    max_high_since_entry = next_high
                                    min_low_since_entry = next_low
                                    
                                    trades.append({
                                        'entry_date': entry_date,
                                        'entry_price': entry_price,
                                        'position': position,
                                        'type': '砖型图信号买入'
                                    })
        
        num_trades = len(completed_trades)
        total_profit = sum([trade.get('profit', 0) for trade in completed_trades])
        avg_profit = total_profit / num_trades if num_trades > 0 else 0
        
        win_trades = [trade for trade in completed_trades if trade.get('profit', 0) > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        
        all_profit_pcts = [trade.get('profit_pct', 0) for trade in completed_trades if 'profit_pct' in trade]
        avg_profit_pct = sum(all_profit_pcts) / len(all_profit_pcts) if all_profit_pcts else 0
        max_profit_pct = max(all_profit_pcts) if all_profit_pcts else 0
        
        cumulative_profit = capital / self.initial_capital
        
        first_date = df['日期'].iloc[0]
        last_date = df['日期'].iloc[-1]
        total_days = (last_date - first_date).days
        years = total_days / 365.0
        if years > 0:
            annual_return = (cumulative_profit - 1) / years
        else:
            annual_return = 0
        
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
            'completed_trades': completed_trades
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
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260226"
    results_file = "/Users/lidongyang/Desktop/Qstrategy/backtest_brick_new_results.csv"
    
    backtest = BrickStrategyBacktest(data_dir, results_file)
    backtest.run(test_mode=False)
