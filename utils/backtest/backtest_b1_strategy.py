import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle


def write_log(content):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    log_file = os.path.join(script_dir, script_name + ".txt")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_record = f"\n========== {timestamp} ==========\n"
    new_record += content + "\n\n"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            old = f.read()
    else:
        old = ""

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(new_record + old)


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
        
        if high == low == close:
            anomaly_dates.add(df.index[i])
            continue
        
        if i > 0:
            prev_close = df.iloc[i-1]['CLOSE']
            if prev_close > 0:
                change_pct = (close - prev_close) / prev_close * 100
                if change_pct > 20 or change_pct < -20:
                    anomaly_dates.add(df.index[i])
                    continue
        
        if volume <= 0:
            anomaly_dates.add(df.index[i])
            continue
        
        if i >= 60:
            rolling_vol = df.iloc[i-60:i]['VOLUME']
            avg_vol = rolling_vol.mean()
            if avg_vol > 0 and volume > avg_vol * 5:
                anomaly_dates.add(df.index[i])
                continue
        
        if open_p < low or open_p > high:
            anomaly_dates.add(df.index[i])
            continue
        
        if high > 0 and low > 0:
            amplitude = (high - low) / low * 100
            if i > 0:
                prev_vol = df.iloc[i-1]['VOLUME']
                vol_change = volume / prev_vol if prev_vol > 0 else 1
                if amplitude > 15 and vol_change < 1.2:
                    anomaly_dates.add(df.index[i])
                    continue
        
        if i >= 1:
            for j in range(1, min(6, i+1)):
                if i - j >= 0:
                    prev_change = (df.iloc[i-j]['CLOSE'] - df.iloc[i-j-1]['CLOSE']) / df.iloc[i-j-1]['CLOSE'] * 100
                    if prev_change < -15:
                        anomaly_dates.add(df.index[i])
                        break
    
    if len(df) >= 2:
        first_price = df.iloc[0]['CLOSE']
        last_price = df.iloc[-1]['CLOSE']
        if first_price > 0 and last_price > 0:
            total_change = (last_price - first_price) / first_price
            if abs(total_change) > 100:
                for idx in df.index:
                    anomaly_dates.add(idx)
        
        if last_price < 0.5:
            for idx in df.index:
                anomaly_dates.add(idx)
        
        if i >= 1 and first_price > 0:
            avg_vol = df.iloc[:-1]['VOLUME'].mean()
            if avg_vol > 0:
                turnover = df.iloc[-1]['VOLUME'] * close
                avg_turnover = avg_vol * first_price
                if avg_turnover > 0 and turnover < avg_turnover * 0.001:
                    anomaly_dates.add(df.index[-1])
    
    return anomaly_dates


def load_stock(path):
    encodings = ['gbk', 'utf-8', 'gb18030', 'latin-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                header=1,
                encoding=encoding
            )
            df.columns = ["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "成交额"][:len(df.columns)]
            df = df[["日期", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df = df[pd.to_numeric(df["OPEN"], errors="coerce").notna()]
            df["日期"] = pd.to_datetime(df["日期"], errors='coerce')
            df = df[df["日期"].notna()]
            df = df.sort_values("日期")
            df.set_index("日期", inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df
        except Exception as e:
            continue
    
    return None


def calculate_indicators(df):
    df = df.copy()
    
    df['知行短期趋势线'] = df['CLOSE'].ewm(span=10, adjust=False).mean()
    df['知行短期趋势线'] = df['知行短期趋势线'].ewm(span=10, adjust=False).mean()

    df['MA14'] = df['CLOSE'].rolling(window=14).mean()
    df['MA28'] = df['CLOSE'].rolling(window=28).mean()
    df['MA57'] = df['CLOSE'].rolling(window=57).mean()
    df['MA114'] = df['CLOSE'].rolling(window=114).mean()

    df['知行多空线'] = (df['MA14'] + df['MA28'] + df['MA57'] + df['MA114']) / 4
    
    df['HHV9'] = df['HIGH'].rolling(9).max()
    df['LLV9'] = df['LOW'].rolling(9).min()

    rng = df['HHV9'] - df['LLV9']
    df['RSV'] = (df['CLOSE'] - df['LLV9']) / rng * 100
    df['RSV'] = df['RSV'].fillna(50)

    df['K'] = df['RSV'].ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    exp1 = df['CLOSE'].ewm(span=12, adjust=False).mean()
    exp2 = df['CLOSE'].ewm(span=26, adjust=False).mean()
    df['MACD_DIF'] = exp1 - exp2
    
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    gain28 = (delta.where(delta > 0, 0)).rolling(window=28).mean()
    loss28 = (-delta.where(delta < 0, 0)).rolling(window=28).mean()
    rs28 = gain28 / loss28
    df['RSI28'] = 100 - (100 / (1 + rs28))
    
    gain57 = (delta.where(delta > 0, 0)).rolling(window=57).mean()
    loss57 = (-delta.where(delta < 0, 0)).rolling(window=57).mean()
    rs57 = gain57 / loss57
    df['RSI57'] = 100 - (100 / (1 + rs57))
    
    df['是否阳线'] = df['CLOSE'] > df['OPEN']
    df['是否阴线'] = df['CLOSE'] < df['OPEN']
    df['涨跌幅'] = df['CLOSE'].pct_change() * 100
    
    return df


FACTOR_SCORES = {
    'MACD_DIF>0': 0.5,
    'RSI14>28>57': 0.5,
    '涨幅-3.5~2%且缩量': 2,
    '跌幅扩大且缩量': 1,
    '跳空下跌': 1,
    '收盘<多空线': -3,
    '前一日回踩趋势线且今日缩量阳线': 2,
    '60日最大量非阴线': -2,
    '前一日回踩多空线且今日缩量阳线': 2,
    '多空线<=收盘<=趋势线': 1,
    '存在倍量柱': 0.5,
    '连续倍量柱': 2,
    '阳量后接2倍阴量': 1,
}


def calculate_b1_score(df, idx):
    if idx < 2:
        return 0, {}
    
    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    
    if pd.isna(row['J']) or pd.isna(row['知行短期趋势线']) or pd.isna(row['知行多空线']):
        return 0, {}
    
    if row['J'] >= 13:
        return 0, {}
    
    if row['知行短期趋势线'] <= row['知行多空线']:
        return 0, {}
    
    score = 0
    details = {}
    
    if not pd.isna(row['MACD_DIF']) and row['MACD_DIF'] > 0:
        score += 0.5
        details['MACD_DIF>0'] = 0.5
    
    if not (pd.isna(row['RSI14']) or pd.isna(row['RSI28']) or pd.isna(row['RSI57'])):
        if row['RSI14'] > row['RSI28'] > row['RSI57']:
            score += 0.5
            details['RSI14>28>57'] = 0.5
    
    change_pct = row['涨跌幅']
    if not pd.isna(change_pct) and not pd.isna(row['VOLUME']) and not pd.isna(prev['VOLUME']):
        if -3.5 < change_pct < 2 and row['VOLUME'] < prev['VOLUME']:
            score += 2
            details['涨幅-3.5~2%且缩量'] = 2
    
    if not pd.isna(prev['涨跌幅']) and not pd.isna(row['涨跌幅']):
        if abs(row['涨跌幅']) > abs(prev['涨跌幅']) and row['VOLUME'] < prev['VOLUME']:
            score += 1
            details['跌幅扩大且缩量'] = 1
    
    if not pd.isna(row['OPEN']) and not pd.isna(prev['CLOSE']):
        if row['OPEN'] < prev['CLOSE']:
            score += 1
            details['跳空下跌'] = 1
    
    if row['CLOSE'] < row['知行多空线']:
        score -= 3
        details['收盘<多空线'] = -3
    
    if idx >= 2 and not pd.isna(prev['知行短期趋势线']) and not pd.isna(row['知行短期趋势线']):
        if prev['CLOSE'] <= prev['知行短期趋势线'] and row['CLOSE'] > row['知行短期趋势线']:
            if row['VOLUME'] < prev['VOLUME'] and row['是否阳线']:
                score += 2
                details['前一日回踩趋势线且今日缩量阳线'] = 2
    
    if idx >= 60:
        rolling_vol = df.iloc[idx-60:idx]['VOLUME']
        max_vol_idx = rolling_vol.idxmax()
        if max_vol_idx in df.index:
            max_vol_row = df.loc[max_vol_idx]
            if not max_vol_row['是否阴线']:
                score -= 2
                details['60日最大量非阴线'] = -2
    
    if idx >= 2 and not pd.isna(prev['知行多空线']) and not pd.isna(row['知行多空线']):
        if prev['CLOSE'] <= prev['知行多空线'] and row['CLOSE'] > row['知行多空线']:
            if row['VOLUME'] < prev['VOLUME'] and row['是否阳线']:
                score += 2
                details['前一日回踩多空线且今日缩量阳线'] = 2
    
    if row['知行多空线'] <= row['CLOSE'] <= row['知行短期趋势线']:
        score += 1
        details['多空线<=收盘<=趋势线'] = 1
    
    has_duplicate_volume = False
    continuous_duplicate = False
    for i in range(max(1, idx-60), idx):
        if i > 0 and not pd.isna(df.iloc[i]['VOLUME']) and not pd.isna(df.iloc[i-1]['VOLUME']):
            if df.iloc[i]['是否阳线'] and df.iloc[i]['VOLUME'] >= df.iloc[i-1]['VOLUME'] * 2:
                has_duplicate_volume = True
                consecutive = 1
                for j in range(i+1, min(i+5, idx)):
                    if df.iloc[j]['是否阳线'] and df.iloc[j]['VOLUME'] >= df.iloc[j-1]['VOLUME'] * 2:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 2:
                    continuous_duplicate = True
                    break
    
    if continuous_duplicate:
        score += 2
        details['连续倍量柱'] = 2
    elif has_duplicate_volume:
        score += 0.5
        details['存在倍量柱'] = 0.5
    
    has_yinyang_pattern = False
    for i in range(max(1, idx-60), idx-1):
        if i > 0:
            if df.iloc[i]['是否阳线'] and df.iloc[i+1]['是否阴线']:
                if df.iloc[i]['VOLUME'] >= df.iloc[i+1]['VOLUME'] * 2:
                    max_vol_in_range = df.iloc[i:idx+1]['VOLUME'].max()
                    if df.iloc[i]['VOLUME'] >= max_vol_in_range * 0.9:
                        has_yinyang_pattern = True
                        break
    
    if has_yinyang_pattern:
        score += 1
        details['阳量后接2倍阴量'] = 1
    
    return score, details


def analyze_factors(stock_data, daily_signals, daily_scores, all_dates, date_to_idx):
    print("\n" + "=" * 60)
    print("因子分析：统计每个因子触发时的平均收益")
    print("=" * 60)
    
    factor_returns = {factor: [] for factor in FACTOR_SCORES.keys()}
    
    for current_date in all_dates:
        if current_date not in daily_scores:
            continue
        
        scores_today = daily_scores[current_date]
        for stock, score, details in scores_today:
            if score <= 0:
                continue
            
            df = stock_data[stock]
            idx = df.index.get_loc(current_date)
            
            if idx + 1 >= len(df):
                continue
            
            entry_p = df.iloc[idx + 1]["OPEN"]
            if entry_p <= 0 or np.isnan(entry_p):
                continue
            
            for factor_name in details.keys():
                if factor_name not in FACTOR_SCORES:
                    continue
                factor_returns[factor_name].append({
                    'date': current_date,
                    'entry_price': entry_p,
                    'stock': stock
                })
    
    factor_analysis = []
    for factor_name, trades in factor_returns.items():
        if len(trades) < 10:
            continue
        
        returns = []
        for trade in trades:
            df = stock_data[trade['stock']]
            date = trade['date']
            entry_price = trade['entry_price']
            
            try:
                idx = df.index.get_loc(date)
                
                max_holding = min(20, len(df) - idx - 1)
                best_return = -100
                exit_price = entry_price
                
                for h in range(1, max_holding + 1):
                    if idx + h >= len(df):
                        break
                    
                    row = df.iloc[idx + h]
                    close = row['CLOSE']
                    low = row['LOW']
                    
                    if low <= entry_price * 0.93:
                        ret = (entry_price * 0.93 - entry_price) / entry_price * 100
                        best_return = max(best_return, ret)
                        break
                    
                    ret = (close - entry_price) / entry_price * 100
                    if ret > best_return:
                        best_return = ret
                
                if best_return > -100:
                    returns.append(best_return)
            except:
                continue
        
        if len(returns) >= 10:
            avg_return = np.mean(returns)
            win_rate = np.mean([r > 0 for r in returns]) * 100
            factor_analysis.append({
                'factor': factor_name,
                'trigger_count': len(returns),
                'avg_return': avg_return,
                'win_rate': win_rate,
                'score': FACTOR_SCORES[factor_name]
            })
    
    factor_analysis.sort(key=lambda x: x['avg_return'], reverse=True)
    
    print(f"\n{'因子名称':<35} {'触发次数':>10} {'平均收益':>12} {'胜率':>10} {'评分':>8}")
    print("-" * 75)
    for item in factor_analysis:
        print(f"{item['factor']:<35} {item['trigger_count']:>10} {item['avg_return']:>11.2f}% {item['win_rate']:>9.1f}% {item['score']:>8.1f}")
    
    return factor_analysis


def run_backtest(data_dir,
                 initial_capital=1_000_000,
                 max_positions=10,
                 min_score=0.5,
                 fail_pause_X=3,
                 fail_pause_Y=2):
    
    cache_file = "/tmp/b1_cache.pkl"
    
    if os.path.exists(cache_file):
        print("加载缓存数据...")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        stock_data = cache['stock_data']
        daily_signals = cache['daily_signals']
        daily_scores = cache['daily_scores']
        all_dates = cache['all_dates']
    else:
        stock_data = {}
        daily_signals = {}
        daily_scores = {}
        
        files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        total = len(files)
        
        print(f"加载 {total} 只股票...")
        
        loaded_count = 0
        for idx, file in enumerate(files, 1):
            if idx % 500 == 0:
                print(f"[加载 {idx}/{total}]")
            
            df = load_stock(os.path.join(data_dir, file))
            if df is None or len(df) < 130:
                continue
            
            loaded_count += 1
            
            df = calculate_indicators(df)
            stock_data[file] = df
            
            for i in range(2, len(df)):
                score, details = calculate_b1_score(df, i)
                if score > 0:
                    date = df.index[i]
                    if date not in daily_signals:
                        daily_signals[date] = []
                        daily_scores[date] = []
                    daily_signals[date].append(file)
                    daily_scores[date].append((file, score, details))
            
            if loaded_count >= 500:
                print(f"已达到测试数量 500，停止加载")
                break
        
        print(f"成功加载 {loaded_count} 只股票")
        
        all_dates = sorted(set().union(*[df.index for df in stock_data.values()]))
        
        print("保存缓存...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'stock_data': stock_data,
                'daily_signals': daily_signals,
                'daily_scores': daily_scores,
                'all_dates': all_dates
            }, f)
    
    print(f"总交易日: {len(all_dates)}")
    print(f"有信号的天数: {len(daily_signals)}")
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    factor_analysis = analyze_factors(stock_data, daily_signals, daily_scores, all_dates, date_to_idx)
    
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
            
            trend_line = row['知行短期趋势线']
            dk_line = row['知行多空线']
            
            exit_flag = False
            exit_price = close
            exit_reason = ""
            
            stop_price = pos["stop_price"]
            if low <= stop_price:
                stop_count = pos.get('stop_count', 0)
                if stop_count == 0:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': stop_price,
                        'ratio': 0.5,
                        'reason': '止损卖出50%'
                    })
                    pos['stop_count'] = 1
                elif stop_count == 1:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': stop_price,
                        'ratio': 0.25,
                        'reason': '止损再次卖出25%'
                    })
                    pos['stop_count'] = 2
                else:
                    exit_flag = True
                    exit_price = stop_price
                    exit_reason = "止损全部卖出"
            
            # 止盈：同一天可以触发多个条件（去掉 exit_flag 限制）
            # J > 100：卖出 20%
            if pos.get('j_take_profit_done', False) == False:
                if not pd.isna(row['J']) and row['J'] > 100:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': 0.2,
                        'reason': 'J>100'
                    })
                    pos['j_take_profit_done'] = True
            
            # 涨幅 > 8%：卖出 20%
            change_pct = (close - entry_price) / entry_price * 100
            if pos.get('gain_take_profit_done', False) == False:
                if change_pct > 8:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': 0.2,
                        'reason': '涨幅>8%'
                    })
                    pos['gain_take_profit_done'] = True
            
            # 偏离趋势线 15%/20%/25%：各卖出 20%
            if not pd.isna(trend_line):
                deviation = (high - trend_line) / trend_line * 100
                if pos.get('dev15_done', False) == False and deviation >= 15:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': 0.2,
                        'reason': '偏离趋势线15%'
                    })
                    pos['dev15_done'] = True
                if pos.get('dev20_done', False) == False and deviation >= 20:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': 0.2,
                        'reason': '偏离趋势线20%'
                    })
                    pos['dev20_done'] = True
                if pos.get('dev25_done', False) == False and deviation >= 25:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': 0.2,
                        'reason': '偏离趋势线25%'
                    })
                    pos['dev25_done'] = True
            
            partial_sells = pos.get('partial_sells', [])
            total_sold_ratio = sum([p['ratio'] for p in partial_sells])
            
            # 止盈全部完成后，连续两天收盘价低于前一日最低则全部平仓
            if pos.get('dev25_done', False) and total_sold_ratio >= 1.0:
                if holding_days >= 2:
                    stock_idx = df.index.get_loc(current_date)
                    if stock_idx > 0:
                        prev_k_low = df.iloc[stock_idx - 1]['LOW']
                    else:
                        prev_k_low = close
                    if close < prev_k_low:
                        if pos.get('consecutive_low_days', 0) >= 1:
                            exit_flag = True
                            exit_price = close
                            exit_reason = "连续两天收盘低于前一日最低"
                            pos['consecutive_low_days'] = pos.get('consecutive_low_days', 0) + 1
                        else:
                            pos['consecutive_low_days'] = 1
                    else:
                        pos['consecutive_low_days'] = 0
            
            if not exit_flag and holding_days >= 1:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0:
                    prev_dk = df.iloc[stock_idx - 1]['知行多空线']
                    prev_trend = df.iloc[stock_idx - 1]['知行短期趋势线']
                    if not pd.isna(prev_dk) and not pd.isna(prev_trend):
                        if prev_dk <= prev_trend and dk_line > trend_line:
                            exit_flag = True
                            exit_price = open_p
                            exit_reason = "多空线金叉趋势线"
            
            # 60日最大量阴线分批卖出
            if not exit_flag and holding_days >= 60:
                stock_idx = df.index.get_loc(current_date)
                rolling_vol = df.iloc[max(0, stock_idx-60):stock_idx]['VOLUME']
                if len(rolling_vol) == 0:
                    continue
                max_vol_idx = rolling_vol.idxmax()
                if max_vol_idx in df.index:
                    max_vol_row = df.loc[max_vol_idx]
                    if max_vol_row['是否阴线']:
                        vol_exit_count = pos.get('vol_exit_count', 0)
                        if vol_exit_count == 0:
                            pos['partial_sells'].append({
                                'date': current_date,
                                'price': open_p,
                                'ratio': 0.5,
                                'reason': '60日最大量阴线卖出50%'
                            })
                            pos['vol_exit_count'] = 1
                        elif vol_exit_count == 1:
                            pos['partial_sells'].append({
                                'date': current_date,
                                'price': open_p,
                                'ratio': 0.25,
                                'reason': '60日最大量阴线再次卖出25%'
                            })
                            pos['vol_exit_count'] = 2
                        else:
                            exit_flag = True
                            exit_price = open_p
                            exit_reason = "60日最大量阴线连续3天全部平仓"
            
            if exit_flag:
                gross = (exit_price - entry_price) / entry_price
                net = gross - 0.0005 * 2 - 0.001
                
                for ps in partial_sells:
                    cash += pos["invested"] * pos.get('hold_ratio', 1.0) * ps['ratio'] * (1 + (ps['price'] - entry_price) / entry_price - 0.0005 - 0.001)
                
                remaining_ratio = 1.0 - sum([p['ratio'] for p in partial_sells])
                cash += pos["invested"] * pos.get('hold_ratio', 1.0) * remaining_ratio * (1 + net)
                
                trade_count += 1
                if gross > 0:
                    win_count += 1
                    current_consecutive_losses = 0
                else:
                    loss_count += 1
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                    if current_consecutive_losses >= fail_pause_X:
                        pause_trading_days = fail_pause_Y
                
                year = current_date.year
                if year not in yearly_returns:
                    yearly_returns[year] = []
                yearly_returns[year].append(gross)
            else:
                pos["current_price"] = close
                pos["holding_days"] = holding_days
                new_positions.append(pos)
        
        positions = new_positions
        
        total_equity = cash if not np.isnan(cash) else 0
        for pos in positions:
            price_ratio = pos["current_price"] / pos["entry_price"]
            if np.isnan(price_ratio) or price_ratio <= 0:
                price_ratio = 1.0
            hold_ratio = pos.get('hold_ratio', 1.0)
            total_equity += pos["invested"] * hold_ratio * price_ratio
        
        if np.isnan(total_equity) or total_equity <= 0:
            stopped = True
            break
        
        equity_curve.append(total_equity)
        
        if current_date not in daily_scores:
            continue
        
        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue
        
        if pause_trading_days > 0:
            continue
        
        scores_today = [s for s in daily_scores[current_date] if s[1] >= min_score]
        scores_today.sort(key=lambda x: x[1], reverse=True)
        
        existing_stocks = {pos["stock"] for pos in positions}
        available_signals = [s for s in scores_today if s[0] not in existing_stocks]
        
        if not available_signals:
            continue
        
        if cash <= 0 or np.isnan(cash):
            continue
        
        num_to_buy = min(len(available_signals), available_slots)
        
        if num_to_buy > 0:
            per_position = cash / num_to_buy
        
        for i in range(num_to_buy):
            if cash < per_position * 0.5:
                break
            
            stock, score, details = available_signals[i]
            
            df = stock_data[stock]
            idx = df.index.get_loc(current_date)
            
            if idx + 1 >= len(df):
                continue
            
            entry_p = df.iloc[idx + 1]["OPEN"]
            stop_p = df.iloc[idx]["LOW"]
            
            if entry_p <= 0 or np.isnan(entry_p):
                continue
            
            gap_up = (entry_p - df.iloc[idx]["CLOSE"]) / df.iloc[idx]["CLOSE"]
            if gap_up >= 0.09:
                continue
            
            invested = per_position
            cash -= invested
            
            positions.append({
                "stock": stock,
                "entry_price": entry_p,
                "entry_date": current_date,
                "entry_idx": idx + 1,
                "invested": invested,
                "current_price": entry_p,
                "stop_price": stop_p * 0.93,
                "hold_ratio": 1.0,
                "j_take_profit_done": False,
                "gain_take_profit_done": False,
                "dev15_done": False,
                "dev20_done": False,
                "dev25_done": False,
                "partial_sells": [],
                "consecutive_low_days": 0
            })
    
    equity_curve = np.array(equity_curve)
    
    if len(equity_curve) == 0:
        print("错误：没有交易数据")
        return
    
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    
    final_multiple = equity_curve[-1] / initial_capital
    
    start_date = all_dates[0]
    end_date = all_dates[-1]
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    if total_years > 0 and final_multiple > 0:
        CAGR = final_multiple ** (1 / total_years) - 1
    else:
        CAGR = 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    sharpe = 0
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    
    success_rate = win_count / trade_count * 100 if trade_count > 0 else 0
    
    print("\n" + "=" * 50)
    print("B1策略回测结果")
    print("=" * 50)
    print(f"初始资金: {initial_capital:,.0f}")
    print(f"最终资金: {equity_curve[-1]:,.2f}")
    print(f"最终倍数: {final_multiple:.4f}")
    print(f"年化收益率(CAGR): {CAGR:.4f} ({CAGR*100:.2f}%)")
    print(f"最大回撤: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"年化夏普: {sharpe:.4f}")
    print(f"总交易次数: {trade_count}")
    print(f"成功次数: {win_count}, 失败次数: {loss_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"最大连续失败次数: {max_consecutive_losses}")
    
    print("\n" + "=" * 50)
    print("每年收益率分布")
    print("=" * 50)
    for year in sorted(yearly_returns.keys()):
        returns = yearly_returns[year]
        if returns:
            avg_ret = np.mean(returns) * 100
            win_rate = np.mean([r > 0 for r in returns]) * 100
            print(f"{year}年: 交易次数={len(returns)}, 平均收益率={avg_ret:.2f}%, 胜率={win_rate:.1f}%")
    
    strategy_desc = """
===== B1 评分系统 (精简版 - 5个核心因子) =====
买入条件（必须）:
- J < 13
- 当日短期趋势线 > 多空线

评分条件:
- MACD DIF > 0: +0.5分
- RSI 14 > 28 > 57: +0.5分
- -3.5% < 涨幅 < 2% 且缩量: +2分
- 前一日回踩趋势线且今日收盘价 > 趋势线且缩量阳线: +2分
- 60日内存在倍量柱: +0.5分

===== 止盈策略 =====
1. J值第一次 > 100: 卖出20%仓位
2. 涨幅第一次 > 8%: 卖出20%仓位
3. 最高价偏离趋势线 >= 15%: 卖出20%仓位
4. 最高价偏离趋势线 >= 20%: 卖出20%仓位
5. 最高价偏离趋势线 >= 25%: 卖出20%仓位
6. 完成前5步止盈后，连续两天收盘价 < 前一日最低: 全部平仓

===== 止损策略 =====
1. 触发止损价 (买入K线最低价 * 0.93): 全部平仓
2. 前一日多空线 < 趋势线，当日多空线 > 趋势线: 全部平仓
3. 持仓期间出现60日内最高成交量的阴线: 全部平仓

===== 风控策略 =====
- 连续失败3次后，暂停交易2天"""
    
    result = f"""
===== B1 策略回测结果 =====

【资金指标】
初始资金: {initial_capital:,.0f}
最终资金: {equity_curve[-1]:,.2f}
最终倍数: {final_multiple:.4f}
年化收益率(CAGR): {CAGR:.4f} ({CAGR*100:.2f}%)
最大回撤: {max_dd:.4f} ({max_dd*100:.2f}%)
年化夏普: {sharpe:.4f}

【交易统计】
总交易次数: {trade_count}
成功次数: {win_count}
失败次数: {loss_count}
成功率: {success_rate:.2f}%
最大连续失败次数: {max_consecutive_losses}

{strategy_desc}
"""
    
    write_log(result)
    print(result)


if __name__ == "__main__":
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"
    run_backtest(data_dir, min_score=0.5)
