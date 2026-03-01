import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import json
import random
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

LOG_FILE = "/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer/optimizer.txt"

def log_result(params, result):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_entry = f"\n========== {timestamp} ==========\n"
    log_entry += f"参数: {params}\n"
    if result:
        log_entry += f"最终倍数: {result.get('final_multiple', 0):.4f}\n"
        log_entry += f"CAGR: {result.get('CAGR', 0)*100:.2f}%\n"
        log_entry += f"最大回撤: {result.get('max_dd', 0)*100:.2f}%\n"
        log_entry += f"夏普比率: {result.get('sharpe', 0):.4f}\n"
        log_entry += f"交易次数: {result.get('trade_count', 0)}\n"
        log_entry += f"成功率: {result.get('success_rate', 0):.2f}%\n"
    else:
        log_entry += "结果: 无效\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

CACHE_FILE = "/tmp/b1_cache_full.pkl"
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

print("="*60)
print("B1策略参数优化系统")
print("="*60)

print("\n加载缓存数据...")
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    stock_data = cache['stock_data']
    daily_signals = cache['daily_signals']
    daily_scores = cache['daily_scores']
    all_dates = cache['all_dates']
    print(f"加载成功: {len(stock_data)} 只股票, {len(all_dates)} 个交易日")
else:
    print("错误: 缓存文件不存在!")
    sys.exit(1)

print("\n创建因子分析索引...")
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

def calculate_b1_score(df, i):
    if i < 2:
        return 0, {}
    
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    
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
    
    if i >= 2 and not pd.isna(prev['知行短期趋势线']) and not pd.isna(row['知行短期趋势线']):
        if prev['CLOSE'] <= prev['知行短期趋势线'] and row['CLOSE'] > row['知行短期趋势线']:
            if row['VOLUME'] < prev['VOLUME'] and row['是否阳线']:
                score += 2
                details['前一日回踩趋势线且今日缩量阳线'] = 2
    
    if i >= 60:
        rolling_vol = df.iloc[i-60:i]['VOLUME']
        max_vol_idx = rolling_vol.idxmax()
        if max_vol_idx in df.index:
            max_vol_row = df.loc[max_vol_idx]
            if not max_vol_row['是否阴线']:
                score -= 2
                details['60日最大量非阴线'] = -2
    
    if i >= 2 and not pd.isna(prev['知行多空线']) and not pd.isna(row['知行多空线']):
        if prev['CLOSE'] <= prev['知行多空线'] and row['CLOSE'] > row['知行多空线']:
            if row['VOLUME'] < prev['VOLUME'] and row['是否阳线']:
                score += 2
                details['前一日回踩多空线且今日缩量阳线'] = 2
    
    if row['知行多空线'] <= row['CLOSE'] <= row['知行短期趋势线']:
        score += 1
        details['多空线<=收盘<=趋势线'] = 1
    
    has_duplicate_volume = False
    continuous_duplicate = False
    for j in range(max(1, i-60), i):
        if j > 0 and not pd.isna(df.iloc[j]['VOLUME']) and not pd.isna(df.iloc[j-1]['VOLUME']):
            if df.iloc[j]['是否阳线'] and df.iloc[j]['VOLUME'] >= df.iloc[j-1]['VOLUME'] * 2:
                has_duplicate_volume = True
                consecutive = 1
                for k in range(j+1, min(j+5, i+1)):
                    if df.iloc[k]['是否阳线'] and df.iloc[k]['VOLUME'] >= df.iloc[k-1]['VOLUME'] * 2:
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
    for j in range(max(1, i-60), i-1):
        if j > 0:
            if df.iloc[j]['是否阳线'] and df.iloc[j+1]['是否阴线']:
                if df.iloc[j]['VOLUME'] >= df.iloc[j+1]['VOLUME'] * 2:
                    max_vol_in_range = df.iloc[j:i+1]['VOLUME'].max()
                    if df.iloc[j]['VOLUME'] >= max_vol_in_range * 0.9:
                        has_yinyang_pattern = True
                        break
    
    if has_yinyang_pattern:
        score += 1
        details['阳量后接2倍阴量'] = 1
    
    return score, details


def run_backtest(params, use_train_test=False, train_ratio=0.7):
    initial_capital = params.get('initial_capital', 1_000_000)
    max_positions = params.get('max_positions', 10)
    min_score = params.get('min_score', 0.5)
    j_take_profit_pct = params.get('j_take_profit_pct', 100)
    j_take_profit_ratio = params.get('j_take_profit_ratio', 0.2)
    gain_take_profit_pct = params.get('gain_take_profit_pct', 8)
    gain_take_profit_ratio = params.get('gain_take_profit_ratio', 0.2)
    dev15_pct = params.get('dev15_pct', 15)
    dev15_ratio = params.get('dev15_ratio', 0.2)
    dev20_pct = params.get('dev20_pct', 20)
    dev20_ratio = params.get('dev20_ratio', 0.2)
    dev25_pct = params.get('dev25_pct', 25)
    dev25_ratio = params.get('dev25_ratio', 0.2)
    stop_loss_pct = params.get('stop_loss_pct', 7)
    gap_up_limit = params.get('gap_up_limit', 0.09)
    
    commission = 0.0005
    stamp = 0.001
    
    if use_train_test:
        train_end_idx = int(len(all_dates) * train_ratio)
        train_dates = all_dates[:train_end_idx]
        test_dates = all_dates[train_end_idx:]
        dates_to_use = test_dates
    else:
        dates_to_use = all_dates
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    daily_signals_filtered = {}
    for date in dates_to_use:
        if date in daily_scores:
            scores_today = [(s[0], s[1], s[2]) for s in daily_scores[date] if s[1] >= min_score]
            if scores_today:
                daily_signals_filtered[date] = scores_today
    
    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    
    trade_count = 0
    win_count = 0
    loss_count = 0
    
    yearly_returns = {}
    
    for current_date in dates_to_use:
        if stopped:
            break
        
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
                        'reason': f'止损卖出{int(stop_loss_pct)}%'
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
            
            if pos.get('j_take_profit_done', False) == False:
                if not pd.isna(row['J']) and row['J'] > j_take_profit_pct:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': j_take_profit_ratio,
                        'reason': f'J>{j_take_profit_pct}'
                    })
                    pos['j_take_profit_done'] = True
            
            change_pct = (close - entry_price) / entry_price * 100
            if pos.get('gain_take_profit_done', False) == False:
                if change_pct > gain_take_profit_pct:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': gain_take_profit_ratio,
                        'reason': f'涨幅>{gain_take_profit_pct}%'
                    })
                    pos['gain_take_profit_done'] = True
            
            if not pd.isna(trend_line):
                deviation = (high - trend_line) / trend_line * 100
                if pos.get('dev15_done', False) == False and deviation >= dev15_pct:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': dev15_ratio,
                        'reason': f'偏离{int(dev15_pct)}%'
                    })
                    pos['dev15_done'] = True
                if pos.get('dev20_done', False) == False and deviation >= dev20_pct:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': dev20_ratio,
                        'reason': f'偏离{int(dev20_pct)}%'
                    })
                    pos['dev20_done'] = True
                if pos.get('dev25_done', False) == False and deviation >= dev25_pct:
                    pos['partial_sells'].append({
                        'date': current_date,
                        'price': close,
                        'ratio': dev25_ratio,
                        'reason': f'偏离{int(dev25_pct)}%'
                    })
                    pos['dev25_done'] = True
            
            partial_sells = pos.get('partial_sells', [])
            total_sold_ratio = sum([p['ratio'] for p in partial_sells])
            
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
            
            if not exit_flag and holding_days >= 60:
                stock_idx = df.index.get_loc(current_date)
                rolling_vol = df.iloc[max(0, stock_idx-60):stock_idx]['VOLUME']
                if len(rolling_vol) > 0:
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
                net = gross - commission * 2 - stamp
                
                for ps in partial_sells:
                    cash += pos["invested"] * pos.get('hold_ratio', 1.0) * ps['ratio'] * (1 + (ps['price'] - entry_price) / entry_price - commission - stamp)
                
                remaining_ratio = 1.0 - sum([p['ratio'] for p in partial_sells])
                cash += pos["invested"] * pos.get('hold_ratio', 1.0) * remaining_ratio * (1 + net)
                
                trade_count += 1
                if gross > 0:
                    win_count += 1
                else:
                    loss_count += 1
                
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
        
        if current_date not in daily_signals_filtered:
            continue
        
        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue
        
        scores_today = daily_signals_filtered[current_date]
        
        existing_stocks = {pos["stock"] for pos in positions}
        available_signals = [s for s in scores_today if s[0] not in existing_stocks]
        
        if not available_signals:
            continue
        
        if cash <= 0 or np.isnan(cash):
            continue
        
        num_to_buy = min(len(available_signals), int(available_slots))
        
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
            if gap_up >= gap_up_limit:
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
                "stop_price": stop_p * (1 - stop_loss_pct/100),
                "hold_ratio": 1.0,
                "j_take_profit_done": False,
                "gain_take_profit_done": False,
                "dev15_done": False,
                "dev20_done": False,
                "dev25_done": False,
                "partial_sells": [],
                "consecutive_low_days": 0
            })
    
    if len(equity_curve) == 0:
        return None
    
    equity_curve = np.array(equity_curve)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    
    final_multiple = equity_curve[-1] / initial_capital
    
    start_date = dates_to_use[0]
    end_date = dates_to_use[-1]
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
    
    return {
        'params': params,
        'final_multiple': final_multiple,
        'CAGR': CAGR,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trade_count': trade_count,
        'win_count': win_count,
        'loss_count': loss_count,
        'success_rate': success_rate,
        'yearly_returns': yearly_returns,
        'final_capital': equity_curve[-1],
        'initial_capital': initial_capital
    }


def grid_search():
    print("\n" + "="*60)
    print("第一阶段: 网格搜索")
    print("="*60)
    
    param_grid = {
        'min_score': [0.5, 1.0, 2.0],
        'max_positions': [4, 8, 12],
        'j_take_profit_pct': [100],
        'j_take_profit_ratio': [0.2],
        'gain_take_profit_pct': [5, 8, 12],
        'gain_take_profit_ratio': [0.2],
        'dev15_pct': [15],
        'dev15_ratio': [0.2],
        'dev20_pct': [20],
        'dev20_ratio': [0.2],
        'dev25_pct': [25],
        'dev25_ratio': [0.2],
        'stop_loss_pct': [5, 7, 10],
        'gap_up_limit': [0.05, 0.09],
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    print(f"总参数组合数: {total_combinations}")
    
    results = []
    processed = 0
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        
        result = run_backtest(params)
        
        log_result(params, result)
        
        if result:
            results.append(result)
        
        processed += 1
        if processed % 50 == 0:
            print(f"进度: {processed}/{total_combinations}")
    
    results.sort(key=lambda x: x['CAGR'], reverse=True)
    
    print("\n网格搜索 Top 20 结果:")
    print("-"*80)
    for i, r in enumerate(results[:20]):
        p = r['params']
        print(f"{i+1}. 倍数:{r['final_multiple']:.2f} CAGR:{r['CAGR']*100:.2f}% 回撤:{r['max_dd']*100:.2f}% 夏普:{r['sharpe']:.2f} 交易:{r['trade_count']} 胜率:{r['success_rate']:.1f}%")
        print(f"   参数: score={p['min_score']}, pos={p['max_positions']}, j_tp={p['j_take_profit_pct']}, gain_tp={p['gain_take_profit_pct']}, stop={p['stop_loss_pct']}%")
    
    return results[:20]


def random_search(n_iter=500):
    print("\n" + "="*60)
    print("第一阶段: 随机搜索")
    print("="*60)
    
    param_distributions = {
        'min_score': (0.5, 4.0),
        'max_positions': (3, 15),
        'j_take_profit_pct': (60, 150),
        'j_take_profit_ratio': (0.05, 0.4),
        'gain_take_profit_pct': (3, 15),
        'gain_take_profit_ratio': (0.05, 0.4),
        'dev15_pct': (5, 25),
        'dev15_ratio': (0.05, 0.3),
        'dev20_pct': (10, 30),
        'dev20_ratio': (0.05, 0.3),
        'dev25_pct': (15, 35),
        'dev25_ratio': (0.1, 0.5),
        'stop_loss_pct': (3, 15),
        'gap_up_limit': (0.03, 0.15),
    }
    
    results = []
    
    for i in range(n_iter):
        params = {
            'min_score': round(random.uniform(*param_distributions['min_score']), 1),
            'max_positions': int(round(random.uniform(param_distributions['max_positions'][0], param_distributions['max_positions'][1]))),
            'j_take_profit_pct': int(round(random.uniform(param_distributions['j_take_profit_pct'][0], param_distributions['j_take_profit_pct'][1]))),
            'j_take_profit_ratio': round(random.uniform(*param_distributions['j_take_profit_ratio']), 2),
            'gain_take_profit_pct': int(round(random.uniform(param_distributions['gain_take_profit_pct'][0], param_distributions['gain_take_profit_pct'][1]))),
            'gain_take_profit_ratio': round(random.uniform(*param_distributions['gain_take_profit_ratio']), 2),
            'dev15_pct': int(round(random.uniform(param_distributions['dev15_pct'][0], param_distributions['dev15_pct'][1]))),
            'dev15_ratio': round(random.uniform(*param_distributions['dev15_ratio']), 2),
            'dev20_pct': int(round(random.uniform(param_distributions['dev20_pct'][0], param_distributions['dev20_pct'][1]))),
            'dev20_ratio': round(random.uniform(*param_distributions['dev20_ratio']), 2),
            'dev25_pct': int(round(random.uniform(param_distributions['dev25_pct'][0], param_distributions['dev25_pct'][1]))),
            'dev25_ratio': round(random.uniform(*param_distributions['dev25_ratio']), 2),
            'stop_loss_pct': int(round(random.uniform(param_distributions['stop_loss_pct'][0], param_distributions['stop_loss_pct'][1]))),
            'gap_up_limit': round(random.uniform(*param_distributions['gap_up_limit']), 2),
        }
        
        result = run_backtest(params)
        
        log_result(params, result)
        
        if result:
            results.append(result)
        
        if (i+1) % 10 == 0:
            print(f"进度: {i+1}/{n_iter}")
    
    results.sort(key=lambda x: x['CAGR'], reverse=True)
    
    print("\n随机搜索 Top 20 结果:")
    print("-"*80)
    for i, r in enumerate(results[:20]):
        p = r['params']
        print(f"{i+1}. 倍数:{r['final_multiple']:.2f} CAGR:{r['CAGR']*100:.2f}% 回撤:{r['max_dd']*100:.2f}% 夏普:{r['sharpe']:.2f} 交易:{r['trade_count']} 胜率:{r['success_rate']:.1f}%")
        print(f"   参数: score={p['min_score']}, pos={p['max_positions']}, j_tp={p['j_take_profit_pct']}, gain_tp={p['gain_take_profit_pct']}, stop={p['stop_loss_pct']}%")
    
    return results[:20]


def bayesian_optimization(n_iter=200):
    print("\n" + "="*60)
    print("第一阶段: 贝叶斯优化")
    print("="*60)
    
    param_bounds = {
        'min_score': (0.5, 4.0),
        'max_positions': (3, 15),
        'j_take_profit_pct': (60, 150),
        'j_take_profit_ratio': (0.05, 0.4),
        'gain_take_profit_pct': (3, 15),
        'gain_take_profit_ratio': (0.05, 0.4),
        'dev15_pct': (5, 25),
        'dev15_ratio': (0.05, 0.3),
        'dev20_pct': (10, 30),
        'dev20_ratio': (0.05, 0.3),
        'dev25_pct': (15, 35),
        'dev25_ratio': (0.1, 0.5),
        'stop_loss_pct': (3, 15),
        'gap_up_limit': (0.03, 0.15),
    }
    
    param_keys = list(param_bounds.keys())
    
    def suggest_random():
        return {k: random.uniform(v[0], v[1]) for k, v in param_bounds.items()}
    
    evaluated = []
    
    for i in range(20):
        params = suggest_random()
        result = run_backtest(params)
        log_result(params, result)
        if result:
            evaluated.append(result)
        print(f"初始点 {i+1}/20")
    
    for iteration in range(n_iter - 20):
        best_result = max(evaluated, key=lambda x: x['CAGR'])
        
        new_params = {}
        for k in param_keys:
            if random.random() < 0.3:
                new_params[k] = random.uniform(param_bounds[k][0], param_bounds[k][1])
            else:
                pert = random.uniform(-0.5, 0.5)
                new_val = best_result['params'][k] + pert * (param_bounds[k][1] - param_bounds[k][0])
                new_params[k] = max(param_bounds[k][0], min(param_bounds[k][1], new_val))
        
        result = run_backtest(new_params)
        log_result(new_params, result)
        if result:
            evaluated.append(result)
        
        if (iteration + 21) % 30 == 0:
            print(f"进度: {iteration+21}/{n_iter}")
    
    evaluated.sort(key=lambda x: x['CAGR'], reverse=True)
    
    print("\n贝叶斯优化 Top 20 结果:")
    print("-"*80)
    for i, r in enumerate(evaluated[:20]):
        p = r['params']
        print(f"{i+1}. 倍数:{r['final_multiple']:.2f} CAGR:{r['CAGR']*100:.2f}% 回撤:{r['max_dd']*100:.2f}% 夏普:{r['sharpe']:.2f} 交易:{r['trade_count']} 胜率:{r['success_rate']:.1f}%")
        print(f"   参数: score={p['min_score']}, pos={p['max_positions']}, j_tp={p['j_take_profit_pct']}, gain_tp={p['gain_take_profit_pct']}, stop={p['stop_loss_pct']}%")
    
    return evaluated[:20]


def run_train_test_optimization():
    print("\n" + "="*60)
    print("训练/测试分离优化")
    print("="*60)
    
    train_ratio = 0.7
    train_end_idx = int(len(all_dates) * train_ratio)
    test_dates = all_dates[train_end_idx:]
    
    print(f"训练集: {all_dates[0]} 到 {all_dates[train_end_idx-1]}")
    print(f"测试集: {all_dates[train_end_idx]} 到 {all_dates[-1]}")
    
    results = []
    
    best_train_results = []
    
    for min_score in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for max_pos in [4, 6, 8, 10, 12]:
            for gain_tp in [5, 8, 10, 12]:
                for stop_loss in [5, 7, 10]:
                    params = {
                        'min_score': min_score,
                        'max_positions': max_pos,
                        'j_take_profit_pct': 100,
                        'j_take_profit_ratio': 0.2,
                        'gain_take_profit_pct': gain_tp,
                        'gain_take_profit_ratio': 0.2,
                        'dev15_pct': 15,
                        'dev15_ratio': 0.2,
                        'dev20_pct': 20,
                        'dev20_ratio': 0.2,
                        'dev25_pct': 25,
                        'dev25_ratio': 0.2,
                        'stop_loss_pct': stop_loss,
                        'gap_up_limit': 0.09,
                    }
                    
                    result = run_backtest(params, use_train_test=True, train_ratio=train_ratio)
                    log_result(params, result)
                    if result:
                        results.append(result)
    
    results.sort(key=lambda x: x['CAGR'], reverse=True)
    
    print("\n训练/测试分离 Top 10 结果:")
    print("-"*80)
    for i, r in enumerate(results[:10]):
        p = r['params']
        print(f"{i+1}. 倍数:{r['final_multiple']:.2f} CAGR:{r['CAGR']*100:.2f}% 回撤:{r['max_dd']*100:.2f}% 夏普:{r['sharpe']:.2f} 交易:{r['trade_count']} 胜率:{r['success_rate']:.1f}%")
        print(f"   参数: score={p['min_score']}, pos={p['max_positions']}, gain_tp={p['gain_take_profit_pct']}, stop={p['stop_loss_pct']}%")
    
    return results[:10]


def save_results(grid_results, random_results, bayesian_results, train_test_results):
    output_dir = "/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer"
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    all_results = {
        'grid_search': grid_results,
        'random_search': random_results,
        'bayesian_optimization': bayesian_results,
        'train_test_split': train_test_results,
        'timestamp': timestamp
    }
    
    with open(f"{output_dir}/optimization_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    json_output = {}
    for k, v in all_results.items():
        if k != 'timestamp':
            json_output[k] = []
            for i in range(min(20, len(v))):
                json_output[k].append({
                    'params': v[i]['params'],
                    'final_multiple': v[i]['final_multiple'],
                    'CAGR': v[i]['CAGR'],
                    'max_dd': v[i]['max_dd'],
                    'sharpe': v[i]['sharpe'],
                    'trade_count': v[i]['trade_count'],
                    'success_rate': v[i]['success_rate']
                })
    
    with open(f"{output_dir}/optimization_results.json", 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n结果已保存到 {output_dir}/")


if __name__ == "__main__":
    start_time = time.time()
    
    grid_results = grid_search()
    
    random_results = random_search(n_iter=100)
    
    bayesian_results = bayesian_optimization(n_iter=50)
    
    train_test_results = run_train_test_optimization()
    
    save_results(grid_results, random_results, bayesian_results, train_test_results)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
