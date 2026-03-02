"""
B1策略完整参数优化系统 V7 - 简化多线程版
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import itertools
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

CACHE_FILE = "/tmp/b1_full_optimization_cache_v2.pkl"
OUTPUT_FILE = "/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer/top100.txt"

print("="*60)
print("B1策略完整参数优化系统 V7")
print("="*60)

NUM_WORKERS = 8
print(f"使用线程数: {NUM_WORKERS}")

# ======================== 参数 ========================

J_BUY_OPTIONS = ['J<13', 'J<10%', 'J<5%']

FACTOR_CONFIGS = {
    'MACD_DIF>0': [0.5, 1],
    'RSI_14_28_57': [1, 2],
    '涨幅缩量': [1, 2, 3],
    '前一日阴线回踩趋势线': [1, 2, 3],
    '多空线_趋势线区间': [1, 2, 3],
    '跌幅扩大缩量': [1, 2, 3],
    '连续倍量柱_阳线': [1, 2],
    '阳量后接小阴量': [1, 2],
    '收盘<多空线': [-2, -1, 1],
}

# 因子名称映射: 代码中使用的名称 -> 数据中的实际名称
FACTOR_NAME_MAPPING = {
    'RSI_14_28_57': 'RSI14>28>57',
    '涨幅缩量': '涨幅-3.5~2%且缩量',
}

TAKE_PROFIT_1_OPTIONS = ['J>100', 'J>90%', 'J>95%']
TAKE_PROFIT_2_OPTIONS = ['涨>7%', '涨>8%', '涨>9%']
STOP_LOSS_1_OPTIONS = ['最低价×0.93', '最低价×0.95', '最低价×0.97', '最低价×0.9']
STOP_LOSS_RATIO_OPTIONS = ['全仓', '半仓']
FAIL_PAUSE_OPTIONS = [(3, 2), (5, 3)]

# ======================== 加载数据 ========================

print("\n加载数据...")
with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

stock_data = cache['stock_data']
all_dates = cache['all_dates']
daily_scores = cache['daily_scores']
stock_j_percentile = cache.get('stock_j_percentile', {})

print(f"股票数: {len(stock_data)}, 交易日: {len(all_dates)}")

date_to_idx = {d: i for i, d in enumerate(all_dates)}

# ======================== 预计算因子矩阵 ========================

print("\n预计算因子分数矩阵...")

# 获取所有在daily_scores中出现的因子
all_factors_in_data = set()
for date in daily_scores:
    for stock, _, details in daily_scores[date]:
        if details:
            all_factors_in_data.update(details.keys())
print(f"数据中的因子: {all_factors_in_data}")

# 初始化因子矩阵
factor_matrices = {}
for factor_name in FACTOR_CONFIGS.keys():
    factor_matrices[factor_name] = {}

for date in tqdm(all_dates, desc="预计算"):
    if date not in daily_scores:
        continue
    for stock, _, details in daily_scores[date]:
        if not details:
            continue
        for factor_name in FACTOR_CONFIGS.keys():
            # 使用映射获取实际因子名称
            actual_name = FACTOR_NAME_MAPPING.get(factor_name, factor_name)
            if actual_name in details and details[actual_name]:
                if stock not in factor_matrices[factor_name]:
                    factor_matrices[factor_name][stock] = set()
                factor_matrices[factor_name][stock].add(date)

print("预计算完成!")

# ======================== 回测函数 ========================

def run_backtest(params):
    max_positions = 4
    min_score = 0.5
    initial_capital = 1_000_000
    commission = 0.0005
    stamp = 0.001
    
    cash = float(initial_capital)
    positions = []
    equity_curve = []
    
    trade_count = 0
    win_count = 0
    pause_trading_days = 0
    current_consecutive_losses = 0
    
    fail_pause_X = params['fail_pause_X']
    fail_pause_Y = params['fail_pause_Y']
    factor_weights = params['factor_weights']
    j_buy = params['j_buy']
    tp1 = params['take_profit_1']
    tp2 = params['take_profit_2']
    sl1 = params['stop_loss_1']
    sl_ratio = params['stop_loss_ratio']
    
    for current_date in all_dates:
        if pause_trading_days > 0:
            pause_trading_days -= 1
        
        new_positions = []
        
        for pos in positions:
            stock = pos["stock"]
            df = stock_data.get(stock)
            if df is None or current_date not in df.index:
                new_positions.append(pos)
                continue
            
            row = df.loc[current_date]
            if pd.isna(row["CLOSE"]) or row["CLOSE"] <= 0:
                new_positions.append(pos)
                continue
            
            entry_price = pos["entry_price"]
            current_idx = date_to_idx[current_date]
            holding_days = current_idx - pos["entry_idx"]
            
            if holding_days < 1:
                new_positions.append(pos)
                continue
            
            close = row["CLOSE"]
            high = row["HIGH"]
            low = row["LOW"]
            trend_line = row.get('知行短期趋势线', close)
            
            exit_flag = False
            exit_price = close
            
            if low <= pos["stop_price"]:
                exit_flag = True
                exit_price = pos["stop_price"]
            
            ratio = 0.2 if sl_ratio == '全仓' else (0.1 if sl_ratio == '半仓' else 0.05)
            
            if not pos.get('tp1_done', False) and tp1 == 'J>100':
                if row.get('J', 0) > 100:
                    pos['partial_sells'].append(ratio)
                    pos['tp1_done'] = True
            
            if not pos.get('tp2_done', False):
                change_pct = (close - entry_price) / entry_price * 100
                tp2_threshold = 7 if tp2 == '涨>7%' else (8 if tp2 == '涨>8%' else 9)
                if change_pct > tp2_threshold:
                    pos['partial_sells'].append(ratio)
                    pos['tp2_done'] = True
            
            if not pd.isna(trend_line):
                deviation = (high - trend_line) / trend_line * 100
                for tp_key, tp_thresh in [('tp15_done', 15), ('tp20_done', 20), ('tp25_done', 25)]:
                    if not pos.get(tp_key, False) and deviation >= tp_thresh:
                        pos['partial_sells'].append(ratio)
                        pos[tp_key] = True
            
            total_sold = sum(pos.get('partial_sells', []))
            if total_sold >= 1.0 and holding_days >= 2:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0 and close < df.iloc[stock_idx - 1]['LOW']:
                    exit_flag = True
                    exit_price = close
            
            if holding_days >= 1 and not exit_flag:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0:
                    prev_dk = df.iloc[stock_idx - 1].get('知行多空线', close)
                    prev_trend = df.iloc[stock_idx - 1].get('知行短期趋势线', close)
                    dk = row.get('知行多空线', close)
                    if not pd.isna(prev_dk) and prev_dk <= prev_trend and dk > trend_line:
                        exit_flag = True
                        exit_price = row["OPEN"]
            
            if exit_flag:
                gross = (exit_price - entry_price) / entry_price
                net = gross - commission * 2 - stamp
                cash += entry_price * (1 + net)
                trade_count += 1
                if gross > 0:
                    win_count += 1
                    current_consecutive_losses = 0
                else:
                    current_consecutive_losses += 1
                    if current_consecutive_losses >= fail_pause_X:
                        pause_trading_days = fail_pause_Y
            else:
                pos["current_price"] = close
                pos["holding_days"] = holding_days
                new_positions.append(pos)
        
        positions = new_positions
        
        total_equity = cash + sum(pos["entry_price"] * (pos["current_price"] / pos["entry_price"]) for pos in positions)
        
        if current_date not in daily_scores or total_equity <= 0:
            equity_curve.append(total_equity)
            continue
        
        available_slots = max_positions - len(positions)
        if available_slots > 0 and pause_trading_days == 0:
            candidates = []
            for stock, _, _ in daily_scores[current_date]:
                score = 0.0
                for fname, fw in factor_weights.items():
                    if fw != 0:
                        actual_name = FACTOR_NAME_MAPPING.get(fname, fname)
                        if stock in factor_matrices.get(fname, {}):
                            if current_date in factor_matrices[fname][stock]:
                                score += fw
                if score >= min_score:
                    candidates.append((stock, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            existing = {p["stock"] for p in positions}
            available = [s for s in candidates if s[0] not in existing]
            
            if available and cash > 0:
                num_to_buy = min(len(available), int(available_slots))
                per_position = cash / num_to_buy
                
                for i in range(num_to_buy):
                    if cash < per_position * 0.5:
                        break
                    stock = available[i][0]
                    df = stock_data[stock]
                    idx = df.index.get_loc(current_date)
                    if idx + 1 >= len(df):
                        continue
                    row = df.iloc[idx]
                    
                    j_val = row.get('J')
                    if pd.isna(j_val):
                        continue
                    if j_buy != 'J<13':
                        pct_data = stock_j_percentile.get(stock, {}).get(current_date)
                        if j_buy == 'J<10%':
                            if not pct_data or 'p10' not in pct_data or j_val >= pct_data['p10']:
                                continue
                        elif j_buy == 'J<5%':
                            if not pct_data or 'p5' not in pct_data or j_val >= pct_data['p5']:
                                continue
                    elif j_val >= 13:
                        continue
                    
                    entry_p = df.iloc[idx + 1]["OPEN"]
                    if entry_p <= 0:
                        continue
                    gap_up = (entry_p - df.iloc[idx]["CLOSE"]) / df.iloc[idx]["CLOSE"]
                    if gap_up >= 0.09:
                        continue
                    
                    cash -= per_position
                    entry_low = df.iloc[idx]["LOW"]
                    sl_map = {'最低价×0.93': 0.93, '最低价×0.95': 0.95, '最低价×0.97': 0.97, '最低价×0.9': 0.9}
                    positions.append({
                        "stock": stock, "entry_price": entry_p, "entry_idx": idx + 1,
                        "current_price": entry_p, "stop_price": entry_low * sl_map.get(sl1, 0.93),
                        "partial_sells": [], "tp1_done": False, "tp2_done": False,
                        "tp15_done": False, "tp20_done": False, "tp25_done": False,
                    })
        
        equity_curve.append(total_equity)
    
    if len(equity_curve) < 10:
        return None
    
    equity_curve = np.array(equity_curve)
    final_multiple = equity_curve[-1] / initial_capital
    total_years = (all_dates[-1] - all_dates[0]).days / 365.25
    CAGR = final_multiple ** (1 / total_years) - 1 if total_years > 0 else 0
    running_max = np.maximum.accumulate(equity_curve)
    max_dd = np.min((equity_curve - running_max) / running_max)
    success_rate = win_count / trade_count * 100 if trade_count > 0 else 0
    
    return {
        'CAGR': CAGR, 'final_multiple': final_multiple, 'max_dd': max_dd,
        'trade_count': trade_count, 'success_rate': success_rate, 'params': params
    }

def main():
    print(f"\n开始优化搜索...")
    
    results = []
    start_time = time.time()
    
    factor_keys = list(FACTOR_CONFIGS.keys())
    factor_values = list(FACTOR_CONFIGS.values())
    
    # 用生成器逐个提交任务
    combo_gen = itertools.product(*factor_values)
    params_list = []
    target_count = 400000  # 约37万组合
    
    for factor_combo in tqdm(combo_gen, total=8640, desc="生成因子组合"):
        factor_weights = dict(zip(factor_keys, factor_combo))
        for j_buy in J_BUY_OPTIONS:
            for tp1 in TAKE_PROFIT_1_OPTIONS:
                for tp2 in TAKE_PROFIT_2_OPTIONS:
                    for sl1 in STOP_LOSS_1_OPTIONS:
                        for sl_ratio in STOP_LOSS_RATIO_OPTIONS:
                            for fp in FAIL_PAUSE_OPTIONS:
                                params_list.append({
                                    'factor_weights': factor_weights.copy(), 'j_buy': j_buy,
                                    'take_profit_1': tp1, 'take_profit_2': tp2,
                                    'stop_loss_1': sl1, 'stop_loss_ratio': sl_ratio,
                                    'fail_pause_X': fp[0], 'fail_pause_Y': fp[1],
                                })
                                if len(params_list) >= target_count:
                                    break
                            if len(params_list) >= target_count:
                                break
                        if len(params_list) >= target_count:
                            break
                    if len(params_list) >= target_count:
                        break
                if len(params_list) >= target_count:
                    break
            if len(params_list) >= target_count:
                break
        if len(params_list) >= target_count:
            break
    
    print(f"生成了 {len(params_list):,} 个任务")
    
    results = []
    start_time = time.time()
    
    # 使用多线程
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_backtest, params): params for params in tqdm(params_list, desc="回测")}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="完成"):
            r = future.result()
            if r and r['trade_count'] >= 10:
                results.append(r)
            
            elapsed = time.time() - start_time
            done = len(results)
            if done > 0 and done % 100 == 0:
                speed = done / elapsed
                best = max([x['CAGR'] for x in results if x['CAGR'] is not None and not np.isnan(x['CAGR'])], default=0) * 100
                print(f"\n完成: {done}, 速度: {speed:.1f}/s, 最佳CAGR: {best:.2f}%")
    
    print(f"\n完成! 总有效结果: {len(results):,}")
    
    results = [r for r in results if r['CAGR'] is not None and not np.isnan(r['CAGR'])]
    results.sort(key=lambda x: x['CAGR'], reverse=True)
    results_by_cagr = results[:100]
    results_by_success = sorted(results, key=lambda x: x['success_rate'], reverse=True)[:100]
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nB1策略参数优化结果 (V7)\n" + "="*60 + "\n\n")
        f.write("【年化收益率前100】\n" + "-"*60 + "\n")
        for i, r in enumerate(results_by_cagr):
            p = r['params']
            f.write(f"\n#{i+1} CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f} 回撤:{r['max_dd']*100:.2f}% 成功率:{r['success_rate']:.2f}%\n")
            f.write(f"J买入:{p['j_buy']} 止盈1:{p['take_profit_1']} 止盈2:{p['take_profit_2']} 止损:{p['stop_loss_1']} {p['stop_loss_ratio']} 风控:{p['fail_pause_X']}次{p['fail_pause_Y']}天\n")
            f.write("因子:" + str(p['factor_weights']) + "\n")
        
        f.write("\n\n【成功率前100】\n" + "-"*60 + "\n")
        for i, r in enumerate(results_by_success):
            p = r['params']
            f.write(f"\n#{i+1} 成功率:{r['success_rate']:.2f}% CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f}\n")
    
    print(f"\n结果已保存到: {OUTPUT_FILE}")
    
    print("\n" + "="*60)
    print("年化收益率 Top 10")
    print("="*60)
    for i, r in enumerate(results_by_cagr[:10]):
        p = r['params']
        print(f"\n#{i+1} CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f}")
        print(f"   J:{p['j_buy']} TP1:{p['take_profit_1']} SL:{p['stop_loss_1']}")

if __name__ == "__main__":
    from concurrent.futures import as_completed
    main()
