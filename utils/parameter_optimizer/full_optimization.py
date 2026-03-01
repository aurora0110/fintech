"""
B1策略完整参数优化系统
- 3种J值买入条件
- 11个因子打分（含新增因子）
- 止盈止损参数搜索
- 风控策略
- 年化收益率前100 + 成功率前100
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import json
import random
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CACHE_FILE = "/tmp/b1_full_optimization_cache.pkl"
OUTPUT_FILE = "/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer/top100.txt"
LOG_FILE = "/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer/full_optimization_log.txt"

print("="*60)
print("B1策略完整参数优化系统")
print("="*60)

# ======================== 参数定义 ========================

J_BUY_OPTIONS = ['J<13', 'J<10%', 'J<5%']

FACTOR_CONFIGS = {
    'MACD_DIF>0': [0.5, 1, 1.5],
    'RSI_14_28_57': [1, 1.5, 2],
    '涨幅缩量': [1, 2, 3],
    '前一日阴线回踩趋势线': [1, 2, 3],
    '倍量柱': [0.5, 1, 1.5],
    '多空线_趋势线区间': [1, 1.5, 2, 2.5],
    '跌幅扩大缩量': [0.5, 1, 1.5, 2],
    '连续倍量柱_阳线': [0.5, 1, 1.5, 2],
    '阳量后接小阴量': [0.5, 1, 1.5, 2],
    '收盘<多空线': [-2, -1, 1],
}

TAKE_PROFIT_1_OPTIONS = ['J>100', 'J>90%', 'J>95%']
TAKE_PROFIT_2_OPTIONS = ['涨>7%', '涨>8%', '涨>9%']

STOP_LOSS_1_OPTIONS = ['最低价×0.93', '最低价×0.95', '最低价×0.97', '最低价×0.9']
STOP_LOSS_RATIO_OPTIONS = ['全仓', '半仓', '1/4仓']

FAIL_PAUSE_OPTIONS = [(3, 2), (5, 3)]

def get_all_combinations():
    """生成所有参数组合"""
    combinations = []
    
    # 因子分数组合
    factor_keys = list(FACTOR_CONFIGS.keys())
    factor_values = list(FACTOR_CONFIGS.values())
    
    import itertools
    for factor_combo in itertools.product(*factor_values):
        factor_scores = dict(zip(factor_keys, factor_combo))
        
        # J买入条件
        for j_buy in J_BUY_OPTIONS:
            # 止盈策略
            for tp1 in TAKE_PROFIT_1_OPTIONS:
                for tp2 in TAKE_PROFIT_2_OPTIONS:
                    # 止损策略
                    for sl1 in STOP_LOSS_1_OPTIONS:
                        for sl_ratio in STOP_LOSS_RATIO_OPTIONS:
                            # 风控策略
                            for fp in FAIL_PAUSE_OPTIONS:
                                combo = {
                                    'factor_scores': factor_scores.copy(),
                                    'j_buy': j_buy,
                                    'take_profit_1': tp1,
                                    'take_profit_2': tp2,
                                    'stop_loss_1': sl1,
                                    'stop_loss_ratio': sl_ratio,
                                    'fail_pause_X': fp[0],
                                    'fail_pause_Y': fp[1],
                                }
                                combinations.append(combo)
    
    return combinations

print(f"\n总组合数: {len(get_all_combinations()):,}")

print("\n加载数据...")
if os.path.exists(CACHE_FILE):
    print("从缓存加载...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    stock_data = cache['stock_data']
    all_dates = cache['all_dates']
    daily_scores = cache['daily_scores']
    j_percentile_index = cache.get('j_percentile_index', {})
    print(f"加载成功!")
else:
    print("需要先生成缓存数据，运行 build_optimization_cache.py")
    sys.exit(1)

print(f"股票数: {len(stock_data)}, 交易日: {len(all_dates)}")

def rebuild_scores(params, daily_scores_original):
    """根据参数重新计算每日评分"""
    new_scores = {}
    for date, signals in daily_scores_original.items():
        new_signals = []
        for stock, old_score, details in signals:
            if not details:
                continue
            
            new_score = 0
            for factor, value in details.items():
                if factor in params['factor_scores']:
                    weight = params['factor_scores'][factor]
                    new_score += weight
            
            if new_score > 0:
                new_signals.append((stock, new_score, details))
        
        if new_signals:
            new_scores[date] = new_signals
    
    return new_scores

def check_j_buy(row, j_buy_option, j_history=None, date=None):
    """检查J值买入条件"""
    if pd.isna(row.get('J')):
        return False
    
    j = row['J']
    
    if j_buy_option == 'J<13':
        return j < 13
    elif j_buy_option == 'J<10%':
        if j_history is not None and date in j_history:
            threshold = j_history[date].get('p10')
            if threshold is not None:
                return j < threshold
        return j < 13
    elif j_buy_option == 'J<5%':
        if j_history is not None and date in j_history:
            threshold = j_history[date].get('p5')
            if threshold is not None:
                return j < threshold
        return j < 13
    
    return j < 13

def check_take_profit_1(row, option, entry_j):
    """检查止盈策略第一步"""
    if option == 'J>100':
        return row.get('J', 0) > 100
    elif option == 'J>90%':
        return False
    elif option == 'J>95%':
        return False
    return False

def check_take_profit_2(row, option):
    """检查止盈策略第二步"""
    if option == '涨>7%':
        return True
    elif option == '涨>8%':
        return True
    elif option == '涨>9%':
        return True
    return False

def get_stop_loss_price(entry_low, option):
    """计算止损价格"""
    if option == '最低价×0.93':
        return entry_low * 0.93
    elif option == '最低价×0.95':
        return entry_low * 0.95
    elif option == '最低价×0.97':
        return entry_low * 0.97
    elif option == '最低价×0.9':
        return entry_low * 0.9
    return entry_low * 0.93

def run_backtest(params, daily_scores_data):
    """运行回测"""
    max_positions = 4
    min_score = 0.5
    
    initial_capital = 1_000_000
    commission = 0.0005
    stamp = 0.001
    
    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    
    trade_count = 0
    win_count = 0
    loss_count = 0
    pause_trading_days = 0
    current_consecutive_losses = 0
    
    fail_pause_X = params['fail_pause_X']
    fail_pause_Y = params['fail_pause_Y']
    
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    for current_date in all_dates:
        if stopped:
            break
        
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
            entry_idx = pos["entry_idx"]
            current_idx = date_to_idx[current_date]
            holding_days = current_idx - entry_idx
            
            if holding_days < 1:
                new_positions.append(pos)
                continue
            
            close = row["CLOSE"]
            high = row["HIGH"]
            low = row["LOW"]
            trend_line = row.get('知行短期趋势线', close)
            
            exit_flag = False
            exit_price = close
            stop_price = pos["stop_price"]
            
            # 止损检查
            if low <= stop_price:
                exit_flag = True
                exit_price = stop_price
            
            # 止盈第一步
            if not pos.get('tp1_done', False):
                if check_take_profit_1(row, params['take_profit_1'], pos.get('entry_j', 100)):
                    ratio = 0.2
                    if params['stop_loss_ratio'] == '半仓':
                        ratio = 0.1
                    elif params['stop_loss_ratio'] == '1/4仓':
                        ratio = 0.05
                    
                    if ratio > 0:
                        pos['partial_sells'].append(ratio)
                        pos['tp1_done'] = True
            
            # 止盈第二步
            if not pos.get('tp2_done', False):
                change_pct = (close - entry_price) / entry_price * 100
                if change_pct > 7:
                    ratio = 0.2
                    if params['stop_loss_ratio'] == '半仓':
                        ratio = 0.1
                    elif params['stop_loss_ratio'] == '1/4仓':
                        ratio = 0.05
                    
                    if ratio > 0:
                        pos['partial_sells'].append(ratio)
                        pos['tp2_done'] = True
            
            # 止盈第三步：偏离趋势线
            if not pd.isna(trend_line):
                deviation = (high - trend_line) / trend_line * 100
                
                if not pos.get('tp15_done', False) and deviation >= 15:
                    ratio = 0.2
                    if params['stop_loss_ratio'] == '半仓':
                        ratio = 0.1
                    elif params['stop_loss_ratio'] == '1/4仓':
                        ratio = 0.05
                    if ratio > 0:
                        pos['partial_sells'].append(ratio)
                        pos['tp15_done'] = True
                
                if not pos.get('tp20_done', False) and deviation >= 20:
                    ratio = 0.2
                    if params['stop_loss_ratio'] == '半仓':
                        ratio = 0.1
                    elif params['stop_loss_ratio'] == '1/4仓':
                        ratio = 0.05
                    if ratio > 0:
                        pos['partial_sells'].append(ratio)
                        pos['tp20_done'] = True
                
                if not pos.get('tp25_done', False) and deviation >= 25:
                    ratio = 0.2
                    if params['stop_loss_ratio'] == '半仓':
                        ratio = 0.1
                    elif params['stop_loss_ratio'] == '1/4仓':
                        ratio = 0.05
                    if ratio > 0:
                        pos['partial_sells'].append(ratio)
                        pos['tp25_done'] = True
            
            # 全部卖出后连续两天最低价平仓
            total_sold = sum(pos.get('partial_sells', []))
            if total_sold >= 1.0 and holding_days >= 2:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0:
                    prev_low = df.iloc[stock_idx - 1]['LOW']
                    if close < prev_low:
                        exit_flag = True
                        exit_price = close
            
            # 多空线金叉趋势线
            if holding_days >= 1 and not exit_flag:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0:
                    prev_dk = df.iloc[stock_idx - 1].get('知行多空线', close)
                    prev_trend = df.iloc[stock_idx - 1].get('知行短期趋势线', close)
                    dk = row.get('知行多空线', close)
                    if not pd.isna(prev_dk) and not pd.isna(prev_trend):
                        if prev_dk <= prev_trend and dk > trend_line:
                            exit_flag = True
                            exit_price = row["OPEN"]
            
            # 60日最大量阴线
            if holding_days >= 60 and not exit_flag:
                stock_idx = df.index.get_loc(current_date)
                rolling_vol = df.iloc[max(0, stock_idx-60):stock_idx]['VOLUME']
                if len(rolling_vol) > 0:
                    max_idx = rolling_vol.idxmax()
                    if max_idx in df.index and df.loc[max_idx].get('是否阴线', False):
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
                    loss_count += 1
                    current_consecutive_losses += 1
                    if current_consecutive_losses >= fail_pause_X:
                        pause_trading_days = fail_pause_Y
            else:
                pos["current_price"] = close
                pos["holding_days"] = holding_days
                new_positions.append(pos)
        
        positions = new_positions
        
        # 计算总权益
        total_equity = cash
        for pos in positions:
            price_ratio = pos["current_price"] / pos["entry_price"]
            if np.isnan(price_ratio) or price_ratio <= 0:
                price_ratio = 1.0
            total_equity += pos["entry_price"] * price_ratio
        
        if np.isnan(total_equity) or total_equity <= 0:
            stopped = True
            break
        
        equity_curve.append(total_equity)
        
        # 买入
        if current_date not in daily_scores_data:
            continue
        
        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue
        
        if pause_trading_days > 0:
            continue
        
        scores_today = [(s[0], s[1]) for s in daily_scores_data[current_date] if s[1] >= min_score]
        scores_today.sort(key=lambda x: x[1], reverse=True)
        
        existing = {p["stock"] for p in positions}
        available = [s for s in scores_today if s[0] not in existing]
        
        if not available or cash <= 0:
            continue
        
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
            
            entry_p = df.iloc[idx + 1]["OPEN"]
            entry_low = df.iloc[idx]["LOW"]
            
            if entry_p <= 0 or np.isnan(entry_p):
                continue
            
            # 跳空限制
            prev_close = df.iloc[idx]["CLOSE"]
            gap_up = (entry_p - prev_close) / prev_close
            if gap_up >= 0.09:
                continue
            
            invested = per_position
            cash -= invested
            
            positions.append({
                "stock": stock,
                "entry_price": entry_p,
                "entry_idx": idx + 1,
                "entry_low": entry_low,
                "entry_j": df.iloc[idx].get('J', 50),
                "current_price": entry_p,
                "stop_price": get_stop_loss_price(entry_low, params['stop_loss_1']),
                "partial_sells": [],
                "tp1_done": False,
                "tp2_done": False,
                "tp15_done": False,
                "tp20_done": False,
                "tp25_done": False,
            })
    
    if len(equity_curve) == 0:
        return None
    
    equity_curve = np.array(equity_curve)
    final_multiple = equity_curve[-1] / initial_capital
    
    start_date = all_dates[0]
    end_date = all_dates[-1]
    total_years = (end_date - start_date).days / 365.25
    
    CAGR = final_multiple ** (1 / total_years) - 1 if total_years > 0 and final_multiple > 0 else 0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    success_rate = win_count / trade_count * 100 if trade_count > 0 else 0
    
    return {
        'final_multiple': final_multiple,
        'CAGR': CAGR,
        'max_dd': max_dd,
        'trade_count': trade_count,
        'win_count': win_count,
        'loss_count': loss_count,
        'success_rate': success_rate,
    }

def main():
    print("\n开始优化搜索...")
    
    # 生成所有组合
    all_combos = get_all_combinations()
    print(f"总组合数: {len(all_combos):,}")
    
    # 随机打乱顺序（更快的找到好结果）
    random.shuffle(all_combos)
    
    results = []
    
    # 使用tqdm显示进度
    for i, params in enumerate(tqdm(all_combos, desc="回测中")):
        try:
            daily_scores_data = rebuild_scores(params, daily_scores)
            result = run_backtest(params, daily_scores_data)
            
            if result and result['trade_count'] >= 10:
                result['params'] = params
                results.append(result)
            
            # 每1000次保存中间结果
            if (i + 1) % 1000 == 0:
                print(f"\n已完成: {i+1:,}, 有效结果: {len(results):,}")
        
        except Exception as e:
            continue
    
    print(f"\n完成! 总有效结果: {len(results):,}")
    
    # 按CAGR排序
    results_by_cagr = sorted(results, key=lambda x: x['CAGR'], reverse=True)[:100]
    
    # 按成功率排序
    results_by_success = sorted(results, key=lambda x: x['success_rate'], reverse=True)[:100]
    
    # 保存到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("B1策略参数优化结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("【年化收益率前100】\n")
        f.write("-"*60 + "\n")
        for i, r in enumerate(results_by_cagr):
            p = r['params']
            f.write(f"\n#{i+1} CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f} 回撤:{r['max_dd']*100:.2f}% 成功率:{r['success_rate']:.2f}%\n")
            f.write(f"J买入:{p['j_buy']}\n")
            f.write(f"止盈1:{p['take_profit_1']} 止盈2:{p['take_profit_2']}\n")
            f.write(f"止损:{p['stop_loss_1']} 止损比例:{p['stop_loss_ratio']}\n")
            f.write(f"风控:失败{p['fail_pause_X']}次暂停{p['fail_pause_Y']}天\n")
            f.write("因子分数:\n")
            for k, v in p['factor_scores'].items():
                f.write(f"  {k}: {v}\n")
        
        f.write("\n\n" + "="*60 + "\n")
        f.write("【成功率前100】\n")
        f.write("-"*60 + "\n")
        for i, r in enumerate(results_by_success):
            p = r['params']
            f.write(f"\n#{i+1} 成功率:{r['success_rate']:.2f}% CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f} 回撤:{r['max_dd']*100:.2f}%\n")
            f.write(f"J买入:{p['j_buy']}\n")
            f.write(f"止盈1:{p['take_profit_1']} 止盈2:{p['take_profit_2']}\n")
            f.write(f"止损:{p['stop_loss_1']} 止损比例:{p['stop_loss_ratio']}\n")
            f.write(f"风控:失败{p['fail_pause_X']}次暂停{p['fail_pause_Y']}天\n")
            f.write("因子分数:\n")
            for k, v in p['factor_scores'].items():
                f.write(f"  {k}: {v}\n")
    
    print(f"\n结果已保存到: {OUTPUT_FILE}")
    
    # 打印Top5
    print("\n" + "="*60)
    print("年化收益率 Top 5")
    print("="*60)
    for i, r in enumerate(results_by_cagr[:5]):
        p = r['params']
        print(f"\n#{i+1} CAGR:{r['CAGR']*100:.2f}% 倍数:{r['final_multiple']:.2f}")
        print(f"   J买入:{p['j_buy']}, 止盈1:{p['take_profit_1']}, 止损:{p['stop_loss_1']}")
        print(f"   因子分数: {p['factor_scores']}")

if __name__ == "__main__":
    main()
