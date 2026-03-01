import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

CACHE_FILE = "/tmp/b1_cache_full.pkl"
DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

print("="*60)
print("单因子打分测试 - 科学验证每个因子的最优分数")
print("="*60)

print("\n加载缓存数据...")
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    stock_data = cache['stock_data']
    all_dates = cache['all_dates']
    daily_scores_original = cache['daily_scores']
    print(f"加载成功: {len(stock_data)} 只股票, {len(all_dates)} 个交易日")
else:
    print("错误: 缓存文件不存在!")
    sys.exit(1)

FACTOR_NAMES = [
    'MACD_DIF>0',
    'RSI14>28>57',
    '涨幅-3.5~2%且缩量',
    '跌幅扩大且缩量',
    '跳空下跌',
    '收盘<多空线',
    '前一日回踩趋势线且今日缩量阳线',
    '60日最大量非阴线',
    '前一日回踩多空线且今日缩量阳线',
    '多空线<=收盘<=趋势线',
    '存在倍量柱',
    '连续倍量柱',
    '阳量后接2倍阴量',
]

DEFAULT_FACTOR_SCORES = {
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


def rebuild_scores_with_factor_scores(factor_scores):
    daily_scores_new = {}
    for date, signals in daily_scores_original.items():
        new_signals = []
        for stock, old_score, details in signals:
            if not details:
                continue
            new_score = 0
            new_details = {}
            for factor, value in details.items():
                if factor in factor_scores:
                    weight = factor_scores[factor]
                    new_score += weight
                    new_details[factor] = weight
            if new_score > 0:
                new_signals.append((stock, new_score, new_details))
        if new_signals:
            daily_scores_new[date] = new_signals
    return daily_scores_new


def run_backtest(params, daily_scores_data):
    initial_capital = params.get('initial_capital', 1_000_000)
    max_positions = params.get('max_positions', 4)
    min_score = params.get('min_score', 0.5)
    fail_pause_X = params.get('fail_pause_X', 3)
    fail_pause_Y = params.get('fail_pause_Y', 2)
    
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
    
    cash = float(initial_capital)
    positions = []
    equity_curve = []
    stopped = False
    
    trade_count = 0
    pause_trading_days = 0
    current_consecutive_losses = 0
    
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
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
            
            stop_price = pos["stop_price"]
            if low <= stop_price:
                stop_count = pos.get('stop_count', 0)
                if stop_count == 0:
                    pos['partial_sells'].append({'price': stop_price, 'ratio': 0.5})
                    pos['stop_count'] = 1
                elif stop_count == 1:
                    pos['partial_sells'].append({'price': stop_price, 'ratio': 0.25})
                    pos['stop_count'] = 2
                else:
                    exit_flag = True
                    exit_price = stop_price
            
            if pos.get('j_take_profit_done', False) == False:
                if not pd.isna(row['J']) and row['J'] > j_take_profit_pct:
                    pos['partial_sells'].append({'price': close, 'ratio': j_take_profit_ratio})
                    pos['j_take_profit_done'] = True
            
            change_pct = (close - entry_price) / entry_price * 100
            if pos.get('gain_take_profit_done', False) == False:
                if change_pct > gain_take_profit_pct:
                    pos['partial_sells'].append({'price': close, 'ratio': gain_take_profit_ratio})
                    pos['gain_take_profit_done'] = True
            
            if not pd.isna(trend_line):
                deviation = (high - trend_line) / trend_line * 100
                if pos.get('dev15_done', False) == False and deviation >= dev15_pct:
                    pos['partial_sells'].append({'price': close, 'ratio': dev15_ratio})
                    pos['dev15_done'] = True
                if pos.get('dev20_done', False) == False and deviation >= dev20_pct:
                    pos['partial_sells'].append({'price': close, 'ratio': dev20_ratio})
                    pos['dev20_done'] = True
                if pos.get('dev25_done', False) == False and deviation >= dev25_pct:
                    pos['partial_sells'].append({'price': close, 'ratio': dev25_ratio})
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
                        exit_flag = True
                        exit_price = close
            
            if not exit_flag and holding_days >= 1:
                stock_idx = df.index.get_loc(current_date)
                if stock_idx > 0:
                    prev_dk = df.iloc[stock_idx - 1]['知行多空线']
                    prev_trend = df.iloc[stock_idx - 1]['知行短期趋势线']
                    if not pd.isna(prev_dk) and not pd.isna(prev_trend):
                        if prev_dk <= prev_trend and dk_line > trend_line:
                            exit_flag = True
                            exit_price = open_p
            
            if not exit_flag and holding_days >= 60:
                stock_idx = df.index.get_loc(current_date)
                rolling_vol = df.iloc[max(0, stock_idx-60):stock_idx]['VOLUME']
                if len(rolling_vol) > 0:
                    max_vol_idx = rolling_vol.idxmax()
                    if max_vol_idx in df.index:
                        max_vol_row = df.loc[max_vol_idx]
                        if max_vol_row['是否阴线']:
                            vol_exit_count = pos.get('vol_exit_count', 0)
                            if vol_exit_count >= 2:
                                exit_flag = True
                                exit_price = open_p
            
            if exit_flag:
                gross = (exit_price - entry_price) / entry_price
                net = gross - commission * 2 - stamp
                
                for ps in partial_sells:
                    cash += pos["invested"] * pos.get('hold_ratio', 1.0) * ps['ratio'] * (1 + (ps['price'] - entry_price) / entry_price - commission - stamp)
                
                remaining_ratio = 1.0 - sum([p['ratio'] for p in partial_sells])
                cash += pos["invested"] * pos.get('hold_ratio', 1.0) * remaining_ratio * (1 + net)
                
                trade_count += 1
                if gross > 0:
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
        
        if current_date not in daily_scores_data:
            continue
        
        available_slots = max_positions - len(positions)
        if available_slots <= 0:
            continue
        
        if pause_trading_days > 0:
            continue
        
        scores_today = [(s[0], s[1], s[2]) for s in daily_scores_data[current_date] if s[1] >= min_score]
        scores_today.sort(key=lambda x: x[1], reverse=True)
        
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
            })
    
    if len(equity_curve) == 0:
        return None
    
    equity_curve = np.array(equity_curve)
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
    
    return {
        'final_multiple': final_multiple,
        'CAGR': CAGR,
        'max_dd': max_dd,
        'trade_count': trade_count,
    }


def test_single_factor(factor_name, score_values):
    print(f"\n{'='*50}")
    print(f"测试因子: {factor_name}")
    print(f"默认分数: {DEFAULT_FACTOR_SCORES[factor_name]}")
    print(f"{'='*50}")
    
    base_scores = DEFAULT_FACTOR_SCORES.copy()
    results = []
    
    for score in score_values:
        test_scores = base_scores.copy()
        test_scores[factor_name] = score
        
        params = {
            'initial_capital': 1_000_000,
            'max_positions': 4,
            'min_score': 0.5,
            'fail_pause_X': 3,
            'fail_pause_Y': 2,
            'j_take_profit_pct': 100,
            'j_take_profit_ratio': 0.2,
            'gain_take_profit_pct': 8,
            'gain_take_profit_ratio': 0.2,
            'dev15_pct': 15,
            'dev15_ratio': 0.2,
            'dev20_pct': 20,
            'dev20_ratio': 0.2,
            'dev25_pct': 25,
            'dev25_ratio': 0.2,
            'stop_loss_pct': 7,
            'gap_up_limit': 0.09,
        }
        
        daily_scores_data = rebuild_scores_with_factor_scores(test_scores)
        result = run_backtest(params, daily_scores_data)
        
        if result:
            results.append({
                'score': score,
                'CAGR': result['CAGR'] * 100,
                'multiple': result['final_multiple'],
                'max_dd': result['max_dd'] * 100,
                'trade_count': result['trade_count'],
            })
            print(f"  分数={score:+.1f}: CAGR={result['CAGR']*100:.2f}%, 倍数={result['final_multiple']:.2f}, 回撤={result['max_dd']*100:.2f}%")
    
    if not results:
        return None
    
    best = max(results, key=lambda x: x['CAGR'])
    print(f"\n  >>> 最佳分数: {best['score']:+0.1f}, CAGR: {best['CAGR']:.2f}%")
    
    return {
        'factor': factor_name,
        'default_score': DEFAULT_FACTOR_SCORES[factor_name],
        'results': results,
        'best_score': best['score'],
        'best_cagr': best['CAGR'],
    }


def main():
    score_values = [-3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    
    all_results = []
    
    for factor_name in FACTOR_NAMES:
        result = test_single_factor(factor_name, score_values)
        if result:
            all_results.append(result)
        time.sleep(0.5)
    
    print("\n" + "="*60)
    print("单因子测试汇总")
    print("="*60)
    
    print(f"\n{'因子':<35} {'默认分':>6} {'最佳分':>6} {'默认CAGR':>10} {'最佳CAGR':>10} {'提升':>8}")
    print("-"*80)
    
    for r in all_results:
        default_cagr = None
        for res in r['results']:
            if res['score'] == r['default_score']:
                default_cagr = res['CAGR']
                break
        
        improvement = ""
        if default_cagr is not None:
            diff = r['best_cagr'] - default_cagr
            if diff > 0:
                improvement = f"+{diff:.2f}%"
            else:
                improvement = f"{diff:.2f}%"
        
        print(f"{r['factor']:<35} {r['default_score']:>+6.1f} {r['best_score']:>+6.1f} {default_cagr if default_cagr else 'N/A':>10} {r['best_cagr']:>10.2f}% {improvement:>8}")
    
    default_scores = {}
    best_scores = {}
    for r in all_results:
        default_scores[r['factor']] = r['default_score']
        best_scores[r['factor']] = r['best_score']
    
    print("\n" + "="*60)
    print("推荐的最优打分 (基于单因子测试)")
    print("="*60)
    for factor, score in best_scores.items():
        print(f"{factor}: {score:+.1f}")
    
    output = {
        'single_factor_results': all_results,
        'recommended_scores': best_scores,
        'default_scores': default_scores,
    }
    
    with open('/Users/lidongyang/Desktop/Qstrategy/utils/parameter_optimizer/single_factor_test.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到 single_factor_test.json")


if __name__ == "__main__":
    main()
