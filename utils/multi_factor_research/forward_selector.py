import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

COST_RATE = 0.0003
SLIPPAGE = 0.001

def is_bullish_increase_volume(row, prev_row):
    if prev_row is None or pd.isna(prev_row.get('成交量')):
        return False
    is_bullish = row['收盘'] > row['开盘']
    vol_increase = row['成交量'] > prev_row['成交量'] * 1.5
    return is_bullish and vol_increase

def is_bearish_increase_volume(row, prev_row):
    if prev_row is None or pd.isna(prev_row.get('成交量')):
        return False
    is_bearish = row['收盘'] < row['开盘']
    vol_increase = row['成交量'] > prev_row['成交量'] * 1.3
    return is_bearish and vol_increase

def calculate_portfolio_returns(df: pd.DataFrame, factor_list: List[str], 
                               weights: Dict[str, float], top_k: int = 10) -> pd.Series:
    dates = sorted(df['日期'].unique())
    portfolio_returns = []
    
    for i, date in enumerate(dates[:-1]):
        day_data = df[df['日期'] == date].copy()
        
        if len(day_data) == 0:
            continue
        
        score = np.zeros(len(day_data))
        for factor in factor_list:
            if factor in day_data.columns:
                score += day_data[factor].fillna(0).values * weights.get(factor, 0)
        
        day_data['score'] = score
        top_stocks = day_data.nlargest(top_k, 'score')
        
        next_day_data = df[df['日期'] == dates[i + 1]]
        
        if len(top_stocks) > 0 and len(next_day_data) > 0:
            merged = top_stocks.merge(next_day_data[['stock', '收盘', '开盘', '最高', '最低', '成交量', 'pct_change']], 
                                    on='stock', suffixes=('', '_next'))
            
            if len(merged) > 0:
                ret = (merged['收盘_next'] / merged['收盘'] - 1) - COST_RATE - SLIPPAGE
                portfolio_returns.append(ret.mean())
    
    return pd.Series(portfolio_returns)

def backtest_factors(df: pd.DataFrame, factor_list: List[str], 
                    weights: Dict[str, float] = None, top_k: int = 10) -> Dict:
    if weights is None:
        weights = {f: 1.0 / len(factor_list) for f in factor_list}
    
    dates = sorted(df['日期'].unique())
    daily_returns = []
    trades = []
    
    position = None
    stop_loss_count = {}
    
    for i, date in enumerate(tqdm(dates[:-1], desc="回测")):
        day_data = df[df['日期'] == date].copy()
        
        if len(day_data) == 0:
            continue
        
        score = np.zeros(len(day_data))
        for factor in factor_list:
            if factor in day_data.columns:
                score += day_data[factor].fillna(0).values * weights.get(factor, 0)
        
        day_data['score'] = score
        top_stocks = set(day_data.nlargest(top_k, 'score')['stock'].tolist())
        
        next_day_data = df[df['日期'] == dates[i + 1]]
        if len(next_day_data) == 0:
            continue
            
        if position:
            stock = position['stock']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            
            stock_next = next_day_data[next_day_data['stock'] == stock]
            if len(stock_next) == 0:
                continue
            
            row = stock_next.iloc[0]
            current_price = row['收盘']
            low_price = row['最低']
            
            should_sell = False
            sell_reason = ""
            
            if position.get('stop_loss') and low_price <= position['stop_price']:
                should_sell = True
                sell_reason = "止损"
            elif stock not in top_stocks:
                should_sell = True
                sell_reason = "不在Top10"
            else:
                prev_row = day_data[day_data['stock'] == stock]
                if len(prev_row) > 0:
                    if is_bearish_increase_volume(row, prev_row.iloc[0]):
                        should_sell = True
                        sell_reason = "放量阴线"
            
            if should_sell:
                ret = (current_price - entry_price) / entry_price - COST_RATE - SLIPPAGE
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'stock': stock,
                    'return': ret,
                    'reason': sell_reason
                })
                
                stop_loss_count[stock] = stop_loss_count.get(stock, 0) + (1 if sell_reason == "止损" else 0)
                position = None
            else:
                position['holding_days'] = position.get('holding_days', 0) + 1
        
        if not position:
            candidates = day_data[day_data['stock'].isin(top_stocks)]
            
            for _, row in candidates.iterrows():
                stock = row['stock']
                
                if stop_loss_count.get(stock, 0) >= 3:
                    continue
                
                stock_next = next_day_data[next_day_data['stock'] == stock]
                if len(stock_next) == 0:
                    continue
                
                entry_price = stock_next.iloc[0]['开盘']
                
                if entry_price <= 0:
                    continue
                
                position = {
                    'stock': stock,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'stop_loss': True,
                    'stop_price': row.get('最低', entry_price * 0.97),
                    'holding_days': 0
                }
                break
    
    if len(daily_returns) > 0:
        daily_ret = pd.Series(daily_returns)
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        annual_return = daily_ret.mean() * 252
    else:
        sharpe = 0
        annual_return = 0
    
    win_trades = [t['return'] for t in trades if t['return'] > 0]
    loss_trades = [t['return'] for t in trades if t['return'] <= 0]
    
    result = {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'total_trades': len(trades),
        'win_rate': len(win_trades) / len(trades) if len(trades) > 0 else 0,
        'avg_return': np.mean([t['return'] for t in trades]) if len(trades) > 0 else 0,
        'trades': trades
    }
    
    return result

def forward_selection(df: pd.DataFrame, factor_list: List[str], 
                     top_k: int = 10, improvement_threshold: float = 0.05) -> Tuple[List[str], Dict]:
    print(f"\n{'='*60}")
    print("阶段三：Forward Selection")
    print(f"{'='*60}")
    
    selected_factors = []
    remaining_factors = factor_list.copy()
    
    best_sharpe = -999
    
    print("\n1. 单因子回测...")
    single_results = {}
    for factor in remaining_factors:
        result = backtest_factors(df, [factor], top_k=top_k)
        single_results[factor] = result['sharpe']
        print(f"  {factor}: Sharpe = {result['sharpe']:.4f}")
    
    best_factor = max(single_results, key=single_results.get)
    selected_factors.append(best_factor)
    remaining_factors.remove(best_factor)
    best_sharpe = single_results[best_factor]
    
    print(f"\n选择第一个因子: {best_factor}, Sharpe = {best_sharpe:.4f}")
    
    iteration = 2
    while remaining_factors:
        print(f"\n{iteration}. 尝试添加因子...")
        
        best_improvement = 0
        best_new_factor = None
        best_new_sharpe = best_sharpe
        
        for factor in remaining_factors:
            test_factors = selected_factors + [factor]
            weights = {f: 1.0 / len(test_factors) for f in test_factors}
            result = backtest_factors(df, test_factors, weights, top_k=top_k)
            
            improvement = (result['sharpe'] - best_sharpe) / abs(best_sharpe) if best_sharpe != 0 else 0
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_new_factor = factor
                best_new_sharpe = result['sharpe']
        
        if best_improvement >= improvement_threshold:
            selected_factors.append(best_new_factor)
            remaining_factors.remove(best_new_factor)
            improvement_pct = best_improvement * 100
            print(f"  + 添加 {best_new_factor}, Sharpe: {best_sharpe:.4f} -> {best_new_sharpe:.4f} (+{improvement_pct:.1f}%)")
            best_sharpe = best_new_sharpe
            iteration += 1
        else:
            print(f"  无因子能提升 Sharpe，停止选择")
            break
    
    print(f"\n最终选择的因子: {selected_factors}")
    print(f"最终 Sharpe: {best_sharpe:.4f}")
    
    return selected_factors, {'best_sharpe': best_sharpe, 'selected': selected_factors}

if __name__ == "__main__":
    from data_processor import prepare_data
    from factor_calculator import calculate_all_factors, get_factor_list
    
    df = prepare_data()
    df = calculate_all_factors(df)
    factor_list = get_factor_list()
    
    selected, _ = forward_selection(df, factor_list[:5])
    print(f"\nForward Selection 结果: {selected}")
