import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from forward_selector import backtest_factors
from weight_optimizer import optimize_weights
import warnings
warnings.filterwarnings('ignore')

def time_split_data(df: pd.DataFrame, train_ratio: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df['日期'].unique())
    split_idx = int(len(dates) * train_ratio)
    train_end_date = dates[split_idx]
    
    train_df = df[df['日期'] <= train_end_date].copy()
    test_df = df[df['日期'] > train_end_date].copy()
    
    print(f"数据切分: 训练期结束于 {train_end_date}")
    print(f"  训练期: {train_df['日期'].min()} ~ {train_df['日期'].max()}, 共 {len(train_df)} 条")
    print(f"  验证期: {test_df['日期'].min()} ~ {test_df['日期'].max()}, 共 {len(test_df)} 条")
    
    return train_df, test_df

def walkforward_validation(df: pd.DataFrame, factor_list: List[str],
                          train_ratio: float = 0.6, top_k: int = 10,
                          n_weight_samples: int = 500) -> Dict:
    print(f"\n{'='*60}")
    print("阶段五：Walk-Forward 验证")
    print(f"{'='*60}")
    
    train_df, test_df = time_split_data(df, train_ratio)
    
    print("\n--- 训练期 ---")
    train_factors = factor_list.copy()
    
    print("\n训练期因子筛选...")
    train_best_factors = train_factors
    
    print(f"\n训练期权重优化...")
    train_weights, _ = optimize_weights(train_df, train_best_factors, top_k=top_k, n_samples=n_weight_samples)
    
    print("\n--- 验证期 ---")
    test_result = backtest_factors(test_df, train_best_factors, train_weights, top_k=top_k)
    
    print(f"\n验证期结果:")
    print(f"  Sharpe: {test_result['sharpe']:.4f}")
    print(f"  年化收益: {test_result['annual_return']*100:.2f}%")
    print(f"  交易次数: {test_result['total_trades']}")
    print(f"  胜率: {test_result['win_rate']*100:.2f}%")
    
    return {
        'train_result': {
            'factors': train_best_factors,
            'weights': train_weights
        },
        'test_result': test_result,
        'train_df': train_df,
        'test_df': test_df
    }

def search_best_stop_loss(df: pd.DataFrame, factor_list: List[str],
                         weights: Dict, top_k: int = 10) -> Dict:
    print(f"\n{'='*60}")
    print("止损参数搜索")
    print(f"{'='*60}")
    
    stop_loss_coefs = [0.90, 0.93, 0.95, 0.97]
    results = []
    
    for coef in stop_loss_coefs:
        print(f"\n测试止损系数: {coef}")
        
        test_weights = weights.copy()
        result = backtest_factors_with_stop_loss(df, factor_list, test_weights, top_k=top_k, stop_coef=coef)
        
        results.append({
            'stop_coef': coef,
            'sharpe': result['sharpe'],
            'annual_return': result['annual_return'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate']
        })
        
        print(f"  Sharpe: {result['sharpe']:.4f}, 交易次数: {result['total_trades']}")
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['sharpe'].idxmax()]
    
    print(f"\n最优止损系数: {best['stop_coef']}")
    
    return best.to_dict()

def backtest_factors_with_stop_loss(df: pd.DataFrame, factor_list: List[str], 
                                   weights: Dict, top_k: int = 10,
                                   stop_coef: float = 0.95) -> Dict:
    from forward_selector import is_bearish_increase_volume
    
    dates = sorted(df['日期'].unique())
    trades = []
    position = None
    stop_loss_count = {}
    
    for i, date in enumerate(tqdm(dates[:-1], desc="止损搜索")):
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
            
            stop_price = entry_price * stop_coef
            if low_price <= stop_price:
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
                ret = (current_price - entry_price) / entry_price - 0.0003 - 0.001
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
                    'holding_days': 0
                }
                break
    
    win_trades = [t['return'] for t in trades if t['return'] > 0]
    loss_trades = [t['return'] for t in trades if t['return'] <= 0]
    
    if len(trades) > 0:
        daily_returns = [t['return'] for t in trades]
        daily_ret = pd.Series(daily_returns)
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        annual_return = daily_ret.mean() * 252
    else:
        sharpe = 0
        annual_return = 0
    
    result = {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'total_trades': len(trades),
        'win_rate': len(win_trades) / len(trades) if len(trades) > 0 else 0,
        'avg_return': np.mean([t['return'] for t in trades]) if len(trades) > 0 else 0,
        'trades': trades
    }
    
    return result

def tqdm_not_available():
    pass

import sys
if 'tqdm' in sys.modules:
    from tqdm import tqdm
else:
    def tqdm(x, desc=''):
        return x

if __name__ == "__main__":
    from data_processor import prepare_data
    from factor_calculator import calculate_all_factors
    
    df = prepare_data()
    df = calculate_all_factors(df)
    factor_list = ['factor_1a', 'factor_2', 'factor_11']
    
    result = walkforward_validation(df, factor_list)
    print(f"\nWalk-Forward 验证完成")
