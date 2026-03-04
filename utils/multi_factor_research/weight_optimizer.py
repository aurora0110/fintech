import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from forward_selector import backtest_factors
import warnings
warnings.filterwarnings('ignore')

def random_weight_search(df: pd.DataFrame, factor_list: List[str], 
                        n_samples: int = 2000, top_k: int = 10) -> Tuple[Dict, pd.DataFrame]:
    print(f"\n{'='*60}")
    print("阶段四：权重随机优化")
    print(f"{'='*60}")
    
    results = []
    
    for _ in range(n_samples):
        weights = np.random.dirichlet(np.ones(len(factor_list)))
        weight_dict = {f: w for f, w in zip(factor_list, weights)}
        
        result = backtest_factors(df, factor_list, weight_dict, top_k=top_k)
        
        results.append({
            'weights': weight_dict,
            'sharpe': result['sharpe'],
            'annual_return': result['annual_return'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    top_10_pct = results_df.head(int(n_samples * 0.1))
    best_weights = top_10_pct.iloc[0]['weights']
    
    print(f"随机搜索 {n_samples} 组权重")
    print(f"最优 Sharpe: {results_df.iloc[0]['sharpe']:.4f}")
    print(f"Top 10% 平均 Sharpe: {top_10_pct['sharpe'].mean():.4f}")
    
    return best_weights, results_df

def perturb_weight_test(df: pd.DataFrame, factor_list: List[str], 
                       base_weights: Dict, top_k: int = 10,
                       perturb_range: float = 0.1) -> Dict:
    print(f"\n权重扰动测试...")
    
    best_sharpe = -999
    best_weights = base_weights.copy()
    
    for f in factor_list:
        for direction in [-1, 1]:
            test_weights = base_weights.copy()
            test_weights[f] = base_weights[f] * (1 + direction * perturb_range)
            
            total_w = sum(test_weights.values())
            test_weights = {k: v/total_w for k, v in test_weights.items()}
            
            result = backtest_factors(df, factor_list, test_weights, top_k=top_k)
            
            if result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_weights = test_weights.copy()
    
    print(f"扰动后最优 Sharpe: {best_sharpe:.4f}")
    
    return best_weights

def optimize_weights(df: pd.DataFrame, factor_list: List[str], 
                   top_k: int = 10, n_samples: int = 2000) -> Dict:
    best_weights, results_df = random_weight_search(df, factor_list, n_samples, top_k)
    
    final_weights = perturb_weight_test(df, factor_list, best_weights, top_k)
    
    print(f"\n最优权重:")
    for f, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {f}: {w:.4f}")
    
    return final_weights, results_df

if __name__ == "__main__":
    from data_processor import prepare_data
    from factor_calculator import calculate_all_factors
    
    df = prepare_data()
    df = calculate_all_factors(df)
    factor_list = ['factor_1a', 'factor_2', 'factor_11']
    
    weights, results = optimize_weights(df, factor_list)
    print(f"\n最终权重: {weights}")
