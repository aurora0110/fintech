import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from data_processor import prepare_data
from factor_calculator import calculate_all_factors, get_factor_list
from ic_analyzer import analyze_factor_ic, filter_by_correlation
from forward_selector import forward_selection, backtest_factors
from weight_optimizer import optimize_weights
from walkforward import walkforward_validation, search_best_stop_loss

OUTPUT_DIR = "/Users/lidongyang/Desktop/Qstrategy/utils/multi_factor_research/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_results(name, data):
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"已保存: {path}")

def main():
    print("="*70)
    print("多因子量化选股研究系统")
    print("="*70)
    
    print("\n[1/7] 准备数据...")
    df = prepare_data()
    
    print("\n[2/7] 计算因子...")
    df = calculate_all_factors(df)
    factor_list = get_factor_list()
    
    print("\n[3/7] IC分析...")
    ic_results, valid_factors = analyze_factor_ic(df, factor_list, ic_threshold=0.0, icir_threshold=0.03)
    
    if len(valid_factors) == 0:
        print("没有满足ICIR>0.03的因子，使用所有因子...")
        valid_factors = [f for f in factor_list if f in df.columns]
    
    print("\n[4/7] 相关性过滤...")
    selected_factors = filter_by_correlation(df, valid_factors)
    
    print("\n[5/7] Forward Selection...")
    final_factors, fs_result = forward_selection(df, selected_factors, top_k=10, improvement_threshold=0.05)
    
    print("\n[6/7] 权重优化...")
    final_weights, weight_results = optimize_weights(df, final_factors, top_k=10, n_samples=100)
    
    print("\n[7/7] Walk-Forward验证...")
    wf_result = walkforward_validation(df, final_factors, train_ratio=0.6, top_k=10, n_weight_samples=50)
    
    print("\n" + "="*70)
    print("最终结果")
    print("="*70)
    
    print(f"\n选定的因子: {final_factors}")
    print(f"最优权重:")
    for f, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {f}: {w:.4f}")
    
    print(f"\n训练期结果:")
    train_result = backtest_factors(wf_result['train_df'], final_factors, final_weights, top_k=10)
    print(f"  Sharpe: {train_result['sharpe']:.4f}")
    print(f"  年化收益: {train_result['annual_return']*100:.2f}%")
    print(f"  交易次数: {train_result['total_trades']}")
    print(f"  胜率: {train_result['win_rate']*100:.2f}%")
    
    print(f"\n验证期结果:")
    test_result = wf_result['test_result']
    print(f"  Sharpe: {test_result['sharpe']:.4f}")
    print(f"  年化收益: {test_result['annual_return']*100:.2f}%")
    print(f"  交易次数: {test_result['total_trades']}")
    print(f"  胜率: {test_result['win_rate']*100:.2f}%")
    
    ic_stats = []
    for f in final_factors:
        if f in ic_results:
            ic_stats.append({
                'factor': f,
                'ic_mean': ic_results[f]['ic_mean'],
                'ic_ir': ic_results[f]['ic_ir']
            })
    
    final_output = {
        'run_time': str(datetime.now()),
        'selected_factors': final_factors,
        'factor_weights': final_weights,
        'ic_statistics': ic_stats,
        'train_result': {
            'sharpe': train_result['sharpe'],
            'annual_return': train_result['annual_return'],
            'total_trades': train_result['total_trades'],
            'win_rate': train_result['win_rate']
        },
        'test_result': {
            'sharpe': test_result['sharpe'],
            'annual_return': test_result['annual_return'],
            'total_trades': test_result['total_trades'],
            'win_rate': test_result['win_rate']
        }
    }
    
    save_results('final_results', final_output)
    
    print("\n" + "="*70)
    print("研究完成！")
    print("="*70)

if __name__ == "__main__":
    main()
