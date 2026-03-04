import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

def calculate_daily_ic(df: pd.DataFrame, factor_col: str) -> pd.Series:
    ics = []
    dates = df['日期'].unique()
    
    for date in dates:
        day_data = df[df['日期'] == date].dropna(subset=[factor_col, 'forward_return'])
        if len(day_data) < 10:
            continue
        
        factor_vals = day_data[factor_col].values
        returns = day_data['forward_return'].values
        
        if len(set(factor_vals)) < 2 or len(set(returns)) < 2:
            continue
            
        corr, _ = spearmanr(factor_vals, returns)
        if not np.isnan(corr):
            ics.append({'date': date, 'ic': corr})
    
    if len(ics) == 0:
        return pd.Series(dtype=float)
    
    return pd.DataFrame(ics).set_index('date')['ic']

def analyze_factor_ic(df: pd.DataFrame, factor_list: List[str], 
                     ic_threshold: float = 0.0, icir_threshold: float = 0.3) -> Dict:
    results = {}
    
    print(f"\n{'='*60}")
    print("阶段一：单因子IC分析")
    print(f"{'='*60}")
    
    for factor in factor_list:
        if factor not in df.columns:
            print(f"因子 {factor} 不存在，跳过")
            continue
            
        ics = calculate_daily_ic(df, factor)
        
        if len(ics) == 0:
            print(f"因子 {factor}: 无有效数据")
            continue
        
        ic_mean = ics.mean()
        ic_std = ics.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        results[factor] = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_series': ics,
            'valid': (ic_mean > ic_threshold) and (ic_ir > icir_threshold)
        }
        
        status = "✓ 保留" if results[factor]['valid'] else "✗ 删除"
        print(f"{factor}: IC均值={ic_mean:.4f}, IC标准差={ic_std:.4f}, ICIR={ic_ir:.4f} {status}")
    
    valid_factors = [f for f, v in results.items() if v['valid']]
    print(f"\n保留的有效因子: {valid_factors}")
    
    return results, valid_factors

def filter_by_correlation(df: pd.DataFrame, factor_list: List[str], 
                         corr_threshold: float = 0.7, max_factors: int = 5) -> List[str]:
    print(f"\n{'='*60}")
    print("阶段二：相关性过滤")
    print(f"{'='*60}")
    
    if len(factor_list) <= max_factors:
        print(f"因子数量({len(factor_list)}) <= {max_factors}，无需过滤")
        return factor_list
    
    corr_matrix = df[factor_list].corr().abs()
    
    to_drop = set()
    for i, f1 in enumerate(factor_list):
        for f2 in factor_list[i+1:]:
            if corr_matrix.loc[f1, f2] > corr_threshold:
                ir1 = df[f1].mean() if f1 in df.columns else 0
                ir2 = df[f2].mean() if f2 in df.columns else 0
                if ir1 >= ir2:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)
    
    selected = [f for f in factor_list if f not in to_drop]
    
    while len(selected) > max_factors:
        most_corr = None
        max_avg_corr = 0
        for f in selected:
            avg_corr = corr_matrix.loc[f, [x for x in selected if x != f]].mean()
            if avg_corr > max_avg_corr:
                max_avg_corr = avg_corr
                most_corr = f
        if most_corr:
            selected.remove(most_corr)
    
    print(f"原始因子数: {len(factor_list)}, 过滤后: {len(selected)}")
    print(f"保留因子: {selected}")
    
    return selected

if __name__ == "__main__":
    from data_processor import prepare_data
    from factor_calculator import calculate_all_factors, get_factor_list
    
    df = prepare_data()
    df = calculate_all_factors(df)
    factor_list = get_factor_list()
    
    results, valid_factors = analyze_factor_ic(df, factor_list)
    selected = filter_by_correlation(df, valid_factors)
    print(f"\n最终因子: {selected}")
