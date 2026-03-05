import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"


def load_stock_data(max_stocks=500):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')][:max_stocks]
    stock_data = {}
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
            continue
        
        if '#' in stock_code:
            code_part = stock_code.split('#')[1]
        else:
            code_part = stock_code
        
        if code_part.startswith('8') and len(code_part) >= 3:
            if code_part[1] == '3' or code_part.startswith('83') or code_part.startswith('87'):
                continue
        
        if code_part.startswith('300'):
            board = '创业板'
        elif code_part.startswith('688'):
            board = '科创板'
        else:
            board = '主板'
        
        path = os.path.join(DATA_DIR, file)
        
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            if len(df) < 120:
                continue
            
            df['board'] = board
            stock_data[stock_code] = df
        except:
            continue
    
    print(f"成功加载 {len(stock_data)} 只股票")
    return stock_data


def calculate_indicators(df):
    close = df['CLOSE']
    high = df['HIGH']
    low = df['LOW']
    
    for window in [5, 10, 20, 30, 60, 120]:
        df[f'MA{window}'] = close.rolling(window).mean()
    
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-6) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df['知行短期趋势线'] = close.ewm(span=5, adjust=False).mean()
    df['知行多空线'] = close.ewm(span=10, adjust=False).mean()
    
    for window in [5, 20, 30, 60, 120]:
        df[f'Slope{window}'] = (df[f'MA{window}'] - df[f'MA{window}'].shift(5)) / 5
    
    return df


def check_entry_signal(row, j_range, structure_func, slope_func, position_func, ema_trend):
    """检查是否满足入场信号"""
    j_value = row.get('J')
    if pd.isna(j_value):
        return False, None
    
    if j_value >= j_range[1] or j_value < j_range[0]:
        return False, None
    
    if structure_func and not structure_func(row):
        return False, None
    
    if slope_func and not slope_func(row):
        return False, None
    
    if position_func and not position_func(row):
        return False, None
    
    if ema_trend:
        trend_line = row.get('知行短期趋势线')
        dk_line = row.get('知行多空线')
        if pd.isna(trend_line) or pd.isna(dk_line):
            return False, None
        if trend_line <= dk_line:
            return False, None
    
    return True, j_value


def check_exit_condition(row, position_type='ma死叉'):
    """检查是否满足出场条件"""
    if position_type == 'ma死叉':
        ma5 = row.get('MA5', 0)
        ma20 = row.get('MA20', 0)
        if pd.notna(ma5) and pd.notna(ma20) and ma5 < ma20:
            return True, 'MA5<MA20死叉'
    elif position_type == 'j超买':
        j_value = row.get('J', 0)
        if pd.notna(j_value) and j_value > 80:
            return True, 'J>80超买'
    elif position_type == '两者取其一':
        ma5 = row.get('MA5', 0)
        ma20 = row.get('MA20', 0)
        j_value = row.get('J', 0)
        if (pd.notna(ma5) and pd.notna(ma20) and ma5 < ma20) or (pd.notna(j_value) and j_value > 80):
            return True, '均线死叉或J>80'
    return False, None


def run_strategy_with_holding(stock_data, strategy_name, j_range, structure_func, 
                               slope_func=None, position_func=None, ema_trend=False,
                               exit_condition='natural'):
    """
    运行策略并计算持有时长
    exit_condition: 
    - 'natural': 自然持有(均线死叉或J>80出场)
    - 'fixed_1', 'fixed_2', etc: 固定持有N天
    """
    trade_results = []
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        
        in_position = False
        entry_idx = None
        entry_price = None
        
        for idx in range(121, len(df) - 1):
            row = df.iloc[idx]
            
            if in_position:
                should_exit = False
                exit_reason = ''
                
                if exit_condition == 'natural':
                    should_exit, exit_reason = check_exit_condition(row, '两者取其一')
                elif exit_condition.startswith('fixed'):
                    days = int(exit_condition.split('_')[1])
                    if idx - entry_idx >= days:
                        should_exit = True
                        exit_reason = f'固定{days}日'
                
                if should_exit:
                    exit_price = row['CLOSE']
                    holding_days = idx - entry_idx
                    ret = (exit_price - entry_price) / entry_price * 100
                    
                    trade_results.append({
                        'strategy': strategy_name,
                        'stock': stock_code,
                        'entry_date': str(df.index[entry_idx])[:10],
                        'exit_date': str(df.index[idx])[:10],
                        'holding_days': holding_days,
                        'return': ret,
                        'exit_reason': exit_reason
                    })
                    in_position = False
            
            if not in_position:
                is_entry, j_val = check_entry_signal(row, j_range, structure_func, 
                                                     slope_func, position_func, ema_trend)
                if is_entry:
                    in_position = True
                    entry_idx = idx
                    entry_price = row['CLOSE']
    
    return trade_results


def main():
    print("="*70)
    print("策略平均持有时长测试")
    print("="*70)
    
    print("\n加载数据...")
    stock_data = load_stock_data(max_stocks=300)
    
    structures = {
        '结构B': lambda r: r.get('MA5', 0) > r.get('MA20', 0) > r.get('MA30', 0) if all(pd.notna(r.get(f'MA{w}')) for w in [5,20,30]) else False,
        '结构G': lambda r: (r.get('MA5', 0) > r.get('MA20', 0)) and (r.get('MA20', 0) < r.get('MA60', 0)) if all(pd.notna(r.get(f'MA{w}')) for w in [5,20,60]) else False,
    }
    
    slopes = {
        '斜率2': lambda r: all(r.get(f'Slope{w}', 0) > 0 for w in [5, 20]),
    }
    
    positions = {
        '位置A': lambda r: r.get('CLOSE', 0) > r.get('MA20', 0) if all(pd.notna(r.get(x)) for x in ['CLOSE', 'MA20']) else False,
    }
    
    strategies = [
        ('策略1: J<-5 + 结构G + 斜率2 + EMA', 
         structures['结构G'], slopes['斜率2'], None, True),
        ('策略2: J<-5 + 结构B + 斜率2 + 位置A', 
         structures['结构B'], slopes['斜率2'], positions['位置A'], False),
        ('策略3: J<-5 + 结构B + 斜率2 + EMA', 
         structures['结构B'], slopes['斜率2'], None, True),
        ('策略4: J<-5 + 结构B + 斜率2 (无过滤)', 
         structures['结构B'], slopes['斜率2'], None, False),
    ]
    
    results_summary = []
    
    for strat_name, struct_func, slope_func, pos_func, ema in strategies:
        print(f"\n{'='*60}")
        print(f"{strat_name}")
        print('='*60)
        
        for exit_cond in ['natural', 'fixed_1', 'fixed_2', 'fixed_3', 'fixed_5']:
            results = run_strategy_with_holding(
                stock_data, strat_name,
                j_range=(-100, -5),
                structure_func=struct_func,
                slope_func=slope_func,
                position_func=pos_func,
                ema_trend=ema,
                exit_condition=exit_cond
            )
            
            if results:
                df = pd.DataFrame(results)
                avg_holding = df['holding_days'].mean()
                median_holding = df['holding_days'].median()
                max_holding = df['holding_days'].max()
                avg_ret = df['return'].mean()
                win_rate = (df['return'] > 0).mean() * 100
                
                if exit_cond == 'natural':
                    print(f"\n  自然持有(死叉或J>80出场):")
                    print(f"    平均持有: {avg_holding:.1f}天, 中位数: {int(median_holding)}天, 最长: {max_holding}天")
                    print(f"    收益: {avg_ret:.2f}%, 胜率: {win_rate:.1f}%, 信号数: {len(df)}")
                    
                    dist = df['holding_days'].value_counts().sort_index()
                    print(f"    持有分布: {dict(dist)}")
                    
                    results_summary.append({
                        'strategy': strat_name,
                        'exit': '自然持有',
                        'avg_holding': avg_holding,
                        'median_holding': median_holding,
                        'return': avg_ret,
                        'win_rate': win_rate,
                        'count': len(df)
                    })
                else:
                    days = exit_cond.split('_')[1]
                    print(f"  固定持有{days}日: 收益={avg_ret:.2f}%, 胜率={win_rate:.1f}%, 信号数={len(df)}")
    
    print("\n" + "="*70)
    print("汇总对比表")
    print("="*70)
    print(f"{'策略':<35} {'持有方式':<10} {'平均天数':<10} {'收益':<10} {'胜率':<8} {'信号数'}")
    print("-"*80)
    for r in results_summary:
        print(f"{r['strategy']:<35} {r['exit']:<10} {r['avg_holding']:.1f}天{'':<5} {r['return']:.2f}%{'':<5} {r['win_rate']:.1f}%{'':<3} {r['count']}")


if __name__ == "__main__":
    main()
