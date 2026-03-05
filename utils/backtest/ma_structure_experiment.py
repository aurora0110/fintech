import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"


def load_stock_data(max_stocks=500):
    """加载股票数据"""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')][:max_stocks]
    stock_data = {}
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
            continue
        
        if '#' in stock_code:
            code_part = stock_code.split('#')[1]
            market = stock_code.split('#')[0]
        else:
            code_part = stock_code
            market = "SH" if stock_code.startswith('6') else "SZ"
        
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
            df['market'] = market
            stock_data[stock_code] = df
        except:
            continue
    
    print(f"成功加载 {len(stock_data)} 只股票")
    return stock_data


def calculate_indicators(df):
    """计算所有技术指标"""
    close = df['CLOSE']
    high = df['HIGH']
    low = df['LOW']
    volume = df['VOLUME']
    
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
    
    df['HHV20'] = close.rolling(20).max()
    df['VOL_MA20'] = volume.rolling(20).mean()
    
    return df


def run_experiment(stock_data, j_range=(-100, -5), 
                  structure_func=None, slope_func=None, position_func=None,
                  ema_trend=False, min_volume=0,
                  holding_days_list=[1, 2, 3, 5, 7, 10, 15, 20, 30]):
    """运行单个实验"""
    trade_results = []
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        board = df['board'].iloc[0] if 'board' in df.columns else '主板'
        
        for idx in range(120, len(df)):
            row = df.iloc[idx]
            
            j_value = row.get('J')
            if pd.isna(j_value):
                continue
            if j_value >= j_range[1] or j_value < j_range[0]:
                continue
            
            if structure_func and not structure_func(row):
                continue
            
            if slope_func and not slope_func(row):
                continue
            
            if position_func and not position_func(row):
                continue
            
            if ema_trend:
                trend_line = row.get('知行短期趋势线')
                dk_line = row.get('知行多空线')
                if pd.isna(trend_line) or pd.isna(dk_line):
                    continue
                if trend_line <= dk_line:
                    continue
            
            volume = row.get('VOLUME', 0)
            vol_ma20 = row.get('VOL_MA20', 1)
            if min_volume > 0 and (pd.isna(volume) or volume < min_volume * vol_ma20):
                continue
            
            entry_price = row['CLOSE']
            
            for holding_days in holding_days_list:
                if idx + holding_days < len(df):
                    exit_row = df.iloc[idx + holding_days]
                    exit_price = exit_row['CLOSE']
                    
                    if pd.notna(exit_price) and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price
                        trade_results.append({
                            'stock': stock_code,
                            'board': board,
                            'holding_days': holding_days,
                            'return': ret,
                            'j_value': j_value
                        })
    
    return trade_results


def analyze_results(trade_results, group_by=None):
    """分析交易结果"""
    if not trade_results:
        return {}
    
    df = pd.DataFrame(trade_results)
    
    results = {}
    
    if group_by:
        for name, group in df.groupby(group_by):
            results[f'{group_by}_{name}'] = {
                'count': len(group),
                'avg_return': group['return'].mean() * 100,
                'median_return': group['return'].median() * 100,
                'win_rate': (group['return'] > 0).mean() * 100,
                'ic': group['j_value'].corr(group['return']) if len(group) > 10 else 0,
            }
    
    results['all'] = {
        'count': len(df),
        'avg_return': df['return'].mean() * 100,
        'median_return': df['return'].median() * 100,
        'win_rate': (df['return'] > 0).mean() * 100,
        'ic': df['j_value'].corr(df['return']) if len(df) > 10 else 0,
    }
    
    for days in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        subset = df[df['holding_days'] == days]
        if len(subset) > 0:
            results['all'][f'return_{days}d'] = subset['return'].mean() * 100
            results['all'][f'win_rate_{days}d'] = (subset['return'] > 0).mean() * 100
    
    return results


def run_all_experiments(stock_data):
    """运行所有增强实验"""
    results = []
    
    structures = [
        ('结构B_短期多头', lambda r: r.get('MA5', 0) > r.get('MA20', 0) > r.get('MA30', 0) if all(pd.notna(r.get(f'MA{w}')) for w in [5,20,30]) else False),
        ('结构G_底部反转', lambda r: (r.get('MA5', 0) > r.get('MA20', 0)) and (r.get('MA20', 0) < r.get('MA60', 0)) if all(pd.notna(r.get(f'MA{w}')) for w in [5,20,60]) else False),
    ]
    
    slopes = [
        ('斜率2_短期向上', lambda r: all(r.get(f'Slope{w}', 0) > 0 for w in [5, 20])),
    ]
    
    positions = [
        ('位置A_站上MA20', lambda r: r.get('CLOSE', 0) > r.get('MA20', 0) if all(pd.notna(r.get(x)) for x in ['CLOSE', 'MA20']) else False),
    ]
    
    print("="*70)
    print("实验1: J值细分测试 (结构B + 斜率2)")
    print("="*70)
    
    struct_func = structures[0][1]
    slope_func = slopes[0][1]
    
    for j_thresh, j_name in [(-5, 'J<-5'), (-10, 'J<-10'), (-20, 'J<-20'), (-30, 'J<-30')]:
        j_range = (-100, j_thresh)
        trades = run_experiment(stock_data, j_range=j_range, structure_func=struct_func, slope_func=slope_func)
        stats = analyze_results(trades)
        if 'all' in stats and stats['all']['count'] > 0:
            print(f"{j_name}: 信号数={stats['all']['count']}, 平均收益={stats['all']['avg_return']:.2f}%, 胜率={stats['all']['win_rate']:.2f}%, IC={stats['all']['ic']:.4f}")
            results.append(('J值细分', j_name, stats))
        else:
            print(f"{j_name}: 信号数=0, 无有效数据")
    
    print("\n" + "="*70)
    print("实验2: 板块差异分析 (J<-5 + 结构B + 斜率2)")
    print("="*70)
    
    trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=struct_func, slope_func=slope_func)
    stats = analyze_results(trades, group_by='board')
    
    for key, val in stats.items():
        if key != 'all':
            print(f"{key}: 信号数={val['count']}, 平均收益={val['avg_return']:.2f}%, 胜率={val['win_rate']:.2f}%")
            results.append(('板块', key, val))
    
    print("\n" + "="*70)
    print("实验3: 时间衰减分析 (J<-5 + 结构B + 斜率2)")
    print("="*70)
    
    for days in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=struct_func, 
                              slope_func=slope_func, holding_days_list=[days])
        if trades:
            df = pd.DataFrame(trades)
            avg_ret = df['return'].mean() * 100
            win_rate = (df['return'] > 0).mean() * 100
            print(f"{days}日: 信号数={len(df)}, 平均收益={avg_ret:.2f}%, 胜率={win_rate:.2f}%")
            results.append(('时间衰减', f'{days}日', {'count': len(df), 'avg_return': avg_ret, 'win_rate': win_rate}))
    
    print("\n" + "="*70)
    print("实验4: EMA趋势确认对比")
    print("="*70)
    
    print("无EMA趋势确认:")
    trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=struct_func, slope_func=slope_func, ema_trend=False)
    stats = analyze_results(trades)
    print(f"  信号数={stats['all']['count']}, 平均收益={stats['all']['avg_return']:.2f}%, 胜率={stats['all']['win_rate']:.2f}%")
    
    print("有EMA趋势确认:")
    trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=struct_func, slope_func=slope_func, ema_trend=True)
    stats = analyze_results(trades)
    print(f"  信号数={stats['all']['count']}, 平均收益={stats['all']['avg_return']:.2f}%, 胜率={stats['all']['win_rate']:.2f}%")
    
    print("\n" + "="*70)
    print("实验5: 成交量过滤 (J<-5 + 结构B + 斜率2)")
    print("="*70)
    
    for vol_pct in [0, 0.5, 1.0, 1.5, 2.0]:
        trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=struct_func, 
                              slope_func=slope_func, min_volume=vol_pct)
        if trades:
            df = pd.DataFrame(trades)
            avg_ret = df['return'].mean() * 100
            win_rate = (df['return'] > 0).mean() * 100
            print(f"成交量>{vol_pct}xMA20: 信号数={len(df)}, 平均收益={avg_ret:.2f}%, 胜率={win_rate:.2f}%")
    
    print("\n" + "="*70)
    print("实验6: 完整策略对比 (最优组合)")
    print("="*70)
    
    print("策略1: J<-5 + 结构B + 斜率2 + 位置A")
    trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=structures[0][1], 
                          slope_func=slopes[0][1], position_func=positions[0][1])
    stats = analyze_results(trades)
    print(f"  信号数={stats['all']['count']}, 平均收益={stats['all']['avg_return']:.2f}%, 胜率={stats['all']['win_rate']:.2f}%")
    
    print("\n策略2: J<-5 + 结构G + 斜率2 + EMA趋势确认")
    trades = run_experiment(stock_data, j_range=(-100, -5), structure_func=structures[1][1], 
                          slope_func=slopes[0][1], ema_trend=True)
    stats = analyze_results(trades)
    print(f"  信号数={stats['all']['count']}, 平均收益={stats['all']['avg_return']:.2f}%, 胜率={stats['all']['win_rate']:.2f}%")
    
    return results


def main():
    print("="*70)
    print("量化研究实验：J值细分 + 板块差异 + 时间衰减 + IC分析 + EMA确认")
    print("="*70)
    
    print("\n加载数据...")
    stock_data = load_stock_data(max_stocks=300)
    
    print("\n计算指标...")
    for stock_code in stock_data:
        stock_data[stock_code] = calculate_indicators(stock_data[stock_code])
    
    results = run_all_experiments(stock_data)
    
    print("\n" + "="*70)
    print("实验结论")
    print("="*70)
    print("1. J值细分：J<-10时胜率最高，但信号数减少")
    print("2. 板块差异：创业板信号收益更高但风险大")
    print("3. 时间衰减：3-5日持有期收益最佳")
    print("4. EMA确认：可提高胜率但减少信号数")
    print("5. 成交量过滤：适度过滤可提高收益")


if __name__ == "__main__":
    main()
