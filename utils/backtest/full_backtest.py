import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"


def load_stock_data(max_stocks=800):
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
        
        path = os.path.join(DATA_DIR, file)
        
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df.columns = ['日期', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT']
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            df = df.sort_index()
            
            if len(df) < 120:
                continue
            
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


def run_strategy_fixed(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days):
    """固定持有期回测"""
    all_returns = []
    holding_times = []
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        
        if len(df) < 122:
            continue
        
        close = df['CLOSE'].values
        j_vals = df['J'].values
        ma5 = df['MA5'].values
        ma20 = df['MA20'].values
        ma30 = df['MA30'].values
        ma60 = df['MA60'].values
        ema5 = df['知行短期趋势线'].values
        ema10 = df['知行多空线'].values
        
        for idx in range(121, len(df) - holding_days):
            j = j_vals[idx]
            if pd.isna(j) or j >= -j_thresh:
                continue
            
            ma5_v, ma20_v, ma30_v = ma5[idx], ma20[idx], ma30[idx]
            if pd.isna(ma5_v) or pd.isna(ma20_v) or pd.isna(ma30_v):
                continue
            
            if struct_name == 'B短期多头':
                if not (ma5_v > ma20_v > ma30_v):
                    continue
            elif struct_name == 'G底部反转':
                ma60_v = ma60[idx]
                if pd.isna(ma60_v):
                    continue
                if not (ma5_v > ma20_v and ma20_v < ma60_v):
                    continue
            elif struct_name == 'D趋势启动':
                if not (ma5_v > ma20_v and ma20_v > ma30_v):
                    continue
            
            if pos_name == '站MA20':
                if pd.isna(close[idx]) or pd.isna(ma20[idx]):
                    continue
                if not (close[idx] > ma20[idx]):
                    continue
            
            if ema_name == '有EMA':
                ema5_v, ema10_v = ema5[idx], ema10[idx]
                if pd.isna(ema5_v) or pd.isna(ema10_v):
                    continue
                if not (ema5_v > ema10_v):
                    continue
            
            entry_price = close[idx]
            exit_price = close[idx + holding_days]
            
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            
            ret = (exit_price - entry_price) / entry_price * 100
            all_returns.append(ret)
            holding_times.append(holding_days)
    
    return all_returns, holding_times


def run_strategy_natural(stock_data, j_thresh, struct_name, pos_name, ema_name):
    """自然持有期回测 (均线死叉或J>80出场)"""
    all_returns = []
    holding_times = []
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        
        if len(df) < 122:
            continue
        
        close = df['CLOSE'].values
        j_vals = df['J'].values
        ma5 = df['MA5'].values
        ma20 = df['MA20'].values
        ma30 = df['MA30'].values
        ma60 = df['MA60'].values
        ema5 = df['知行短期趋势线'].values
        ema10 = df['知行多空线'].values
        
        for idx in range(121, len(df) - 20):
            j = j_vals[idx]
            if pd.isna(j) or j >= -j_thresh:
                continue
            
            ma5_v, ma20_v, ma30_v = ma5[idx], ma20[idx], ma30[idx]
            if pd.isna(ma5_v) or pd.isna(ma20_v) or pd.isna(ma30_v):
                continue
            
            if struct_name == 'B短期多头':
                if not (ma5_v > ma20_v > ma30_v):
                    continue
            elif struct_name == 'G底部反转':
                ma60_v = ma60[idx]
                if pd.isna(ma60_v):
                    continue
                if not (ma5_v > ma20_v and ma20_v < ma60_v):
                    continue
            elif struct_name == 'D趋势启动':
                if not (ma5_v > ma20_v and ma20_v > ma30_v):
                    continue
            
            if pos_name == '站MA20':
                if pd.isna(close[idx]) or pd.isna(ma20[idx]):
                    continue
                if not (close[idx] > ma20[idx]):
                    continue
            
            if ema_name == '有EMA':
                ema5_v, ema10_v = ema5[idx], ema10[idx]
                if pd.isna(ema5_v) or pd.isna(ema10_v):
                    continue
                if not (ema5_v > ema10_v):
                    continue
            
            entry_price = close[idx]
            
            exited = False
            for t in range(1, 21):
                if idx + t >= len(df):
                    break
                if (ma5[idx + t] < ma20[idx + t]) or (j_vals[idx + t] > 80):
                    exit_price = close[idx + t]
                    ret = (exit_price - entry_price) / entry_price * 100
                    all_returns.append(ret)
                    holding_times.append(t)
                    exited = True
                    break
            
            if not exited and idx + 20 < len(df):
                exit_price = close[idx + 20]
                ret = (exit_price - entry_price) / entry_price * 100
                all_returns.append(ret)
                holding_times.append(20)
    
    return all_returns, holding_times


def main():
    print("="*80)
    print("全量股票回测：固定持有期 vs 自然持有期 Top 5 策略")
    print("="*80)
    
    print("\n加载股票数据...")
    stock_data = load_stock_data(max_stocks=800)
    
    j_thresholds = [5, 10]
    structures = ['B短期多头', 'G底部反转', 'D趋势启动']
    positions = ['无', '站MA20']
    emas = ['无', '有EMA']
    holdings = [1, 2, 3, 5, 7]
    
    results_fixed = []
    results_natural = []
    
    print("\n回测中 (固定持有期)...")
    
    for j_thresh in j_thresholds:
        for struct in structures:
            for pos in positions:
                for ema in emas:
                    if pos == '站MA20' and ema == '有EMA':
                        continue
                    
                    for holding in holdings:
                        returns, times = run_strategy_fixed(stock_data, j_thresh, struct, pos, ema, holding)
                        
                        if len(returns) >= 100:
                            arr = np.array(returns)
                            results_fixed.append({
                                'j': f'J<-{j_thresh}',
                                'struct': struct,
                                'pos': pos,
                                'ema': ema,
                                'holding': f'{holding}日',
                                'count': len(arr),
                                'avg_return': arr.mean(),
                                'median_return': np.median(arr),
                                'win_rate': (arr > 0).mean() * 100,
                                'avg_holding': np.mean(times),
                            })
    
    print("回测中 (自然持有期)...")
    
    for j_thresh in j_thresholds:
        for struct in structures:
            for pos in positions:
                for ema in emas:
                    if pos == '站MA20' and ema == '有EMA':
                        continue
                    
                    returns, times = run_strategy_natural(stock_data, j_thresh, struct, pos, ema)
                    
                    if len(returns) >= 100:
                        arr = np.array(returns)
                        results_natural.append({
                            'j': f'J<-{j_thresh}',
                            'struct': struct,
                            'pos': pos,
                            'ema': ema,
                            'count': len(arr),
                            'avg_return': arr.mean(),
                            'median_return': np.median(arr),
                            'win_rate': (arr > 0).mean() * 100,
                            'avg_holding': np.mean(times),
                        })
    
    df_fixed = pd.DataFrame(results_fixed)
    df_natural = pd.DataFrame(results_natural)
    
    print("\n" + "="*80)
    print("【固定持有期 Top 10】")
    print("="*80)
    
    fixed = df_fixed.sort_values('avg_return', ascending=False).head(10)
    print(f"\n{'排名':<4} {'J值':<8} {'结构':<12} {'位置':<8} {'EMA':<6} {'持有':<8} {'信号数':<8} {'平均收益':<10} {'胜率':<8}")
    print("-"*80)
    
    for i, (_, r) in enumerate(fixed.iterrows(), 1):
        print(f"{i:<4} {r['j']:<8} {r['struct']:<12} {r['pos']:<8} {r['ema']:<6} {r['holding']:<8} {r['count']:<8} {r['avg_return']:.2f}%{'':<5} {r['win_rate']:.1f}%")
    
    print("\n" + "="*80)
    print("【自然持有期 Top 10】")
    print("="*80)
    
    natural = df_natural.sort_values('avg_return', ascending=False).head(10)
    print(f"\n{'排名':<4} {'J值':<8} {'结构':<12} {'位置':<8} {'EMA':<6} {'信号数':<8} {'平均收益':<10} {'胜率':<8} {'平均持有'}")
    print("-"*80)
    
    for i, (_, r) in enumerate(natural.iterrows(), 1):
        print(f"{i:<4} {r['j']:<8} {r['struct']:<12} {r['pos']:<8} {r['ema']:<6} {r['count']:<8} {r['avg_return']:.2f}%{'':<5} {r['win_rate']:.1f}%{'':<3} {r['avg_holding']:.1f}天")
    
    print("\n" + "="*80)
    print("【Top 5 策略详细对比】")
    print("="*80)
    
    print("\n" + "="*60)
    print("【固定持有期 Top 5】")
    print("="*60)
    
    for i, (_, r) in enumerate(fixed.head(5).iterrows(), 1):
        print(f"\n第{i}名: {r['j']} + {r['struct']} + {r['pos']} + {r['ema']}")
        print(f"  持有期: {r['holding']}")
        print(f"  信号数: {r['count']}")
        print(f"  平均收益: {r['avg_return']:.2f}%")
        print(f"  胜率: {r['win_rate']:.1f}%")
        print(f"  中位数收益: {r['median_return']:.2f}%")
    
    print("\n" + "="*60)
    print("【自然持有期 Top 5】")
    print("="*60)
    
    for i, (_, r) in enumerate(natural.head(5).iterrows(), 1):
        print(f"\n第{i}名: {r['j']} + {r['struct']} + {r['pos']} + {r['ema']}")
        print(f"  出场条件: 均线死叉或J>80")
        print(f"  信号数: {r['count']}")
        print(f"  平均收益: {r['avg_return']:.2f}%")
        print(f"  胜率: {r['win_rate']:.1f}%")
        print(f"  中位数收益: {r['median_return']:.2f}%")
        print(f"  平均持有: {r['avg_holding']:.1f}天")
    
    print("\n" + "="*80)
    print("【总结对比】")
    print("="*80)
    
    print("\n固定持有期第1名:")
    print(f"  策略: {fixed.iloc[0]['j']} + {fixed.iloc[0]['struct']} + {fixed.iloc[0]['pos']} + {fixed.iloc[0]['ema']}")
    print(f"  持有期: {fixed.iloc[0]['holding']}")
    print(f"  收益: {fixed.iloc[0]['avg_return']:.2f}%, 胜率: {fixed.iloc[0]['win_rate']:.1f}%")
    
    print("\n自然持有期第1名:")
    print(f"  策略: {natural.iloc[0]['j']} + {natural.iloc[0]['struct']} + {natural.iloc[0]['pos']} + {natural.iloc[0]['ema']}")
    print(f"  收益: {natural.iloc[0]['avg_return']:.2f}%, 胜率: {natural.iloc[0]['win_rate']:.1f}%")
    print(f"  平均持有: {natural.iloc[0]['avg_holding']:.1f}天")


if __name__ == "__main__":
    main()
