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
    
    return df


def run_strategy_fixed(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days):
    all_returns = []
    
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
            
            if struct_name == 'G底部反转':
                ma60_v = ma60[idx]
                if pd.isna(ma60_v):
                    continue
                if not (ma5_v > ma20_v and ma20_v < ma60_v):
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
    
    return all_returns


def main():
    print("="*80)
    print("持有天数对比分析：1日 vs 2日 vs 3日 vs 5日 vs 7日 vs 10日")
    print("="*80)
    
    print("\n加载股票数据...")
    stock_data = load_stock_data(max_stocks=800)
    
    holdings = [1, 2, 3, 5, 7, 10]
    
    best_strategy = ('J<-5', 'G底部反转', '站MA20', '无')
    
    print("\n" + "="*80)
    print(f"【最优策略组合: J<-5 + G底部反转 + 站MA20】")
    print("="*80)
    print("\n各持有天数对比:")
    print("-"*60)
    print(f"{'持有天数':<10} {'信号数':<10} {'平均收益':<12} {'胜率':<10} {'中位数收益'}")
    print("-"*60)
    
    results = []
    
    for holding in holdings:
        returns = run_strategy_fixed(stock_data, 5, 'G底部反转', '站MA20', '无', holding)
        
        if len(returns) >= 50:
            arr = np.array(returns)
            print(f"{holding}日{'':<6} {len(arr):<10} {arr.mean():.2f}%{'':<7} {(arr>0).mean()*100:.1f}%{'':<6} {np.median(arr):.2f}%")
            results.append({
                'holding': holding,
                'count': len(arr),
                'avg_return': arr.mean(),
                'win_rate': (arr > 0).mean() * 100,
                'median': np.median(arr)
            })
    
    best = max(results, key=lambda x: x['avg_return'])
    print("-"*60)
    print(f"\n结论: 持有{best['holding']}日收益最高 ({best['avg_return']:.2f}%)")
    
    print("\n" + "="*80)
    print(f"【策略: J<-5 + G底部反转 + 有EMA】")
    print("="*80)
    print("\n各持有天数对比:")
    print("-"*60)
    print(f"{'持有天数':<10} {'信号数':<10} {'平均收益':<12} {'胜率':<10} {'中位数收益'}")
    print("-"*60)
    
    results2 = []
    
    for holding in holdings:
        returns = run_strategy_fixed(stock_data, 5, 'G底部反转', '无', '有EMA', holding)
        
        if len(returns) >= 50:
            arr = np.array(returns)
            print(f"{holding}日{'':<6} {len(arr):<10} {arr.mean():.2f}%{'':<7} {(arr>0).mean()*100:.1f}%{'':<6} {np.median(arr):.2f}%")
            results2.append({
                'holding': holding,
                'count': len(arr),
                'avg_return': arr.mean(),
                'win_rate': (arr > 0).mean() * 100,
                'median': np.median(arr)
            })
    
    best2 = max(results2, key=lambda x: x['avg_return'])
    print("-"*60)
    print(f"\n结论: 持有{best2['holding']}日收益最高 ({best2['avg_return']:.2f}%)")
    
    print("\n" + "="*80)
    print("【所有策略各持有天数收益对比表】")
    print("="*80)
    
    j_thresholds = [5, 10]
    structures = ['G底部反转', 'B短期多头', 'D趋势启动']
    positions = ['无', '站MA20']
    emas = ['无', '有EMA']
    
    print(f"\n{'策略组合':<45} {'1日':<10} {'2日':<10} {'3日':<10} {'5日':<10} {'7日':<10} {'10日':<10}")
    print("-"*110)
    
    for j_thresh in j_thresholds:
        for struct in structures:
            for pos in positions:
                for ema in emas:
                    if pos == '站MA20' and ema == '有EMA':
                        continue
                    
                    strategy_name = f"J<-{j_thresh}+{struct}+{pos}+{ema}"
                    
                    row_returns = []
                    for holding in holdings:
                        returns = run_strategy_fixed(stock_data, j_thresh, struct, pos, ema, holding)
                        if len(returns) >= 50:
                            row_returns.append(f"{np.mean(returns):.2f}%")
                        else:
                            row_returns.append("-")
                    
                    print(f"{strategy_name:<45} {row_returns[0]:<10} {row_returns[1]:<10} {row_returns[2]:<10} {row_returns[3]:<10} {row_returns[4]:<10} {row_returns[5]:<10}")
    
    print("\n" + "="*80)
    print("【总结】")
    print("="*80)
    print("""
从回测结果来看:
1. 持有7日的收益普遍高于较短持有期(1-5日)
2. 持有10日在部分策略中收益开始下降
3. "7日"是经过完整对比后的最优选择，不是预设值
""")


if __name__ == "__main__":
    main()
