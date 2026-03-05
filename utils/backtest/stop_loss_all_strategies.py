import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"


def load_stock_data(max_stocks=5000):
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')][:max_stocks]
    stock_data = {}
    
    for file in files:
        stock_code = file.replace('.txt', '')
        
        if stock_code.startswith('BJ'):
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


def run_strategy(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days, use_stop_loss=False):
    all_returns = []
    stop_loss_count = 0
    
    for stock_code, df in stock_data.items():
        df = calculate_indicators(df)
        
        if len(df) < 122:
            continue
        
        close = df['CLOSE'].values
        open_prices = df['OPEN'].values
        low_prices = df['LOW'].values
        j_vals = df['J'].values
        ma5 = df['MA5'].values
        ma20 = df['MA20'].values
        ma30 = df['MA30'].values
        ma60 = df['MA60'].values
        ema5 = df['知行短期趋势线'].values
        ema10 = df['知行多空线'].values
        
        for idx in range(121, len(df) - holding_days - 1):
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
            elif struct_name == 'B短期多头':
                if not (ma5_v > ma20_v > ma30_v):
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
            
            signal_low = low_prices[idx]
            entry_open = open_prices[idx + 1]
            
            if pd.isna(entry_open) or entry_open <= 0:
                continue
            
            if use_stop_loss:
                buy_day_low = low_prices[idx + 1]
                stop_loss_price = min(signal_low, buy_day_low) * 0.95
                
                stopped = False
                for day in range(1, holding_days + 1):
                    if idx + day + 1 >= len(df):
                        break
                    
                    day_close = close[idx + day]
                    if pd.isna(day_close):
                        continue
                    
                    if day_close < stop_loss_price:
                        exit_open = open_prices[idx + day + 1]
                        if pd.isna(exit_open) or exit_open <= 0:
                            continue
                        ret = (exit_open - entry_open) / entry_open * 100
                        all_returns.append(ret)
                        stop_loss_count += 1
                        stopped = True
                        break
                
                if stopped:
                    continue
            
            exit_open = open_prices[idx + holding_days + 1]
            if pd.isna(exit_open) or exit_open <= 0:
                continue
            
            ret = (exit_open - entry_open) / entry_open * 100
            all_returns.append(ret)
    
    return all_returns, stop_loss_count


def main():
    print("="*100)
    print("全策略对比：止损 vs 无止损 | 持有期: 5/7/10/15/20/30/60日")
    print("="*100)
    print("""
入场: 信号次日开盘价买入
出场: 固定持有期后次日开盘卖出，或触及止损则次日开盘卖出
止损价: min(信号日最低价, 买入日最低价) * 0.95
""")
    
    print("加载股票数据...")
    stock_data = load_stock_data(max_stocks=5000)
    
    j_thresholds = [5, 10]
    structures = ['G底部反转', 'B短期多头', 'D趋势启动']
    positions = ['无', '站MA20']
    emas = ['无', '有EMA']
    holdings = [5, 7, 10, 15, 20, 30, 60]
    
    results = []
    
    print("\n回测中...")
    
    for j_thresh in j_thresholds:
        for struct in structures:
            for pos in positions:
                for ema in emas:
                    if pos == '站MA20' and ema == '有EMA':
                        continue
                    
                    for holding in holdings:
                        returns_no_sl, sl_count = run_strategy(stock_data, j_thresh, struct, pos, ema, holding, use_stop_loss=False)
                        
                        if len(returns_no_sl) >= 50:
                            returns_with_sl, _ = run_strategy(stock_data, j_thresh, struct, pos, ema, holding, use_stop_loss=True)
                            
                            if len(returns_with_sl) >= 50:
                                results.append({
                                    'j': f'J<-{j_thresh}',
                                    'struct': struct,
                                    'pos': pos,
                                    'ema': ema,
                                    'holding': holding,
                                    'count_no_sl': len(returns_no_sl),
                                    'avg_no_sl': np.mean(returns_no_sl),
                                    'win_rate_no_sl': (np.array(returns_no_sl) > 0).mean() * 100,
                                    'avg_with_sl': np.mean(returns_with_sl),
                                    'win_rate_with_sl': (np.array(returns_with_sl) > 0).mean() * 100,
                                    'sl_count': sl_count,
                                    'change': np.mean(returns_with_sl) - np.mean(returns_no_sl),
                                })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("【全部策略对比结果 - 按无止损收益排序 Top 30】")
    print("="*100)
    print(f"\n{'排名':<4} {'策略组合':<45} {'持有':<6} {'无止损':<10} {'有止损':<10} {'止损次':<8} {'收益变化':<10} {'结论'}")
    print("-"*100)
    
    df_sorted = df.sort_values('avg_no_sl', ascending=False)
    
    for i, (_, r) in enumerate(df_sorted.head(30).iterrows(), 1):
        strategy = f"{r['j']}+{r['struct']}+{r['pos']}+{r['ema']}"
        
        if r['change'] > 0.1:
            conclusion = "✓止损优"
        elif r['change'] < -0.1:
            conclusion = "✗无止损优"
        else:
            conclusion = "≈持平"
        
        print(f"{i:<4} {strategy:<45} {r['holding']}日{'':<2} {r['avg_no_sl']:.2f}%{'':<5} {r['avg_with_sl']:.2f}%{'':<5} {r['sl_count']:<8} {r['change']:+.2f}%{'':<5} {conclusion}")
    
    print("\n" + "="*100)
    print("【止损有效的策略 Top 10】")
    print("="*100)
    
    df_sl_better = df[df['change'] > 0.1].sort_values('change', ascending=False).head(10)
    print(f"\n{'排名':<4} {'策略组合':<45} {'持有':<6} {'无止损':<10} {'有止损':<10} {'收益变化'}")
    print("-"*90)
    
    for i, (_, r) in enumerate(df_sl_better.iterrows(), 1):
        strategy = f"{r['j']}+{r['struct']}+{r['pos']}+{r['ema']}"
        print(f"{i:<4} {strategy:<45} {r['holding']}日{'':<2} {r['avg_no_sl']:.2f}%{'':<5} {r['avg_with_sl']:.2f}%{'':<5} {r['change']:+.2f}%")
    
    print("\n" + "="*100)
    print("【无止损更优的策略 Top 10】")
    print("="*100)
    
    df_no_sl_better = df[df['change'] < -0.1].sort_values('change', ascending=True).head(10)
    print(f"\n{'排名':<4} {'策略组合':<45} {'持有':<6} {'无止损':<10} {'有止损':<10} {'收益变化'}")
    print("-"*90)
    
    for i, (_, r) in enumerate(df_no_sl_better.iterrows(), 1):
        strategy = f"{r['j']}+{r['struct']}+{r['pos']}+{r['ema']}"
        print(f"{i:<4} {strategy:<45} {r['holding']}日{'':<2} {r['avg_no_sl']:.2f}%{'':<5} {r['avg_with_sl']:.2f}%{'':<5} {r['change']:+.2f}%")
    
    print("\n" + "="*100)
    print("【总结统计】")
    print("="*100)
    
    sl_better_count = (df['change'] > 0.1).sum()
    no_sl_better_count = (df['change'] < -0.1).sum()
    equal_count = len(df) - sl_better_count - no_sl_better_count
    
    print(f"\n总策略数: {len(df)}")
    print(f"止损更优 (收益提升>0.1%): {sl_better_count} 个 ({sl_better_count/len(df)*100:.1f}%)")
    print(f"无止损更优 (收益下降>0.1%): {no_sl_better_count} 个 ({no_sl_better_count/len(df)*100:.1f}%)")
    print(f"基本持平 (差异<0.1%): {equal_count} 个 ({equal_count/len(df)*100:.1f}%)")
    
    best_no_sl = df_sorted.iloc[0]
    print(f"\n无止损最优策略: {best_no_sl['j']}+{best_no_sl['struct']}+{best_no_sl['pos']}+{best_no_sl['ema']}")
    print(f"  持有期: {best_no_sl['holding']}日, 收益: {best_no_sl['avg_no_sl']:.2f}%, 胜率: {best_no_sl['win_rate_no_sl']:.1f}%")


if __name__ == "__main__":
    main()
