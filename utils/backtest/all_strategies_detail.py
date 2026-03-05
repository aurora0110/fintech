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


def run_strategy(stock_data, j_thresh, struct_name, pos_name, ema_name, holding_days):
    all_returns = []
    all_drawdowns = []
    
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
            
            entry_open = open_prices[idx + 1]
            
            if pd.isna(entry_open) or entry_open <= 0:
                continue
            
            min_during_hold = float('inf')
            for day in range(1, holding_days + 1):
                if idx + day + 1 >= len(df):
                    break
                day_close = close[idx + day]
                if pd.isna(day_close):
                    continue
                drawdown = (day_close - entry_open) / entry_open * 100
                if drawdown < min_during_hold:
                    min_during_hold = drawdown
            
            exit_open = open_prices[idx + holding_days + 1]
            if pd.isna(exit_open) or exit_open <= 0:
                continue
            
            ret = (exit_open - entry_open) / entry_open * 100
            all_returns.append(ret)
            
            if min_during_hold != float('inf'):
                all_drawdowns.append(min_during_hold)
            else:
                all_drawdowns.append(ret)
    
    return all_returns, all_drawdowns


def main():
    print("="*100)
    print("各持有期最优策略详细信息")
    print("="*100)
    
    print("\n加载股票数据...")
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
                        returns, drawdowns = run_strategy(stock_data, j_thresh, struct, pos, ema, holding)
                        
                        if len(returns) >= 50:
                            arr = np.array(returns)
                            dd_arr = np.array(drawdowns)
                            
                            results.append({
                                'j': j_thresh,
                                'struct': struct,
                                'pos': pos,
                                'ema': ema,
                                'holding': holding,
                                'count': len(arr),
                                'avg_return': arr.mean(),
                                'median_return': np.median(arr),
                                'win_rate': (arr > 0).mean() * 100,
                                'max_return': arr.max(),
                                'min_return': arr.min(),
                                'std_return': arr.std(),
                                'max_drawdown': dd_arr.min(),
                                'avg_drawdown': dd_arr.mean(),
                            })
    
    df = pd.DataFrame(results)
    
    struct_desc = {
        'G底部反转': 'MA5 > MA20，且 MA20 < MA60（底部反转形态）',
        'B短期多头': 'MA5 > MA20 > MA30（短期多头排列）',
        'D趋势启动': 'MA5 > MA20 > MA30（上升趋势启动）'
    }
    
    print("\n" + "="*100)
    print("【各持有期 Top 3 汇总表】")
    print("="*100)
    
    print(f"\n{'持有期':<8} {'排名':<6} {'策略':<45} {'信号数':<8} {'收益':<10} {'胜率':<8} {'最大回撤'}")
    print("-"*90)
    
    for holding in holdings:
        df_h = df[df['holding'] == holding].sort_values('avg_return', ascending=False).head(3)
        
        for i, (_, r) in enumerate(df_h.iterrows(), 1):
            strategy = f"J<-{r['j']}+{r['struct']}+{r['pos']}+{r['ema']}"
            print(f"{holding}日{'':<2} {i:<6} {strategy:<45} {r['count']:<8} {r['avg_return']:.2f}%{'':<5} {r['win_rate']:.1f}%{'':<3} {r['max_drawdown']:.2f}%")
    
    print("\n" + "="*100)
    print("【策略选择建议】")
    print("="*100)
    print("""
┌──────────┬─────────────────────────────┬──────────┬──────────┬────────────┐
│ 持有期   │ 最优策略                    │ 平均收益  │ 胜率     │ 最大回撤   │
├──────────┼─────────────────────────────┼──────────┼──────────┼────────────┤""")
    
    for holding in holdings:
        best = df[df['holding'] == holding].sort_values('avg_return', ascending=False).iloc[0]
        strategy = f"J<-{best['j']}+{best['struct']}+{best['pos']}+{best['ema']}"
        print(f"│ {holding}日    │ {strategy:<27} │ {best['avg_return']:>6.2f}% │ {best['win_rate']:>6.1f}% │ {best['max_drawdown']:>8.2f}% │")
    
    print("└──────────┴─────────────────────────────┴──────────┴──────────┴────────────┘")


if __name__ == "__main__":
    main()
