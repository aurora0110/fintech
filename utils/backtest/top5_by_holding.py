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
    print("各持有期 Top 5 策略详情")
    print("="*100)
    print("""
入场: 信号次日开盘价买入
出场: 固定持有期后次日开盘卖出
止损价(可选): min(信号日最低价, 买入日最低价) * 0.95
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
                                arr_no_sl = np.array(returns_no_sl)
                                arr_with_sl = np.array(returns_with_sl)
                                
                                results.append({
                                    'j': j_thresh,
                                    'struct': struct,
                                    'pos': pos,
                                    'ema': ema,
                                    'holding': holding,
                                    'count': len(arr_no_sl),
                                    'avg_no_sl': arr_no_sl.mean(),
                                    'median_no_sl': np.median(arr_no_sl),
                                    'win_rate_no_sl': (arr_no_sl > 0).mean() * 100,
                                    'max_ret_no_sl': arr_no_sl.max(),
                                    'min_ret_no_sl': arr_no_sl.min(),
                                    'std_no_sl': arr_no_sl.std(),
                                    'avg_with_sl': arr_with_sl.mean(),
                                    'win_rate_with_sl': (arr_with_sl > 0).mean() * 100,
                                    'sl_count': sl_count,
                                    'change': arr_with_sl.mean() - arr_no_sl.mean(),
                                })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("【各持有期 Top 5 策略详细信息】")
    print("="*100)
    
    for holding in holdings:
        print(f"\n{'='*100}")
        print(f"【持有期: {holding}日】")
        print("="*100)
        
        df_holding = df[df['holding'] == holding].sort_values('avg_no_sl', ascending=False).head(5)
        
        print(f"\n{'排名':<4} {'J值':<8} {'结构':<12} {'位置':<8} {'EMA':<6} {'信号数':<8} {'平均收益':<10} {'胜率':<8} {'中位数':<10} {'最大收益':<10} {'最大亏损'}")
        print("-"*100)
        
        for i, (_, r) in enumerate(df_holding.iterrows(), 1):
            print(f"{i:<4} J<-{r['j']:<5} {r['struct']:<12} {r['pos']:<8} {r['ema']:<6} {r['count']:<8} {r['avg_no_sl']:.2f}%{'':<5} {r['win_rate_no_sl']:.1f}%{'':<4} {r['median_no_sl']:.2f}%{'':<5} {r['max_ret_no_sl']:.2f}%{'':<5} {r['min_ret_no_sl']:.2f}%")
        
        print(f"\n{'第i名':<6} {'策略描述'}")
        print("-"*100)
        
        for i, (_, r) in enumerate(df_holding.iterrows(), 1):
            struct_desc = {
                'G底部反转': 'MA5>MA20，且MA20<MA60（底部反转形态）',
                'B短期多头': 'MA5>MA20>MA30（短期多头排列）',
                'D趋势启动': 'MA5>MA20>MA30，且MA20>MA30（上升趋势启动）'
            }
            
            print(f"\n第{i}名: J<-{r['j']} + {r['struct']} + {r['pos']} + {r['ema']}")
            print(f"  入场条件:")
            print(f"    - J值 < -{r['j']} (超卖)")
            print(f"    - 结构: {struct_desc.get(r['struct'], r['struct'])}")
            if r['pos'] == '站MA20':
                print(f"    - 收盘价 > MA20")
            if r['ema'] == '有EMA':
                print(f"    - EMA5 > EMA10")
            print(f"  持有期: {r['holding']}日后次日开盘卖出")
            print(f"  回测结果: 信号数={r['count']}, 平均收益={r['avg_no_sl']:.2f}%, 胜率={r['win_rate_no_sl']:.1f}%")
            print(f"           中位数={r['median_no_sl']:.2f}%, 最大收益={r['max_ret_no_sl']:.2f}%, 最大亏损={r['min_ret_no_sl']:.2f}%")
            print(f"           标准差={r['std_no_sl']:.2f}%")
            print(f"  加止损后: 收益={r['avg_with_sl']:.2f}%, 胜率={r['win_rate_with_sl']:.1f}%, 止损触发={r['sl_count']}次")
            if r['change'] > 0.1:
                print(f"  结论: ✓ 加止损更优 (收益+{r['change']:.2f}%)")
            elif r['change'] < -0.1:
                print(f"  结论: ✗ 无止损更优 (收益-{abs(r['change']):.2f}%)")
            else:
                print(f"  结论: ≈ 持平")
    
    print("\n" + "="*100)
    print("【总结: 各持有期最优策略汇总】")
    print("="*100)
    
    print(f"\n{'持有期':<8} {'最优策略':<50} {'收益':<10} {'胜率'}")
    print("-"*80)
    
    for holding in holdings:
        best = df[df['holding'] == holding].sort_values('avg_no_sl', ascending=False).iloc[0]
        strategy = f"J<-{best['j']}+{best['struct']}+{best['pos']}+{best['ema']}"
        print(f"{holding}日{'':<3} {strategy:<50} {best['avg_no_sl']:.2f}%{'':<5} {best['win_rate_no_sl']:.1f}%")


if __name__ == "__main__":
    main()
