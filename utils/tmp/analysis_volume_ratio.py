import os
import pandas as pd
import numpy as np

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

def load_all_data(data_dir):
    data_dict = {}
    for file in os.listdir(data_dir):
        if not file.endswith(".txt"):
            continue
        stock_code = file.replace(".txt", "")
        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(path, sep='\t', encoding='utf-8')
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            df['stock'] = stock_code
            if len(df) >= 60:
                data_dict[stock_code] = df
        except:
            continue
    return data_dict

def main():
    print("加载数据...")
    data_dict = load_all_data(DATA_DIR)
    print(f"加载了 {len(data_dict)} 只股票")
    
    print("\n" + "="*90)
    print("放量 + 30日/60日最高量倍数测试 (下跌定义: 收盘价 < 前一日收盘价)")
    print("="*90)
    
    results_summary = []
    
    for is_bullish in [True, False]:
        signal_name = "放量阳线" if is_bullish else "放量阴线"
        print(f"\n{'='*45}")
        print(f"{signal_name}")
        print(f"{'='*45}")
        
        for ratio in [1.0, 1.1, 1.2, 1.3, 1.4]:
            print(f"\n--- 放量倍数: {ratio}x ---")
            
            for w in [30, 60]:
                total_count = 0
                drop_count = 0
                returns_list = []
                
                for stock, df in data_dict.items():
                    df = df.copy()
                    df['vol_max'] = df['成交量'].rolling(w, min_periods=w).max()
                    
                    for i in range(w, len(df) - 10):
                        row = df.iloc[i]
                        prev_row = df.iloc[i - 1]
                        
                        if prev_row is None or pd.isna(prev_row.get('成交量')):
                            continue
                        
                        if is_bullish:
                            is_signal = row['收盘'] > row['开盘']
                        else:
                            is_signal = row['收盘'] < row['开盘']
                        
                        vol_increase = row['成交量'] > prev_row['成交量'] * ratio
                        is_max = row['成交量'] >= row['vol_max'] * 0.99
                        
                        if not (is_signal and vol_increase and is_max):
                            continue
                        
                        for fd in [1, 3, 5, 10]:
                            if i + fd < len(df):
                                future_price = df.iloc[i + fd]['收盘']
                                current_price = row['收盘']
                                if current_price <= 0 or pd.isna(future_price):
                                    continue
                                ret = (future_price - current_price) / current_price
                                total_count += 1
                                returns_list.append(ret)
                                if ret < 0:
                                    drop_count += 1
                
                if total_count > 0:
                    prob = drop_count / total_count * 100
                    avg_ret = np.mean(returns_list) * 100
                    print(f"  {w}日最高量: 样本={total_count}, 下跌={drop_count}, 概率={prob:.2f}%, 均收益={avg_ret:.2f}%")
                    results_summary.append({
                        'signal': signal_name,
                        'ratio': ratio,
                        'window': w,
                        'total': total_count,
                        'drop': drop_count,
                        'prob': prob,
                        'avg_ret': avg_ret
                    })
    
    print("\n" + "="*90)
    print("对比总结")
    print("="*90)
    
    print(f"\n{'放量倍数':<10} {'周期':<8} {'放量阳线下跌概率':<18} {'放量阴线下跌概率':<18}")
    print("-" * 60)
    
    for ratio in [1.0, 1.1, 1.2, 1.3, 1.4]:
        for w in [30, 60]:
            b = [r for r in results_summary if r['ratio']==ratio and r['window']==w and r['signal']=='放量阳线']
            d = [r for r in results_summary if r['ratio']==ratio and r['window']==w and r['signal']=='放量阴线']
            bp = b[0]['prob'] if b else 0
            dp = d[0]['prob'] if d else 0
            print(f"{ratio}x       {w}日     {bp:>12.2f}%         {dp:>12.2f}%")

if __name__ == "__main__":
    main()
