import os
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "/Users/lidongyang/Desktop/Qstrategy/data/forward_data"

VOLUME_WINDOWS = [30, 60, 120, 240]
FUTURE_DAYS = [1, 3, 5, 10]

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

def calculate_volume_percentile(df, windows):
    for w in windows:
        df[f'vol_percentile{w}'] = df['成交量'].rolling(w, min_periods=w).apply(
            lambda x: (x[-1] < x).sum() / len(x) if len(x) == w else np.nan,
            raw=True
        )
    return df

def is_bullish_increase_volume(row, prev_row):
    if prev_row is None or pd.isna(prev_row.get('成交量')):
        return False
    is_bullish = row['收盘'] > row['开盘']
    vol_increase = row['成交量'] > prev_row['成交量'] * 1.5
    return is_bullish and vol_increase

def is_bearish_increase_volume(row, prev_row):
    if prev_row is None or pd.isna(prev_row.get('成交量')):
        return False
    is_bearish = row['收盘'] < row['开盘']
    vol_increase = row['成交量'] > prev_row['成交量'] * 1.3
    return is_bearish and vol_increase

def analyze_volume_percentile_drop_prob(data_dict, signal_type='bullish'):
    results = {w: defaultdict(lambda: {'total': 0, 'drop': 0, 'returns': []}) for w in VOLUME_WINDOWS}

    signal_check = is_bullish_increase_volume if signal_type == 'bullish' else is_bearish_increase_volume

    for stock, df in data_dict.items():
        df = calculate_volume_percentile(df, VOLUME_WINDOWS)

        for i in range(60, len(df) - 10):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if not signal_check(row, prev_row):
                continue

            for w in VOLUME_WINDOWS:
                pct = row.get(f'vol_percentile{w}')
                if pd.isna(pct) or pct < 0.3:
                    continue

                bucket = round(pct * 10) / 10
                if bucket > 0.9:
                    bucket = 0.9

                for fd in FUTURE_DAYS:
                    if i + fd < len(df):
                        future_price = df.iloc[i + fd]['收盘']
                        current_price = row['收盘']
                        if current_price <= 0 or pd.isna(future_price):
                            continue
                        ret = (future_price - current_price) / current_price

                        key = f'future_{fd}d_bucket_{bucket}'
                        results[w][key]['total'] += 1
                        results[w][key]['returns'].append(ret)
                        if ret < 0:
                            results[w][key]['drop'] += 1

    return results

def print_results(results, signal_name):
    print("\n" + "=" * 80)
    print(f"{signal_name}后，后续下跌概率分析")
    print("(下跌定义: 收盘价 < 前一日收盘价)")
    print("=" * 80)

    summary = []

    for w in VOLUME_WINDOWS:
        print(f"\n{'='*40}")
        print(f"成交量分位数统计周期: {w}日")
        print(f"{'='*40}")

        for fd in FUTURE_DAYS:
            print(f"\n--- 未来{fd}日 ---")

            buckets = []
            for key, val in results[w].items():
                if f'future_{fd}d' in key:
                    total = val['total']
                    drop = val['drop']
                    if total >= 10:
                        pct = drop / total * 100
                        avg_ret = np.mean(val['returns']) * 100
                        bucket = key.split('_')[-1]
                        buckets.append((bucket, total, drop, pct, avg_ret))

            buckets.sort(key=lambda x: x[3], reverse=True)

            print(f"{'分位区间':<12} {'样本数':<8} {'下跌数':<8} {'下跌概率':<12} {'平均收益':<10}")
            print("-" * 55)
            for b, t, d, prob, ret in buckets:
                print(f"{b:<12} {t:<8} {d:<8} {prob:>8.2f}%    {ret:>8.2f}%")

            if buckets:
                worst = buckets[0]
                print(f"\n结论: 当成交量处于{w}日分位数为{worst[0]}时，未来{fd}日下跌概率最高({worst[3]:.2f}%)")
                summary.append((w, fd, worst[0], worst[3], worst[4]))

    return summary

def main():
    print("加载数据...")
    data_dict = load_all_data(DATA_DIR)
    print(f"加载了 {len(data_dict)} 只股票")

    print("\n" + "="*80)
    print("分析1: 放量阳线")
    print("="*80)
    results_bullish = analyze_volume_percentile_drop_prob(data_dict, 'bullish')
    summary_bullish = print_results(results_bullish, "放量阳线")

    print("\n" + "="*80)
    print("分析2: 放量阴线")
    print("="*80)
    results_bearish = analyze_volume_percentile_drop_prob(data_dict, 'bearish')
    summary_bearish = print_results(results_bearish, "放量阴线")

    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    print(f"\n{'周期':<8} {'未来天数':<10} {'放量阳线最优分位':<16} {'下跌概率':<12} {'放量阴线最优分位':<16} {'下跌概率':<12}")
    print("-" * 80)

    for i in range(len(summary_bullish)):
        w1, fd1, b1, p1, r1 = summary_bullish[i]
        w2, fd2, b2, p2, r2 = summary_bearish[i]
        print(f"{w1}日     {fd1}日       {b1:<16} {p1:>8.2f}%     {b2:<16} {p2:>8.2f}%")

if __name__ == "__main__":
    main()
