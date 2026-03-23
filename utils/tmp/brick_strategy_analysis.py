"""
砖型图策略时序稳定性分析
- 滚动窗口胜率分析
- 游程检验
- 马尔可夫转移概率分析
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/data")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/strategy_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLD_DAYS = 3
TARGET_RETURN = 0.035
STOP_LOSS = 0.99
ROLLING_WINDOW = 20
STEP = 1
MIN_BARS = 160


def safe_div(a, b, default=np.nan):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.shape(a_arr), default, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (np.abs(b_arr) > 1e-12)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    return series.ewm(alpha=m / n, adjust=False).mean()


def calc_green_streak(green_flag: np.ndarray) -> np.ndarray:
    out = np.zeros(len(green_flag), dtype=np.int32)
    for i in range(1, len(green_flag)):
        out[i] = out[i - 1] + 1 if green_flag[i] else 0
    return out


def load_stock_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < MIN_BARS + 1:
            return None
        data_lines = lines[1:]
        records = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    records.append({
                        'date': parts[0],
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]),
                    })
                except ValueError:
                    continue
        if not records:
            return None
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < MIN_BARS:
            return None
        return df
    except Exception:
        return None


def compute_brick_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['trend_line'] = df['close'].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    
    df['ma14'] = df['close'].rolling(14).mean()
    df['ma28'] = df['close'].rolling(28).mean()
    df['ma57'] = df['close'].rolling(57).mean()
    df['ma114'] = df['close'].rolling(114).mean()
    df['long_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4.0
    
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    hhv4 = df['high'].rolling(4).max()
    llv4 = df['low'].rolling(4).min()
    den4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = safe_div((hhv4 - df['close']), den4) * 100 - 90
    var2a = tdx_sma(pd.Series(var1a, index=df.index), 4, 1) + 100
    var3a = safe_div((df['close'] - llv4), den4) * 100
    var4a = tdx_sma(pd.Series(var3a, index=df.index), 6, 1)
    var5a = tdx_sma(var4a, 6, 1) + 100
    var6a = var5a - var2a
    df['brick'] = np.where(var6a > 4, var6a - 4, 0.0)
    df['brick_prev'] = df['brick'].shift(1)
    df['brick_red_len'] = np.where(df['brick'] > df['brick_prev'], df['brick'] - df['brick_prev'], 0.0)
    df['brick_green_len'] = np.where(df['brick'] < df['brick_prev'], df['brick_prev'] - df['brick'], 0.0)
    df['brick_red'] = df['brick_red_len'] > 0
    df['brick_green'] = df['brick_green_len'] > 0
    df['prev_green_streak'] = pd.Series(calc_green_streak(df['brick_green'].to_numpy()), index=df.index).shift(1)
    
    df['vol_ma5_prev'] = df['volume'].shift(1).rolling(5).mean()
    df['signal_vs_ma5'] = safe_div(df['volume'], df['vol_ma5_prev'])
    df['signal_vs_ma5_valid'] = df['signal_vs_ma5'].between(1, 2.2, inclusive="both")
    
    df['close_slope_10'] = df['close'].rolling(10).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if np.isfinite(s).all() else np.nan, raw=False
    )
    df['not_sideways'] = np.abs(safe_div(df['close_slope_10'], df['close'].rolling(10).mean())) > 0.002
    
    df['up_leg_avg_vol'] = df['volume'].shift(4).rolling(3).mean()
    df['pullback_avg_vol'] = df['volume'].shift(1).rolling(3).mean()
    df['pullback_shrinking'] = df['pullback_avg_vol'] < df['up_leg_avg_vol']
    
    df['close_pullback_white'] = df['close'] < df['trend_line'] * 1.01
    df['close_above_white'] = df['close'] > df['trend_line']
    
    df['rebound_ratio'] = safe_div(df['brick_red_len'], df['brick_green_len'].shift(1))
    
    return df


def detect_pattern_a(df: pd.DataFrame) -> pd.Series:
    return (
        (df['prev_green_streak'] >= 3)
        & df['brick_red']
        & df['close_pullback_white'].shift(1).fillna(False)
        & df['close_above_white']
    )


def detect_pattern_b(df: pd.DataFrame) -> pd.Series:
    green_streak = pd.Series(calc_green_streak(df['brick_green'].to_numpy()), index=df.index)
    return (
        (green_streak.shift(3) >= 3)
        & df['brick_red']
        & df['brick_green'].shift(1).fillna(False)
        & df['brick_red'].shift(2).fillna(False)
        & df['close_pullback_white'].shift(1).fillna(False)
        & df['close_above_white']
    )


def compute_trade_result(df: pd.DataFrame, signal_idx: int, hold_days: int) -> Optional[Dict]:
    if signal_idx + hold_days >= len(df):
        return None
    
    entry_price = df.iloc[signal_idx]['close']
    entry_low = df.iloc[signal_idx]['low']
    stop_loss_price = entry_low * STOP_LOSS
    
    exit_prices = df.iloc[signal_idx+1:signal_idx+hold_days+1]
    
    max_high = exit_prices['high'].max()
    min_close = exit_prices['close'].min()
    
    if max_high >= entry_price * (1 + TARGET_RETURN):
        return {'result': 'success', 'return': TARGET_RETURN, 'exit_type': '止盈'}
    elif min_close <= stop_loss_price:
        return {'result': 'failure', 'return': (stop_loss_price - entry_price) / entry_price, 'exit_type': '止损'}
    else:
        holding_return = (exit_prices.iloc[-1]['close'] - entry_price) / entry_price
        return {'result': 'hold', 'return': holding_return, 'exit_type': '持有'}


def process_stock(file_path: str) -> List[Dict]:
    df = load_stock_data(file_path)
    if df is None:
        return []
    
    df = compute_brick_features(df)
    
    pattern_a = detect_pattern_a(df)
    pattern_b = detect_pattern_b(df)
    signal = (pattern_a | pattern_b) & df['pullback_shrinking'].fillna(False) & df['signal_vs_ma5_valid'].fillna(False) & df['not_sideways'].fillna(False)
    
    results = []
    for idx in range(MIN_BARS, len(df) - HOLD_DAYS):
        if signal.iloc[idx]:
            trade = compute_trade_result(df, idx, HOLD_DAYS)
            if trade:
                results.append({
                    'date': df.iloc[idx]['date'],
                    'signal_idx': idx,
                    **trade
                })
    
    return results


def load_all_trades() -> pd.DataFrame:
    all_dates = []
    for date_dir in sorted(DATA_DIR.glob("20*")):
        normal_dir = date_dir / "normal"
        if normal_dir.exists():
            all_dates.append(date_dir.name)
    
    if not all_dates:
        print("未找到数据目录")
        return pd.DataFrame()
    
    date_str = all_dates[-1]
    normal_dir = DATA_DIR / date_str / "normal"
    
    file_paths = list(normal_dir.glob("*.txt"))
    print(f"加载 {len(file_paths)} 个股票文件...")
    
    all_trades = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for trades in executor.map(process_stock, [str(fp) for fp in file_paths], chunksize=20):
            all_trades.extend(trades)
    
    if not all_trades:
        print("未找到任何交易信号")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    df = df.sort_values('date').reset_index(drop=True)
    print(f"共找到 {len(df)} 个交易信号")
    
    return df


def rolling_win_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['win'] = (df['result'] == 'success').astype(int)
    df['loss'] = (df['result'] == 'failure').astype(int)
    
    rolling_results = []
    unique_dates = df['date'].unique()
    
    for i in range(0, len(unique_dates) - ROLLING_WINDOW + 1, STEP):
        window_dates = unique_dates[i:i+ROLLING_WINDOW]
        window_df = df[df['date'].isin(window_dates)]
        
        if len(window_df) < 3:
            continue
        
        win_count = window_df['win'].sum()
        loss_count = window_df['loss'].sum()
        settled_count = win_count + loss_count
        
        if settled_count > 0:
            win_rate = win_count / settled_count
        else:
            win_rate = np.nan
        
        avg_return = window_df['return'].mean()
        
        rolling_results.append({
            'window_start': window_dates[0],
            'window_end': window_dates[-1],
            'trade_count': len(window_df),
            'win_count': win_count,
            'loss_count': loss_count,
            'settled_count': settled_count,
            'win_rate': win_rate,
            'avg_return': avg_return
        })
    
    return pd.DataFrame(rolling_results)


def runs_test(binary_sequence: np.ndarray) -> Dict:
    n = len(binary_sequence)
    if n < 10:
        return {'p_value': np.nan, 'z_stat': np.nan, 'significant': False}
    
    n1 = np.sum(binary_sequence)
    n2 = n - n1
    
    if n1 == 0 or n2 == 0:
        return {'p_value': np.nan, 'z_stat': np.nan, 'significant': False}
    
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    mean_r = (2 * n1 * n2) / (n1 + n2) + 1
    var_r = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    if var_r > 0:
        z_stat = (runs - mean_r) / np.sqrt(var_r)
    else:
        z_stat = 0
    
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'n': n,
        'n1': n1,
        'n2': n2,
        'runs': runs,
        'mean_r': mean_r,
        'z_stat': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def markov_analysis(df: pd.DataFrame) -> Dict:
    df = df.copy()
    df['win'] = (df['result'] == 'success').astype(int)
    df['loss'] = (df['result'] == 'failure').astype(int)
    df['outcome'] = np.where(df['win'] == 1, 1, 0)
    
    states = df['outcome'].values
    n = len(states)
    
    transition_matrix = np.zeros((2, 2))
    
    for i in range(n - 1):
        current_state = int(states[i])
        next_state = int(states[i + 1])
        transition_matrix[current_state, next_state] += 1
    
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_prob = transition_matrix / row_sums
    
    return {
        'transition_count': transition_matrix.tolist(),
        'transition_prob': transition_prob.tolist(),
        'P_win_given_win': transition_prob[1, 1] if transition_prob[1, 1] > 0 else np.nan,
        'P_win_given_loss': transition_prob[0, 1] if transition_prob[0, 1] > 0 else np.nan,
        'P_loss_given_win': transition_prob[1, 0] if transition_prob[1, 0] > 0 else np.nan,
        'P_loss_given_loss': transition_prob[0, 0] if transition_prob[0, 0] > 0 else np.nan,
    }


def generate_report(rolling_df: pd.DataFrame, all_trades: pd.DataFrame, markov_result: Dict, runs_result: Dict) -> str:
    report = []
    report.append("=" * 80)
    report.append("砖型图策略时序稳定性分析报告")
    report.append("=" * 80)
    report.append(f"\n分析参数:")
    report.append(f"  持有天数: {HOLD_DAYS}天")
    report.append(f"  止盈阈值: {TARGET_RETURN*100}%")
    report.append(f"  止损阈值: {(1-STOP_LOSS)*100}%")
    report.append(f"  滚动窗口: {ROLLING_WINDOW}个交易日")
    report.append(f"  步长: {STEP}个交易日")
    
    report.append(f"\n总体统计:")
    report.append(f"  总交易信号数: {len(all_trades)}")
    
    win_count = len(all_trades[all_trades['result']=='success'])
    loss_count = len(all_trades[all_trades['result']=='failure'])
    hold_count = len(all_trades[all_trades['result']=='hold'])
    settled_count = win_count + loss_count
    
    report.append(f"  止盈次数: {win_count}")
    report.append(f"  止损次数: {loss_count}")
    report.append(f"  持有到期次数: {hold_count}")
    report.append(f"  已结算交易数: {settled_count}")
    
    if settled_count > 0:
        overall_win_rate = win_count / settled_count
        report.append(f"  总体胜率 (止盈/已结算): {overall_win_rate*100:.2f}%")
        report.append(f"  平均收益: {all_trades['return'].mean()*100:.2f}%")
    
    report.append(f"\n滚动窗口分析:")
    if len(rolling_df) > 0:
        report.append(f"  有效窗口数: {len(rolling_df)}")
        report.append(f"  平均胜率: {rolling_df['win_rate'].mean()*100:.2f}%")
        report.append(f"  胜率标准差: {rolling_df['win_rate'].std()*100:.2f}%")
        report.append(f"  最高胜率: {rolling_df['win_rate'].max()*100:.2f}%")
        report.append(f"  最低胜率: {rolling_df['win_rate'].min()*100:.2f}%")
        
        high_win_periods = rolling_df[rolling_df['win_rate'] >= 0.6]
        low_win_periods = rolling_df[rolling_df['win_rate'] <= 0.4]
        
        report.append(f"\n  高胜率时段 (>=60%): {len(high_win_periods)}个窗口")
        if len(high_win_periods) > 0:
            report.append(f"    时间范围: {high_win_periods['window_start'].min()} ~ {high_win_periods['window_end'].max()}")
        
        report.append(f"  低胜率时段 (<=40%): {len(low_win_periods)}个窗口")
        if len(low_win_periods) > 0:
            report.append(f"    时间范围: {low_win_periods['window_start'].min()} ~ {low_win_periods['window_end'].max()}")
    
    report.append(f"\n游程检验 (Runs Test):")
    if not np.isnan(runs_result.get('p_value', np.nan)):
        report.append(f"  序列长度: {runs_result['n']}")
        report.append(f"  盈利次数: {runs_result['n1']}")
        report.append(f"  亏损次数: {runs_result['n2']}")
        report.append(f"  游程数: {runs_result['runs']}")
        report.append(f"  期望游程数: {runs_result['mean_r']:.2f}")
        report.append(f"  Z统计量: {runs_result['z_stat']:.4f}")
        report.append(f"  P值: {runs_result['p_value']:.4f}")
        report.append(f"  显著性 (p<0.05): {'是 - 存在序列依赖性' if runs_result['significant'] else '否 - 符合随机序列'}")
    else:
        report.append(f"  数据不足，无法进行游程检验")
    
    report.append(f"\n马尔可夫转移概率分析:")
    report.append(f"  P(盈利|盈利) = {markov_result.get('P_win_given_win', 'N/A'):.4f}")
    report.append(f"  P(盈利|亏损) = {markov_result.get('P_win_given_loss', 'N/A'):.4f}")
    report.append(f"  P(亏损|盈利) = {markov_result.get('P_loss_given_win', 'N/A'):.4f}")
    report.append(f"  P(亏损|亏损) = {markov_result.get('P_loss_given_loss', 'N/A'):.4f}")
    
    if not np.isnan(markov_result.get('P_win_given_win', np.nan)):
        if markov_result['P_win_given_win'] > markov_result['P_win_given_loss']:
            report.append(f"  解读: 盈利后更容易继续盈利 (趋势持续)")
        else:
            report.append(f"  解读: 亏损后更容易转为盈利 (均值回归)")
    
    report.append("\n" + "=" * 80)
    report.append("结论:")
    if len(rolling_df) > 0:
        win_rate_cv = rolling_df['win_rate'].std() / rolling_df['win_rate'].mean() if rolling_df['win_rate'].mean() > 0 else 0
        if win_rate_cv > 0.3:
            report.append("  - 胜率波动较大，策略在不同市场状态下表现不稳定")
        else:
            report.append("  - 胜率相对稳定，策略具有较好的一致性")
    
    if not np.isnan(runs_result.get('p_value', np.nan)):
        if runs_result['significant']:
            report.append("  - 存在显著的序列依赖性，策略效果受市场环境影响")
        else:
            report.append("  - 未检测到显著的序列依赖性，策略相对稳定")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    print("开始加载数据...")
    all_trades = load_all_trades()
    
    if all_trades.empty:
        print("没有交易数据可分析")
        return
    
    print("\n进行滚动窗口分析...")
    rolling_df = rolling_win_analysis(all_trades)
    
    print("进行游程检验...")
    settled_trades = all_trades[all_trades['result'].isin(['success', 'failure'])]
    binary_sequence = (settled_trades['result'] == 'success').astype(int).values
    runs_result = runs_test(binary_sequence)
    
    print("进行马尔可夫分析...")
    markov_result = markov_analysis(settled_trades)
    
    report = generate_report(rolling_df, all_trades, markov_result, runs_result)
    print(report)
    
    report_file = OUTPUT_DIR / f"brick_strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {report_file}")
    
    rolling_file = OUTPUT_DIR / "rolling_window_results.csv"
    rolling_df.to_csv(rolling_file, index=False, encoding='utf-8-sig')
    print(f"滚动窗口数据已保存到: {rolling_file}")


if __name__ == "__main__":
    main()
