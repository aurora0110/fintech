from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/Users/lidongyang/Desktop/Qstrategy')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tmp.run_b2_type14_exit_and_param_opt import (  # type: ignore
    add_base_features,
    load_one_csv,
    DATA_DIR,
    EXCLUDE_START,
    EXCLUDE_END,
    TRADING_DAYS_PER_YEAR,
)
from utils.tmp.run_b2_type14_param_search_cached import (  # type: ignore
    select_type1,
    select_type4,
)
from utils.tmp.run_b2_type14_split_account_opt_20260314 import (  # type: ignore
    simulate_portfolio,
    AccountConfig,
)

RESULT_DIR = ROOT / 'results/b2_distribution_exit_ab_20260314'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TYPE1_PARAMS = {
    'ret1_min': 0.04,
    'upper_shadow_body_ratio': 0.40,
    'j_max': 90.0,
    'type1_near_ratio': 1.02,
    'type1_j_rank20_max': 0.10,
}
TYPE4_PARAMS = {
    'ret1_min': 0.03,
    'upper_shadow_body_ratio': 0.80,
    'j_max': 100.0,
    'type4_touch_ratio': 1.01,
}

TYPE1_ACCOUNT = AccountConfig(
    name='pos5_new3_b100_cap20_equal',
    max_positions=5,
    daily_new_limit=3,
    daily_budget_frac=1.0,
    position_cap_frac=0.2,
    allocation_mode='equal',
)
TYPE4_ACCOUNT = TYPE1_ACCOUNT

TYPE1_CANDIDATES_PATH = ROOT / 'results/b2_type14_param_search_cached_20260313/type1_candidates.csv'
TYPE4_CANDIDATES_PATH = ROOT / 'results/b2_type14_param_search_cached_20260313/type4_candidates.csv'


@dataclass(frozen=True)
class ExitVariant:
    name: str
    max_hold_days: int
    take_profit: float | None = None
    use_distribution_exit: bool = False


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def load_all_data() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob('*.txt'))
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        raw = load_one_csv(path)
        if raw is None:
            continue
        x = add_base_features(raw)
        x = add_distribution_features(x)
        x = x[(x['date'] < EXCLUDE_START) | (x['date'] > EXCLUDE_END)].reset_index(drop=True)
        if len(x) >= 150:
            dfs[path.stem] = x
        if idx % 500 == 0:
            print(f'数据加载进度: {idx}/{total}')
    return dfs


def add_distribution_features(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x['vol_ma20'] = x['volume'].rolling(20).mean()
    x['vol_rank30'] = x['volume'].rolling(30, min_periods=30).apply(lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False)
    x['vol_rank60'] = x['volume'].rolling(60, min_periods=60).apply(lambda v: pd.Series(v).rank(pct=True).iloc[-1], raw=False)
    x['high60_prev'] = x['high'].shift(1).rolling(60).max()
    x['high30_prev'] = x['high'].shift(1).rolling(30).max()
    x['close60_prev'] = x['close'].shift(1).rolling(60).max()
    x['low20_prev'] = x['low'].shift(1).rolling(20).min()
    x['upper_range_ratio'] = _safe_div(x['upper_shadow'], (x['high'] - x['low']).replace(0.0, np.nan))
    x['ret5'] = x['close'] / x['close'].shift(5) - 1.0
    x['ret10'] = x['close'] / x['close'].shift(10) - 1.0
    x['is_bear'] = x['close'] < x['open']
    x['is_bull'] = x['close'] > x['open']
    x['bear_vol'] = np.where(x['is_bear'], x['volume'], 0.0)
    x['bull_vol'] = np.where(x['is_bull'], x['volume'], 0.0)
    x['bear_vol_sum8'] = pd.Series(x['bear_vol']).rolling(8).sum()
    x['bull_vol_sum8'] = pd.Series(x['bull_vol']).rolling(8).sum()
    x['bear_days_8'] = pd.Series(x['is_bear'].astype(int)).rolling(8).sum()
    x['new_high_5d'] = x['high'].rolling(5).max() >= x['high60_prev'] * 0.995

    accel_top_heavy_bear = (
        x['is_bear']
        & (x['body_ratio'] >= 0.45)
        & (x['close_position'] <= 0.35)
        & (x['volume'] >= x['vol_ma20'] * 2.0)
        & ((x['vol_rank30'] >= 0.93) | (x['vol_rank60'] >= 0.95))
        & ((x['ret10'] >= 0.15) | (x['trend_slope_5'] >= 0.06))
        & (x['high'] >= x['high60_prev'] * 0.97)
    )

    top_distribution = (
        (x['bear_days_8'] >= 4)
        & (x['bear_vol_sum8'] >= x['bull_vol_sum8'] * 1.6)
        & (x['close'].rolling(8).max() >= x['close60_prev'] * 0.95)
        & (x['close'] >= x['low20_prev'] * 1.10)
    )

    failed_breakout = (
        (x['high'] >= x['high30_prev'] * 0.995)
        & (x['close'] <= x['high30_prev'] * 0.985)
        & (x['upper_range_ratio'] >= 0.33)
        & (x['volume'] >= x['vol_ma20'] * 1.5)
    )

    post_new_high_heavy_selloff = (
        x['new_high_5d'].shift(1).fillna(False)
        & (pd.Series(x['is_bear'].astype(int)).rolling(3).sum() >= 2)
        & (x['volume'] >= x['vol_ma20'] * 1.2)
        & (x['close'] <= x['close'].rolling(5).max() * 0.94)
    )

    stair_bear = pd.Series(False, index=x.index)
    accel_anchor = accel_top_heavy_bear.fillna(False).to_numpy(dtype=bool)
    vols = x['volume'].to_numpy(dtype=float)
    closes = x['close'].to_numpy(dtype=float)
    opens = x['open'].to_numpy(dtype=float)
    for i in range(2, len(x)):
        left = max(0, i - 4)
        anchor_idx = None
        for j in range(i - 1, left - 1, -1):
            if accel_anchor[j]:
                anchor_idx = j
                break
        if anchor_idx is None or i - anchor_idx < 2:
            continue
        sub = x.iloc[anchor_idx + 1 : i + 1]
        if len(sub) < 2:
            continue
        if not bool((sub['close'] < sub['open']).all()):
            continue
        if not bool((sub['volume'].diff().fillna(0) < 0).iloc[1:].all()):
            continue
        if not bool((sub['close'].diff().fillna(0) < 0).iloc[1:].all()):
            continue
        stair_bear.iat[i] = True

    x['dist_accel_heavy_bear'] = accel_top_heavy_bear.fillna(False)
    x['dist_top_distribution'] = top_distribution.fillna(False)
    x['dist_failed_breakout'] = failed_breakout.fillna(False)
    x['dist_post_new_high_selloff'] = post_new_high_heavy_selloff.fillna(False)
    x['dist_stair_bear'] = stair_bear.fillna(False)
    x['dist_any'] = (
        x['dist_accel_heavy_bear']
        | x['dist_top_distribution']
        | x['dist_failed_breakout']
        | x['dist_post_new_high_selloff']
        | x['dist_stair_bear']
    )
    return x


def load_candidates(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['signal_date', 'entry_date'])


def _exit_trade_distribution(x: pd.DataFrame, signal_idx: int, variant: ExitVariant):
    entry_idx = signal_idx + 1
    if entry_idx >= len(x):
        return len(x) - 1, float(x.iloc[-1]['close']), 'insufficient_data'
    entry_open = float(x.at[entry_idx, 'open'])
    max_exit_idx = min(entry_idx + variant.max_hold_days, len(x) - 1)
    for i in range(entry_idx, max_exit_idx + 1):
        row = x.iloc[i]
        if variant.take_profit is not None:
            tp_price = entry_open * (1.0 + variant.take_profit)
            if float(row['high']) >= tp_price:
                next_idx = min(i + 1, len(x) - 1)
                return next_idx, float(x.at[next_idx, 'open']), f'tp_{variant.take_profit:.2f}'
        if variant.use_distribution_exit and bool(row['dist_any']):
            next_idx = min(i + 1, len(x) - 1)
            return next_idx, float(x.at[next_idx, 'open']), 'distribution_exit'
    return max_exit_idx, float(x.iloc[max_exit_idx]['close']), f'hold_{variant.max_hold_days}_close'


def build_trade_table(signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], variant: ExitVariant, tag: str) -> pd.DataFrame:
    rows: List[dict] = []
    for rec in signals.itertuples(index=False):
        x = dfs[rec.code]
        exit_idx, exit_price, reason = _exit_trade_distribution(x, int(rec.signal_idx), variant)
        entry_open = float(rec.entry_open)
        ret = exit_price / entry_open - 1.0
        path = x.iloc[int(rec.entry_idx): exit_idx + 1].copy()
        rows.append({
            'tag': tag,
            'variant': variant.name,
            'code': rec.code,
            'signal_idx': int(rec.signal_idx),
            'signal_date': rec.signal_date,
            'entry_idx': int(rec.entry_idx),
            'entry_date': rec.entry_date,
            'exit_idx': int(exit_idx),
            'exit_date': x.at[exit_idx, 'date'],
            'entry_open': entry_open,
            'exit_price': exit_price,
            'return': ret,
            'reason': reason,
            'sort_score': float(rec.sort_score),
            'max_favorable': float(path['high'].max() / entry_open - 1.0),
            'max_adverse': float(path['low'].min() / entry_open - 1.0),
        })
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'sample_count': 0, 'success_rate': np.nan, 'avg_return': np.nan, 'avg_max_favorable': np.nan, 'avg_max_adverse': np.nan}
    return {
        'sample_count': int(len(trades)),
        'success_rate': float((trades['return'] > 0).mean()),
        'avg_return': float(trades['return'].mean()),
        'avg_max_favorable': float(trades['max_favorable'].mean()),
        'avg_max_adverse': float(trades['max_adverse'].mean()),
    }


def run_type(tag: str, signals: pd.DataFrame, dfs: Dict[str, pd.DataFrame], variants: List[ExitVariant], account_cfg: AccountConfig) -> pd.DataFrame:
    rows = []
    for variant in variants:
        trades = build_trade_table(signals, dfs, variant, tag)
        trade_summary = summarize_trades(trades)
        account_summary = simulate_portfolio(trades, dfs, account_cfg)
        row = {'tag': tag, 'variant': variant.name, **trade_summary, **account_summary}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    dfs = load_all_data()
    type1_candidates = load_candidates(TYPE1_CANDIDATES_PATH)
    type4_candidates = load_candidates(TYPE4_CANDIDATES_PATH)
    type1_signals = select_type1(type1_candidates, TYPE1_PARAMS)
    type4_signals = select_type4(type4_candidates, TYPE4_PARAMS)

    type1_variants = [
        ExitVariant('baseline_tp10_hold30', 30, take_profit=0.10, use_distribution_exit=False),
        ExitVariant('distribution_only_hold30', 30, take_profit=None, use_distribution_exit=True),
        ExitVariant('tp10_plus_distribution_hold30', 30, take_profit=0.10, use_distribution_exit=True),
    ]
    type4_variants = [
        ExitVariant('baseline_hold20', 20, take_profit=None, use_distribution_exit=False),
        ExitVariant('distribution_only_hold20', 20, take_profit=None, use_distribution_exit=True),
    ]

    type1_res = run_type('type1', type1_signals, dfs, type1_variants, TYPE1_ACCOUNT)
    type4_res = run_type('type4', type4_signals, dfs, type4_variants, TYPE4_ACCOUNT)
    result = pd.concat([type1_res, type4_res], ignore_index=True)
    result.to_csv(RESULT_DIR / 'comparison.csv', index=False)

    dist_counts = []
    for code, x in dfs.items():
        dist_counts.append({
            'code': code,
            'dist_accel_heavy_bear': int(x['dist_accel_heavy_bear'].sum()),
            'dist_top_distribution': int(x['dist_top_distribution'].sum()),
            'dist_failed_breakout': int(x['dist_failed_breakout'].sum()),
            'dist_post_new_high_selloff': int(x['dist_post_new_high_selloff'].sum()),
            'dist_stair_bear': int(x['dist_stair_bear'].sum()),
            'dist_any': int(x['dist_any'].sum()),
        })
    pd.DataFrame(dist_counts).to_csv(RESULT_DIR / 'distribution_tag_counts.csv', index=False)

    summary = {
        'type1_signal_count': int(len(type1_signals)),
        'type4_signal_count': int(len(type4_signals)),
        'type1_variants': type1_res.to_dict(orient='records'),
        'type4_variants': type4_res.to_dict(orient='records'),
    }
    (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'type1_signal_count': len(type1_signals), 'type4_signal_count': len(type4_signals)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
