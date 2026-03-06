from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from core.data_loader import load_price_directory
from strategies.unified_strategies import B1Strategy, B2Strategy, B3Strategy, BrickStrategy, PinStrategy
from utils.backtest import backtest_b1_strategy as legacy_b1
from utils.backtest import backtest_b2_strategy as legacy_b2
from utils.backtest import backtest_b3_strategy as legacy_b3
from utils.backtest import backtest_brick_strategy as legacy_brick
from utils.backtest import backtest_pin_strategy as legacy_pin


def _to_legacy_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().rename(
        columns={
            "date": "日期",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "volume": "VOLUME",
            "amount": "AMOUNT",
        }
    )
    if "日期" not in out.columns:
        out = out.rename(columns={"index": "日期"})
    return out


def legacy_signal_dates(strategy_name: str, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, List[pd.Timestamp]]:
    result: Dict[str, List[pd.Timestamp]] = {}
    for code, raw_df in stock_data.items():
        df = _to_legacy_df(raw_df)
        if strategy_name == "B1":
            df = df.set_index("日期")
            df = legacy_b1.calculate_indicators(df)
            result[code] = [df.index[i] for i in range(2, len(df)) if legacy_b1.calculate_b1_score(df, i)[0] >= 0.5]
        elif strategy_name == "B2":
            df = df.set_index("日期")
            df = legacy_b2.calculate_indicators(df)
            result[code] = list(df.index[legacy_b2.generate_signal(df, 2.0).fillna(False)])
        elif strategy_name == "B3":
            df = df.set_index("日期")
            df = legacy_b3.calculate_indicators(df)
            result[code] = [df.index[i] for i in range(4, len(df)) if legacy_b3.check_b3_signal(df, i)[0]]
        elif strategy_name == "PIN":
            df = df.set_index("日期")
            df = legacy_pin.calculate_trend(df)
            df = legacy_pin.brick_chart_indicator(df)
            result[code] = list(df.index[df["买入信号"] == 1])
        elif strategy_name == "BRICK":
            df = df.set_index("日期")
            df = legacy_brick.calculate_trend(df)
            df = legacy_brick.brick_chart_indicator(df)
            result[code] = list(df.index[df["买入信号"] == 1])
        else:
            raise ValueError(strategy_name)
    return result


def unified_signal_dates(strategy_name: str, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, List[pd.Timestamp]]:
    strategies = {
        "B1": B1Strategy(),
        "B2": B2Strategy(),
        "B3": B3Strategy(),
        "PIN": PinStrategy(),
        "BRICK": BrickStrategy(),
    }
    strategy = strategies[strategy_name]
    prepared = strategy.prepare(stock_data)
    signal_map = strategy.generate_signals(prepared)
    by_code: Dict[str, List[pd.Timestamp]] = {}
    for signal_date, signals in signal_map.items():
        for signal in signals:
            by_code.setdefault(signal.code, []).append(signal_date)
    return by_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unified strategy signal dates against legacy definitions")
    parser.add_argument("strategy", choices=["B1", "B2", "B3", "PIN", "BRICK"])
    parser.add_argument("data_dir")
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    stock_data, _ = load_price_directory(args.data_dir)
    unified = unified_signal_dates(args.strategy, stock_data)
    legacy = legacy_signal_dates(args.strategy, stock_data)

    rows = []
    for code in sorted(stock_data):
        new_dates = set(unified.get(code, []))
        old_dates = set(legacy.get(code, []))
        only_new = sorted(new_dates - old_dates)[: args.sample]
        only_old = sorted(old_dates - new_dates)[: args.sample]
        if only_new or only_old:
            rows.append(
                {
                    "code": code,
                    "only_new_count": len(new_dates - old_dates),
                    "only_old_count": len(old_dates - new_dates),
                    "only_new_preview": [d.strftime("%Y-%m-%d") for d in only_new],
                    "only_old_preview": [d.strftime("%Y-%m-%d") for d in only_old],
                }
            )

    if not rows:
        print("All sampled signal dates match.")
        return

    for row in rows[: args.sample]:
        print(row)


if __name__ == "__main__":
    main()
