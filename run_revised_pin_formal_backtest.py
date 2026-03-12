from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_loader import load_price_directory
from core.engine import BacktestEngine, BaseSignalStrategy, EngineConfig
from core.models import Signal
from strategies.common import calc_trend_cn


def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_up = up.ewm(span=period, adjust=False).mean()
    avg_down = down.ewm(span=period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def add_pin_shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_3 = out["low"].rolling(window=3).min()
    close_high_3 = out["close"].rolling(window=3).max()
    low_21 = out["low"].rolling(window=21).min()
    close_high_21 = out["close"].rolling(window=21).max()
    short_denom = (close_high_3 - low_3).replace(0.0, np.nan)
    long_denom = (close_high_21 - low_21).replace(0.0, np.nan)
    out["pin_short"] = ((out["close"] - low_3) / short_denom) * 100.0
    out["pin_long"] = ((out["close"] - low_21) / long_denom) * 100.0
    out["pin_shape"] = out["pin_short"].le(30.0) & out["pin_long"].ge(85.0)
    return out


@dataclass
class RevisedPinConfig:
    name: str
    gem_mode: str = "none"
    gem_score_bonus: float = 0.0
    take_profit_pin_long: float = 100.0
    stop_pin_long: float = 80.0


class RevisedPinFormalStrategy(BaseSignalStrategy):
    def __init__(self, config: RevisedPinConfig):
        self.config = config
        self.name = config.name

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for code, df in stock_data.items():
            out = add_pin_shape(df)
            out["short_trend"], out["long_trend"] = calc_trend_cn(out["close"])
            out["return_5d"] = out["close"].pct_change(5)
            out["signal_day_return"] = out["close"] / out["close"].shift(1) - 1.0
            out["volume_ratio_prev"] = out["volume"] / out["volume"].shift(1).replace(0.0, np.nan)
            out["rsi14"] = calc_rsi(out["close"], 14)
            out["rsi28"] = calc_rsi(out["close"], 28)
            out["rsi57"] = calc_rsi(out["close"], 57)
            out["rsi_stack_bull"] = out["rsi14"].gt(out["rsi28"]) & out["rsi28"].gt(out["rsi57"])
            out["trend_ok"] = out["short_trend"].gt(out["long_trend"])
            out["strong_5d_momentum"] = out["return_5d"].gt(0.10)
            out["deep_drop"] = out["signal_day_return"].le(-0.05)
            out["moderate_shrink"] = out["volume_ratio_prev"].gt(0.7) & out["volume_ratio_prev"].le(0.9)
            out["mild_expand"] = out["volume_ratio_prev"].gt(1.1) & out["volume_ratio_prev"].le(1.5)
            out["volume_band_ok"] = out["moderate_shrink"] | out["mild_expand"]
            gem_ok = True if self.config.gem_mode != "hard" else out["board"].eq("GEM")
            out["buy_signal"] = (
                out["pin_shape"]
                & out["trend_ok"]
                & out["strong_5d_momentum"]
                & out["deep_drop"]
                & out["volume_band_ok"]
                & gem_ok
            )
            prepared[code] = out
        return prepared

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for dt in df.index[df["buy_signal"].fillna(False)]:
                row = df.loc[dt]
                score = float(row["return_5d"] - row["signal_day_return"])
                if self.config.gem_mode == "bonus" and row.get("board") == "GEM":
                    score += self.config.gem_score_bonus
                signal = Signal(
                    date=dt,
                    code=code,
                    score=score,
                    reason=self.config.name,
                    metadata={},
                )
                signal_map.setdefault(dt, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        return {}

    def should_exit(
        self,
        current_date: pd.Timestamp,
        position,
        row: pd.Series,
        stock_df: pd.DataFrame,
    ) -> str | None:
        pin_long = row.get("pin_long")
        if pd.notna(pin_long) and float(pin_long) >= self.config.take_profit_pin_long:
            return "pin_long_100"
        if pd.notna(pin_long) and float(pin_long) < self.config.stop_pin_long:
            return "pin_long_below_80"
        return None


def run_variant(config: RevisedPinConfig, stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]) -> dict:
    engine = BacktestEngine(EngineConfig(initial_capital=1_000_000, max_positions=10))
    strategy = RevisedPinFormalStrategy(config)
    result = engine.run(strategy, stock_data, all_dates)
    win_rate = float(sum(1 for trade in result.trades if trade.return_pct > 0) / len(result.trades)) if result.trades else 0.0
    avg_trade_return = float(np.mean([trade.return_pct for trade in result.trades])) if result.trades else 0.0
    return {
        "variant": config.name,
        "gem_mode": config.gem_mode,
        "gem_score_bonus": config.gem_score_bonus,
        "take_profit_pin_long": config.take_profit_pin_long,
        "stop_pin_long": config.stop_pin_long,
        "signal_count": len(result.signals),
        "trade_count": len(result.trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        **result.metrics,
        **result.diagnostics,
    }


def main() -> None:
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal"
    output_dir = Path("/Users/lidongyang/Desktop/Qstrategy/results/revised_pin_formal_backtest_20260311")
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data, all_dates = load_price_directory(data_dir)
    variants = [
        RevisedPinConfig(name="pin_repair_pinlong_exit_formal", gem_mode="none"),
        RevisedPinConfig(name="pin_repair_pinlong_exit_gem_bonus_formal", gem_mode="bonus", gem_score_bonus=0.05),
        RevisedPinConfig(name="pin_repair_pinlong_exit_gem_hard_formal", gem_mode="hard"),
    ]
    rows = [run_variant(config, stock_data, all_dates) for config in variants]
    summary = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(output_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    payload = {
        "data_scope": data_dir,
        "variants": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
