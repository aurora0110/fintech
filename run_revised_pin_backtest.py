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
class PinVariant:
    name: str
    holding_days: int
    require_strong_5d: bool
    drop_threshold: float | None
    volume_mode: str
    gem_only: bool = False
    friday_only: bool = False
    require_rsi_stack: bool = False


class RevisedPinStrategy(BaseSignalStrategy):
    def __init__(self, variant: PinVariant):
        self.variant = variant
        self.name = variant.name

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared: Dict[str, pd.DataFrame] = {}
        for code, df in stock_data.items():
            out = df.copy()
            out["deep_drop"] = out["signal_day_return"].le(self.variant.drop_threshold) if self.variant.drop_threshold is not None else True
            out["volume_ok"] = self._volume_mask(out)
            out["board_ok"] = True if not self.variant.gem_only else out["board"].eq("GEM")
            out["weekday_ok"] = True if not self.variant.friday_only else (out.index.weekday == 4)
            out["rsi_ok"] = True if not self.variant.require_rsi_stack else out["rsi_stack_bull"]
            strong_momentum = out["strong_5d_momentum"] if self.variant.require_strong_5d else True
            out["buy_signal"] = out["pin_shape"] & out["trend_ok"] & strong_momentum & out["deep_drop"] & out["volume_ok"] & out["board_ok"] & out["weekday_ok"] & out["rsi_ok"]
            prepared[code] = out
        return prepared

    def _volume_mask(self, df: pd.DataFrame) -> pd.Series | bool:
        if self.variant.volume_mode == "any":
            return True
        if self.variant.volume_mode == "band":
            return df["moderate_shrink"] | df["mild_expand"]
        if self.variant.volume_mode == "shrink":
            return df["moderate_shrink"]
        if self.variant.volume_mode == "expand":
            return df["mild_expand"]
        raise ValueError(self.variant.volume_mode)

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for dt in df.index[df["buy_signal"].fillna(False)]:
                row = df.loc[dt]
                score = float(row.get("return_5d", 0.0)) - float(row.get("signal_day_return", 0.0))
                if row.get("board") == "GEM":
                    score += 0.01
                signal = Signal(
                    date=dt,
                    code=code,
                    score=score,
                    reason=self.variant.name,
                    metadata={},
                )
                signal_map.setdefault(dt, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        return {
            "max_holding_days": self.variant.holding_days,
        }


def build_variants() -> List[PinVariant]:
    return [
        PinVariant(
            name="pin_core_hold2",
            holding_days=2,
            require_strong_5d=False,
            drop_threshold=None,
            volume_mode="any",
        ),
        PinVariant(
            name="pin_core_hold3",
            holding_days=3,
            require_strong_5d=False,
            drop_threshold=None,
            volume_mode="any",
        ),
        PinVariant(
            name="pin_repair_hold2",
            holding_days=2,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
        ),
        PinVariant(
            name="pin_repair_hold3",
            holding_days=3,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
        ),
        PinVariant(
            name="pin_repair_gem_hold2",
            holding_days=2,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            gem_only=True,
        ),
        PinVariant(
            name="pin_repair_gem_hold3",
            holding_days=3,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            gem_only=True,
        ),
        PinVariant(
            name="pin_repair_friday_hold2",
            holding_days=2,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            friday_only=True,
        ),
        PinVariant(
            name="pin_repair_friday_hold3",
            holding_days=3,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            friday_only=True,
        ),
        PinVariant(
            name="pin_repair_gem_rsi_hold2",
            holding_days=2,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            gem_only=True,
            require_rsi_stack=True,
        ),
        PinVariant(
            name="pin_repair_gem_rsi_hold3",
            holding_days=3,
            require_strong_5d=True,
            drop_threshold=-0.05,
            volume_mode="band",
            gem_only=True,
            require_rsi_stack=True,
        ),
    ]


def run_variant(variant: PinVariant, stock_data: Dict[str, pd.DataFrame], all_dates: List[pd.Timestamp]) -> dict:
    engine = BacktestEngine(EngineConfig(initial_capital=1_000_000, max_positions=10))
    strategy = RevisedPinStrategy(variant)
    result = engine.run(strategy, stock_data, all_dates)
    wins = [trade for trade in result.trades if trade.return_pct > 0]
    avg_trade_return = float(np.mean([trade.return_pct for trade in result.trades])) if result.trades else 0.0
    win_rate = float(len(wins) / len(result.trades)) if result.trades else 0.0
    return {
        "variant": variant.name,
        "holding_days": variant.holding_days,
        "require_strong_5d": variant.require_strong_5d,
        "drop_threshold": variant.drop_threshold,
        "volume_mode": variant.volume_mode,
        "gem_only": variant.gem_only,
        "friday_only": variant.friday_only,
        "require_rsi_stack": variant.require_rsi_stack,
        "signal_count": len(result.signals),
        "trade_count": len(result.trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        **result.metrics,
        **result.diagnostics,
    }


def build_base_dataset(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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
        out["moderate_shrink"] = out["volume_ratio_prev"].gt(0.7) & out["volume_ratio_prev"].le(0.9)
        out["mild_expand"] = out["volume_ratio_prev"].gt(1.1) & out["volume_ratio_prev"].le(1.5)
        prepared[code] = out
    return prepared


def main() -> None:
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260311/normal"
    output_dir = Path("/Users/lidongyang/Desktop/Qstrategy/results/revised_pin_backtest_20260311")
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_data, all_dates = load_price_directory(data_dir)
    base_data = build_base_dataset(stock_data)
    rows = [run_variant(variant, base_data, all_dates) for variant in build_variants()]
    summary = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "sharpe"],
        ascending=[False, True, False],
    )
    summary.to_csv(output_dir / "summary_table.csv", index=False, encoding="utf-8-sig")

    payload = {
        "data_scope": data_dir,
        "top_variants": summary.head(5).to_dict(orient="records"),
        "bottom_variants": summary.tail(5).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
