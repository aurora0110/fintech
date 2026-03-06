from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from core.engine import BaseSignalStrategy
from core.models import Signal
from strategies.common import base_prepare, calc_brick_signal, calc_pin_buy_signal


@dataclass
class ExitProfile:
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_holding_days: int = 10


class B1Strategy(BaseSignalStrategy):
    name = "B1"

    def __init__(self, use_bullish_ma: bool = False, exit_profile: ExitProfile | None = None):
        self.use_bullish_ma = use_bullish_ma
        self.exit_profile = exit_profile or ExitProfile()

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared = {}
        for code, df in stock_data.items():
            out = base_prepare(df)
            low_9 = out["low"].rolling(window=9).min()
            high_9 = out["high"].rolling(window=9).max()
            rsv = (out["close"] - low_9) / (high_9 - low_9 + 1e-6) * 100
            out["K"] = rsv.ewm(com=2, adjust=False).mean()
            out["D"] = out["K"].ewm(com=2, adjust=False).mean()
            out["J"] = 3 * out["K"] - 2 * out["D"]
            out["MA5"] = out["close"].rolling(window=5).mean()
            out["MA10"] = out["close"].rolling(window=10).mean()
            out["MA30"] = out["close"].rolling(window=30).mean()
            prepared[code] = out
        return prepared

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for idx in range(2, len(df)):
                row = df.iloc[idx]
                if pd.isna(row["J"]) or row["J"] >= -5:
                    continue
                if self.use_bullish_ma:
                    if not (row["MA5"] > row["MA10"] > row["MA30"] and row["close"] > row["MA30"]):
                        continue
                score = max(1.0, (-float(row["J"]) - 5.0) / 10.0)
                signal = Signal(date=df.index[idx], code=code, score=score, reason="J<-5", metadata={"J": float(row["J"])})
                signal_map.setdefault(signal.date, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        entry_price = float(exec_row["open"])
        return {
            "stop_price": entry_price * (1.0 - self.exit_profile.stop_loss_pct),
            "take_profit_price": entry_price * (1.0 + self.exit_profile.take_profit_pct),
            "max_holding_days": self.exit_profile.max_holding_days,
        }


class B2Strategy(BaseSignalStrategy):
    name = "B2"

    def __init__(self, volume_multiplier: float = 2.0, exit_profile: ExitProfile | None = None):
        self.volume_multiplier = volume_multiplier
        self.exit_profile = exit_profile or ExitProfile(stop_loss_pct=0.04, take_profit_pct=0.08, max_holding_days=5)

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {code: base_prepare(df) for code, df in stock_data.items()}

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            trend_ok = df["long_trend"] <= df["short_trend"]
            j_ok = (df["J"].shift(1) <= 20) & (df["J"] <= 50)
            volatility_ok = (df["close"] / df["close"].shift(1) - 1.0) * 100 >= 4
            if self.volume_multiplier <= 1.0:
                volume_ok = df["volume"] > df["volume"].shift(1)
            else:
                volume_ok = df["volume"] > df["volume"].shift(1) * self.volume_multiplier
            bullish_ok = df["close"] > df["open"]
            body = (df["close"] - df["open"]).abs()
            upper_shadow = df["high"] - np.maximum(df["open"], df["close"])
            shadow_ratio = np.where(body > 0, upper_shadow / body, 0)
            shadow_ok = shadow_ratio < 0.3
            signal_mask = (trend_ok & j_ok & volatility_ok & volume_ok & bullish_ok & shadow_ok).fillna(False)
            for dt in df.index[signal_mask]:
                signal = Signal(date=dt, code=code, score=1.0, reason="B2 entry", metadata={})
                signal_map.setdefault(dt, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        entry_price = float(exec_row["open"])
        return {
            "stop_price": entry_price * (1.0 - self.exit_profile.stop_loss_pct),
            "take_profit_price": entry_price * (1.0 + self.exit_profile.take_profit_pct),
            "max_holding_days": self.exit_profile.max_holding_days,
        }


class B3Strategy(BaseSignalStrategy):
    name = "B3"

    def __init__(self, exit_profile: ExitProfile | None = None):
        self.exit_profile = exit_profile or ExitProfile(stop_loss_pct=0.05, take_profit_pct=0.05, max_holding_days=5)

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {code: base_prepare(df) for code, df in stock_data.items()}

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for idx in range(4, len(df)):
                a = df.iloc[idx]
                a1 = df.iloc[idx - 1]
                a2 = df.iloc[idx - 2]
                a3 = df.iloc[idx - 3]
                a4 = df.iloc[idx - 4]
                if pd.isna(a["J"]) or pd.isna(a1["J"]) or pd.isna(a2["J"]):
                    continue
                change_pct = (a["close"] - a1["close"]) / a1["close"] * 100 if a1["close"] > 0 else 0.0
                amplitude = (a["high"] - a["low"]) / a["low"] * 100 if a["low"] > 0 else 0.0
                cond = (
                    a["J"] < 80
                    and a["close"] > a["open"]
                    and change_pct < 2
                    and amplitude < 4
                    and a["volume"] < a1["volume"]
                    and a1["close"] > a1["open"]
                    and a1["volume"] >= a2["volume"] * 1.8
                    and a2["J"] < 30
                    and a3["close"] < a3["open"]
                    and a4["close"] < a4["open"]
                )
                if not cond:
                    continue
                signal = Signal(
                    date=df.index[idx],
                    code=code,
                    score=1.0,
                    reason="B3 entry",
                    metadata={"signal_low": float(a["low"])},
                )
                signal_map.setdefault(signal.date, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        entry_price = float(exec_row["open"])
        signal_low = float(signal.metadata.get("signal_low", exec_row["low"]))
        entry_low = float(exec_row["low"])
        stop_ref = min(entry_low, signal_low)
        return {
            "stop_price": stop_ref * (1.0 - self.exit_profile.stop_loss_pct),
            "take_profit_price": entry_price * (1.0 + self.exit_profile.take_profit_pct),
            "max_holding_days": self.exit_profile.max_holding_days,
        }


class PinStrategy(BaseSignalStrategy):
    name = "PIN"

    def __init__(self, exit_profile: ExitProfile | None = None):
        self.exit_profile = exit_profile or ExitProfile(stop_loss_pct=0.03, take_profit_pct=0.08, max_holding_days=10)

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {code: calc_pin_buy_signal(base_prepare(df)) for code, df in stock_data.items()}

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for dt in df.index[df["pin_buy_signal"].fillna(False)]:
                signal = Signal(date=dt, code=code, score=1.0, reason="PIN buy_signal", metadata={})
                signal_map.setdefault(dt, []).append(signal)
        return signal_map

    def build_position(self, signal: Signal, exec_row: pd.Series) -> Dict[str, float]:
        entry_price = float(exec_row["open"])
        return {
            "stop_price": entry_price * (1.0 - self.exit_profile.stop_loss_pct),
            "take_profit_price": entry_price * (1.0 + self.exit_profile.take_profit_pct),
            "max_holding_days": self.exit_profile.max_holding_days,
        }


class BrickStrategy(PinStrategy):
    name = "BRICK"

    def prepare(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {code: calc_brick_signal(base_prepare(df)) for code, df in stock_data.items()}

    def generate_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[pd.Timestamp, List[Signal]]:
        signal_map: Dict[pd.Timestamp, List[Signal]] = {}
        for code, df in stock_data.items():
            for dt in df.index[df["brick_buy_signal"].fillna(False)]:
                signal = Signal(date=dt, code=code, score=1.0, reason="BRICK buy_signal", metadata={})
                signal_map.setdefault(dt, []).append(signal)
        return signal_map
