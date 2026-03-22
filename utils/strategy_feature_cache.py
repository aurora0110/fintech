from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils import b1filter, b2filter, b3filter, brick_filter, pinfilter, technical_indicators
from utils.market_risk_tags import add_risk_features, latest_risk_snapshot
from utils.shared_market_features import compute_base_features


@dataclass
class StrategyFeatureCache:
    file_path: str
    _raw_df: Optional[pd.DataFrame] = None
    _daily_cn_df: Optional[pd.DataFrame] = None
    _base_features: Optional[pd.DataFrame] = None
    _weekly_bundle: Optional[pd.DataFrame] = None
    _weekly_screen: Optional[tuple[bool, str]] = None
    _b1_daily_bundle: Optional[dict] = None
    _b2_features: Optional[pd.DataFrame] = None
    _b3_features: Optional[pd.DataFrame] = None
    _brick_features: Optional[pd.DataFrame] = None
    _pin_today_features: Optional[dict] = None
    _risk_features: Optional[pd.DataFrame] = None
    _risk_snapshot: Optional[dict] = None

    def raw_df(self) -> Optional[pd.DataFrame]:
        if self._raw_df is None:
            self._raw_df = b2filter.load_one_csv(self.file_path)
        return self._raw_df

    def daily_cn_df(self) -> Optional[pd.DataFrame]:
        if self._daily_cn_df is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._daily_cn_df = pd.DataFrame(
                {
                    "日期": raw["date"],
                    "开盘": raw["open"],
                    "最高": raw["high"],
                    "最低": raw["low"],
                    "收盘": raw["close"],
                    "成交量": raw["volume"],
                    "成交额": 0.0,
                }
            )
        return self._daily_cn_df

    def weekly_screen(self) -> tuple[bool, str]:
        if self._weekly_screen is None:
            weekly_df = self.weekly_bundle()
            self._weekly_screen = b1filter.weekly_screen_from_weekly_df(weekly_df) if weekly_df is not None else (False, "")
        return self._weekly_screen

    def base_features(self) -> Optional[pd.DataFrame]:
        if self._base_features is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._base_features = compute_base_features(raw.copy())
        return self._base_features

    def weekly_bundle(self) -> Optional[pd.DataFrame]:
        if self._weekly_bundle is None:
            daily_df = self.daily_cn_df()
            if daily_df is None or daily_df.empty:
                return None
            weekly_df = technical_indicators.calculate_week_price_from_df(daily_df)
            if weekly_df is None or weekly_df.empty:
                return None
            weekly_df = technical_indicators.calculate_trend(weekly_df.copy())
            weekly_df = technical_indicators.calculate_kdj(weekly_df)
            weekly_df = technical_indicators.calculate_daily_ma(weekly_df)
            self._weekly_bundle = weekly_df
        return self._weekly_bundle

    def b1_daily_bundle(self) -> Optional[dict]:
        if self._b1_daily_bundle is None:
            daily_df = self.daily_cn_df()
            if daily_df is None or len(daily_df) < 120:
                return None
            base_df = daily_df.copy()
            df_trend = technical_indicators.calculate_trend(base_df.copy())
            df_kdj = technical_indicators.calculate_kdj(base_df.copy())
            df_ma = technical_indicators.calculate_daily_ma(base_df.copy())
            self._b1_daily_bundle = {
                "df": base_df,
                "df_trend": df_trend,
                "df_kdj": df_kdj,
                "df_ma": df_ma,
            }
        return self._b1_daily_bundle

    def b2_features(self) -> Optional[pd.DataFrame]:
        if self._b2_features is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._b2_features = b2filter.add_features(
                raw.copy(),
                precomputed_base=self.base_features(),
            )
        return self._b2_features

    def b3_features(self) -> Optional[pd.DataFrame]:
        if self._b3_features is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._b3_features = b3filter.add_features(
                raw.copy(),
                precomputed_b2=self.b2_features(),
            )
        return self._b3_features

    def brick_features(self) -> Optional[pd.DataFrame]:
        if self._brick_features is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._brick_features = brick_filter.add_features(raw.copy())
        return self._brick_features

    def pin_today_features(self) -> Optional[dict]:
        if self._pin_today_features is None:
            brick_df = self.brick_features()
            if brick_df is None or brick_df.empty:
                return None
            self._pin_today_features = pinfilter.build_today_features_from_feature_df(brick_df.copy())
        return self._pin_today_features

    def risk_features(self) -> Optional[pd.DataFrame]:
        if self._risk_features is None:
            raw = self.raw_df()
            if raw is None or raw.empty:
                return None
            self._risk_features = add_risk_features(raw.copy(), precomputed_base=self.base_features())
        return self._risk_features

    def risk_snapshot(self) -> dict:
        if self._risk_snapshot is None:
            risk_df = self.risk_features()
            self._risk_snapshot = latest_risk_snapshot(risk_df) if risk_df is not None else {}
        return self._risk_snapshot
