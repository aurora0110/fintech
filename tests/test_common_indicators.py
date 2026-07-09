from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.common.brick import calc_brick_values
from utils.common.indicators import (
    calc_ema,
    calc_kdj,
    calc_long_dev,
    calc_ma,
    calc_macd,
    calc_rsi,
    calc_trend_dev,
    calc_zhixing_long_line,
    calc_zhixing_trend_line,
)


def make_price_df(n: int = 30) -> pd.DataFrame:
    close = pd.Series(np.linspace(10, 20, n))
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000, 5000, n),
        }
    )


class CommonIndicatorsTest(unittest.TestCase):
    def test_ma_ema_length_same_as_input(self):
        close = pd.Series([1, 2, 3, 4, 5], dtype=float)
        self.assertEqual(len(calc_ma(close, 3)), len(close))
        self.assertEqual(len(calc_ema(close, 3)), len(close))

    def test_kdj_no_future_data(self):
        df = make_price_df(20)
        short = calc_kdj(df.iloc[:10].copy())
        full = calc_kdj(df.copy()).iloc[:10]
        pd.testing.assert_frame_equal(short.reset_index(drop=True), full.reset_index(drop=True))

    def test_macd_output_columns(self):
        macd = calc_macd(pd.Series([1, 2, 3, 4, 5], dtype=float))
        self.assertEqual(list(macd.columns), ["dif", "dea", "macd"])

    def test_zhixing_lines_formula(self):
        close = pd.Series(np.arange(1, 121, dtype=float))
        trend = calc_zhixing_trend_line(close)
        expected_trend = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
        pd.testing.assert_series_equal(trend, expected_trend)

        long_line = calc_zhixing_long_line(close)
        expected_long = (
            close.rolling(14).mean()
            + close.rolling(28).mean()
            + close.rolling(57).mean()
            + close.rolling(114).mean()
        ) / 4.0
        pd.testing.assert_series_equal(long_line, expected_long)

    def test_trend_long_dev_zero_denominator_returns_nan(self):
        close = pd.Series([1.0, 2.0, 3.0])
        zeros = pd.Series([0.0, 0.0, 0.0])
        self.assertTrue(calc_trend_dev(close, zeros).isna().all())
        self.assertTrue(calc_long_dev(close, zeros).isna().all())

    def test_sample_insufficient_stays_nan(self):
        close = pd.Series([1.0, 2.0, 3.0])
        ma = calc_ma(close, 5)
        self.assertTrue(ma.isna().all())

    def test_rsi_length_and_fill(self):
        rsi = calc_rsi(pd.Series([1, 2, 3, 4, 5], dtype=float), 14)
        self.assertEqual(len(rsi), 5)
        self.assertTrue(rsi.notna().all())

    def test_brick_output_columns_exist(self):
        df = make_price_df(40)
        out = calc_brick_values(df, n=3, m1=5, m2=5)
        for col in ["brick", "brick_prev", "brick_red_len", "brick_green_len", "brick_red", "brick_green"]:
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
