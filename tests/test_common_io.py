from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.common.io import (
    extract_stock_code_from_filename,
    read_stock_daily,
    read_stock_name_from_file,
    standardize_ohlcv_columns,
)


class CommonIoTest(unittest.TestCase):
    def test_standardize_ohlcv_columns_from_chinese_names(self):
        raw = pd.DataFrame(
            {
                "日期": ["2026-01-02"],
                "开盘": ["10"],
                "最高": ["11"],
                "最低": ["9"],
                "收盘": ["10.5"],
                "成交量": ["1000"],
                "成交额": ["10000"],
            }
        )
        df = standardize_ohlcv_columns(raw, path="SZ#000001.txt")
        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume", "amount", "code"])
        self.assertEqual(df.loc[0, "code"], "000001")

    def test_extract_stock_code_from_filename(self):
        self.assertEqual(extract_stock_code_from_filename("data/20260616/normal/SH#600000.txt"), "600000")

    def test_read_stock_daily_csv_standardized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "SZ#000001.csv"
            path.write_text(
                "日期,开盘,最高,最低,收盘,成交量,成交额\n"
                "2026-01-03,11,12,10,11.5,2000,22000\n"
                "2026-01-02,10,11,9,10.5,1000,10000\n",
                encoding="utf-8",
            )
            df = read_stock_daily(path, min_bars=2, allow_zero_volume=False)
            self.assertEqual(list(df["date"].dt.strftime("%Y-%m-%d")), ["2026-01-02", "2026-01-03"])
            self.assertEqual(float(df.iloc[0]["open"]), 10.0)

    def test_read_stock_daily_txt_standardized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "SH#600000.txt"
            path.write_text(
                "日期 开盘 最高 最低 收盘 成交量 成交额\n"
                "2026/01/02 10 11 9 10.5 1000 10000\n"
                "2026/01/03 11 12 10 11.5 2000 22000\n",
                encoding="utf-8",
            )
            df = read_stock_daily(path, min_bars=2)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[-1]["close"], 11.5)

    def test_read_stock_name_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "SZ#000001.txt"
            path.write_bytes("000001 平安银行 日线\n2026/01/02 10 11 9 10.5 1000 10000\n".encode("gbk"))
            self.assertEqual(read_stock_name_from_file(path), "平安银行")


if __name__ == "__main__":
    unittest.main()
