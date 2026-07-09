from __future__ import annotations

import unittest

from utils.common.dates import get_next_trade_date, get_prev_trade_date, get_trade_date_offset


class CommonDatesTest(unittest.TestCase):
    def test_trade_date_offset_forward_backward(self):
        trade_dates = ["2026-01-02", "2026-01-05", "2026-01-06"]
        self.assertEqual(str(get_next_trade_date(trade_dates, "2026-01-02", 1).date()), "2026-01-05")
        self.assertEqual(str(get_prev_trade_date(trade_dates, "2026-01-06", 1).date()), "2026-01-05")
        self.assertEqual(str(get_trade_date_offset(trade_dates, "2026-01-05", -1).date()), "2026-01-02")

    def test_trade_date_offset_missing_returns_none(self):
        trade_dates = ["2026-01-02", "2026-01-05"]
        self.assertIsNone(get_trade_date_offset(trade_dates, "2026-01-03", 1))


if __name__ == "__main__":
    unittest.main()
