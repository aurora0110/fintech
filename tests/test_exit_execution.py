from __future__ import annotations

import pandas as pd

from utils.backtest_pipeline.exits.execution import (
    close_trail_next_open_exit,
    corrected_intraday_trail_exit,
    fixed_close_exit,
    next_valid_open,
    stop_loss_gap_corrected,
    take_profit_gap_corrected,
)


def _row(**kwargs) -> pd.Series:
    return pd.Series(kwargs, name=0)


def test_take_profit_gap_corrected_uses_open_when_gap_above_target() -> None:
    fill = take_profit_gap_corrected(_row(date="2026-01-02", open=10.5, high=10.8), 10.0)

    assert fill is not None
    assert fill.exit_price == 10.5
    assert fill.exit_reason == "TAKE_PROFIT_GAP_OPEN"
    assert fill.gap_adjusted is True


def test_stop_loss_gap_corrected_uses_open_when_gap_below_stop() -> None:
    fill = stop_loss_gap_corrected(_row(date="2026-01-02", open=9.5, low=9.2), 10.0)

    assert fill is not None
    assert fill.exit_price == 9.5
    assert fill.exit_reason == "STOP_LOSS_GAP_OPEN"
    assert fill.gap_adjusted is True


def test_corrected_intraday_trail_does_not_use_same_day_new_high_for_trail() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1},
            {"date": "2026-01-02", "open": 10.1, "high": 10.3, "low": 9.95, "close": 10.2, "volume": 1},
            {"date": "2026-01-03", "open": 10.2, "high": 10.5, "low": 10.25, "close": 10.4, "volume": 1},
            {"date": "2026-01-04", "open": 10.4, "high": 11.0, "low": 10.7, "close": 10.8, "volume": 1},
        ]
    )

    fill = corrected_intraday_trail_exit(df, 0, 10.0, 3, trail_start=0.04, trail_drawdown=0.02)

    assert fill.is_settled is False


def test_corrected_intraday_trail_uses_gap_open_below_trail() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 10.0, "high": 10.6, "low": 10.0, "close": 10.5, "volume": 1},
            {"date": "2026-01-02", "open": 10.1, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1},
        ]
    )

    fill = corrected_intraday_trail_exit(df, 0, 10.0, 1, trail_start=0.05, trail_drawdown=0.02)

    assert fill.is_settled is True
    assert fill.exit_price == 10.1
    assert fill.exit_reason == "TRAIL_GAP_OPEN"
    assert fill.gap_adjusted is True


def test_close_trail_sells_next_valid_open_after_close_confirmation() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1},
            {"date": "2026-01-02", "open": 10.1, "high": 10.7, "low": 10.1, "close": 10.6, "volume": 1},
            {"date": "2026-01-03", "open": 10.4, "high": 10.5, "low": 10.2, "close": 10.3, "volume": 1},
            {"date": "2026-01-04", "open": 0.0, "high": 10.1, "low": 10.0, "close": 10.0, "volume": 0},
            {"date": "2026-01-05", "open": 10.2, "high": 10.4, "low": 10.1, "close": 10.3, "volume": 1},
        ]
    )

    fill = close_trail_next_open_exit(df, 0, 10.0, 4, trail_start=0.05, trail_drawdown=0.02)

    assert fill.is_settled is True
    assert fill.exit_date == pd.Timestamp("2026-01-05")
    assert fill.exit_price == 10.2
    assert fill.delayed_exit_days == 1


def test_next_valid_open_skips_invalid_open_and_zero_volume() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "open": 0.0, "volume": 1},
            {"date": "2026-01-02", "open": 10.0, "volume": 0},
            {"date": "2026-01-03", "open": 10.5, "volume": 1},
        ]
    )

    fill = next_valid_open(df, 0)

    assert fill.is_settled is True
    assert fill.exit_date == pd.Timestamp("2026-01-03")
    assert fill.exit_price == 10.5
    assert fill.delayed_exit_days == 2


def test_fixed_close_exit_uses_signal_idx_plus_hold_days() -> None:
    df = pd.DataFrame(
        [
            {"date": "2026-01-01", "close": 10.0},
            {"date": "2026-01-02", "close": 10.1},
            {"date": "2026-01-03", "close": 10.2},
            {"date": "2026-01-04", "close": 10.3},
        ]
    )

    fill = fixed_close_exit(df, signal_idx=0, hold_days=3)

    assert fill.is_settled is True
    assert fill.exit_date == pd.Timestamp("2026-01-04")
    assert fill.exit_price == 10.3
