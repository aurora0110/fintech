from __future__ import annotations

import math


def calc_commission(amount: float, rate: float = 0.0003, minimum: float = 0.0) -> float:
    fee = float(amount) * float(rate)
    return max(fee, float(minimum))


def calc_stamp_tax(amount: float, is_sell: bool = True, rate: float = 0.0005) -> float:
    return float(amount) * float(rate) if is_sell else 0.0


def round_lot_size(shares: float, lot_size: int = 100) -> int:
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    return int(math.floor(float(shares) / lot_size) * lot_size)


def equal_weight_cash(total_cash: float, max_positions: int) -> float:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    return float(total_cash) / float(max_positions)

