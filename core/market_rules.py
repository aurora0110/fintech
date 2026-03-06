from __future__ import annotations

import pandas as pd


def extract_code_number(code: str) -> str:
    if "#" in code:
        return code.split("#", 1)[1]
    return code


def detect_board(code: str) -> str:
    num = extract_code_number(code)
    if num.startswith("688"):
        return "STAR"
    if num.startswith("300"):
        return "GEM"
    if num.startswith("8") or num.startswith("83") or num.startswith("87"):
        return "BSE"
    return "MAIN"


def detect_st(code: str) -> bool:
    text = str(code).upper()
    return "ST" in text or "*ST" in text


def limit_pct_for(code: str, date: pd.Timestamp, is_st: bool = False) -> float:
    num = extract_code_number(code)
    if is_st:
        return 0.05
    if num.startswith("688"):
        return 0.20
    if num.startswith("300"):
        gem_switch = pd.Timestamp("2020-08-24")
        return 0.20 if date >= gem_switch else 0.10
    if num.startswith("8") or num.startswith("83") or num.startswith("87"):
        return 0.30
    return 0.10


def is_limit_up(open_price: float, prev_close: float, limit_pct: float) -> bool:
    if prev_close <= 0:
        return False
    return open_price >= prev_close * (1.0 + limit_pct)


def is_limit_down(open_price: float, prev_close: float, limit_pct: float) -> bool:
    if prev_close <= 0:
        return False
    return open_price <= prev_close * (1.0 - limit_pct)
