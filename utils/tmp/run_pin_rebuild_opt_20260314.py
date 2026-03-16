from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import stoploss, technical_indicators


DATA_DIR = ROOT / "data" / "forward_data"
NAME_MAP_DIR = ROOT / "data" / "20260313"
CASE_ROOT = ROOT / "data" / "完美图" / "单针"
OUT_DIR = ROOT / "results" / "pin_rebuild_opt_20260314"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS_PER_YEAR = 252
MAX_POSITIONS = 10
MAX_SINGLE_WEIGHT = 0.2
MIN_BARS = 160
EPS = 1e-12
PIN_NAME_RE = re.compile(r"^(?P<name>.+?)(?P<date>20\d{6})\.[^.]+$")


MANUAL_SUCCESS = [
    ("博源化工", "20260303"),
    ("开山股份", "20260303"),
    ("雪榕生物", "20260303"),
    ("中国船舶", "20260303"),
    ("锡业股份", "20260119"),
    ("天山铝业", "20260116"),
    ("赛微电子", "20251217"),
    ("三花智控", "20250827"),
    ("三花智控", "20260108"),
    ("三花智控", "20260115"),
    ("三花智控", "20250926"),
    ("罗曼股份", "20250926"),
    ("长盈精密", "20250926"),
    ("兖矿能源", "20250926"),
    ("奥普光电", "20250926"),
    ("大悦城", "20250717"),
    ("太阳能", "20260211"),
    ("天奇股份", "20260213"),
    ("威尔高", "20260211"),
    ("凯格精机", "20260211"),
    ("中体产业", "20260211"),
    ("快克智能", "20260120"),
    ("天齐锂业", "20251111"),
]

MANUAL_FAIL = [
    ("东方电气", "20260310"),
    ("燕麦科技", "20260305"),
    ("星宇股份", "20260303"),
    ("中新赛克", "20260116"),
    ("中新赛克", "20260119"),
    ("东富龙", "20260116"),
    ("鼎捷数智", "20260116"),
    ("深信服", "20260116"),
    ("佰奥智能", "20250926"),
    ("宇晶股份", "20250717"),
    ("厦门象屿", "20250717"),
    ("上汽集团", "20250717"),
    ("嘉诚国际", "20250717"),
    ("丰原药业", "20260211"),
    ("渤海租赁", "20260211"),
    ("广弘控股", "20260211"),
    ("箭牌家居", "20260211"),
    ("得利斯", "20260211"),
    ("西点药业", "20260211"),
    ("普莱得", "20260211"),
    ("宁波联合", "20260211"),
    ("哈空调", "20260211"),
    ("中化装备", "20260211"),
    ("金晶科技", "20260211"),
    ("玲珑轮胎", "20260211"),
    ("威奥股份", "20260211"),
    ("岱美股份", "20260211"),
    ("华鲁恒升", "20251118"),
    ("华阳股份", "20251111"),
]


@dataclass(frozen=True)
class EntryCombo:
    trend_slope_3_min: float
    trend_slope_5_min: float
    long_slope_5_min: float
    trend_lead_min: float
    ret10_min: float
    ret3_max: float
    signal_vs_ma20_max: float
    vol_vs_prev_max: float
    close_position_max: Optional[float] = None
    lower_shadow_max: Optional[float] = None

    @property
    def name(self) -> str:
        parts = [
            f"s3_{self.trend_slope_3_min:.3f}",
            f"s5_{self.trend_slope_5_min:.3f}",
            f"l5_{self.long_slope_5_min:.3f}",
            f"lead_{self.trend_lead_min:.3f}",
            f"r10_{self.ret10_min:.2f}",
            f"r3max_{self.ret3_max:.2f}",
            f"ma20_{self.signal_vs_ma20_max:.2f}",
            f"vp_{self.vol_vs_prev_max:.2f}",
            "cp_none" if self.close_position_max is None else f"cp_{self.close_position_max:.2f}",
            "ls_none" if self.lower_shadow_max is None else f"ls_{self.lower_shadow_max:.2f}",
        ]
        return "__".join(parts)


@dataclass(frozen=True)
class ExitConfig:
    name: str
    max_hold_days: int
    forced_exit: str
    tp_pct: Optional[float] = None
    sl_mode: str = "none"
    trigger_exec: str = "same_day"
    trend_break_mode: str = "none"
    vol_bear_exit: bool = False


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    if isinstance(a, float) and not math.isfinite(a):
        return np.nan
    if isinstance(b, float) and (not math.isfinite(b) or abs(b) <= EPS):
        return np.nan
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return np.nan
    if not math.isfinite(a_f) or not math.isfinite(b_f) or abs(b_f) <= EPS:
        return np.nan
    return a_f / b_f


def load_name_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in sorted(NAME_MAP_DIR.glob("*.txt")):
        try:
            first = path.read_bytes().splitlines()[0]
        except Exception:
            continue
        line = None
        for enc in ("gb18030", "gbk", "utf-8"):
            try:
                line = first.decode(enc).strip()
                break
            except Exception:
                continue
        if not line:
            continue
        m = re.match(r"^(\d{6})\s+(.+?)\s+日线", line)
        if m:
            mapping[m.group(2)] = path.stem
    return mapping


def parse_perfect_success(name_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for path in sorted(CASE_ROOT.iterdir()):
        if not path.is_file():
            continue
        m = PIN_NAME_RE.match(path.name)
        if not m:
            continue
        code = name_map.get(m.group("name"))
        rows.append(
            {
                "label": "success",
                "source": "perfect",
                "name": m.group("name"),
                "date": pd.to_datetime(m.group("date"), format="%Y%m%d"),
                "code": code,
            }
        )
    return pd.DataFrame(rows)


def parse_manual_cases(items: List[Tuple[str, str]], label: str, name_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, date_str in items:
        rows.append(
            {
                "label": label,
                "source": "manual",
                "name": name,
                "date": pd.to_datetime(date_str, format="%Y%m%d"),
                "code": name_map.get(name),
            }
        )
    return pd.DataFrame(rows)


def build_case_df(name_map: Dict[str, str]) -> pd.DataFrame:
    parts = [
        parse_perfect_success(name_map),
        parse_manual_cases(MANUAL_SUCCESS, "success", name_map),
        parse_manual_cases(MANUAL_FAIL, "fail", name_map),
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["label", "name", "date"], keep="first").reset_index(drop=True)
    return out


def add_forward_metrics(x: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = x.copy()
    n = len(out)
    entry_open = np.full(n, np.nan, dtype=float)
    max_high = {h: np.full(n, np.nan, dtype=float) for h in horizons}
    close_ret = {h: np.full(n, np.nan, dtype=float) for h in horizons}
    up_days = {h: np.full(n, np.nan, dtype=float) for h in horizons}
    for i in range(n):
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        entry_px = float(out.at[entry_idx, "open"])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        entry_open[i] = entry_px
        for h in horizons:
            last_idx = entry_idx + h - 1
            if last_idx >= n:
                continue
            hi = float(out.loc[entry_idx:last_idx, "high"].max())
            close_px = float(out.at[last_idx, "close"])
            daily = out.loc[entry_idx:last_idx, "close"].astype(float).to_numpy()
            prev = np.r_[entry_px, daily[:-1]]
            up_days[h][i] = float(np.mean(daily > prev))
            max_high[h][i] = hi / entry_px - 1.0
            close_ret[h][i] = close_px / entry_px - 1.0
    out["entry_open"] = entry_open
    for h in horizons:
        out[f"max_float_{h}d"] = max_high[h]
        out[f"close_ret_{h}d"] = close_ret[h]
        out[f"up_days_ratio_{h}d"] = up_days[h]
    return out


def build_feature_df(file_path: Path) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(str(file_path))
    if err or df is None or len(df) < MIN_BARS:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < MIN_BARS:
        return None
    df = technical_indicators.calculate_trend(df)
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(df["日期"]),
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "volume": df["成交量"].astype(float),
            "code": file_path.stem,
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
        }
    ).reset_index(drop=True)

    short_llv = x["low"].rolling(3).min()
    short_hhv = x["close"].rolling(3).max()
    short_den = (short_hhv - short_llv).replace(0, np.nan)
    short_value = (x["close"] - short_llv) / short_den * 100
    long_llv = x["low"].rolling(21).min()
    long_hhv = x["close"].rolling(21).max()
    long_den = (long_hhv - long_llv).replace(0, np.nan)
    long_value = (x["close"] - long_llv) / long_den * 100

    full_range = (x["high"] - x["low"]).replace(0, np.nan)
    body = (x["close"] - x["open"]).abs()
    body_low = np.minimum(x["open"], x["close"])
    body_high = np.maximum(x["open"], x["close"])
    x["body_ratio"] = body / full_range
    x["lower_shadow_ratio"] = (body_low - x["low"]) / full_range
    x["upper_shadow_ratio"] = (x["high"] - body_high) / full_range
    x["close_position"] = (x["close"] - x["low"]) / full_range
    x["trend_ok"] = x["trend_line"] > x["long_line"]
    x["pin_ok"] = (short_value <= 30) & (long_value >= 85)
    x["base_pin"] = x["trend_ok"] & x["pin_ok"]
    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["long_slope_5"] = x["long_line"] / x["long_line"].shift(5) - 1.0
    x["trend_line_lead"] = (x["trend_line"] - x["long_line"]) / x["close"]
    x["dist_trend"] = (x["close"] - x["trend_line"]) / x["close"]
    x["dist_long"] = (x["close"] - x["long_line"]) / x["close"]
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"] / x["close"].shift(3) - 1.0
    x["ret5"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret10"] = x["close"] / x["close"].shift(10) - 1.0
    x["vol_ma5"] = x["volume"].rolling(5).mean()
    x["vol_ma10"] = x["volume"].rolling(10).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["signal_vs_ma5"] = x["volume"] / x["vol_ma5"]
    x["signal_vs_ma10"] = x["volume"] / x["vol_ma10"]
    x["signal_vs_ma20"] = x["volume"] / x["vol_ma20"]
    x["vol_vs_prev"] = x["volume"] / x["volume"].shift(1)
    x["near_20d_high_ratio"] = x["close"] / x["high"].rolling(20).max()
    x["near_20d_low_ratio"] = x["close"] / x["low"].rolling(20).min()
    x["near_60d_high_ratio"] = x["close"] / x["high"].rolling(60).max()
    x["near_60d_low_ratio"] = x["close"] / x["low"].rolling(60).min()

    x = add_forward_metrics(x, horizons=[3, 5])

    score = (
        x["trend_slope_3"].clip(lower=0, upper=0.06).fillna(0.0) / 0.06 * 0.22
        + x["trend_slope_5"].clip(lower=0, upper=0.10).fillna(0.0) / 0.10 * 0.22
        + x["long_slope_5"].clip(lower=0, upper=0.06).fillna(0.0) / 0.06 * 0.15
        + x["trend_line_lead"].clip(lower=0, upper=0.12).fillna(0.0) / 0.12 * 0.20
        + x["ret10"].clip(lower=0, upper=0.25).fillna(0.0) / 0.25 * 0.10
        + (1.2 - x["signal_vs_ma20"].clip(lower=0, upper=1.2).fillna(1.2)) / 1.2 * 0.04
        + (1.2 - x["vol_vs_prev"].clip(lower=0, upper=1.2).fillna(1.2)) / 1.2 * 0.04
        + (0.25 - x["close_position"].clip(lower=0, upper=0.25).fillna(0.25)) / 0.25 * 0.015
        + (0.25 - x["lower_shadow_ratio"].clip(lower=0, upper=0.25).fillna(0.25)) / 0.25 * 0.015
    )
    x["sort_score"] = score.clip(lower=0.0)
    return x


def load_universe() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    all_rows: List[pd.DataFrame] = []
    per_code: Dict[str, pd.DataFrame] = {}
    files = sorted(DATA_DIR.glob("*.txt"))
    for idx, file_path in enumerate(files, 1):
        feat = build_feature_df(file_path)
        if feat is None:
            continue
        per_code[file_path.stem] = feat
        all_rows.append(feat)
        if idx % 500 == 0:
            print(f"特征进度: {idx}/{len(files)}")
    universe = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return universe, per_code


def build_coarse_combos() -> List[EntryCombo]:
    combos = []
    for s3, s5, l5, lead, ret10, ret3max, ma20max, vpmax in product(
        [0.00, 0.01, 0.02],
        [0.03, 0.05, 0.07],
        [0.00, 0.02, 0.04],
        [0.03, 0.05, 0.07],
        [0.05, 0.10, 0.15],
        [0.00, 0.01],
        [1.00, 1.10, 1.20],
        [1.00, 1.10, 1.20],
    ):
        combos.append(EntryCombo(s3, s5, l5, lead, ret10, ret3max, ma20max, vpmax))
    return combos


def refine_combos(base: EntryCombo) -> List[EntryCombo]:
    combos = []
    for cp, ls in product([None, 0.12, 0.16, 0.20, 0.25], [None, 0.10, 0.15, 0.20, 0.25]):
        combos.append(
            EntryCombo(
                base.trend_slope_3_min,
                base.trend_slope_5_min,
                base.long_slope_5_min,
                base.trend_lead_min,
                base.ret10_min,
                base.ret3_max,
                base.signal_vs_ma20_max,
                base.vol_vs_prev_max,
                cp,
                ls,
            )
        )
    return combos


def mask_combo(df: pd.DataFrame, combo: EntryCombo) -> pd.Series:
    mask = df["base_pin"].fillna(False)
    mask &= df["trend_slope_3"].fillna(-np.inf) >= combo.trend_slope_3_min
    mask &= df["trend_slope_5"].fillna(-np.inf) >= combo.trend_slope_5_min
    mask &= df["long_slope_5"].fillna(-np.inf) >= combo.long_slope_5_min
    mask &= df["trend_line_lead"].fillna(-np.inf) >= combo.trend_lead_min
    mask &= df["ret10"].fillna(-np.inf) >= combo.ret10_min
    mask &= df["ret3"].fillna(np.inf) <= combo.ret3_max
    mask &= df["signal_vs_ma20"].fillna(np.inf) <= combo.signal_vs_ma20_max
    mask &= df["vol_vs_prev"].fillna(np.inf) <= combo.vol_vs_prev_max
    if combo.close_position_max is not None:
        mask &= df["close_position"].fillna(np.inf) <= combo.close_position_max
    if combo.lower_shadow_max is not None:
        mask &= df["lower_shadow_ratio"].fillna(np.inf) <= combo.lower_shadow_max
    return mask


def attach_cases(case_df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "code",
        "date",
        "base_pin",
        "trend_ok",
        "pin_ok",
        "trend_slope_3",
        "trend_slope_5",
        "long_slope_5",
        "trend_line_lead",
        "dist_trend",
        "dist_long",
        "ret1",
        "ret3",
        "ret5",
        "ret10",
        "signal_vs_ma5",
        "signal_vs_ma10",
        "signal_vs_ma20",
        "vol_vs_prev",
        "lower_shadow_ratio",
        "close_position",
        "sort_score",
        "entry_open",
        "max_float_3d",
        "close_ret_3d",
        "max_float_5d",
        "close_ret_5d",
    ]
    return case_df.merge(universe[keep_cols], on=["code", "date"], how="left")


def combo_case_stats(case_features: pd.DataFrame, combo: EntryCombo) -> dict:
    hit = mask_combo(case_features, combo)
    df = case_features.loc[hit].copy()
    success_total = int((case_features["label"] == "success").sum())
    fail_total = int((case_features["label"] == "fail").sum())
    success_hit = int((df["label"] == "success").sum())
    fail_hit = int((df["label"] == "fail").sum())
    precision = success_hit / (success_hit + fail_hit) if (success_hit + fail_hit) > 0 else 0.0
    return {
        "success_hit": success_hit,
        "fail_hit": fail_hit,
        "success_capture": success_hit / success_total if success_total else np.nan,
        "fail_select_rate": fail_hit / fail_total if fail_total else np.nan,
        "sample_precision": precision,
    }


def combo_hist_stats(universe: pd.DataFrame, combo: EntryCombo) -> dict:
    hit = mask_combo(universe, combo)
    df = universe.loc[hit].copy()
    if df.empty:
        return {
            "universe_count": 0,
            "avg_max_float_3d": np.nan,
            "avg_max_float_5d": np.nan,
            "avg_close_ret_5d": np.nan,
            "escape_rate_3d": np.nan,
            "escape_rate_5d": np.nan,
            "close_positive_5d": np.nan,
        }
    return {
        "universe_count": int(len(df)),
        "avg_max_float_3d": float(df["max_float_3d"].mean()),
        "avg_max_float_5d": float(df["max_float_5d"].mean()),
        "avg_close_ret_5d": float(df["close_ret_5d"].mean()),
        "escape_rate_3d": float((df["max_float_3d"] >= 0.03).mean()),
        "escape_rate_5d": float((df["max_float_5d"] >= 0.05).mean()),
        "close_positive_5d": float((df["close_ret_5d"] > 0).mean()),
    }


def combo_objective(row: dict) -> float:
    if row["universe_count"] < 80:
        return -1.0
    avg_close_scaled = np.clip((row["avg_close_ret_5d"] + 0.02) / 0.12, 0.0, 1.0)
    return (
        0.25 * row["success_capture"]
        + 0.15 * row["sample_precision"]
        + 0.15 * (1.0 - row["fail_select_rate"])
        + 0.20 * row["escape_rate_3d"]
        + 0.15 * row["escape_rate_5d"]
        + 0.10 * avg_close_scaled
    )


def evaluate_entry_combos(universe: pd.DataFrame, case_features: pd.DataFrame, combos: List[EntryCombo]) -> pd.DataFrame:
    rows = []
    for idx, combo in enumerate(combos, 1):
        case_stats = combo_case_stats(case_features, combo)
        hist_stats = combo_hist_stats(universe, combo)
        row = {**asdict(combo), **case_stats, **hist_stats, "combo_name": combo.name}
        row["objective"] = combo_objective(row)
        rows.append(row)
        if idx % 500 == 0:
            print(f"买点组合进度: {idx}/{len(combos)}")
    out = pd.DataFrame(rows).sort_values(
        ["objective", "success_capture", "sample_precision", "avg_max_float_5d"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out


def stop_price(df: pd.DataFrame, signal_idx: int, sl_mode: str) -> Optional[float]:
    signal_low = float(df.at[signal_idx, "low"])
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_day_low = float(df.at[entry_idx, "low"])
    if sl_mode == "none":
        return None
    if sl_mode == "signal_low":
        return signal_low
    if sl_mode == "signal_low_x_0.99":
        return signal_low * 0.99
    if sl_mode == "entry_day_low":
        return entry_day_low
    if sl_mode == "entry_day_low_x_0.99":
        return entry_day_low * 0.99
    return None


def is_vol_bear(df: pd.DataFrame, idx: int) -> bool:
    if idx <= 0:
        return False
    open_px = float(df.at[idx, "open"])
    close_px = float(df.at[idx, "close"])
    vol = float(df.at[idx, "volume"])
    prev_vol = float(df.at[idx - 1, "volume"])
    ma10 = float(df["volume"].rolling(10).mean().iloc[idx])
    return (
        np.isfinite(open_px)
        and np.isfinite(close_px)
        and close_px < open_px
        and np.isfinite(vol)
        and np.isfinite(prev_vol)
        and np.isfinite(ma10)
        and vol >= prev_vol * 1.5
        and vol >= ma10 * 1.5
    )


def simulate_trade(df: pd.DataFrame, signal_idx: int, exit_cfg: ExitConfig) -> Optional[dict]:
    entry_idx = signal_idx + 1
    last_hold_idx = entry_idx + exit_cfg.max_hold_days - 1
    if last_hold_idx >= len(df):
        return None
    entry_px = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    tp_px = entry_px * (1.0 + exit_cfg.tp_pct) if exit_cfg.tp_pct is not None else None
    sl_px = stop_price(df, signal_idx, exit_cfg.sl_mode)
    exit_idx = last_hold_idx
    exit_px = float(df.at[last_hold_idx, "close"]) if exit_cfg.forced_exit == "close" else float(df.at[last_hold_idx, "open"])
    exit_reason = f"hold_{exit_cfg.max_hold_days}_{exit_cfg.forced_exit}"

    for j in range(entry_idx + 1, last_hold_idx + 1):
        stop_hit = sl_px is not None and float(df.at[j, "low"]) <= sl_px
        tp_hit = tp_px is not None and float(df.at[j, "high"]) >= tp_px
        trend_break = False
        if exit_cfg.trend_break_mode == "close_below_trend":
            trend_val = float(df.at[j, "trend_line"])
            close_px = float(df.at[j, "close"])
            trend_break = np.isfinite(trend_val) and np.isfinite(close_px) and close_px < trend_val
        elif exit_cfg.trend_break_mode == "two_close_below_trend" and j >= entry_idx + 2:
            close_j = float(df.at[j, "close"])
            close_1 = float(df.at[j - 1, "close"])
            trend_j = float(df.at[j, "trend_line"])
            trend_1 = float(df.at[j - 1, "trend_line"])
            trend_break = (
                np.isfinite(close_j)
                and np.isfinite(close_1)
                and np.isfinite(trend_j)
                and np.isfinite(trend_1)
                and close_j < trend_j
                and close_1 < trend_1
            )
        vol_bear = exit_cfg.vol_bear_exit and is_vol_bear(df, j)

        if exit_cfg.trigger_exec == "same_day":
            if stop_hit:
                exit_idx = j
                exit_px = sl_px
                exit_reason = "stop_same_day"
                break
            if tp_hit:
                exit_idx = j
                exit_px = tp_px
                exit_reason = "take_profit_same_day"
                break
            if trend_break:
                exit_idx = j
                exit_px = float(df.at[j, "close"])
                exit_reason = f"{exit_cfg.trend_break_mode}_same_close"
                break
            if vol_bear:
                exit_idx = j
                exit_px = float(df.at[j, "close"])
                exit_reason = "vol_bear_same_close"
                break
        else:
            if j >= last_hold_idx:
                continue
            if stop_hit:
                exit_idx = j + 1
                exit_px = float(df.at[exit_idx, "open"])
                exit_reason = "stop_next_open"
                break
            if tp_hit:
                exit_idx = j + 1
                exit_px = float(df.at[exit_idx, "open"])
                exit_reason = "take_profit_next_open"
                break
            if trend_break:
                exit_idx = j + 1
                exit_px = float(df.at[exit_idx, "open"])
                exit_reason = f"{exit_cfg.trend_break_mode}_next_open"
                break
            if vol_bear:
                exit_idx = j + 1
                exit_px = float(df.at[exit_idx, "open"])
                exit_reason = "vol_bear_next_open"
                break

    ret = exit_px / entry_px - 1.0
    return {
        "signal_date": df.at[signal_idx, "date"],
        "entry_date": df.at[entry_idx, "date"],
        "exit_date": df.at[exit_idx, "date"],
        "code": df.at[signal_idx, "code"],
        "ret": ret,
        "success": ret > 0,
        "sort_score": float(df.at[signal_idx, "sort_score"]),
        "exit_reason": exit_reason,
    }


def build_exit_configs() -> List[ExitConfig]:
    configs = [
        ExitConfig("hold3_close", 3, "close"),
        ExitConfig("hold5_close", 5, "close"),
        ExitConfig("trend_break_next_open", 5, "close", trend_break_mode="close_below_trend", trigger_exec="next_open"),
        ExitConfig("two_trend_break_next_open", 5, "close", trend_break_mode="two_close_below_trend", trigger_exec="next_open"),
        ExitConfig("vol_bear_next_open", 5, "close", vol_bear_exit=True, trigger_exec="next_open"),
    ]
    for tp, sl_mode, exec_mode in product(
        [0.03, 0.05, 0.08, 0.10],
        ["none", "signal_low", "signal_low_x_0.99", "entry_day_low", "entry_day_low_x_0.99"],
        ["same_day", "next_open"],
    ):
        configs.append(
            ExitConfig(
                name=f"tp_{tp:.2f}__sl_{sl_mode}__{exec_mode}",
                max_hold_days=5,
                forced_exit="close",
                tp_pct=tp,
                sl_mode=sl_mode,
                trigger_exec=exec_mode,
            )
        )
    return configs


def build_trade_df(per_code: Dict[str, pd.DataFrame], combo: EntryCombo, exit_cfg: ExitConfig) -> pd.DataFrame:
    trades = []
    for code, df in per_code.items():
        hit_idxs = np.flatnonzero(mask_combo(df, combo).to_numpy())
        if len(hit_idxs) == 0:
            continue
        for signal_idx in hit_idxs:
            trade = simulate_trade(df, int(signal_idx), exit_cfg)
            if trade is not None:
                trades.append(trade)
    if not trades:
        return pd.DataFrame()
    out = pd.DataFrame(trades)
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.sort_values(["signal_date", "sort_score", "code"], ascending=[True, False, True]).reset_index(drop=True)


def build_portfolio_curve(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame(columns=["signal_date", "portfolio_ret", "equity"])
    rows = []
    equity = INITIAL_CAPITAL
    for signal_date, group in trade_df.groupby("signal_date", sort=True):
        g = group.head(MAX_POSITIONS).copy()
        score = g["sort_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        if score.sum() <= 0:
            weights = np.repeat(1.0 / len(g), len(g))
        else:
            weights = (score / score.sum()).clip(upper=MAX_SINGLE_WEIGHT).to_numpy()
            weights = weights / weights.sum()
        basket_ret = float(np.sum(g["ret"].to_numpy() * weights))
        equity *= 1.0 + basket_ret
        rows.append({"signal_date": signal_date, "portfolio_ret": basket_ret, "equity": equity})
    return pd.DataFrame(rows)


def max_consecutive_failures(flags: List[bool]) -> int:
    cur = 0
    worst = 0
    for flag in flags:
        if flag:
            cur = 0
        else:
            cur += 1
            worst = max(worst, cur)
    return worst


def summarize_exit(combo: EntryCombo, exit_cfg: ExitConfig, trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    if portfolio_df.empty:
        annual = np.nan
        max_dd = np.nan
        final_equity = np.nan
        equity_days = 0
    else:
        eq = portfolio_df["equity"].astype(float)
        max_dd = float((eq / eq.cummax() - 1.0).min())
        final_equity = float(eq.iloc[-1])
        equity_days = len(portfolio_df)
        annual = float((final_equity / INITIAL_CAPITAL) ** (TRADING_DAYS_PER_YEAR / equity_days) - 1) if final_equity > 0 and equity_days > 0 else np.nan
    return {
        "combo_name": combo.name,
        "exit_name": exit_cfg.name,
        "sample_count": int(len(trade_df)),
        "avg_trade_return": float(trade_df["ret"].mean()) if not trade_df.empty else np.nan,
        "success_rate": float(trade_df["success"].mean()) if not trade_df.empty else np.nan,
        "max_consecutive_failures": int(max_consecutive_failures(trade_df["success"].tolist())) if not trade_df.empty else np.nan,
        "annual_return": annual,
        "max_drawdown": max_dd,
        "equity_days": int(equity_days),
        "final_equity": final_equity,
    }


def main() -> None:
    name_map = load_name_map()
    case_df = build_case_df(name_map)
    case_df.to_csv(OUT_DIR / "case_labels.csv", index=False, encoding="utf-8-sig")

    universe, per_code = load_universe()
    universe.to_csv(OUT_DIR / "pin_universe_features.csv", index=False, encoding="utf-8-sig")
    case_features = attach_cases(case_df, universe)
    case_features.to_csv(OUT_DIR / "case_features.csv", index=False, encoding="utf-8-sig")

    coarse = build_coarse_combos()
    coarse_res = evaluate_entry_combos(universe, case_features, coarse)
    coarse_res.to_csv(OUT_DIR / "entry_coarse_results.csv", index=False, encoding="utf-8-sig")

    top_base = [
        EntryCombo(
            row["trend_slope_3_min"],
            row["trend_slope_5_min"],
            row["long_slope_5_min"],
            row["trend_lead_min"],
            row["ret10_min"],
            row["ret3_max"],
            row["signal_vs_ma20_max"],
            row["vol_vs_prev_max"],
        )
        for _, row in coarse_res.head(15).iterrows()
    ]
    refined_list: List[EntryCombo] = []
    seen = set()
    for base in top_base:
        for combo in refine_combos(base):
            if combo.name not in seen:
                seen.add(combo.name)
                refined_list.append(combo)

    refined_res = evaluate_entry_combos(universe, case_features, refined_list)
    refined_res.to_csv(OUT_DIR / "entry_refined_results.csv", index=False, encoding="utf-8-sig")

    best_row = refined_res.iloc[0]
    best_combo = EntryCombo(
        best_row["trend_slope_3_min"],
        best_row["trend_slope_5_min"],
        best_row["long_slope_5_min"],
        best_row["trend_lead_min"],
        best_row["ret10_min"],
        best_row["ret3_max"],
        best_row["signal_vs_ma20_max"],
        best_row["vol_vs_prev_max"],
        None if pd.isna(best_row["close_position_max"]) else float(best_row["close_position_max"]),
        None if pd.isna(best_row["lower_shadow_max"]) else float(best_row["lower_shadow_max"]),
    )

    exit_rows = []
    exit_cfgs = build_exit_configs()
    for idx, exit_cfg in enumerate(exit_cfgs, 1):
        trades = build_trade_df(per_code, best_combo, exit_cfg)
        portfolio = build_portfolio_curve(trades)
        row = summarize_exit(best_combo, exit_cfg, trades, portfolio)
        exit_rows.append(row)
        if idx % 10 == 0:
            print(f"退出组合进度: {idx}/{len(exit_cfgs)}")
    exit_res = pd.DataFrame(exit_rows).sort_values(
        ["annual_return", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    exit_res.to_csv(OUT_DIR / "exit_results.csv", index=False, encoding="utf-8-sig")

    best_exit = exit_res.iloc[0].to_dict() if not exit_res.empty else None
    summary = {
        "case_count": int(len(case_df)),
        "matched_case_count": int(case_features["code"].notna().sum()),
        "success_count": int((case_df["label"] == "success").sum()),
        "fail_count": int((case_df["label"] == "fail").sum()),
        "universe_count": int(len(universe)),
        "best_entry_combo": best_row.to_dict(),
        "best_exit": best_exit,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
