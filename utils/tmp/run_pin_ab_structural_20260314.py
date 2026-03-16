from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brick_filter, stoploss, technical_indicators


DATA_DIR = ROOT / "data" / "forward_data"
BASE_RESULT_DIR = ROOT / "results" / "pin_rebuild_opt_20260314"
OUT_DIR = ROOT / "results" / "pin_ab_structural_20260314_fast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
MIN_BARS = 160
N_LOOKBACK = 80
EPS = 1e-12


@dataclass(frozen=True)
class BCombo:
    trend_slope_3_min: float
    trend_slope_5_min: float
    long_slope_5_min: float
    trend_lead_min: float
    ret10_min: float
    ret3_max: float
    signal_vs_ma20_min: float
    signal_vs_ma20_max: float
    vol_vs_prev_max: float
    close_position_max: Optional[float]
    lower_shadow_max: Optional[float]

    @property
    def name(self) -> str:
        parts = [
            f"s3_{self.trend_slope_3_min:.3f}",
            f"s5_{self.trend_slope_5_min:.3f}",
            f"l5_{self.long_slope_5_min:.3f}",
            f"lead_{self.trend_lead_min:.3f}",
            f"r10_{self.ret10_min:.2f}",
            f"r3max_{self.ret3_max:.2f}",
            f"ma20min_{self.signal_vs_ma20_min:.2f}",
            f"ma20max_{self.signal_vs_ma20_max:.2f}",
            f"vp_{self.vol_vs_prev_max:.2f}",
            "cp_none" if self.close_position_max is None else f"cp_{self.close_position_max:.2f}",
            "ls_none" if self.lower_shadow_max is None else f"ls_{self.lower_shadow_max:.2f}",
        ]
        return "__".join(parts)


def safe_div(a, b):
    if a is None or b is None:
        return np.nan
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return np.nan
    if not math.isfinite(af) or not math.isfinite(bf) or abs(bf) <= EPS:
        return np.nan
    return af / bf


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float)

    def _pct_last(arr: np.ndarray) -> float:
        if len(arr) == 0 or not np.isfinite(arr[-1]):
            return np.nan
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= arr[-1]) / len(valid))

    return values.rolling(window, min_periods=window).apply(_pct_last, raw=True)


def identify_low_zones(mask_series: pd.Series) -> List[Tuple[int, int]]:
    mask = mask_series.fillna(False).to_numpy(dtype=bool)
    zones: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            zones.append((start, i - 1))
            start = None
    if start is not None:
        zones.append((start, len(mask) - 1))
    return zones


def build_n_up_feature(df: pd.DataFrame, rank_col: str, rank_threshold: float) -> pd.Series:
    out = np.zeros(len(df), dtype=bool)
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    rank_values = df[rank_col].astype(float)
    for idx in range(len(df)):
        left = max(0, idx - N_LOOKBACK + 1)
        sub_rank = rank_values.iloc[left : idx + 1].reset_index(drop=True)
        zones = identify_low_zones(sub_rank <= rank_threshold)
        if len(zones) < 2:
            continue
        z1, z2 = zones[-2], zones[-1]
        z1_start, z1_end = left + z1[0], left + z1[1]
        z2_start, z2_end = left + z2[0], left + z2[1]
        first_low = float(np.min(lows[z1_start : z1_end + 1]))
        second_low = float(np.min(lows[z2_start : z2_end + 1]))
        if not (second_low > first_low):
            continue
        mid_left = z1_end + 1
        mid_right = z2_start - 1
        if mid_right < mid_left:
            continue
        rebound_high = float(np.max(highs[mid_left : mid_right + 1]))
        if closes[idx] > rebound_high:
            out[idx] = True
    return pd.Series(out, index=df.index)


def add_structure_tags(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["j_rank_20"] = rolling_last_percentile(out["J"], 20)
    out["j_rank_30"] = rolling_last_percentile(out["J"], 30)
    out["n_up_rank20_p10"] = build_n_up_feature(out, "j_rank_20", 0.10)
    out["n_up_rank30_p10"] = build_n_up_feature(out, "j_rank_30", 0.10)
    out["n_up_any"] = out["n_up_rank20_p10"] | out["n_up_rank30_p10"]

    anchor_df = brick_filter.last_double_bull_anchor(out, lookback=60)
    out = pd.concat([out, anchor_df], axis=1)
    keyk_df = brick_filter.derive_keyk_states(out)
    out = pd.concat([out, keyk_df], axis=1)

    trend_slope_10 = out["trend_line"] / out["trend_line"].shift(10) - 1.0
    above_trend = (out["close"] >= out["trend_line"] * 0.99).astype(float)
    above_ratio_15 = above_trend.rolling(15, min_periods=15).mean()
    dist_trend = (out["close"] - out["trend_line"]) / out["close"].replace(0, np.nan)
    dist_min_15 = dist_trend.rolling(15, min_periods=15).min()
    out["along_trend_up"] = (
        (trend_slope_10 > 0.02)
        & (above_ratio_15 >= 0.80)
        & (dist_min_15 > -0.03)
    )

    full_range = (out["high"] - out["low"]).replace(0, np.nan)
    body_ratio = (out["close"] - out["open"]).abs() / full_range
    prev_vol = out["volume"].shift(1)
    vol_rank30 = (
        out["volume"]
        .rolling(30, min_periods=1)
        .apply(lambda s: pd.Series(s).rank(method="min", ascending=False).iloc[-1], raw=False)
    )
    giant_bear = (
        (out["close"] < out["open"])
        & ((out["close"] / out["close"].shift(1) - 1.0) < -0.03)
        & (body_ratio > 0.40)
        & (
            (vol_rank30 <= 3.0)
            | ((out["volume"] / prev_vol.replace(0, np.nan)) >= 2.0)
        )
    )
    giant_bear_30 = giant_bear.rolling(30, min_periods=1).max().fillna(0.0).astype(bool)
    out["no_giant_bear_30"] = ~giant_bear_30
    return out


def load_tag_rows(file_path: Path) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(str(file_path))
    if err or df is None or len(df) < MIN_BARS:
        return None
    df = df[(df["日期"] < EXCLUDE_START) | (df["日期"] > EXCLUDE_END)].copy()
    if len(df) < MIN_BARS:
        return None
    df = technical_indicators.calculate_trend(df)
    df = technical_indicators.calculate_kdj(df)
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(df["日期"]),
            "open": df["开盘"].astype(float),
            "high": df["最高"].astype(float),
            "low": df["最低"].astype(float),
            "close": df["收盘"].astype(float),
            "volume": df["成交量"].astype(float),
            "trend_line": df["知行短期趋势线"].astype(float),
            "long_line": df["知行多空线"].astype(float),
            "J": df["J"].astype(float),
            "code": file_path.stem,
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
    x["base_pin"] = (x["trend_line"] > x["long_line"]) & (short_value <= 30) & (long_value >= 85)
    if not x["base_pin"].any():
        return None

    x = add_structure_tags(x)
    out = x.loc[
        x["base_pin"],
        [
            "code",
            "date",
            "n_up_rank20_p10",
            "n_up_rank30_p10",
            "n_up_any",
            "keyk_support_active",
            "along_trend_up",
            "no_giant_bear_30",
        ],
    ].copy()
    return out


def load_tag_df(target_codes: set[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    files = [fp for fp in sorted(DATA_DIR.glob("*.txt")) if fp.stem in target_codes]
    for idx, fp in enumerate(files, 1):
        tag_rows = load_tag_rows(fp)
        if tag_rows is not None and not tag_rows.empty:
            rows.append(tag_rows)
        if idx % 500 == 0 or idx == len(files):
            print(f"结构标签进度: {idx}/{len(files)}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def subtype_a_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["base_pin"].fillna(False)
    mask &= df["trend_slope_5"].fillna(-np.inf) >= 0.03
    mask &= df["trend_line_lead"].fillna(-np.inf) >= 0.03
    mask &= df["ret10"].fillna(-np.inf) >= 0.05
    mask &= df["ret3"].fillna(np.inf) <= 0.01
    mask &= df["signal_vs_ma20"].fillna(np.inf) <= 1.0
    mask &= df["vol_vs_prev"].fillna(np.inf) <= 1.1
    mask &= df["lower_shadow_ratio"].fillna(np.inf) <= 0.25
    return mask


def subtype_b_mask(df: pd.DataFrame, combo: BCombo) -> pd.Series:
    mask = df["base_pin"].fillna(False)
    mask &= df["trend_slope_3"].fillna(-np.inf) >= combo.trend_slope_3_min
    mask &= df["trend_slope_5"].fillna(-np.inf) >= combo.trend_slope_5_min
    mask &= df["long_slope_5"].fillna(-np.inf) >= combo.long_slope_5_min
    mask &= df["trend_line_lead"].fillna(-np.inf) >= combo.trend_lead_min
    mask &= df["ret10"].fillna(-np.inf) >= combo.ret10_min
    mask &= df["ret3"].fillna(np.inf) <= combo.ret3_max
    mask &= df["signal_vs_ma20"].fillna(-np.inf) >= combo.signal_vs_ma20_min
    mask &= df["signal_vs_ma20"].fillna(np.inf) <= combo.signal_vs_ma20_max
    mask &= df["vol_vs_prev"].fillna(np.inf) <= combo.vol_vs_prev_max
    if combo.close_position_max is not None:
        mask &= df["close_position"].fillna(np.inf) <= combo.close_position_max
    if combo.lower_shadow_max is not None:
        mask &= df["lower_shadow_ratio"].fillna(np.inf) <= combo.lower_shadow_max
    return mask


def build_b_combos() -> List[BCombo]:
    combos: List[BCombo] = []
    for s3, s5, l5, lead, ret10, r3max, ma20min, ma20max, vpmax, cp, ls in product(
        [0.02, 0.03, 0.04],
        [0.04, 0.05, 0.06],
        [0.00, 0.02],
        [0.02, 0.03, 0.04],
        [0.05, 0.08, 0.10, 0.15],
        [0.00, 0.03, 0.05],
        [0.90, 1.00],
        [1.30, 1.50, 1.80],
        [1.10, 1.30, 1.50],
        [0.10, 0.20, None],
        [0.10, 0.20, None],
    ):
        if ma20max <= ma20min:
            continue
        combos.append(BCombo(s3, s5, l5, lead, ret10, r3max, ma20min, ma20max, vpmax, cp, ls))
    return combos


def tag_variants(df: pd.DataFrame) -> Dict[str, pd.Series]:
    n_up_any = df["n_up_any"].fillna(False)
    keyk = df["keyk_support_active"].fillna(False)
    along = df["along_trend_up"].fillna(False)
    clean = df["no_giant_bear_30"].fillna(False)
    return {
        "none": pd.Series(True, index=df.index),
        "n_up_any": n_up_any,
        "keyk_support": keyk,
        "along_trend": along,
        "clean_history": clean,
        "n_or_keyk": n_up_any | keyk,
        "keyk_or_along": keyk | along,
        "trend_clean": along & clean,
        "quality_any": (n_up_any | keyk | along) & clean,
    }


def summarize(mask: pd.Series, df: pd.DataFrame) -> Dict[str, float]:
    selected = df[mask.fillna(False)]
    return {
        "universe_count": int(len(selected)),
        "avg_max_float_5d": float(selected["max_float_5d"].mean()) if len(selected) else np.nan,
        "avg_close_ret_5d": float(selected["close_ret_5d"].mean()) if len(selected) else np.nan,
        "escape_rate_5d": float((selected["max_float_5d"] > 0.03).mean()) if len(selected) else np.nan,
        "close_positive_5d": float((selected["close_ret_5d"] > 0).mean()) if len(selected) else np.nan,
    }


def main():
    universe = pd.read_csv(BASE_RESULT_DIR / "pin_universe_features.csv", parse_dates=["date"])
    case_features = pd.read_csv(BASE_RESULT_DIR / "case_features.csv", parse_dates=["date"])
    target_codes = set(universe.loc[universe["base_pin"].fillna(False), "code"].astype(str).unique())

    tag_df = load_tag_df(target_codes)
    tag_df.to_csv(OUT_DIR / "tag_rows.csv", index=False, encoding="utf-8-sig")

    universe = universe.merge(tag_df, on=["code", "date"], how="left")
    case_features = case_features.merge(tag_df, on=["code", "date"], how="left")

    a_univ = subtype_a_mask(universe)
    a_case = subtype_a_mask(case_features)
    a_summary = summarize(a_univ, universe)

    base_success = (case_features["label"] == "success") & case_features["base_pin"].fillna(False)
    base_fail = (case_features["label"] == "fail") & case_features["base_pin"].fillna(False)
    a_success_hit = int((a_case & base_success).sum())
    a_fail_hit = int((a_case & base_fail).sum())

    baseline = {
        "variant": "A_only",
        "combo_name": "current_balanced",
        "tag_variant": "none",
        "success_hit_total": a_success_hit,
        "fail_hit_total": a_fail_hit,
        "success_added": 0,
        "fail_added": 0,
        **a_summary,
    }

    combos = build_b_combos()
    tag_map_univ = tag_variants(universe)
    tag_map_case = tag_variants(case_features)

    rows: List[dict] = [baseline]
    for idx, combo in enumerate(combos, 1):
        bmask_univ_base = subtype_b_mask(universe, combo)
        bmask_case_base = subtype_b_mask(case_features, combo)
        for tag_name in tag_map_univ.keys():
            bmask_univ = bmask_univ_base & tag_map_univ[tag_name]
            bmask_case = bmask_case_base & tag_map_case[tag_name]
            union_univ = a_univ | bmask_univ
            union_case = a_case | bmask_case

            success_total = int((union_case & base_success).sum())
            fail_total = int((union_case & base_fail).sum())
            success_added = int((~a_case & bmask_case & base_success).sum())
            fail_added = int((~a_case & bmask_case & base_fail).sum())
            row = {
                "variant": "A_or_B",
                "combo_name": combo.name,
                "tag_variant": tag_name,
                "success_hit_total": success_total,
                "fail_hit_total": fail_total,
                "success_added": success_added,
                "fail_added": fail_added,
                **summarize(union_univ, universe),
            }
            row["objective"] = (
                row["success_hit_total"] * 100.0
                - row["fail_hit_total"] * 120.0
                + row["success_added"] * 35.0
                - row["fail_added"] * 45.0
                + (0.0 if pd.isna(row["avg_close_ret_5d"]) else row["avg_close_ret_5d"] * 500.0)
                + (0.0 if pd.isna(row["escape_rate_5d"]) else row["escape_rate_5d"] * 10.0)
            )
            rows.append(row)
        if idx % 500 == 0 or idx == len(combos):
            print(f"B型组合进度: {idx}/{len(combos)}")

    res = pd.DataFrame(rows).sort_values(
        ["success_hit_total", "fail_hit_total", "success_added", "avg_close_ret_5d", "objective"],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)
    res.to_csv(OUT_DIR / "ab_results.csv", index=False, encoding="utf-8-sig")

    top = res.head(30).copy()
    top.to_csv(OUT_DIR / "ab_top30.csv", index=False, encoding="utf-8-sig")

    summary = {
        "baseline": baseline,
        "best_union": res.iloc[0].to_dict(),
        "second_union": res.iloc[1].to_dict() if len(res) > 1 else None,
        "third_union": res.iloc[2].to_dict() if len(res) > 2 else None,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
