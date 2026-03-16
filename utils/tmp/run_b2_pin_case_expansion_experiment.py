from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b2filter, stoploss, technical_indicators

CASE_ROOT = ROOT / "data" / "完美图"
NORMAL_DIR = ROOT / "data" / "20260313" / "normal"
NAME_MAP_DIR = ROOT / "data" / "20260313"
OUT_DIR = ROOT / "results" / "b2_pin_case_expansion_experiment_20260313"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
DATE_RE = re.compile(r"^(?P<name>.+?)(?P<date>20\d{6})\.[^.]+$")
EPS = 1e-12


def safe_div(a, b):
    if b is None or (isinstance(b, float) and not math.isfinite(b)) or abs(float(b)) <= EPS:
        return np.nan
    if a is None or (isinstance(a, float) and not math.isfinite(a)):
        return np.nan
    return float(a) / float(b)


def exclude_disaster(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return df[(df[date_col] < EXCLUDE_START) | (df[date_col] > EXCLUDE_END)].copy()


def load_name_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in NAME_MAP_DIR.glob("*.txt"):
        try:
            first = path.read_bytes().splitlines()[0]
        except Exception:
            continue
        line = None
        for enc in ("gb18030", "gbk", "utf-8"):
            try:
                line = first.decode(enc)
                break
            except Exception:
                continue
        if not line:
            continue
        m = re.match(r"^(\d{6})\s+(.+?)\s+日线", line.strip())
        if not m:
            continue
        mapping[m.group(2)] = path.stem
    return mapping


def parse_cases(subdir: str, name_map: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    bad = []
    for path in sorted((CASE_ROOT / subdir).iterdir()):
        if not path.is_file():
            continue
        m = DATE_RE.match(path.name)
        if not m:
            bad.append(path.name)
            continue
        rows.append(
            {
                "folder": subdir,
                "case_file": path.name,
                "name": m.group("name"),
                "date": pd.to_datetime(m.group("date"), format="%Y%m%d"),
                "code": name_map.get(m.group("name")),
            }
        )
    return pd.DataFrame(rows), bad


def body_shadow_flags(open_s, close_s, high_s, low_s):
    body = (close_s - open_s).abs()
    upper = high_s - np.maximum(open_s, close_s)
    lower = np.minimum(open_s, close_s) - low_s
    full = (high_s - low_s).replace(0, np.nan)
    return body, upper, lower, full


def add_future_metrics(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    x = df.copy()
    entry = x["open"].shift(-1)
    x[f"{prefix}entry_open"] = entry
    for h in (3, 5, 10, 20, 30):
        max_high = x["high"].shift(-1).rolling(h, min_periods=1).max()
        close_h = x["close"].shift(-h)
        x[f"{prefix}max_float_{h}d"] = max_high / entry - 1.0
        x[f"{prefix}close_ret_{h}d"] = close_h / entry - 1.0
    return x


def build_b2_features_for_one(path: Path) -> Optional[pd.DataFrame]:
    df = b2filter.load_one_csv(str(path))
    if df is None or df.empty:
        return None
    x = b2filter.add_features(df)
    x = exclude_disaster(x, "date")
    if x.empty:
        return None
    body = (x["close"] - x["open"]).abs()
    upper = x["high"] - np.maximum(x["open"], x["close"])
    full = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = body / full
    x["upper_shadow_body_ratio"] = upper / body.replace(0, np.nan)
    x["vol_vs_prev"] = x["volume"] / x["volume"].shift(1)
    x["vol_ma10"] = x["volume"].rolling(10).mean()
    x["vol_vs_ma10"] = x["volume"] / x["vol_ma10"]
    x["near_20d_high_ratio"] = x["close"] / x["high"].rolling(20).max()
    x["near_20d_low_ratio"] = x["close"] / x["low"].rolling(20).min()
    x["trend_slope_3"] = x["trend_line"] / x["trend_line"].shift(3) - 1.0
    x["trend_slope_5"] = x["trend_line"] / x["trend_line"].shift(5) - 1.0
    x["long_slope_3"] = x["long_line"] / x["long_line"].shift(3) - 1.0
    x["long_slope_5"] = x["long_line"] / x["long_line"].shift(5) - 1.0
    x["trend_line_lead"] = (x["trend_line"] - x["long_line"]) / x["close"]
    x["prev_ret1"] = x["ret1"].shift(1)
    x["ret3"] = x["close"] / x["close"].shift(3) - 1.0
    x["ret5"] = x["close"] / x["close"].shift(5) - 1.0
    x["ret10"] = x["close"] / x["close"].shift(10) - 1.0
    x["any_line_start"] = x["trend_start"] | x["long_start"]
    x["small_upper_shadow_05"] = (body <= EPS) | (upper <= body * 0.5 + EPS)
    x["small_upper_shadow_08"] = (body <= EPS) | (upper <= body * 0.8 + EPS)
    x = add_future_metrics(x, prefix="")
    x["base_relaxed"] = (
        x["trend_ok"]
        & (x["ret1"] >= 0.03)
        & (x["close"] > x["open"])
        & x["b2_volume_ok"]
        & x["b2_j_ok"]
    )
    x["code"] = path.stem
    return x


def build_pin_features_for_one(path: Path) -> Optional[pd.DataFrame]:
    df, err = stoploss.load_data(str(path))
    if err or df is None or len(df) < 25:
        return None
    df = technical_indicators.calculate_trend(df)
    x = df.copy().reset_index(drop=True)
    x["日期"] = pd.to_datetime(x["日期"], errors="coerce")
    x = exclude_disaster(x, "日期")
    if x.empty:
        return None

    llv_l_n1 = x["最低"].rolling(window=3).min()
    hhv_c_n1 = x["收盘"].rolling(window=3).max()
    x["短期"] = (x["收盘"] - llv_l_n1) / (hhv_c_n1 - llv_l_n1) * 100
    llv_l_n2 = x["最低"].rolling(window=21).min()
    hhv_l_n2 = x["收盘"].rolling(window=21).max()
    x["长期"] = (x["收盘"] - llv_l_n2) / (hhv_l_n2 - llv_l_n2) * 100

    body, upper, lower, full = body_shadow_flags(x["开盘"], x["收盘"], x["最高"], x["最低"])
    x["body_ratio"] = body / full
    x["upper_shadow_ratio"] = upper / full
    x["lower_shadow_ratio"] = lower / full
    x["close_position"] = (x["收盘"] - x["最低"]) / full
    x["trend_slope_3"] = x["知行短期趋势线"] / x["知行短期趋势线"].shift(3) - 1.0
    x["trend_slope_5"] = x["知行短期趋势线"] / x["知行短期趋势线"].shift(5) - 1.0
    x["long_slope_5"] = x["知行多空线"] / x["知行多空线"].shift(5) - 1.0
    x["trend_ok"] = x["知行短期趋势线"] > x["知行多空线"]
    x["dist_trend"] = x["收盘"] / x["知行短期趋势线"] - 1.0
    x["dist_long"] = x["收盘"] / x["知行多空线"] - 1.0
    x["ret1"] = x["收盘"].pct_change()
    x["ret3"] = x["收盘"] / x["收盘"].shift(3) - 1.0
    x["ret5"] = x["收盘"] / x["收盘"].shift(5) - 1.0
    x["vol_ma5"] = x["成交量"].rolling(5).mean()
    x["signal_vs_ma5"] = x["成交量"] / x["vol_ma5"]
    x["vol_vs_prev"] = x["成交量"] / x["成交量"].shift(1)
    x["near_20d_high_ratio"] = x["收盘"] / x["最高"].rolling(20).max()
    x["near_20d_low_ratio"] = x["收盘"] / x["最低"].rolling(20).min()
    x["pin_ok"] = (x["短期"] <= 30) & (x["长期"] >= 85)

    # rename to common future metric columns using next-open entry
    temp = pd.DataFrame(
        {
            "date": x["日期"],
            "open": x["开盘"],
            "high": x["最高"],
            "low": x["最低"],
            "close": x["收盘"],
        }
    )
    temp = add_future_metrics(temp, prefix="")
    for col in [c for c in temp.columns if c not in {"date", "open", "high", "low", "close"}]:
        x[col] = temp[col].values
    x["base_relaxed"] = x["trend_ok"] & x["pin_ok"]
    x["code"] = path.stem
    return x


def collect_universe_features(kind: str) -> pd.DataFrame:
    rows = []
    for path in sorted(NORMAL_DIR.glob("*.txt")):
        if kind == "b2":
            x = build_b2_features_for_one(path)
            if x is None:
                continue
            use = x[x["base_relaxed"]].copy()
            if use.empty:
                continue
            rows.append(use)
        else:
            x = build_pin_features_for_one(path)
            if x is None:
                continue
            use = x[x["base_relaxed"]].copy()
            if use.empty:
                continue
            rows.append(use)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    return all_df


def build_case_features(kind: str, cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases.iterrows():
        code = row["code"]
        if pd.isna(code) or not str(code).strip():
            continue
        code = str(code).strip()
        path = NORMAL_DIR / f"{code}.txt"
        x = build_b2_features_for_one(path) if kind == "b2" else build_pin_features_for_one(path)
        if x is None or x.empty:
            continue
        date_col = "date" if kind == "b2" else "日期"
        hit = x[x[date_col] == row["date"]]
        if hit.empty:
            continue
        rec = hit.iloc[0].to_dict()
        rec.update({"case_name": row["name"], "case_date": row["date"], "code": code})
        rows.append(rec)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class B2Combo:
    start_rule: str
    vol_rule: str
    near_rule: str
    close_rule: str
    shadow_rule: str
    jmax_rule: str
    trend_rule: str
    prev_rule: str


@dataclass(frozen=True)
class PinCombo:
    lower_rule: str
    slope_rule: str
    close_rule: str
    upper_rule: str
    ret_rule: str
    dist_rule: str
    vol_rule: str


def mask_b2(df: pd.DataFrame, combo: B2Combo) -> pd.Series:
    mask = df["trend_ok"].fillna(False)
    if combo.start_rule == "dual":
        mask &= df["dual_start"].fillna(False)
    elif combo.start_rule == "any":
        mask &= df["any_line_start"].fillna(False)
    elif combo.start_rule == "trend":
        mask &= df["trend_start"].fillna(False)
    elif combo.start_rule == "long":
        mask &= df["long_start"].fillna(False)

    if combo.vol_rule == "strict":
        mask &= df["signal_vs_ma5"].between(1.5, 2.5, inclusive="both")
    elif combo.vol_rule == "mid":
        mask &= df["signal_vs_ma5"].between(1.2, 2.2, inclusive="both")
    elif combo.vol_rule == "wide":
        mask &= df["signal_vs_ma5"].between(1.0, 2.5, inclusive="both")
    elif combo.vol_rule == "loose":
        mask &= df["signal_vs_ma5"] >= 1.0

    if combo.near_rule == "095":
        mask &= df["near_20d_high_ratio"] >= 0.95
    elif combo.near_rule == "093":
        mask &= df["near_20d_high_ratio"] >= 0.93
    elif combo.near_rule == "090":
        mask &= df["near_20d_high_ratio"] >= 0.90

    if combo.close_rule == "070":
        mask &= df["close_position"] >= 0.70
    elif combo.close_rule == "060":
        mask &= df["close_position"] >= 0.60

    if combo.shadow_rule == "03":
        mask &= df["small_upper_shadow"].fillna(False)
    elif combo.shadow_rule == "05":
        mask &= df["small_upper_shadow_05"].fillna(False)
    elif combo.shadow_rule == "08":
        mask &= df["small_upper_shadow_08"].fillna(False)

    if combo.jmax_rule == "90":
        mask &= df["J"] < 90
        mask &= df["b2_j_ok"].fillna(False)
    elif combo.jmax_rule == "100":
        mask &= df["J"] < 100
        mask &= (df["J"] > df["J"].shift(1)) & (df["J"].shift(1) < df["J"].shift(2))

    if combo.trend_rule == "s3_pos":
        mask &= df["trend_slope_3"] > 0
    elif combo.trend_rule == "s5_pos":
        mask &= df["trend_slope_5"] > 0
    elif combo.trend_rule == "lead_003":
        mask &= df["trend_line_lead"] >= 0.03

    if combo.prev_rule == "prev_le_0":
        mask &= df["prev_ret1"] <= 0
    elif combo.prev_rule == "prev_le_2":
        mask &= df["prev_ret1"] <= 0.02
    return mask.fillna(False)


def mask_pin(df: pd.DataFrame, combo: PinCombo) -> pd.Series:
    mask = df["trend_ok"].fillna(False) & df["pin_ok"].fillna(False)
    if combo.lower_rule == "005":
        mask &= df["lower_shadow_ratio"] <= 0.05
    elif combo.lower_rule == "010":
        mask &= df["lower_shadow_ratio"] <= 0.10
    elif combo.lower_rule == "020":
        mask &= df["lower_shadow_ratio"] <= 0.20
    elif combo.lower_rule == "030":
        mask &= df["lower_shadow_ratio"] <= 0.30
    elif combo.lower_rule == "010_035":
        mask &= df["lower_shadow_ratio"].between(0.10, 0.35, inclusive="both")

    if combo.slope_rule == "008":
        mask &= df["trend_slope_3"] > 0.008
    elif combo.slope_rule == "003":
        mask &= df["trend_slope_3"] > 0.003
    elif combo.slope_rule == "000":
        mask &= df["trend_slope_3"] > 0

    if combo.close_rule == "035":
        mask &= df["close_position"] <= 0.35
    elif combo.close_rule == "045":
        mask &= df["close_position"] <= 0.45
    elif combo.close_rule == "060":
        mask &= df["close_position"] <= 0.60

    if combo.upper_rule == "060":
        mask &= df["upper_shadow_ratio"] <= 0.60
    elif combo.upper_rule == "080":
        mask &= df["upper_shadow_ratio"] <= 0.80

    if combo.ret_rule == "le0":
        mask &= df["ret1"] <= 0
    elif combo.ret_rule == "m4_0":
        mask &= df["ret1"].between(-0.04, 0.0, inclusive="both")
    elif combo.ret_rule == "le2":
        mask &= df["ret1"] <= 0.02

    if combo.dist_rule == "010":
        mask &= df["dist_trend"] <= 0.10
    elif combo.dist_rule == "015":
        mask &= df["dist_trend"] <= 0.15

    if combo.vol_rule == "v12":
        mask &= df["vol_vs_prev"] <= 1.2
    elif combo.vol_rule == "ma12":
        mask &= df["signal_vs_ma5"] <= 1.2
    return mask.fillna(False)


def evaluate_b2(universe: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    combos = [
        B2Combo(*args)
        for args in product(
            ["dual", "any", "trend", "long"],
            ["strict", "mid", "wide", "loose"],
            ["095", "093", "090", "none"],
            ["070", "060", "none"],
            ["03", "05", "08"],
            ["90", "100"],
            ["none", "s3_pos", "s5_pos", "lead_003"],
            ["none", "prev_le_0", "prev_le_2"],
        )
    ]
    rows = []
    base_target = (
        0.4 * universe["max_float_10d"].fillna(0)
        + 0.4 * universe["max_float_20d"].fillna(0)
        + 0.2 * universe["close_ret_20d"].fillna(0)
    )
    for combo in combos:
        u_mask = mask_b2(universe, combo)
        c_mask = mask_b2(cases, combo)
        selected = universe[u_mask]
        if len(selected) < 20:
            continue
        target = (
            0.4 * selected["max_float_10d"].fillna(0)
            + 0.4 * selected["max_float_20d"].fillna(0)
            + 0.2 * selected["close_ret_20d"].fillna(0)
        )
        rows.append(
            {
                "start_rule": combo.start_rule,
                "vol_rule": combo.vol_rule,
                "near_rule": combo.near_rule,
                "close_rule": combo.close_rule,
                "shadow_rule": combo.shadow_rule,
                "jmax_rule": combo.jmax_rule,
                "trend_rule": combo.trend_rule,
                "prev_rule": combo.prev_rule,
                "case_hits": int(c_mask.sum()),
                "case_total": int(len(cases)),
                "coverage": float(c_mask.mean()) if len(cases) else 0.0,
                "selected_count": int(len(selected)),
                "avg_target": float(target.mean()),
                "avg_max10": float(selected["max_float_10d"].mean()),
                "avg_max20": float(selected["max_float_20d"].mean()),
                "avg_close20": float(selected["close_ret_20d"].mean()),
                "win20": float((selected["close_ret_20d"] > 0).mean()),
                "baseline_target_delta": float(target.mean() - base_target.mean()),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["case_hits", "coverage", "avg_target", "win20", "selected_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return df


def evaluate_pin(universe: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    combos = [
        PinCombo(*args)
        for args in product(
            ["005", "010", "020", "030", "010_035", "none"],
            ["008", "003", "000", "none"],
            ["035", "045", "060", "none"],
            ["060", "080", "none"],
            ["le0", "m4_0", "le2", "none"],
            ["010", "015", "none"],
            ["v12", "ma12", "none"],
        )
    ]
    rows = []
    base_target = (
        0.5 * universe["max_float_3d"].fillna(0)
        + 0.3 * universe["max_float_5d"].fillna(0)
        + 0.2 * universe["close_ret_5d"].fillna(0)
    )
    for combo in combos:
        u_mask = mask_pin(universe, combo)
        c_mask = mask_pin(cases, combo)
        selected = universe[u_mask]
        if len(selected) < 15:
            continue
        target = (
            0.5 * selected["max_float_3d"].fillna(0)
            + 0.3 * selected["max_float_5d"].fillna(0)
            + 0.2 * selected["close_ret_5d"].fillna(0)
        )
        rows.append(
            {
                "lower_rule": combo.lower_rule,
                "slope_rule": combo.slope_rule,
                "close_rule": combo.close_rule,
                "upper_rule": combo.upper_rule,
                "ret_rule": combo.ret_rule,
                "dist_rule": combo.dist_rule,
                "vol_rule": combo.vol_rule,
                "case_hits": int(c_mask.sum()),
                "case_total": int(len(cases)),
                "coverage": float(c_mask.mean()) if len(cases) else 0.0,
                "selected_count": int(len(selected)),
                "avg_target": float(target.mean()),
                "avg_max3": float(selected["max_float_3d"].mean()),
                "avg_max5": float(selected["max_float_5d"].mean()),
                "avg_close5": float(selected["close_ret_5d"].mean()),
                "win5": float((selected["close_ret_5d"] > 0).mean()),
                "baseline_target_delta": float(target.mean() - base_target.mean()),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["case_hits", "coverage", "avg_target", "win5", "selected_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return df


def main():
    name_map = load_name_map()
    b2_cases, b2_bad = parse_cases("B2", name_map)
    pin_cases, pin_bad = parse_cases("单针", name_map)
    b2_case_feats = build_case_features("b2", b2_cases)
    pin_case_feats = build_case_features("pin", pin_cases)
    b2_case_feats.to_csv(OUT_DIR / "b2_case_features.csv", index=False, encoding="utf-8-sig")
    pin_case_feats.to_csv(OUT_DIR / "pin_case_features.csv", index=False, encoding="utf-8-sig")

    b2_universe = collect_universe_features("b2")
    pin_universe = collect_universe_features("pin")
    b2_universe.to_csv(OUT_DIR / "b2_relaxed_universe.csv", index=False, encoding="utf-8-sig")
    pin_universe.to_csv(OUT_DIR / "pin_relaxed_universe.csv", index=False, encoding="utf-8-sig")

    b2_res = evaluate_b2(b2_universe, b2_case_feats)
    pin_res = evaluate_pin(pin_universe, pin_case_feats)
    b2_res.to_csv(OUT_DIR / "b2_combo_results.csv", index=False, encoding="utf-8-sig")
    pin_res.to_csv(OUT_DIR / "pin_combo_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "b2": {
            "cases_total": int(len(b2_cases)),
            "cases_mapped": int(b2_case_feats["code"].nunique()) if not b2_case_feats.empty else 0,
            "bad_files": b2_bad,
            "relaxed_universe_count": int(len(b2_universe)),
            "best_combo": b2_res.iloc[0].to_dict() if not b2_res.empty else None,
            "top_10": b2_res.head(10).to_dict(orient="records"),
        },
        "pin": {
            "cases_total": int(len(pin_cases)),
            "cases_mapped": int(pin_case_feats["code"].nunique()) if not pin_case_feats.empty else 0,
            "bad_files": pin_bad,
            "relaxed_universe_count": int(len(pin_universe)),
            "best_combo": pin_res.iloc[0].to_dict() if not pin_res.empty else None,
            "top_10": pin_res.head(10).to_dict(orient="records"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
