from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
BASE_SCRIPT_PATH = ROOT / "utils" / "tmp" / "run_b1_j_atr_factor_rolling_lab_v1_20260408.py"
BASE_RESULT_DIR = RESULTS_DIR / "b1_j_atr_factor_rolling_lab_v1_full_20260408_105621"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_SCRIPT_PATH, "b1_j_atr_factor_rolling_lab_v1_20260408_base")

WINDOW_START = BASE.WINDOW_START
WINDOW_END = BASE.WINDOW_END
SMOKE_HORIZONS = BASE.SMOKE_HORIZONS
FULL_HORIZONS = BASE.FULL_HORIZONS
MIN_SAMPLE_EXP1 = BASE.MIN_SAMPLE_EXP1
MIN_COVERAGE_EXP1 = BASE.MIN_COVERAGE_EXP1
MIN_SAMPLE_EXP2 = BASE.MIN_SAMPLE_EXP2
PIN_DEFAULT_EXIT_HOLD_DAYS = BASE.PIN_DEFAULT_EXIT_HOLD_DAYS

INITIAL_J_WINDOWS = (5, 7, 10, 12, 15, 20, 30, 40, 60)
INITIAL_J_QUANTILES = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
HARD_J_WINDOWS = tuple(range(3, 121))
HARD_J_QUANTILES = tuple(round(x, 3) for x in np.arange(0.01, 0.301, 0.01))

INITIAL_SHRINK_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
EXPANDED_SHRINK_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
INITIAL_ABS_RET_THRESHOLDS = (0.01, 0.02, 0.03, 0.04)
EXPANDED_ABS_RET_THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
INITIAL_BAND3_THRESHOLDS = (0.03, 0.04, 0.05, 0.06)
EXPANDED_BAND3_THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08)

INITIAL_TREND_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
EXPANDED_TREND_THRESHOLDS = (0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08)


@dataclass
class SearchState:
    values: tuple
    hard_values: tuple


def _serialize_value(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _load_cached_csv(path: Path, max_files: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signal_date", "entry_date"])
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if max_files > 0:
        codes = sorted(df["code"].dropna().astype(str).unique())[:max_files]
        df = df[df["code"].astype(str).isin(codes)].copy()
    df = df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    return df


def load_or_build_base_candidates(data_dir: Path, max_files: int, horizons: tuple[int, ...], result_dir: Path) -> pd.DataFrame:
    cache_path = BASE_RESULT_DIR / "base_candidates.csv"
    reuse_ok = cache_path.exists() and BASE_RESULT_DIR.joinpath("summary.json").exists()
    if reuse_ok:
        meta = json.loads((BASE_RESULT_DIR / "summary.json").read_text())
        if meta.get("data_dir") == str(data_dir):
            df = _load_cached_csv(cache_path, max_files=max_files)
            df.to_csv(result_dir / "base_candidates.csv", index=False, encoding="utf-8-sig")
            return df
    df = BASE.build_base_candidate_df(data_dir, max_files=max_files, horizons=horizons)
    df.to_csv(result_dir / "base_candidates.csv", index=False, encoding="utf-8-sig")
    return df


def load_or_build_pin_candidates(data_dir: Path, max_files: int, result_dir: Path) -> pd.DataFrame:
    cache_path = BASE_RESULT_DIR / "pin_candidates.csv"
    reuse_ok = cache_path.exists()
    if reuse_ok:
        df = _load_cached_csv(cache_path, max_files=max_files)
        df.to_csv(result_dir / "pin_candidates.csv", index=False, encoding="utf-8-sig")
        return df
    df = BASE.compute_pin_candidate_df(data_dir, max_files=max_files)
    df.to_csv(result_dir / "pin_candidates.csv", index=False, encoding="utf-8-sig")
    return df


def build_price_map(data_dir: Path, base_df: pd.DataFrame, pin_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    codes = sorted(set(base_df["code"].astype(str)).union(set(pin_df["code"].astype(str))))
    return BASE.build_price_map(data_dir, codes)


def summarize_trade_and_signal(signal_df: pd.DataFrame, trade_df: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    signal_summary = BASE.summarize_signal_layer(signal_df, horizons)
    trade_summary = BASE.summarize_trade_layer(trade_df)
    return {**signal_summary, **trade_summary}


def rank_generic(df: pd.DataFrame, min_sample: int, min_coverage: int, extra_sort_cols: list[str] | None = None) -> pd.DataFrame:
    ranked = df.copy()
    ranked["valid_main_filter"] = (ranked["signal_count"] >= min_sample) & (ranked["coverage_days"] >= min_coverage)
    extra_sort_cols = extra_sort_cols or []
    sort_cols = ["valid_main_filter", "avg_trade_return", "avg_ret_20d", "win_rate_20d", "signal_count"] + extra_sort_cols
    ascending = [False, False, False, False, False] + [True] * len(extra_sort_cols)
    ranked = ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return ranked


def build_percentile_rules(base_df: pd.DataFrame, windows: tuple[int, ...], quantiles: tuple[float, ...]) -> list[dict]:
    rules = []
    for w in windows:
        col = f"j_rank_{w}"
        if col not in base_df.columns:
            continue
        series = pd.to_numeric(base_df[col], errors="coerce")
        for q in quantiles:
            rules.append(
                {
                    "window": int(w),
                    "quantile": float(q),
                    "rule_name": f"j_rank_{w}_le_{q:.3f}",
                    "mask": (series <= q).fillna(False),
                }
            )
    return rules


def run_j_refine_round(base_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], windows: tuple[int, ...], quantiles: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for rule in build_percentile_rules(base_df, windows, quantiles):
        picked = base_df[rule["mask"]].copy()
        trade_df = BASE.simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
        rows.append(
            {
                "window": int(rule["window"]),
                "quantile": float(rule["quantile"]),
                "rule_name": rule["rule_name"],
                **summarize_trade_and_signal(picked, trade_df, horizons),
            }
        )
    ranked = rank_generic(pd.DataFrame(rows), MIN_SAMPLE_EXP1, MIN_COVERAGE_EXP1, extra_sort_cols=["window", "quantile"])
    return ranked


def expand_j_search(best_row: pd.Series, windows: tuple[int, ...], quantiles: tuple[float, ...]) -> tuple[tuple[int, ...], tuple[float, ...], bool]:
    new_windows = set(windows)
    new_quantiles = set(quantiles)
    expanded = False
    best_w = int(best_row["window"])
    best_q = round(float(best_row["quantile"]), 3)
    if best_w == min(windows):
        for v in (3, 4, 5, 6, 7, 8, 9):
            if v in HARD_J_WINDOWS:
                new_windows.add(v)
        expanded = True
    if best_w == max(windows):
        for v in (80, 100, 120):
            if v in HARD_J_WINDOWS:
                new_windows.add(v)
        expanded = True
    if best_q == round(min(quantiles), 3):
        for v in (0.01, 0.02, 0.03, 0.04):
            if v in HARD_J_QUANTILES:
                new_quantiles.add(v)
        expanded = True
    if best_q == round(max(quantiles), 3):
        for v in (0.25, 0.30):
            if v in HARD_J_QUANTILES:
                new_quantiles.add(v)
        expanded = True
    return tuple(sorted(new_windows)), tuple(sorted(new_quantiles)), expanded


def run_exp1_j_refine(base_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    windows = INITIAL_J_WINDOWS
    quantiles = INITIAL_J_QUANTILES
    trace = []
    final_ranked = pd.DataFrame()
    max_rounds = 6
    for round_id in range(1, max_rounds + 1):
        ranked = run_j_refine_round(base_df, price_map, horizons, windows, quantiles)
        ranked.to_csv(result_dir / f"j_grid_round_{round_id}.csv", index=False, encoding="utf-8-sig")
        final_ranked = ranked
        best = ranked.iloc[0]
        hit_boundary = (
            int(best["window"]) in {min(windows), max(windows)}
            or round(float(best["quantile"]), 3) in {round(min(quantiles), 3), round(max(quantiles), 3)}
        )
        trace.append(
            {
                "round": round_id,
                "windows": list(windows),
                "quantiles": list(quantiles),
                "best_window": int(best["window"]),
                "best_quantile": float(best["quantile"]),
                "hit_boundary": bool(hit_boundary),
            }
        )
        if not hit_boundary:
            break
        next_windows, next_quantiles, expanded = expand_j_search(best, windows, quantiles)
        if not expanded or (next_windows == windows and next_quantiles == quantiles):
            break
        windows, quantiles = next_windows, next_quantiles

    best_row = final_ranked.iloc[0].to_dict()
    secondary_row = final_ranked.iloc[1].to_dict() if len(final_ranked) > 1 else None
    payload = {"best": best_row, "secondary": secondary_row}
    (result_dir / "best_j_refined.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "j_boundary_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    best_mask = (pd.to_numeric(base_df[f"j_rank_{int(best_row['window'])}"], errors="coerce") <= float(best_row["quantile"])).fillna(False)
    best_df = base_df[best_mask].copy()
    return final_ranked, payload, best_df


def simulate_pin_hold3(signal_df: pd.DataFrame, price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trade_df = BASE.simulate_pin_hold3(signal_df[["code", "signal_date", "entry_date", "entry_open"]], price_map)
    if trade_df.empty:
        return pd.DataFrame(columns=["code", "signal_date", "entry_date", "exit_date", "holding_return", "holding_days"])
    return trade_df


def summarize_fixed_window(signal_df: pd.DataFrame, trade_df: pd.DataFrame, signal_horizon: int) -> dict:
    return BASE.summarize_window(signal_df, trade_df, signal_horizon)


def detect_significant_periods(compare_df: pd.DataFrame) -> pd.DataFrame:
    periods = []
    for window_days, window_df in compare_df.groupby("window_days"):
        for direction in ("pin85_stronger", "pin80_stronger"):
            run_start = None
            run_end = None
            run_count = 0
            for row in window_df.sort_values("window_end").itertuples(index=False):
                flag = bool(getattr(row, direction))
                current_date = pd.Timestamp(row.window_end)
                if flag:
                    if run_start is None:
                        run_start = current_date
                    run_end = current_date
                    run_count += 1
                else:
                    if run_count >= 10:
                        periods.append(
                            {
                                "window_days": int(window_days),
                                "direction": direction,
                                "start_date": run_start,
                                "end_date": run_end,
                                "window_points": int(run_count),
                            }
                        )
                    run_start = None
                    run_end = None
                    run_count = 0
            if run_count >= 10:
                periods.append(
                    {
                        "window_days": int(window_days),
                        "direction": direction,
                        "start_date": run_start,
                        "end_date": run_end,
                        "window_points": int(run_count),
                    }
                )
    if not periods:
        return pd.DataFrame(columns=["window_days", "direction", "start_date", "end_date", "window_points"])
    return pd.DataFrame(periods)


def run_exp2_pin_windows(pin_df: pd.DataFrame, base_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pin85 = pin_df[(pin_df["short_value"] <= 30) & (pin_df["long_value"] >= 85)].copy()
    pin80 = pin_df[(pin_df["short_value"] <= 20) & (pin_df["long_value"] >= 80)].copy()
    trade85 = simulate_pin_hold3(pin85, price_map)
    trade80 = simulate_pin_hold3(pin80, price_map)
    calendar = sorted(pd.Series(base_df["signal_date"].dropna().unique()).tolist())
    rows = []
    for window_days in (30, 60, 120):
        for end_date in calendar:
            end_ts = pd.Timestamp(end_date)
            start_ts = end_ts - pd.offsets.BDay(window_days - 1)
            sig85 = pin85[(pin85["signal_date"] >= start_ts) & (pin85["signal_date"] <= end_ts)].copy()
            sig80 = pin80[(pin80["signal_date"] >= start_ts) & (pin80["signal_date"] <= end_ts)].copy()
            tr85 = trade85[(trade85["signal_date"] >= start_ts) & (trade85["signal_date"] <= end_ts)].copy()
            tr80 = trade80[(trade80["signal_date"] >= start_ts) & (trade80["signal_date"] <= end_ts)].copy()
            m85 = summarize_fixed_window(sig85, tr85, 3)
            m80 = summarize_fixed_window(sig80, tr80, 3)
            strong_checks = [
                np.isfinite(m85["trade_avg_return"]) and np.isfinite(m80["trade_avg_return"]) and m85["trade_avg_return"] > m80["trade_avg_return"] * 1.2,
                np.isfinite(m85["trade_win_rate"]) and np.isfinite(m80["trade_win_rate"]) and m85["trade_win_rate"] > m80["trade_win_rate"] + 0.05,
                np.isfinite(m85["signal_avg_return"]) and np.isfinite(m80["signal_avg_return"]) and (m85["signal_avg_return"] - m80["signal_avg_return"]) >= 0.003,
            ]
            weak_checks = [
                np.isfinite(m80["trade_avg_return"]) and np.isfinite(m85["trade_avg_return"]) and m80["trade_avg_return"] > m85["trade_avg_return"] * 1.2,
                np.isfinite(m80["trade_win_rate"]) and np.isfinite(m85["trade_win_rate"]) and m80["trade_win_rate"] > m85["trade_win_rate"] + 0.05,
                np.isfinite(m80["signal_avg_return"]) and np.isfinite(m85["signal_avg_return"]) and (m80["signal_avg_return"] - m85["signal_avg_return"]) >= 0.003,
            ]
            eligible = m85["signal_count"] >= 30 and m80["signal_count"] >= 30
            rows.append(
                {
                    "window_days": int(window_days),
                    "window_end": end_ts,
                    "pin85_signal_count": int(m85["signal_count"]),
                    "pin80_signal_count": int(m80["signal_count"]),
                    "pin85_signal_avg_return": m85["signal_avg_return"],
                    "pin80_signal_avg_return": m80["signal_avg_return"],
                    "pin85_signal_win_rate": m85["signal_win_rate"],
                    "pin80_signal_win_rate": m80["signal_win_rate"],
                    "pin85_trade_avg_return": m85["trade_avg_return"],
                    "pin80_trade_avg_return": m80["trade_avg_return"],
                    "pin85_trade_win_rate": m85["trade_win_rate"],
                    "pin80_trade_win_rate": m80["trade_win_rate"],
                    "pin85_stronger": bool(eligible and sum(strong_checks) >= 2),
                    "pin80_stronger": bool(eligible and sum(weak_checks) >= 2),
                }
            )
    compare_df = pd.DataFrame(rows).sort_values(["window_days", "window_end"]).reset_index(drop=True)
    compare_df.to_csv(result_dir / "pin_window_compare.csv", index=False, encoding="utf-8-sig")
    periods_df = detect_significant_periods(compare_df)
    periods_df.to_csv(result_dir / "pin_regime_periods.csv", index=False, encoding="utf-8-sig")
    summary = {
        "window_summary": {
            str(window): {
                "pin85_stronger_periods": int(len(periods_df[(periods_df["window_days"] == window) & (periods_df["direction"] == "pin85_stronger")])),
                "pin80_stronger_periods": int(len(periods_df[(periods_df["window_days"] == window) & (periods_df["direction"] == "pin80_stronger")])),
            }
            for window in (30, 60, 120)
        }
    }
    (result_dir / "pin_window_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return compare_df, periods_df, summary


def build_shrink_consistent_candidates(best_j_df: pd.DataFrame, shrink_values: tuple[float, ...], abs_values: tuple[float, ...], band_values: tuple[float, ...]) -> list[dict]:
    vol_ma5 = pd.to_numeric(best_j_df["vol_ratio_ma5"], errors="coerce")
    vol_ma10 = pd.to_numeric(best_j_df["vol_ratio_ma10"], errors="coerce")
    vol_prev = pd.to_numeric(best_j_df["vol_ratio_prev"], errors="coerce")
    abs_ret1 = pd.to_numeric(best_j_df["abs_ret1"], errors="coerce")
    band3 = pd.to_numeric(best_j_df["return_band_3d"], errors="coerce")
    items = []
    for s in shrink_values:
        for a in abs_values:
            for b in band_values:
                cons_mask = (abs_ret1 <= a) & (band3 <= b)
                items.append({"family": "ma5_cons", "shrink_value": float(s), "abs_ret1_max": float(a), "band3_max": float(b), "definition": f"ma5_le_{s:.3f}__absret_le_{a:.3f}__band3_le_{b:.3f}", "mask": (vol_ma5 <= s) & cons_mask})
                items.append({"family": "ma10_cons", "shrink_value": float(s), "abs_ret1_max": float(a), "band3_max": float(b), "definition": f"ma10_le_{s:.3f}__absret_le_{a:.3f}__band3_le_{b:.3f}", "mask": (vol_ma10 <= s) & cons_mask})
                items.append({"family": "prev_ma5_cons", "shrink_value": float(s), "abs_ret1_max": float(a), "band3_max": float(b), "definition": f"prev_lt_1__ma5_le_{s:.3f}__absret_le_{a:.3f}__band3_le_{b:.3f}", "mask": (vol_prev < 1.0) & (vol_ma5 <= s) & cons_mask})
    for a in abs_values:
        for b in band_values:
            cons_mask = (abs_ret1 <= a) & (band3 <= b)
            items.append({"family": "prev_cons", "shrink_value": None, "abs_ret1_max": float(a), "band3_max": float(b), "definition": f"prev_lt_1__absret_le_{a:.3f}__band3_le_{b:.3f}", "mask": (vol_prev < 1.0) & cons_mask})
    return items


def expand_shrink_search(best_row: pd.Series, shrink_values: tuple[float, ...], abs_values: tuple[float, ...], band_values: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], bool]:
    new_shrink = set(shrink_values)
    new_abs = set(abs_values)
    new_band = set(band_values)
    expanded = False
    if pd.notna(best_row["shrink_value"]):
        best_s = round(float(best_row["shrink_value"]), 3)
        if best_s == round(min(shrink_values), 3):
            new_shrink.add(0.4)
            expanded = True
        if best_s == round(max(shrink_values), 3):
            new_shrink.add(1.0)
            expanded = True
    best_a = round(float(best_row["abs_ret1_max"]), 3)
    if best_a == round(min(abs_values), 3):
        new_abs.add(0.005)
        expanded = True
    if best_a == round(max(abs_values), 3):
        new_abs.add(0.05)
        expanded = True
    best_b = round(float(best_row["band3_max"]), 3)
    if best_b == round(min(band_values), 3):
        new_band.add(0.02)
        expanded = True
    if best_b == round(max(band_values), 3):
        new_band.add(0.08)
        expanded = True
    return tuple(sorted(new_shrink)), tuple(sorted(new_abs)), tuple(sorted(new_band)), expanded


def run_exp3_shrink_consistent(best_j_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict]:
    shrink_values = INITIAL_SHRINK_THRESHOLDS
    abs_values = INITIAL_ABS_RET_THRESHOLDS
    band_values = INITIAL_BAND3_THRESHOLDS
    trace = []
    final_ranked = pd.DataFrame()
    for round_id in range(1, 6):
        rows = []
        for item in build_shrink_consistent_candidates(best_j_df, shrink_values, abs_values, band_values):
            picked = best_j_df[item["mask"].fillna(False)].copy()
            trade_df = BASE.simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
            rows.append({k: item[k] for k in ("family", "definition", "shrink_value", "abs_ret1_max", "band3_max")} | summarize_trade_and_signal(picked, trade_df, horizons))
        ranked = rank_generic(pd.DataFrame(rows), MIN_SAMPLE_EXP2, 30)
        ranked.to_csv(result_dir / f"shrink_consistent_grid_round_{round_id}.csv", index=False, encoding="utf-8-sig")
        final_ranked = ranked
        best = ranked.iloc[0]
        hit_boundary = False
        if pd.notna(best["shrink_value"]):
            hit_boundary = hit_boundary or round(float(best["shrink_value"]), 3) in {round(min(shrink_values), 3), round(max(shrink_values), 3)}
        hit_boundary = hit_boundary or round(float(best["abs_ret1_max"]), 3) in {round(min(abs_values), 3), round(max(abs_values), 3)}
        hit_boundary = hit_boundary or round(float(best["band3_max"]), 3) in {round(min(band_values), 3), round(max(band_values), 3)}
        trace.append(
            {
                "round": round_id,
                "shrink_values": [None if v is None else float(v) for v in shrink_values],
                "abs_values": list(abs_values),
                "band_values": list(band_values),
                "best_definition": best["definition"],
                "hit_boundary": bool(hit_boundary),
            }
        )
        if not hit_boundary:
            break
        shrink_values, abs_values, band_values, expanded = expand_shrink_search(best, shrink_values, abs_values, band_values)
        if not expanded:
            break
    payload = {
        "best": final_ranked.iloc[0].to_dict(),
        "secondary": final_ranked.iloc[1].to_dict() if len(final_ranked) > 1 else None,
    }
    (result_dir / "best_shrink_consistent.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "boundary_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_ranked, payload


def expand_trend_thresholds(best_threshold: float, thresholds: tuple[float, ...]) -> tuple[tuple[float, ...], bool]:
    vals = set(thresholds)
    expanded = False
    best_t = round(float(best_threshold), 4)
    if best_t == round(min(thresholds), 4):
        vals.add(0.003)
        expanded = True
    if best_t == round(max(thresholds), 4):
        vals.update((0.06, 0.08))
        expanded = True
    return tuple(sorted(vals)), expanded


def run_exp4_trend_refine(best_j_df: pd.DataFrame, price_map: dict[str, pd.DataFrame], horizons: tuple[int, ...], result_dir: Path) -> tuple[pd.DataFrame, dict]:
    thresholds = INITIAL_TREND_THRESHOLDS
    trace = []
    final_ranked = pd.DataFrame()
    for round_id in range(1, 6):
        rows = []
        for t in thresholds:
            picked = best_j_df[pd.to_numeric(best_j_df["low_to_trend_abs"], errors="coerce") <= t].copy()
            trade_df = BASE.simulate_trade_layer(picked[["code", "signal_date", "entry_date", "entry_open", "signal_low"]], price_map)
            rows.append({"threshold": float(t), "definition": f"low_to_trend_abs_le_{t:.4f}", **summarize_trade_and_signal(picked, trade_df, horizons)})
        ranked = rank_generic(pd.DataFrame(rows), MIN_SAMPLE_EXP1, MIN_COVERAGE_EXP1)
        ranked.to_csv(result_dir / f"trend_pullback_grid_round_{round_id}.csv", index=False, encoding="utf-8-sig")
        final_ranked = ranked
        best = ranked.iloc[0]
        hit_boundary = round(float(best["threshold"]), 4) in {round(min(thresholds), 4), round(max(thresholds), 4)}
        trace.append({"round": round_id, "thresholds": list(thresholds), "best_threshold": float(best["threshold"]), "hit_boundary": bool(hit_boundary)})
        if not hit_boundary:
            break
        thresholds, expanded = expand_trend_thresholds(best["threshold"], thresholds)
        if not expanded:
            break
    payload = {
        "best": final_ranked.iloc[0].to_dict(),
        "secondary": final_ranked.iloc[1].to_dict() if len(final_ranked) > 1 else None,
    }
    (result_dir / "best_trend_pullback.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "boundary_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_ranked, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 J 边界扩搜、pin 固定窗口比较、缩量一致与回踩趋势线分开再量化")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"b1_pin_boundary_refine_v1_{args.mode}_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    horizons = SMOKE_HORIZONS if args.mode == "smoke" else FULL_HORIZONS
    max_files = args.max_files if args.max_files is not None else (400 if args.mode == "smoke" else 0)

    data_dir, selection_meta = BASE.choose_data_dir()
    (result_dir / "data_selection.json").write_text(json.dumps(selection_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    base_df = load_or_build_base_candidates(data_dir, max_files=max_files, horizons=horizons, result_dir=result_dir)
    pin_df = load_or_build_pin_candidates(data_dir, max_files=max_files, result_dir=result_dir)
    price_map = build_price_map(data_dir, base_df, pin_df)

    exp1_dir = result_dir / "exp1_j_refine"
    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp1_ranked, exp1_payload, best_j_df = run_exp1_j_refine(base_df, price_map, horizons, exp1_dir)
    best_j_df.to_csv(result_dir / "best_j_refined_signals.csv", index=False, encoding="utf-8-sig")

    exp2_dir = result_dir / "exp2_pin_windows"
    exp2_dir.mkdir(parents=True, exist_ok=True)
    pin_compare_df, pin_periods_df, pin_summary = run_exp2_pin_windows(pin_df, base_df, price_map, exp2_dir)

    exp3_dir = result_dir / "exp3_shrink_consistent"
    exp3_dir.mkdir(parents=True, exist_ok=True)
    exp3_ranked, exp3_payload = run_exp3_shrink_consistent(best_j_df, price_map, horizons, exp3_dir)

    exp4_dir = result_dir / "exp4_trend_pullback_refine"
    exp4_dir.mkdir(parents=True, exist_ok=True)
    exp4_ranked, exp4_payload = run_exp4_trend_refine(best_j_df, price_map, horizons, exp4_dir)

    summary = {
        "mode": args.mode,
        "data_dir": str(data_dir),
        "window_start": str(WINDOW_START.date()),
        "window_end": str(WINDOW_END.date()),
        "base_candidate_count": int(len(base_df)),
        "pin_candidate_count": int(len(pin_df)),
        "price_map_codes": int(len(price_map)),
        "exp1_best": exp1_payload["best"],
        "exp2_pin_window_summary": pin_summary,
        "exp3_best": exp3_payload["best"],
        "exp4_best": exp4_payload["best"],
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
