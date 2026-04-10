from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b1filter, technical_indicators

DATA_DIR = ROOT / "data" / "20260402"
RESULTS_DIR = ROOT / "results"
START_DATE = pd.Timestamp("2024-04-02")
END_DATE = pd.Timestamp("2026-04-02")
HOLD_DAYS = (1, 3, 5, 10)
BODY_ATR_THRESHOLD = 1.0
SHORT_VOL_THRESHOLD = 0.8
LONG_VOL_THRESHOLD = 1.2


@dataclass(frozen=True)
class GroupRule:
    name: str
    description: str


GROUPS = (
    GroupRule("long_bear_short_vol", "长阴短柱"),
    GroupRule("long_bear_long_vol", "长阴长柱"),
    GroupRule("short_bear_short_vol", "短阴短柱"),
    GroupRule("short_bear_long_vol", "短阴长柱"),
)


def _stock_code(file_path: Path) -> str:
    return file_path.stem.split("#")[-1]


def _calc_atr_prev(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["收盘"].shift(1)
    tr = pd.concat(
        [
            (df["最高"] - df["最低"]).abs(),
            (df["最高"] - prev_close).abs(),
            (df["最低"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean().shift(1)


def _calc_prev_vol_ma(df: pd.DataFrame, period: int = 5) -> pd.Series:
    return df["成交量"].shift(1).rolling(period, min_periods=period).mean()


def _future_return(df: pd.DataFrame, signal_idx: int, hold_days: int) -> float:
    entry_idx = signal_idx + 1
    exit_idx = entry_idx + hold_days - 1
    if exit_idx >= len(df):
        return np.nan
    entry_open = float(df["开盘"].iloc[entry_idx])
    exit_close = float(df["收盘"].iloc[exit_idx])
    if not np.isfinite(entry_open) or not np.isfinite(exit_close) or entry_open <= 0:
        return np.nan
    return exit_close / entry_open - 1.0


def _group_name(body_atr: float, vol_ratio: float) -> str | None:
    if not np.isfinite(body_atr) or not np.isfinite(vol_ratio):
        return None
    if body_atr >= BODY_ATR_THRESHOLD and vol_ratio <= SHORT_VOL_THRESHOLD:
        return "long_bear_short_vol"
    if body_atr >= BODY_ATR_THRESHOLD and vol_ratio >= LONG_VOL_THRESHOLD:
        return "long_bear_long_vol"
    if 0 < body_atr < BODY_ATR_THRESHOLD and vol_ratio <= SHORT_VOL_THRESHOLD:
        return "short_bear_short_vol"
    if 0 < body_atr < BODY_ATR_THRESHOLD and vol_ratio >= LONG_VOL_THRESHOLD:
        return "short_bear_long_vol"
    return None


def _process_file(file_path_str: str) -> list[dict]:
    file_path = Path(file_path_str)
    df = technical_indicators._load_price_data(str(file_path))
    if df.empty or len(df) < 60:
        return []

    df = technical_indicators.calculate_trend(df)
    df = technical_indicators.calculate_kdj(df)
    df["atr14_prev"] = _calc_atr_prev(df, 14)
    df["vol_ma5_prev"] = _calc_prev_vol_ma(df, 5)
    df["j_rank20"] = b1filter.rolling_last_percentile(df["J"], 20)

    code = _stock_code(file_path)
    rows: list[dict] = []

    for i in range(len(df)):
        signal_date = pd.Timestamp(df["日期"].iloc[i])
        if signal_date < START_DATE or signal_date > END_DATE:
            continue

        trend_line = float(df["知行短期趋势线"].iloc[i]) if pd.notna(df["知行短期趋势线"].iloc[i]) else np.nan
        long_line = float(df["知行多空线"].iloc[i]) if pd.notna(df["知行多空线"].iloc[i]) else np.nan
        j_rank20 = float(df["j_rank20"].iloc[i]) if pd.notna(df["j_rank20"].iloc[i]) else np.nan
        if not np.isfinite(trend_line) or not np.isfinite(long_line) or not np.isfinite(j_rank20):
            continue
        if trend_line <= long_line or j_rank20 >= 0.10:
            continue

        open_i = float(df["开盘"].iloc[i])
        close_i = float(df["收盘"].iloc[i])
        volume_i = float(df["成交量"].iloc[i]) if pd.notna(df["成交量"].iloc[i]) else np.nan
        atr_prev = float(df["atr14_prev"].iloc[i]) if pd.notna(df["atr14_prev"].iloc[i]) else np.nan
        vol_ma5_prev = float(df["vol_ma5_prev"].iloc[i]) if pd.notna(df["vol_ma5_prev"].iloc[i]) else np.nan

        if not np.isfinite(open_i) or not np.isfinite(close_i) or not np.isfinite(volume_i):
            continue
        if close_i >= open_i:
            continue
        if not np.isfinite(atr_prev) or atr_prev <= 0 or not np.isfinite(vol_ma5_prev) or vol_ma5_prev <= 0:
            continue

        body_atr = (open_i - close_i) / atr_prev
        vol_ratio = volume_i / vol_ma5_prev
        group_name = _group_name(body_atr, vol_ratio)

        rec = {
            "code": code,
            "signal_date": signal_date,
            "group_name": group_name or "neutral_other",
            "body_atr": body_atr,
            "vol_ratio": vol_ratio,
            "trend_line": trend_line,
            "long_line": long_line,
            "j_rank20": j_rank20,
            "ret_signal": close_i / open_i - 1.0,
        }
        for hold_days in HOLD_DAYS:
            rec[f"ret_h{hold_days}"] = _future_return(df, i, hold_days)
        rows.append(rec)
    return rows


def _build_summary(signal_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    group_order = ["all_context"] + [g.name for g in GROUPS] + ["neutral_other"]

    for group_name in group_order:
        if group_name == "all_context":
            sub = signal_df.copy()
            description = "所有满足趋势线>多空线且J<20日历史10%分位的阴线样本"
        else:
            sub = signal_df[signal_df["group_name"] == group_name].copy()
            desc_map = {g.name: g.description for g in GROUPS}
            description = desc_map.get(group_name, "其他未归类阴线")

        rec = {
            "group_name": group_name,
            "description": description,
            "signal_count": int(len(sub)),
            "mean_body_atr": float(sub["body_atr"].mean()) if not sub.empty else np.nan,
            "mean_vol_ratio": float(sub["vol_ratio"].mean()) if not sub.empty else np.nan,
        }
        for hold_days in HOLD_DAYS:
            ret_col = f"ret_h{hold_days}"
            valid = sub[ret_col].dropna()
            rec[f"h{hold_days}_count"] = int(len(valid))
            rec[f"h{hold_days}_avg_return"] = float(valid.mean()) if not valid.empty else np.nan
            rec[f"h{hold_days}_win_rate"] = float((valid > 0).mean()) if not valid.empty else np.nan
            rec[f"h{hold_days}_median_return"] = float(valid.median()) if not valid.empty else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"b1_long_bear_short_volume_compare_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(str(p) for p in DATA_DIR.glob("*.txt"))
    max_workers = min(10, max(1, cpu_count() - 1))

    all_rows: list[dict] = []
    with Pool(processes=max_workers) as pool:
        for rows in pool.imap_unordered(_process_file, file_paths, chunksize=8):
            if rows:
                all_rows.extend(rows)

    signal_df = pd.DataFrame(all_rows)
    if signal_df.empty:
        raise RuntimeError("没有找到满足基础前提的阴线样本")

    signal_df = signal_df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    summary_df = _build_summary(signal_df)

    signal_path = result_dir / "signal_detail.csv"
    summary_path = result_dir / "summary.csv"
    json_path = result_dir / "summary.json"

    signal_df.to_csv(signal_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "result_dir": str(result_dir),
        "data_dir": str(DATA_DIR),
        "date_range": [str(START_DATE.date()), str(END_DATE.date())],
        "base_condition": "趋势线>多空线 且 J<近20天历史10%分位",
        "long_bear_definition": f"(开盘-收盘)/ATR14_prev >= {BODY_ATR_THRESHOLD}",
        "short_bear_definition": f"0 < (开盘-收盘)/ATR14_prev < {BODY_ATR_THRESHOLD}",
        "short_volume_definition": f"volume / prev_ma5_volume <= {SHORT_VOL_THRESHOLD}",
        "long_volume_definition": f"volume / prev_ma5_volume >= {LONG_VOL_THRESHOLD}",
        "hold_days": list(HOLD_DAYS),
        "max_workers": max_workers,
        "groups": summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"结果目录: {result_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
