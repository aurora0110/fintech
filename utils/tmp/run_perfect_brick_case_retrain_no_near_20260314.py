from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
sys.path.insert(0, str(ROOT))
from utils import brick_filter


CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
NAME_MAP_DIR = ROOT / "data" / "20260313"
NORMAL_DIR = NAME_MAP_DIR / "normal"
OUT_DIR = ROOT / "results" / "perfect_brick_case_retrain_no_near_20260314"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")

CASE_NAME_ALIASES = {
    "特变电气": "特变电工",
    "创意通": "创益通",
}


@dataclass(frozen=True)
class Combo:
    require_close_above_trend: bool
    require_bull_close: bool
    require_pullback_shrinking: bool
    require_not_sideways: bool
    require_trend_gt_long: bool
    signal_vs_ma5_low: Optional[float]
    signal_vs_ma5_high: Optional[float]
    ret_cap: Optional[float]

    @property
    def combo_id(self) -> str:
        parts = [
            f"above{int(self.require_close_above_trend)}",
            f"bull{int(self.require_bull_close)}",
            f"shrink{int(self.require_pullback_shrinking)}",
            f"side{int(self.require_not_sideways)}",
            f"trendgt{int(self.require_trend_gt_long)}",
            f"vlo{'none' if self.signal_vs_ma5_low is None else self.signal_vs_ma5_low}",
            f"vhi{'none' if self.signal_vs_ma5_high is None else self.signal_vs_ma5_high}",
            f"ret{'none' if self.ret_cap is None else self.ret_cap}",
        ]
        return "_".join(parts)


def parse_case_files() -> pd.DataFrame:
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    rows: List[dict] = []
    for path in sorted(CASE_DIR.glob("*.png")):
        m = pat.match(path.name)
        if not m:
            continue
        rows.append(
            {
                "股票名称": m.group(1),
                "信号日期": pd.to_datetime(m.group(2), format="%Y%m%d"),
                "案例文件": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_name_code_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in sorted(NAME_MAP_DIR.glob("*.txt")):
        try:
            first_line = path.read_text(encoding="gbk", errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
        parts = first_line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[parts[1]] = path.stem
    return mapping


def resolve_code(stock_name: str, mapping: Dict[str, str]) -> Optional[str]:
    if stock_name in mapping:
        return mapping[stock_name]
    alias = CASE_NAME_ALIASES.get(stock_name)
    if alias and alias in mapping:
        return mapping[alias]
    return None


def load_case_feature_frames(case_codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for code in sorted(set(case_codes)):
        path = NORMAL_DIR / f"{code}.txt"
        if not path.exists():
            continue
        df = brick_filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        x = brick_filter.add_features(df)
        out[code] = x
    return out


def load_all_feature_frames() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    files = sorted(p for p in NORMAL_DIR.iterdir() if p.suffix.lower() in {".txt", ".csv"})
    total = len(files)
    for idx, path in enumerate(files, 1):
        if idx % 500 == 0 or idx == total:
            print(f"全市场特征进度: {idx}/{total}")
        df = brick_filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        out[path.stem] = brick_filter.add_features(df)
    return out


def build_mask(df: pd.DataFrame, combo: Combo) -> pd.Series:
    mask = df["brick_red"].fillna(False)
    if combo.require_close_above_trend:
        mask &= df["close_above_white"].fillna(False)
    if combo.require_bull_close:
        mask &= df["bull_close"].fillna(False)
    if combo.require_pullback_shrinking:
        mask &= df["pullback_shrinking"].fillna(False)
    if combo.require_not_sideways:
        mask &= df["not_sideways"].fillna(False)
    if combo.require_trend_gt_long:
        mask &= (df["trend_line"] > df["long_line"]).fillna(False)
    if combo.signal_vs_ma5_low is not None:
        mask &= df["signal_vs_ma5"].ge(combo.signal_vs_ma5_low).fillna(False)
    if combo.signal_vs_ma5_high is not None:
        mask &= df["signal_vs_ma5"].le(combo.signal_vs_ma5_high).fillna(False)
    mask &= df["ret1"].notna()
    if combo.ret_cap is not None:
        mask &= df["ret1"].le(combo.ret_cap)
    return mask.fillna(False)


def calc_future_metrics(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry_price = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    out = {
        "entry_date": pd.Timestamp(df.at[entry_idx, "date"]),
        "entry_price": entry_price,
        "vol_vs_prev": float(df.at[signal_idx, "vol_vs_prev"]) if pd.notna(df.at[signal_idx, "vol_vs_prev"]) else np.nan,
    }

    for horizon in (3, 5):
        end_idx = signal_idx + horizon
        if end_idx >= len(df):
            out[f"ret_close_{horizon}d"] = np.nan
            out[f"max_ret_{horizon}d"] = np.nan
            out[f"positive_close_{horizon}d"] = np.nan
            continue
        end_close = float(df.at[end_idx, "close"])
        max_high = float(df.loc[entry_idx:end_idx, "high"].max())
        out[f"ret_close_{horizon}d"] = end_close / entry_price - 1.0
        out[f"max_ret_{horizon}d"] = max_high / entry_price - 1.0
        out[f"positive_close_{horizon}d"] = float((end_close / entry_price - 1.0) > 0)
    return out


def search_combos_on_cases(cases: pd.DataFrame, case_frames: Dict[str, pd.DataFrame], combos: List[Combo]) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, combo in enumerate(combos, 1):
        if idx % 200 == 0 or idx == len(combos):
            print(f"案例搜索进度: {idx}/{len(combos)}")
        hit_cases = 0
        total_signals = 0
        metrics_rows: List[dict] = []
        for _, case in cases.iterrows():
            code = str(case["股票代码"])
            signal_date = pd.Timestamp(case["信号日期"])
            x = case_frames.get(code)
            if x is None:
                continue
            row_match = x.index[x["date"] == signal_date]
            if len(row_match) == 0:
                continue
            signal_idx = int(row_match[-1])
            mask = build_mask(x, combo)
            if bool(mask.iat[signal_idx]):
                hit_cases += 1
            signal_idxs = np.flatnonzero(mask.to_numpy())
            total_signals += int(len(signal_idxs))
            for sig_idx in signal_idxs:
                fm = calc_future_metrics(x, int(sig_idx))
                if fm is None:
                    continue
                metrics_rows.append(fm)
        metrics_df = pd.DataFrame(metrics_rows)
        rows.append(
            {
                **asdict(combo),
                "combo_id": combo.combo_id,
                "case_hit_count": int(hit_cases),
                "case_count": int(len(cases)),
                "case_hit_rate": float(hit_cases / len(cases)) if len(cases) else np.nan,
                "case_code_signal_count": int(total_signals),
                "avg_ret_close_3d_casecodes": float(metrics_df["ret_close_3d"].mean()) if not metrics_df.empty else np.nan,
                "avg_ret_close_5d_casecodes": float(metrics_df["ret_close_5d"].mean()) if not metrics_df.empty else np.nan,
                "positive_close_3d_rate_casecodes": float(metrics_df["positive_close_3d"].mean()) if not metrics_df.empty else np.nan,
                "positive_close_5d_rate_casecodes": float(metrics_df["positive_close_5d"].mean()) if not metrics_df.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["case_hit_count", "positive_close_5d_rate_casecodes", "avg_ret_close_5d_casecodes", "case_code_signal_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return out


def eval_top_combos_on_universe(top_combos: List[Combo], all_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, combo in enumerate(top_combos, 1):
        print(f"全市场评估进度: {idx}/{len(top_combos)} -> {combo.combo_id}")
        metrics_rows: List[dict] = []
        signal_count = 0
        for code, x in all_frames.items():
            mask = build_mask(x, combo)
            if EXCLUDE_START <= x["date"].max() and x["date"].min() <= EXCLUDE_END:
                mask &= ~x["date"].between(EXCLUDE_START, EXCLUDE_END)
            signal_idxs = np.flatnonzero(mask.to_numpy())
            signal_count += int(len(signal_idxs))
            for sig_idx in signal_idxs:
                fm = calc_future_metrics(x, int(sig_idx))
                if fm is None:
                    continue
                metrics_rows.append(fm)
        metrics_df = pd.DataFrame(metrics_rows)
        rows.append(
            {
                **asdict(combo),
                "combo_id": combo.combo_id,
                "signal_count": int(signal_count),
                "avg_ret_close_3d": float(metrics_df["ret_close_3d"].mean()) if not metrics_df.empty else np.nan,
                "avg_ret_close_5d": float(metrics_df["ret_close_5d"].mean()) if not metrics_df.empty else np.nan,
                "avg_max_ret_5d": float(metrics_df["max_ret_5d"].mean()) if not metrics_df.empty else np.nan,
                "positive_close_3d_rate": float(metrics_df["positive_close_3d"].mean()) if not metrics_df.empty else np.nan,
                "positive_close_5d_rate": float(metrics_df["positive_close_5d"].mean()) if not metrics_df.empty else np.nan,
                "double_prev_vol_share": float((metrics_df["vol_vs_prev"] >= 2.0).mean()) if not metrics_df.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["positive_close_5d_rate", "avg_ret_close_5d", "signal_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_case_hit_table(cases: pd.DataFrame, case_frames: Dict[str, pd.DataFrame], combo: Combo) -> pd.DataFrame:
    rows: List[dict] = []
    for _, case in cases.iterrows():
        code = str(case["股票代码"])
        signal_date = pd.Timestamp(case["信号日期"])
        x = case_frames.get(code)
        status = "无数据"
        if x is not None:
            row_match = x.index[x["date"] == signal_date]
            if len(row_match) > 0:
                signal_idx = int(row_match[-1])
                mask = build_mask(x, combo)
                status = "命中" if bool(mask.iat[signal_idx]) else "未命中"
        rows.append(
            {
                "股票名称": case["股票名称"],
                "股票代码": code,
                "信号日期": signal_date.strftime("%Y-%m-%d"),
                "状态": status,
            }
        )
    return pd.DataFrame(rows)


def analyze_double_prev_vol_effect(combo: Combo, all_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for code, x in all_frames.items():
        mask = build_mask(x, combo)
        mask &= ~x["date"].between(EXCLUDE_START, EXCLUDE_END)
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for sig_idx in signal_idxs:
            fm = calc_future_metrics(x, int(sig_idx))
            if fm is None:
                continue
            rows.append(
                {
                    "code": code,
                    "signal_date": pd.Timestamp(x.at[int(sig_idx), "date"]),
                    "vol_vs_prev": fm["vol_vs_prev"],
                    "double_prev_vol": bool(pd.notna(fm["vol_vs_prev"]) and fm["vol_vs_prev"] >= 2.0),
                    "ret_close_3d": fm["ret_close_3d"],
                    "ret_close_5d": fm["ret_close_5d"],
                    "max_ret_5d": fm["max_ret_5d"],
                    "positive_close_3d": fm["positive_close_3d"],
                    "positive_close_5d": fm["positive_close_5d"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("double_prev_vol")
        .agg(
            signal_count=("code", "size"),
            positive_close_3d_rate=("positive_close_3d", "mean"),
            positive_close_5d_rate=("positive_close_5d", "mean"),
            avg_ret_close_3d=("ret_close_3d", "mean"),
            avg_ret_close_5d=("ret_close_5d", "mean"),
            avg_max_ret_5d=("max_ret_5d", "mean"),
        )
        .reset_index()
    )
    return out


def main() -> None:
    cases = parse_case_files()
    name_code_map = build_name_code_map()
    cases["股票代码"] = cases["股票名称"].map(lambda n: resolve_code(str(n), name_code_map))
    cases["名称映射成功"] = cases["股票代码"].notna()

    case_codes = [str(v) for v in cases["股票代码"].dropna().unique()]
    case_frames = load_case_feature_frames(case_codes)

    data_flags: List[bool] = []
    for _, row in cases.iterrows():
        code = row["股票代码"]
        dt = pd.Timestamp(row["信号日期"])
        x = case_frames.get(str(code)) if pd.notna(code) else None
        data_flags.append(bool(x is not None and (x["date"] == dt).any()))
    cases["案例日期可用"] = data_flags
    eligible_cases = cases[cases["名称映射成功"] & cases["案例日期可用"]].copy().reset_index(drop=True)
    cases.to_csv(OUT_DIR / "case_mapping.csv", index=False, encoding="utf-8-sig")

    combos = [
        Combo(*vals)
        for vals in itertools.product(
            [True, False],   # require_close_above_trend
            [True, False],   # require_bull_close
            [True, False],   # require_pullback_shrinking
            [True, False],   # require_not_sideways
            [True, False],   # require_trend_gt_long
            [0.8, 0.9, 1.0, None],
            [2.2, 2.5, 3.0, None],
            [0.08, 0.09, 0.10, 0.12],
        )
        if not (vals[5] is not None and vals[6] is not None and vals[5] > vals[6])
    ]

    stage1 = search_combos_on_cases(eligible_cases, case_frames, combos)
    stage1.to_csv(OUT_DIR / "stage1_case_combo_results.csv", index=False, encoding="utf-8-sig")

    if stage1.empty:
        raise SystemExit("未找到任何组合结果")

    max_hit = int(stage1["case_hit_count"].max())
    top_case = stage1[stage1["case_hit_count"] == max_hit].copy()
    top_case = top_case.sort_values(
        ["positive_close_5d_rate_casecodes", "avg_ret_close_5d_casecodes", "case_code_signal_count"],
        ascending=[False, False, True],
    ).head(20)

    top_combos = [
        Combo(
            require_close_above_trend=bool(r["require_close_above_trend"]),
            require_bull_close=bool(r["require_bull_close"]),
            require_pullback_shrinking=bool(r["require_pullback_shrinking"]),
            require_not_sideways=bool(r["require_not_sideways"]),
            require_trend_gt_long=bool(r["require_trend_gt_long"]),
            signal_vs_ma5_low=None if pd.isna(r["signal_vs_ma5_low"]) else float(r["signal_vs_ma5_low"]),
            signal_vs_ma5_high=None if pd.isna(r["signal_vs_ma5_high"]) else float(r["signal_vs_ma5_high"]),
            ret_cap=None if pd.isna(r["ret_cap"]) else float(r["ret_cap"]),
        )
        for _, r in top_case.iterrows()
    ]

    all_frames = load_all_feature_frames()
    stage2 = eval_top_combos_on_universe(top_combos, all_frames)
    stage2.to_csv(OUT_DIR / "stage2_universe_top_combo_results.csv", index=False, encoding="utf-8-sig")

    merged_top = top_case.merge(
        stage2[
            [
                "combo_id",
                "signal_count",
                "avg_ret_close_3d",
                "avg_ret_close_5d",
                "avg_max_ret_5d",
                "positive_close_3d_rate",
                "positive_close_5d_rate",
                "double_prev_vol_share",
            ]
        ],
        on="combo_id",
        how="left",
    ).sort_values(
        ["case_hit_count", "positive_close_5d_rate", "avg_ret_close_5d", "signal_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    merged_top.to_csv(OUT_DIR / "top_coverage_combos.csv", index=False, encoding="utf-8-sig")

    best = merged_top.iloc[0]
    best_combo = Combo(
        require_close_above_trend=bool(best["require_close_above_trend"]),
        require_bull_close=bool(best["require_bull_close"]),
        require_pullback_shrinking=bool(best["require_pullback_shrinking"]),
        require_not_sideways=bool(best["require_not_sideways"]),
        require_trend_gt_long=bool(best["require_trend_gt_long"]),
        signal_vs_ma5_low=None if pd.isna(best["signal_vs_ma5_low"]) else float(best["signal_vs_ma5_low"]),
        signal_vs_ma5_high=None if pd.isna(best["signal_vs_ma5_high"]) else float(best["signal_vs_ma5_high"]),
        ret_cap=None if pd.isna(best["ret_cap"]) else float(best["ret_cap"]),
    )

    case_hits = build_case_hit_table(eligible_cases, case_frames, best_combo)
    case_hits.to_csv(OUT_DIR / "best_combo_case_hits.csv", index=False, encoding="utf-8-sig")

    vol_effect = analyze_double_prev_vol_effect(best_combo, all_frames)
    vol_effect.to_csv(OUT_DIR / "double_prev_vol_effect.csv", index=False, encoding="utf-8-sig")

    summary = {
        "case_total": int(len(cases)),
        "case_name_mapped": int(cases["名称映射成功"].sum()),
        "case_date_available": int(cases["案例日期可用"].sum()),
        "eligible_case_count": int(len(eligible_cases)),
        "best_case_hit_count": int(best["case_hit_count"]),
        "best_case_hit_rate": float(best["case_hit_rate"]),
        "best_combo": {
            "require_close_above_trend": bool(best_combo.require_close_above_trend),
            "require_bull_close": bool(best_combo.require_bull_close),
            "require_pullback_shrinking": bool(best_combo.require_pullback_shrinking),
            "require_not_sideways": bool(best_combo.require_not_sideways),
            "require_trend_gt_long": bool(best_combo.require_trend_gt_long),
            "signal_vs_ma5_low": best_combo.signal_vs_ma5_low,
            "signal_vs_ma5_high": best_combo.signal_vs_ma5_high,
            "ret_cap": best_combo.ret_cap,
        },
        "best_combo_universe_metrics": {
            "signal_count": int(best["signal_count"]) if pd.notna(best["signal_count"]) else 0,
            "positive_close_3d_rate": float(best["positive_close_3d_rate"]) if pd.notna(best["positive_close_3d_rate"]) else np.nan,
            "positive_close_5d_rate": float(best["positive_close_5d_rate"]) if pd.notna(best["positive_close_5d_rate"]) else np.nan,
            "avg_ret_close_3d": float(best["avg_ret_close_3d"]) if pd.notna(best["avg_ret_close_3d"]) else np.nan,
            "avg_ret_close_5d": float(best["avg_ret_close_5d"]) if pd.notna(best["avg_ret_close_5d"]) else np.nan,
            "avg_max_ret_5d": float(best["avg_max_ret_5d"]) if pd.notna(best["avg_max_ret_5d"]) else np.nan,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
