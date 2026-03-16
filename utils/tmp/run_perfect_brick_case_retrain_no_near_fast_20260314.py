from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
sys.path.insert(0, str(ROOT))

from utils import brick_filter


CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
NAME_MAP_DIR = ROOT / "data" / "20260313"
NORMAL_DIR = NAME_MAP_DIR / "normal"
OUT_DIR = ROOT / "results" / "perfect_brick_case_retrain_no_near_fast_20260314"
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
        return "|".join(
            [
                f"above={int(self.require_close_above_trend)}",
                f"bull={int(self.require_bull_close)}",
                f"shrink={int(self.require_pullback_shrinking)}",
                f"side={int(self.require_not_sideways)}",
                f"trendgt={int(self.require_trend_gt_long)}",
                f"vlo={self.signal_vs_ma5_low}",
                f"vhi={self.signal_vs_ma5_high}",
                f"ret={self.ret_cap}",
            ]
        )


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


def resolve_code(name: str, mapping: Dict[str, str]) -> Optional[str]:
    if name in mapping:
        return mapping[name]
    alias = CASE_NAME_ALIASES.get(name)
    if alias and alias in mapping:
        return mapping[alias]
    return None


def load_case_frames(case_codes: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for code in sorted(set(case_codes)):
        fp = NORMAL_DIR / f"{code}.txt"
        if not fp.exists():
            continue
        df = brick_filter.load_one_csv(str(fp))
        if df is None or df.empty:
            continue
        out[code] = brick_filter.add_features(df)
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


def future_metrics(df: pd.DataFrame, signal_idx: int) -> Optional[dict]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry = float(df.at[entry_idx, "open"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    out: dict = {"vol_vs_prev": float(df.at[signal_idx, "vol_vs_prev"]) if pd.notna(df.at[signal_idx, "vol_vs_prev"]) else np.nan}
    for horizon in (3, 5):
        end_idx = signal_idx + horizon
        if end_idx >= len(df):
            out[f"ret_close_{horizon}d"] = np.nan
            out[f"positive_close_{horizon}d"] = np.nan
            continue
        close_ret = float(df.at[end_idx, "close"]) / entry - 1.0
        out[f"ret_close_{horizon}d"] = close_ret
        out[f"positive_close_{horizon}d"] = float(close_ret > 0)
    end_idx = signal_idx + 5
    if end_idx < len(df):
        out["max_ret_5d"] = float(df.loc[entry_idx:end_idx, "high"].max()) / entry - 1.0
    else:
        out["max_ret_5d"] = np.nan
    return out


def main() -> None:
    cases = parse_case_files()
    name_map = build_name_code_map()
    cases["股票代码"] = cases["股票名称"].map(lambda n: resolve_code(str(n), name_map))
    cases["名称映射成功"] = cases["股票代码"].notna()

    case_frames = load_case_frames([str(c) for c in cases["股票代码"].dropna().unique()])
    cases["案例日期可用"] = cases.apply(
        lambda r: bool(
            pd.notna(r["股票代码"])
            and str(r["股票代码"]) in case_frames
            and (case_frames[str(r["股票代码"])]["date"] == pd.Timestamp(r["信号日期"])).any()
        ),
        axis=1,
    )
    eligible = cases[cases["名称映射成功"] & cases["案例日期可用"]].copy().reset_index(drop=True)
    cases.to_csv(OUT_DIR / "case_mapping.csv", index=False, encoding="utf-8-sig")

    combos = [
        Combo(*vals)
        for vals in itertools.product(
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [0.8, 0.9, 1.0, None],
            [2.2, 2.5, 3.0, None],
            [0.08, 0.09, 0.10, 0.12],
        )
        if not (vals[5] is not None and vals[6] is not None and vals[5] > vals[6])
    ]

    # precompute case-date mask booleans per combo quickly
    rows: List[dict] = []
    case_hit_by_combo: Dict[str, List[bool]] = {}
    for idx, combo in enumerate(combos, 1):
        if idx % 200 == 0 or idx == len(combos):
            print(f"阶段1案例覆盖进度: {idx}/{len(combos)}", flush=True)
        hits: List[bool] = []
        for _, case in eligible.iterrows():
            x = case_frames[str(case["股票代码"])]
            signal_date = pd.Timestamp(case["信号日期"])
            sig_idx = int(x.index[x["date"] == signal_date][-1])
            hit = bool(build_mask(x, combo).iat[sig_idx])
            hits.append(hit)
        case_hit_by_combo[combo.combo_id] = hits
        rows.append(
            {
                "combo_id": combo.combo_id,
                "require_close_above_trend": combo.require_close_above_trend,
                "require_bull_close": combo.require_bull_close,
                "require_pullback_shrinking": combo.require_pullback_shrinking,
                "require_not_sideways": combo.require_not_sideways,
                "require_trend_gt_long": combo.require_trend_gt_long,
                "signal_vs_ma5_low": combo.signal_vs_ma5_low,
                "signal_vs_ma5_high": combo.signal_vs_ma5_high,
                "ret_cap": combo.ret_cap,
                "case_hit_count": int(sum(hits)),
                "case_count": int(len(hits)),
                "case_hit_rate": float(sum(hits) / len(hits)) if hits else np.nan,
            }
        )
    stage1 = pd.DataFrame(rows).sort_values(["case_hit_count", "case_hit_rate"], ascending=[False, False]).reset_index(drop=True)
    stage1.to_csv(OUT_DIR / "stage1_case_combo_results.csv", index=False, encoding="utf-8-sig")

    max_hit = int(stage1["case_hit_count"].max())
    top_ids = stage1[stage1["case_hit_count"] == max_hit]["combo_id"].tolist()[:20]
    top_combos: List[Combo] = []
    for cid in top_ids:
        row = stage1[stage1["combo_id"] == cid].iloc[0]
        top_combos.append(
            Combo(
                require_close_above_trend=bool(row["require_close_above_trend"]),
                require_bull_close=bool(row["require_bull_close"]),
                require_pullback_shrinking=bool(row["require_pullback_shrinking"]),
                require_not_sideways=bool(row["require_not_sideways"]),
                require_trend_gt_long=bool(row["require_trend_gt_long"]),
                signal_vs_ma5_low=None if pd.isna(row["signal_vs_ma5_low"]) else float(row["signal_vs_ma5_low"]),
                signal_vs_ma5_high=None if pd.isna(row["signal_vs_ma5_high"]) else float(row["signal_vs_ma5_high"]),
                ret_cap=None if pd.isna(row["ret_cap"]) else float(row["ret_cap"]),
            )
        )

    # full-universe eval only for top coverage combos
    files = sorted(p for p in NORMAL_DIR.iterdir() if p.suffix.lower() in {".txt", ".csv"})
    top_rows: List[dict] = []
    best_case_hits = None
    best_case_hit_count = -1
    best_combo_id = None
    for combo_idx, combo in enumerate(top_combos, 1):
        print(f"阶段2全市场评估进度: {combo_idx}/{len(top_combos)} -> {combo.combo_id}", flush=True)
        metrics: List[dict] = []
        signal_count = 0
        for file_idx, path in enumerate(files, 1):
            if combo_idx == 1 and (file_idx % 800 == 0 or file_idx == len(files)):
                print(f"  全市场特征进度: {file_idx}/{len(files)}", flush=True)
            df = brick_filter.load_one_csv(str(path))
            if df is None or df.empty:
                continue
            x = brick_filter.add_features(df)
            mask = build_mask(x, combo)
            mask &= ~x["date"].between(EXCLUDE_START, EXCLUDE_END)
            signal_idxs = np.flatnonzero(mask.to_numpy())
            signal_count += int(len(signal_idxs))
            for sig_idx in signal_idxs:
                fm = future_metrics(x, int(sig_idx))
                if fm is not None:
                    metrics.append(fm)
        mdf = pd.DataFrame(metrics)
        row = {
            "combo_id": combo.combo_id,
            "case_hit_count": int(stage1.loc[stage1["combo_id"] == combo.combo_id, "case_hit_count"].iloc[0]),
            "signal_count": int(signal_count),
            "positive_close_3d_rate": float(mdf["positive_close_3d"].mean()) if not mdf.empty else np.nan,
            "positive_close_5d_rate": float(mdf["positive_close_5d"].mean()) if not mdf.empty else np.nan,
            "avg_ret_close_3d": float(mdf["ret_close_3d"].mean()) if not mdf.empty else np.nan,
            "avg_ret_close_5d": float(mdf["ret_close_5d"].mean()) if not mdf.empty else np.nan,
            "avg_max_ret_5d": float(mdf["max_ret_5d"].mean()) if not mdf.empty else np.nan,
            "double_prev_vol_share": float((mdf["vol_vs_prev"] >= 2.0).mean()) if not mdf.empty else np.nan,
        }
        top_rows.append(row)
        if row["case_hit_count"] > best_case_hit_count:
            best_case_hit_count = row["case_hit_count"]
            best_case_hits = case_hit_by_combo[combo.combo_id]
            best_combo_id = combo.combo_id
    top_df = pd.DataFrame(top_rows).sort_values(
        ["case_hit_count", "positive_close_5d_rate", "avg_ret_close_5d", "signal_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    top_df.to_csv(OUT_DIR / "top_coverage_combos.csv", index=False, encoding="utf-8-sig")

    best_id = str(top_df.iloc[0]["combo_id"])
    best_row = stage1[stage1["combo_id"] == best_id].iloc[0]
    best_combo = Combo(
        require_close_above_trend=bool(best_row["require_close_above_trend"]),
        require_bull_close=bool(best_row["require_bull_close"]),
        require_pullback_shrinking=bool(best_row["require_pullback_shrinking"]),
        require_not_sideways=bool(best_row["require_not_sideways"]),
        require_trend_gt_long=bool(best_row["require_trend_gt_long"]),
        signal_vs_ma5_low=None if pd.isna(best_row["signal_vs_ma5_low"]) else float(best_row["signal_vs_ma5_low"]),
        signal_vs_ma5_high=None if pd.isna(best_row["signal_vs_ma5_high"]) else float(best_row["signal_vs_ma5_high"]),
        ret_cap=None if pd.isna(best_row["ret_cap"]) else float(best_row["ret_cap"]),
    )

    # case hit table
    hit_flags = case_hit_by_combo[best_id]
    case_hit_table = eligible[["股票名称", "股票代码", "信号日期"]].copy()
    case_hit_table["状态"] = ["命中" if x else "未命中" for x in hit_flags]
    case_hit_table["信号日期"] = case_hit_table["信号日期"].dt.strftime("%Y-%m-%d")
    case_hit_table.to_csv(OUT_DIR / "best_combo_case_hits.csv", index=False, encoding="utf-8-sig")

    # volume>=2x previous-day effect
    vol_rows: List[dict] = []
    for path in files:
        df = brick_filter.load_one_csv(str(path))
        if df is None or df.empty:
            continue
        x = brick_filter.add_features(df)
        mask = build_mask(x, best_combo)
        mask &= ~x["date"].between(EXCLUDE_START, EXCLUDE_END)
        signal_idxs = np.flatnonzero(mask.to_numpy())
        for sig_idx in signal_idxs:
            fm = future_metrics(x, int(sig_idx))
            if fm is None:
                continue
            vol_rows.append(
                {
                    "double_prev_vol": bool(pd.notna(fm["vol_vs_prev"]) and fm["vol_vs_prev"] >= 2.0),
                    "ret_close_3d": fm["ret_close_3d"],
                    "ret_close_5d": fm["ret_close_5d"],
                    "max_ret_5d": fm["max_ret_5d"],
                    "positive_close_3d": fm["positive_close_3d"],
                    "positive_close_5d": fm["positive_close_5d"],
                }
            )
    vol_df = pd.DataFrame(vol_rows)
    vol_summary = (
        vol_df.groupby("double_prev_vol")
        .agg(
            signal_count=("double_prev_vol", "size"),
            positive_close_3d_rate=("positive_close_3d", "mean"),
            positive_close_5d_rate=("positive_close_5d", "mean"),
            avg_ret_close_3d=("ret_close_3d", "mean"),
            avg_ret_close_5d=("ret_close_5d", "mean"),
            avg_max_ret_5d=("max_ret_5d", "mean"),
        )
        .reset_index()
    )
    vol_summary.to_csv(OUT_DIR / "double_prev_vol_effect.csv", index=False, encoding="utf-8-sig")

    summary = {
        "case_total": int(len(cases)),
        "case_name_mapped": int(cases["名称映射成功"].sum()),
        "case_date_available": int(cases["案例日期可用"].sum()),
        "eligible_case_count": int(len(eligible)),
        "best_case_hit_count": int(best_row["case_hit_count"]),
        "best_case_hit_rate": float(best_row["case_hit_rate"]),
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
        "best_combo_universe_metrics": top_df.iloc[0].to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
