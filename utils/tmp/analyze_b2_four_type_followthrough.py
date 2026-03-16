from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b2filter


NORMAL_DIR = ROOT / "data" / "20260313" / "normal"
CASE_DIR = ROOT / "data" / "完美图" / "B2"
NAME_MAP_DIR = ROOT / "data" / "20260313"
OUT_DIR = ROOT / "results" / "b2_four_type_followthrough_20260313"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")
DATE_RE = re.compile(r"^(?P<name>.+?)(?P<date>20\d{6})\.[^.]+$")
EPS = 1e-12


FEATURE_COLS = [
    "ret1",
    "prev_ret1",
    "signal_vs_ma5",
    "vol_vs_prev",
    "vol_vs_ma10",
    "close_position",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_ratio",
    "near_20d_high_ratio",
    "near_20d_low_ratio",
    "trend_slope_3",
    "trend_slope_5",
    "long_slope_3",
    "long_slope_5",
    "trend_line_lead",
    "j_rank20_prev",
    "j_rank30_prev",
    "box_range40",
    "box_net40",
    "box_slope20",
    "anchor_vol_rank30",
    "middle_bear_ratio",
]


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


def parse_cases(name_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for path in sorted(CASE_DIR.iterdir()):
        if not path.is_file():
            continue
        m = DATE_RE.match(path.name)
        if not m:
            continue
        rows.append(
            {
                "case_file": path.name,
                "name": m.group("name"),
                "date": pd.to_datetime(m.group("date"), format="%Y%m%d"),
                "code": name_map.get(m.group("name")),
            }
        )
    return pd.DataFrame(rows)


def rolling_rank_ratio(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def _last_rank(win: np.ndarray) -> float:
        arr = pd.Series(win).dropna()
        if arr.empty:
            return np.nan
        return arr.rank(pct=True).iloc[-1]

    return values.rolling(window, min_periods=window).apply(_last_rank, raw=True)


def longest_up_streak(values: Iterable[float]) -> int:
    arr = list(values)
    streak = 0
    best = 0
    for i in range(1, len(arr)):
        if pd.notna(arr[i]) and pd.notna(arr[i - 1]) and arr[i] > arr[i - 1]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def normalized_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or not np.isfinite(arr).all():
        return np.nan
    x = np.arange(len(arr), dtype=float)
    slope, _ = np.polyfit(x, arr, 1)
    base = np.nanmean(np.abs(arr))
    if not np.isfinite(base) or base <= EPS:
        return np.nan
    return slope / base


def add_forward_labels(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    entry_open = out["open"].shift(-1)
    out["entry_open"] = entry_open
    out["entry_close"] = out["close"].shift(-1)
    out["close5_ret"] = out["close"].shift(-5) / entry_open - 1.0
    out["close10_ret"] = out["close"].shift(-10) / entry_open - 1.0

    close_cols_5 = [out["close"].shift(-k) for k in range(1, 6)]
    close_cols_10 = [out["close"].shift(-k) for k in range(1, 11)]
    close_frame_5 = pd.concat(close_cols_5, axis=1)
    close_frame_10 = pd.concat(close_cols_10, axis=1)

    out["up5_close"] = out["close5_ret"] > 0
    out["up10_close"] = out["close10_ret"] > 0
    out["up_streak3_in_5d"] = close_frame_5.apply(lambda r: longest_up_streak(r.values) >= 3, axis=1)

    def _trend_up_10d(row: pd.Series) -> bool:
        vals = row.values.astype(float)
        if np.isfinite(vals).sum() < 8:
            return False
        slope = normalized_slope(vals[np.isfinite(vals)])
        diffs = np.diff(vals[np.isfinite(vals)])
        up_ratio = np.mean(diffs > 0) if len(diffs) else 0.0
        return bool(
            np.isfinite(slope)
            and slope > 0
            and vals[-1] > vals[0]
            and up_ratio >= 0.55
        )

    out["trend_up_10d"] = close_frame_10.apply(_trend_up_10d, axis=1)
    return out


def build_type_features(path: Path) -> Optional[pd.DataFrame]:
    df = b2filter.load_one_csv(str(path))
    if df is None or df.empty:
        return None
    x = b2filter.add_features(df)
    x = exclude_disaster(x, "date")
    if x.empty:
        return None
    x = x.sort_values("date").reset_index(drop=True)

    body = (x["close"] - x["open"]).abs()
    upper = x["high"] - np.maximum(x["open"], x["close"])
    lower = np.minimum(x["open"], x["close"]) - x["low"]
    full = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = body / full
    x["upper_shadow_ratio"] = upper / full
    x["lower_shadow_ratio"] = lower / full
    x["upper_shadow_body_ratio_08"] = upper <= body * 0.8 + EPS
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
    x["j_rank20"] = rolling_rank_ratio(x["J"], 20)
    x["j_rank30"] = rolling_rank_ratio(x["J"], 30)
    x["j_rank20_prev"] = x["j_rank20"].shift(1)
    x["j_rank30_prev"] = x["j_rank30"].shift(1)

    # 宽口径基础 B2，用于类型研究，不沿用严格版所有硬过滤。
    x["base_b2"] = (
        x["trend_ok"]
        & (x["close"] > x["open"])
        & (x["ret1"] >= 0.04)
        & x["b2_volume_ok"]
        & x["b2_j_ok"]
        & x["upper_shadow_body_ratio_08"]
    )

    # type1: T-1 贴近多空线 + J进入20日历史10%低位
    x["type1"] = (
        (x["close"].shift(1) <= x["long_line"].shift(1) * 1.02)
        & (x["j_rank20_prev"] <= 0.10)
    )

    # type2: 40日箱体横盘后放量B2
    x["box_high40"] = x["high"].shift(1).rolling(40).max()
    x["box_low40"] = x["low"].shift(1).rolling(40).min()
    x["box_range40"] = x["box_high40"] / x["box_low40"] - 1.0
    x["box_net40"] = x["close"].shift(1) / x["close"].shift(40) - 1.0
    x["box_slope20"] = x["trend_line"].shift(1) / x["trend_line"].shift(20) - 1.0
    x["type2"] = (
        (x["box_range40"] <= 0.45)
        & (x["box_net40"].abs() <= 0.12)
        & (x["box_slope20"].abs() <= 0.05)
        & (x["volume"] > x["vol_ma5"])
    )

    # type3: 最近30日内有巨量阳量锚点 + 中间阴量被锚点和当前B2包住
    x["type3"] = False
    x["anchor_vol_rank30"] = np.nan
    x["middle_bear_ratio"] = np.nan
    vol = x["volume"].astype(float).tolist()
    opens = x["open"].astype(float).tolist()
    closes = x["close"].astype(float).tolist()
    dates = x["date"].tolist()
    for i in range(len(x)):
        if i < 31:
            continue
        left = max(1, i - 30)
        anchor_idx = None
        anchor_rank = np.nan
        for j in range(i - 1, left - 1, -1):
            if closes[j] <= opens[j]:
                continue
            if vol[j] < vol[j - 1] * 2:
                continue
            recent = x.loc[left:i - 1, "volume"].nlargest(3)
            if vol[j] + EPS < recent.min():
                continue
            anchor_idx = j
            recent30 = x.loc[max(0, j - 29):j, "volume"]
            if len(recent30) >= 5:
                rank = recent30.rank(pct=True).iloc[-1]
            else:
                rank = np.nan
            anchor_rank = rank
            break
        if anchor_idx is None or anchor_idx >= i - 1:
            continue
        middle = x.iloc[anchor_idx + 1 : i]
        bears = middle[middle["close"] < middle["open"]]
        if bears.empty:
            x.at[i, "type3"] = True
            x.at[i, "anchor_vol_rank30"] = anchor_rank
            x.at[i, "middle_bear_ratio"] = 0.0
            continue
        cond = (bears["volume"] < vol[anchor_idx]) & (bears["volume"] < vol[i])
        if bool(cond.all()):
            x.at[i, "type3"] = True
            x.at[i, "anchor_vol_rank30"] = anchor_rank
            x.at[i, "middle_bear_ratio"] = len(bears) / max(len(middle), 1)

    # type4: 趋势线第一次上穿多空线后，T-1第一次回踩白线，T日出B2
    bull_cross = (x["trend_line"] > x["long_line"]) & (x["trend_line"].shift(1) <= x["long_line"].shift(1))
    x["type4"] = False
    for i in range(len(x)):
        if i < 10:
            continue
        left = max(1, i - 20)
        crosses = np.where(bull_cross.iloc[left:i].to_numpy())[0]
        if len(crosses) == 0:
            continue
        cross_idx = left + int(crosses[-1])
        # T-1 必须贴近或盘中回踩趋势线，并且是跨越后第一次回踩
        prev_touch = (
            (x.at[i - 1, "low"] <= x.at[i - 1, "trend_line"] * 1.01)
            or (x.at[i - 1, "close"] <= x.at[i - 1, "trend_line"] * 1.01)
        )
        if not prev_touch:
            continue
        if i - 2 > cross_idx:
            between = x.iloc[cross_idx + 1 : i - 1]
            had_touch = (
                (between["low"] <= between["trend_line"] * 1.01)
                | (between["close"] <= between["trend_line"] * 1.01)
            ).any()
            if had_touch:
                continue
        x.at[i, "type4"] = True

    x["any_type"] = x[["type1", "type2", "type3", "type4"]].any(axis=1)
    x["ordinary"] = x["base_b2"] & (~x["any_type"])

    x = add_forward_labels(x)
    x["code"] = path.stem
    return x


def collect_universe() -> pd.DataFrame:
    rows = []
    files = sorted(NORMAL_DIR.glob("*.txt"))
    total = len(files)
    for idx, path in enumerate(files, 1):
        if idx % 400 == 0 or idx == total:
            print(f"B2分类特征进度: {idx}/{total}")
        x = build_type_features(path)
        if x is None:
            continue
        use = x[x["base_b2"]].copy()
        if use.empty:
            continue
        rows.append(use)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_case_hits(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cases.iterrows():
        code = row["code"]
        if pd.isna(code) or not str(code).strip():
            continue
        path = NORMAL_DIR / f"{str(code).strip()}.txt"
        x = build_type_features(path)
        if x is None or x.empty:
            continue
        hit = x[(x["date"] == row["date"]) & (x["base_b2"])]
        if hit.empty:
            continue
        rec = hit.iloc[0].to_dict()
        rec.update({"case_name": row["name"], "case_file": row["case_file"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    groups = ["type1", "type2", "type3", "type4", "ordinary", "any_type"]
    rows = []
    for group in groups:
        sub = df[df[group]]
        if sub.empty:
            continue
        rows.append(
            {
                "group": group,
                "count": int(len(sub)),
                "rate_5d_close_up": float(sub["up5_close"].mean()),
                "rate_10d_close_up": float(sub["up10_close"].mean()),
                "rate_streak3_5d": float(sub["up_streak3_in_5d"].mean()),
                "rate_trend_up_10d": float(sub["trend_up_10d"].mean()),
                "avg_close5_ret": float(sub["close5_ret"].mean()),
                "avg_close10_ret": float(sub["close10_ret"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    out.to_csv(OUT_DIR / f"{label_col}_group_summary.csv", index=False, encoding="utf-8-sig")
    return out


def feature_diff_by_type(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    groups = ["type1", "type2", "type3", "type4", "ordinary", "any_type"]
    rows = []
    for group in groups:
        sub = df[df[group]].copy()
        if len(sub) < 30:
            continue
        good = sub[sub[target_col]]
        bad = sub[~sub[target_col]]
        if len(good) < 10 or len(bad) < 10:
            continue
        for feat in FEATURE_COLS:
            g = pd.to_numeric(good[feat], errors="coerce").dropna()
            b = pd.to_numeric(bad[feat], errors="coerce").dropna()
            if len(g) < 10 or len(b) < 10:
                continue
            mean_g = g.mean()
            mean_b = b.mean()
            pooled = math.sqrt(max((g.var(ddof=1) + b.var(ddof=1)) / 2.0, EPS))
            effect = (mean_g - mean_b) / pooled
            rows.append(
                {
                    "group": group,
                    "target": target_col,
                    "feature": feat,
                    "good_mean": mean_g,
                    "bad_mean": mean_b,
                    "delta": mean_g - mean_b,
                    "effect_size": effect,
                    "good_count": int(len(g)),
                    "bad_count": int(len(b)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group", "target", "effect_size"], ascending=[True, True, False]).reset_index(drop=True)
        out.to_csv(OUT_DIR / "feature_effects_by_type.csv", index=False, encoding="utf-8-sig")
    return out


def top_features_snapshot(effect_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if effect_df.empty:
        return pd.DataFrame()
    for (group, target), sub in effect_df.groupby(["group", "target"], sort=False):
        top = sub.sort_values("effect_size", ascending=False).head(5)
        for _, row in top.iterrows():
            rows.append(row.to_dict())
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "top_feature_snapshots.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    name_map = load_name_map()
    cases = parse_cases(name_map)
    case_hits = build_case_hits(cases)
    case_hits.to_csv(OUT_DIR / "b2_case_hits.csv", index=False, encoding="utf-8-sig")

    universe = collect_universe()
    universe.to_csv(OUT_DIR / "b2_universe_tagged.csv", index=False, encoding="utf-8-sig")

    group_summary = summarize_group(universe, "future")
    effects = feature_diff_by_type(universe, "up5_close")
    extra_targets = [
        feature_diff_by_type(universe, "up10_close"),
        feature_diff_by_type(universe, "up_streak3_in_5d"),
        feature_diff_by_type(universe, "trend_up_10d"),
    ]
    effects_all = pd.concat([effects, *extra_targets], ignore_index=True)
    effects_all.to_csv(OUT_DIR / "feature_effects_all_targets.csv", index=False, encoding="utf-8-sig")
    top_snapshot = top_features_snapshot(effects_all)

    case_type_counts = (
        case_hits[["type1", "type2", "type3", "type4", "ordinary", "any_type"]]
        .sum()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "group"})
    )
    case_type_counts.to_csv(OUT_DIR / "case_type_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "case_total_files": int(len(cases)),
        "case_mapped_hits": int(len(case_hits)),
        "universe_count": int(len(universe)),
        "type_counts_universe": {
            group: int(universe[group].sum()) for group in ["type1", "type2", "type3", "type4", "ordinary", "any_type"]
        },
        "type_counts_cases": {
            group: int(case_hits[group].sum()) if not case_hits.empty else 0
            for group in ["type1", "type2", "type3", "type4", "ordinary", "any_type"]
        },
        "best_groups": group_summary.sort_values("avg_close10_ret", ascending=False).head(3).to_dict(orient="records"),
        "top_feature_rows": top_snapshot.head(20).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
