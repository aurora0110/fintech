from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PERFECT_DIR = ROOT / "data/完美图"
RAW_DIR = ROOT / "data/20260317"
NORMAL_DIR = ROOT / "data/20260317/normal"
RESULT_DIR = ROOT / "results/market_risk_sample_build_20260319"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

from utils import b2filter  # type: ignore
from utils.market_risk_tags import add_risk_features  # type: ignore


POSITIVE_FOLDERS = ["B1", "B2", "B3", "单针", "砖型图"]
NEGATIVE_FOLDERS = ["出货图", "绝对不能碰", "砖型图反例"]
COUNTERPARTY_FOLDERS = ["击穿对手盘"]

SKIP_BASENAMES = {
    "image",
    "image (1)",
    "image (2)",
    "案例图完美",
    "案例图完美2",
    "平地起惊雷图",
}

# 手工别名只保留目前已知的名字差异。
CASE_NAME_ALIASES: Dict[str, str] = {
    "亨通股份": "600226",
    "鼎捷数智": "300378",
}


def parse_case_files(folder: Path, label: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        m = re.match(r"(.+?)(\d{8})\.[^.]+$", p.name)
        if not m:
            continue
        stock_name = m.group(1).strip()
        if stock_name in SKIP_BASENAMES:
            continue
        rows.append(
            {
                "folder": folder.name,
                "label": label,
                "stock_name": stock_name,
                "signal_date": m.group(2),
                "image_name": p.name,
            }
        )
    return rows


def read_first_line(path: Path) -> str:
    for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
    return ""


def build_name_code_map() -> Dict[str, str]:
    code_to_name: Dict[str, str] = {}
    for fp in RAW_DIR.glob("*.txt"):
        first = read_first_line(fp)
        m = re.match(r"([A-Z0-9#]+)\s+(.+?)\s+", first)
        if not m:
            continue
        raw_code, stock_name = m.group(1).strip(), m.group(2).strip()
        code_to_name[raw_code] = stock_name

    normal_tail_map: Dict[str, str] = {}
    for fp in NORMAL_DIR.glob("*.txt"):
        m = re.search(r"(\d{6})", fp.stem)
        if m:
            normal_tail_map[m.group(1)] = fp.stem

    name_to_normal: Dict[str, str] = {}
    for raw_code, stock_name in code_to_name.items():
        tail = re.search(r"(\d{6})", raw_code)
        if not tail:
            continue
        normal_code = normal_tail_map.get(tail.group(1))
        if normal_code:
            name_to_normal[stock_name] = normal_code
    return name_to_normal


def resolve_code(stock_name: str, mapping: Dict[str, str]) -> Optional[str]:
    alias = CASE_NAME_ALIASES.get(stock_name)
    if alias:
        if alias.startswith(("SH#", "SZ#", "BJ#")):
            return alias
        for fp in NORMAL_DIR.glob("*.txt"):
            if alias in fp.stem:
                return fp.stem
    if stock_name in mapping:
        return mapping[stock_name]
    for mapped_name, code in mapping.items():
        if stock_name in mapped_name or mapped_name in stock_name:
            return code
    return None


def enrich_case(case: Dict[str, str], name_code_map: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = dict(case)
    code = resolve_code(case["stock_name"], name_code_map)
    out["stock_code"] = code
    if not code:
        out["status"] = "name_unmapped"
        return out

    path = NORMAL_DIR / f"{code}.txt"
    if not path.exists():
        out["status"] = "normal_missing"
        return out

    df = b2filter.load_one_csv(str(path))
    if df is None or df.empty:
        out["status"] = "load_failed"
        return out

    date_str = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    idxs = np.flatnonzero(date_str.to_numpy() == case["signal_date"])
    if len(idxs) == 0:
        out["status"] = "date_missing"
        return out

    idx = int(idxs[-1])
    feat = add_risk_features(df)
    row = feat.iloc[idx]

    out["status"] = "ok"
    out["row_index"] = idx
    out["close"] = float(row["close"])
    out["ret_5"] = float(row["ret_5"]) if np.isfinite(row["ret_5"]) else np.nan
    out["ret_10"] = float(row["ret_10"]) if np.isfinite(row["ret_10"]) else np.nan
    out["max_up_streak_10"] = int(row["max_up_streak_10"]) if np.isfinite(row["max_up_streak_10"]) else 0
    out["segment_rise_return_10"] = float(row["segment_rise_return_10"]) if np.isfinite(row["segment_rise_return_10"]) else np.nan
    out["segment_rise_slope_10"] = float(row["segment_rise_slope_10"]) if np.isfinite(row["segment_rise_slope_10"]) else np.nan
    out["dist_20d_high"] = float(row["dist_20d_high"]) if np.isfinite(row["dist_20d_high"]) else np.nan
    out["recent_heavy_bear_top_20"] = bool(row["recent_heavy_bear_top_20"])
    out["recent_failed_breakout_20"] = bool(row["recent_failed_breakout_20"])
    out["top_distribution_20"] = bool(row["top_distribution_20"])
    out["recent_stair_bear_20"] = bool(row["recent_stair_bear_20"])
    out["risk_fast_rise_5d_30"] = bool(row["risk_fast_rise_5d_30"])
    out["risk_fast_rise_5d_40"] = bool(row["risk_fast_rise_5d_40"])
    out["risk_fast_rise_10d_40"] = bool(row["risk_fast_rise_10d_40"])
    out["risk_segment_rise_slope_10_006"] = bool(row["risk_segment_rise_slope_10_006"])
    out["risk_segment_rise_slope_15_005"] = bool(row["risk_segment_rise_slope_15_005"])
    out["risk_distribution_any_20"] = bool(row["risk_distribution_any_20"])
    return out


def main() -> None:
    rows: List[Dict[str, str]] = []
    for name in POSITIVE_FOLDERS:
        rows.extend(parse_case_files(PERFECT_DIR / name, "positive"))
    for name in NEGATIVE_FOLDERS:
        rows.extend(parse_case_files(PERFECT_DIR / name, "negative"))
    for name in COUNTERPARTY_FOLDERS:
        rows.extend(parse_case_files(PERFECT_DIR / name, "counterparty"))

    name_code_map = build_name_code_map()
    enriched = [enrich_case(row, name_code_map) for row in rows]
    df = pd.DataFrame(enriched).sort_values(["label", "folder", "signal_date", "stock_name"]).reset_index(drop=True)
    df.to_csv(RESULT_DIR / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        pd.DataFrame().to_csv(RESULT_DIR / "group_summary.csv", index=False, encoding="utf-8-sig")
        print({"total_cases": len(df), "ok_cases": 0})
        return

    summary_rows: List[Dict[str, object]] = []
    binary_cols = [
        "recent_heavy_bear_top_20",
        "recent_failed_breakout_20",
        "top_distribution_20",
        "recent_stair_bear_20",
        "risk_fast_rise_5d_30",
        "risk_fast_rise_5d_40",
        "risk_fast_rise_10d_40",
        "risk_segment_rise_slope_10_006",
        "risk_segment_rise_slope_15_005",
        "risk_distribution_any_20",
    ]
    numeric_cols = [
        "ret_5",
        "ret_10",
        "max_up_streak_10",
        "segment_rise_return_10",
        "segment_rise_slope_10",
        "dist_20d_high",
    ]

    for label, g in ok.groupby("label"):
        row: Dict[str, object] = {"label": label, "sample_count": int(len(g))}
        for col in binary_cols:
            row[col + "_rate"] = round(float(g[col].astype(float).mean()), 4)
        for col in numeric_cols:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col + "_median"] = round(float(s.median()), 4) if not s.empty else np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("label").reset_index(drop=True)
    summary.to_csv(RESULT_DIR / "group_summary.csv", index=False, encoding="utf-8-sig")
    print({"total_cases": int(len(df)), "ok_cases": int(len(ok)), "labels": ok["label"].value_counts().to_dict()})


if __name__ == "__main__":
    main()
