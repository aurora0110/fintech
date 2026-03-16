from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
sys.path.insert(0, str(ROOT))

from utils import brick_filter


TARGET_DATES = {pd.Timestamp("2026-03-11"), pd.Timestamp("2026-03-12")}
NAME_DIR = ROOT / "data" / "20260313"


def build_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(NAME_DIR.glob("*.txt")):
        try:
            first = path.read_text(encoding="gbk", errors="ignore").splitlines()[0].strip()
        except Exception:
            continue
        parts = first.split()
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[path.stem] = parts[1]
    return mapping


NAME_MAP = build_name_map()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: scan_brick_dates_no_limits_20260315.py <txt_path>")
    path = Path(sys.argv[1])
    df = brick_filter.load_one_csv(str(path))
    if df is None or df.empty:
        return

    code = str(df["code"].iloc[0])
    x = brick_filter.add_features(df)
    x_dates = pd.to_datetime(x["date"])

    for signal_idx in x.index[x_dates.isin(TARGET_DATES)]:
        latest = x.loc[signal_idx]
        signal_ok = (
            bool(latest["signal_base"])
            and float(latest["ret1"]) <= 0.10
            and float(latest["trend_line"]) > float(latest["long_line"])
        )
        if not signal_ok:
            continue

        close = float(latest["close"])
        trend_line = float(latest["trend_line"])
        long_line = float(latest["long_line"])
        trend_spread = max((trend_line - long_line) / max(close, brick_filter.EPS), 0.0)
        trend_component = brick_filter.clip01(pd.Series([trend_spread * 20.0])).iloc[0]
        candle_component = (
            0.35 * brick_filter.clip01(pd.Series([float(latest["body_ratio"])]).fillna(0.0)).iloc[0]
            + 0.35 * brick_filter.clip01(pd.Series([float(latest["close_position"])]).fillna(0.0)).iloc[0]
            + 0.20 * brick_filter.clip01(pd.Series([1.0 - float(latest["upper_shadow_ratio"])]).fillna(0.0)).iloc[0]
            + 0.10 * brick_filter.clip01(pd.Series([1.0 - float(latest["lower_shadow_ratio"])]).fillna(0.0)).iloc[0]
        )
        volume_component = (
            0.45 * float(latest["signal_vs_ma5_quality"])
            + 0.20 * float(latest["signal_vs_ma10_quality"])
            + 0.20 * float(latest["vol_vs_prev_quality"])
            + 0.15 * float(latest["shrink_quality"])
        )
        keyk_component = float(latest["keyk_quality"])
        slope_component = brick_filter.clip01(pd.Series([(float(latest["trend_slope_5"]) + 0.02) / 0.06]).fillna(0.0)).iloc[0]
        sort_score = (
            0.30 * volume_component
            + 0.25 * candle_component
            + 0.20 * trend_component
            + 0.15 * keyk_component
            + 0.10 * slope_component
        )

        reason = "brick宽口径续冲"
        if bool(latest["double_prev_volume"]):
            reason += "+2倍量"

        print(
            "\t".join(
                [
                    pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
                    code,
                    NAME_MAP.get(code, ""),
                    f"{sort_score:.4f}",
                    f"{float(latest['signal_vs_ma5']):.6f}",
                    f"{float(latest['vol_vs_prev']):.6f}",
                    "1" if bool(latest["double_prev_volume"]) else "0",
                    reason,
                    f"{float(latest['close']):.2f}",
                    f"{float(latest['low']) * 0.99:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
