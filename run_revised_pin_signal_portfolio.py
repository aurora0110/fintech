from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.metrics import compute_metrics


SIGNAL_CSV = "/Users/lidongyang/Desktop/Qstrategy/results/pin_minimal_signal_analysis_20260311/historical_signals.csv"
CALENDAR_FILE = "/Users/lidongyang/Desktop/Qstrategy/data/SH#000001_INDEX.txt"
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/results/revised_pin_signal_portfolio_20260311")


@dataclass
class Variant:
    name: str
    holding_days: int
    filters: Dict[str, object]


def build_variants() -> List[Variant]:
    return [
        Variant("pin_core_hold2", 2, {}),
        Variant("pin_core_hold3", 3, {}),
        Variant(
            "pin_repair_hold2",
            2,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
            },
        ),
        Variant(
            "pin_repair_hold3",
            3,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
            },
        ),
        Variant(
            "pin_repair_gem_hold2",
            2,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
                "gem_board": True,
            },
        ),
        Variant(
            "pin_repair_gem_hold3",
            3,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
                "gem_board": True,
            },
        ),
        Variant(
            "pin_repair_friday_hold2",
            2,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
                "friday_signal": True,
            },
        ),
        Variant(
            "pin_repair_friday_hold3",
            3,
            {
                "strong_5d_momentum": True,
                "deep_signal_drop": True,
                "volume_band": True,
                "friday_signal": True,
            },
        ),
    ]


def load_calendar() -> List[pd.Timestamp]:
    df = pd.read_csv(
        CALENDAR_FILE,
        sep=r"\s+|\t+",
        engine="python",
        skiprows=1,
        header=None,
        usecols=[0],
        names=["date"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    return sorted(df["date"].dropna().unique().tolist())


def filter_signals(signals: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    subset = signals.copy()
    if filters.get("strong_5d_momentum"):
        subset = subset.loc[subset["strong_5d_momentum"].fillna(False)]
    if filters.get("deep_signal_drop"):
        subset = subset.loc[subset["deep_signal_drop"].fillna(False)]
    if filters.get("volume_band"):
        subset = subset.loc[(subset["moderate_shrink_volume"].fillna(False)) | (subset["mild_expand_volume"].fillna(False))]
    if filters.get("gem_board"):
        subset = subset.loc[subset["gem_board"].fillna(False)]
    if filters.get("friday_signal"):
        subset = subset.loc[subset["friday_signal"].fillna(False)]
    return subset.copy()


def run_variant(signals: pd.DataFrame, variant: Variant, calendar: List[pd.Timestamp]) -> dict:
    ret_col = f"hold_{variant.holding_days}d_return"
    win_col = f"hold_{variant.holding_days}d_win"
    subset = filter_signals(signals, variant.filters)
    subset = subset.loc[subset[ret_col].notna()].copy()
    subset["entry_date"] = pd.to_datetime(subset["next_date"])
    date_to_idx = {dt: idx for idx, dt in enumerate(calendar)}
    exit_dates = []
    for signal_date in pd.to_datetime(subset["signal_date"]):
        idx = date_to_idx.get(signal_date)
        exit_idx = None if idx is None else idx + variant.holding_days
        exit_dates.append(calendar[exit_idx] if exit_idx is not None and exit_idx < len(calendar) else pd.NaT)
    subset["exit_date"] = exit_dates
    subset = subset.loc[subset["exit_date"].notna()].copy()
    subset["score"] = subset["return_5d"] - subset["signal_day_return"] + subset["gem_board"].astype(float) * 0.01
    subset = subset.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)

    initial_capital = 1_000_000.0
    max_positions = 10
    daily_cash = initial_capital
    positions: List[dict] = []
    trades: List[dict] = []
    all_dates = sorted(set(pd.to_datetime(signals["next_date"]).tolist()) | set(pd.to_datetime(subset["exit_date"]).tolist()))
    equity_points: List[float] = []

    for current_date in all_dates:
        remaining: List[dict] = []
        for pos in positions:
            if pos["exit_date"] == current_date:
                pnl = pos["capital"] * pos["return"]
                daily_cash += pos["capital"] + pnl
                trades.append(
                    {
                        "entry_date": pos["entry_date"],
                        "exit_date": pos["exit_date"],
                        "code": pos["code"],
                        "return_pct": pos["return"],
                        "pnl": pnl,
                    }
                )
            else:
                remaining.append(pos)
        positions = remaining

        todays = subset.loc[subset["entry_date"] == current_date].copy()
        if not todays.empty:
            held_codes = {pos["code"] for pos in positions}
            todays = todays.loc[~todays["code"].isin(held_codes)].head(max_positions - len(positions))
            if not todays.empty and len(positions) < max_positions:
                allocation = daily_cash / max(len(todays), 1)
                for _, row in todays.iterrows():
                    if allocation <= 0:
                        break
                    positions.append(
                        {
                            "code": row["code"],
                            "entry_date": current_date,
                            "exit_date": row["exit_date"],
                            "capital": allocation,
                            "return": float(row[ret_col]),
                        }
                    )
                    daily_cash -= allocation

        equity = daily_cash + sum(pos["capital"] for pos in positions)
        equity_points.append(equity)

    equity_curve = pd.Series(equity_points, index=pd.Index(all_dates, name="date"), name="equity")
    metrics = compute_metrics(equity_curve)
    trade_df = pd.DataFrame(trades)
    win_rate = float((trade_df["return_pct"] > 0).mean()) if not trade_df.empty else 0.0
    avg_trade_return = float(trade_df["return_pct"].mean()) if not trade_df.empty else 0.0
    return {
        "variant": variant.name,
        "holding_days": variant.holding_days,
        "signal_count": int(len(subset)),
        "trade_count": int(len(trade_df)),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        **metrics,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(
        SIGNAL_CSV,
        parse_dates=["signal_date", "next_date"],
    )
    calendar = load_calendar()
    rows = [run_variant(signals, variant, calendar) for variant in build_variants()]
    summary = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "summary_table.csv", index=False, encoding="utf-8-sig")

    payload = {
        "top_variants": summary.head(5).to_dict(orient="records"),
        "all_variants": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
