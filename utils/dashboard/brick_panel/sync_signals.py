from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .paths import RESULTS_DIR
    from .pricing import build_basket_snapshot, get_signal_bar
    from .storage import (
        init_db,
        list_signals,
        replace_basket_snapshot,
        upsert_signals,
    )
except ImportError:  # pragma: no cover - for direct script execution
    from paths import RESULTS_DIR
    from pricing import build_basket_snapshot, get_signal_bar
    from storage import (
        init_db,
        list_signals,
        replace_basket_snapshot,
        upsert_signals,
    )


STRATEGY_LIST_MAP = {
    "brick": "brick_list",
    "brick_case_rank": "brick_case_rank_lgbm_top20_list",
}


def _iter_result_files() -> list[Path]:
    files = []
    for path in RESULTS_DIR.glob("*.json"):
        if path.stem.isdigit() and len(path.stem) == 8:
            files.append(path)
    return sorted(files, key=lambda p: p.stem)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_signal_records() -> list[dict[str, Any]]:
    import_batch = datetime.now().strftime("%Y%m%d%H%M%S")
    records: list[dict[str, Any]] = []
    for path in _iter_result_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        signal_date = datetime.strptime(path.stem, "%Y%m%d").strftime("%Y-%m-%d")
        for strategy, list_key in STRATEGY_LIST_MAP.items():
            for item in payload.get(list_key, []) or []:
                if not item:
                    continue
                code = str(item[0]).zfill(6)
                bar = get_signal_bar(code, signal_date)
                signal_close = bar["signal_close"] if bar else _safe_float(item[2] if len(item) > 2 else None)
                signal_low = bar["signal_low"] if bar else None
                records.append(
                    {
                        "signal_date": signal_date,
                        "code": code,
                        "name": bar["name"] if bar and bar.get("name") else "",
                        "strategy": strategy,
                        "source_list": list_key,
                        "signal_close": signal_close,
                        "signal_low": signal_low,
                        "stop_loss_price": signal_low if signal_low is not None else _safe_float(item[1] if len(item) > 1 else None),
                        "raw_score": _safe_float(item[3] if len(item) > 3 else None),
                        "raw_reason": str(item[4]) if len(item) > 4 else "",
                        "import_batch": import_batch,
                    }
                )
    return records


def refresh_recent_baskets(days: int = 5) -> None:
    signals = list_signals()
    if signals.empty:
        return
    for strategy in ("brick", "brick_case_rank"):
        strategy_df = signals[signals["strategy"] == strategy].copy()
        if strategy_df.empty:
            continue
        recent_dates = (
            strategy_df["signal_date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .head(days)
            .tolist()
        )
        for signal_date in recent_dates:
            codes = strategy_df.loc[strategy_df["signal_date"] == signal_date, "code"].tolist()
            summary, members = build_basket_snapshot(signal_date, strategy, codes)
            replace_basket_snapshot(
                signal_date=signal_date,
                strategy=strategy,
                stock_count=summary["stock_count"],
                avg_return_to_latest_close=summary["avg_return_to_latest_close"],
                latest_price_date=summary["latest_price_date"],
                members=members,
            )


def sync_all_signals() -> int:
    init_db()
    records = build_signal_records()
    count = upsert_signals(records)
    refresh_recent_baskets()
    return count


if __name__ == "__main__":
    synced = sync_all_signals()
    print(f"synced_signals={synced}")
