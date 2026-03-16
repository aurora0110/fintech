from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import b3filter  # type: ignore
from utils.tmp.run_b3_exit_combo_search_v2_20260314 import (  # type: ignore
    RESULT_DIR as SOURCE_RESULT_DIR,
    add_extra_features,
    AccountConfig,
    SingleCombo,
    PartialCombo,
    build_account_configs,
    evaluate_single,
    evaluate_partial,
    run_account_backtest,
)


RESULT_DIR = ROOT / "results/b3_account_focus_20260314"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data/forward_data"


def load_signal_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    signal_df = pd.read_csv(SOURCE_RESULT_DIR / "signals.csv", parse_dates=["signal_date", "entry_date"])
    codes = sorted(signal_df["code"].unique())
    data_map: dict[str, pd.DataFrame] = {}
    for idx, code in enumerate(codes, start=1):
        df = b3filter.load_one_csv(str(DATA_DIR / f"{code}.txt"))
        if df is None or df.empty:
            continue
        data_map[code] = add_extra_features(df)
        if idx % 100 == 0:
            print({"focus_load_progress": idx, "codes": len(codes)}, flush=True)
    return data_map, signal_df


def main() -> None:
    data_map, signal_df = load_signal_data()
    single_df = pd.read_csv(SOURCE_RESULT_DIR / "single_exit_results.csv")
    partial_df = pd.read_csv(SOURCE_RESULT_DIR / "partial_exit_results.csv")

    top_single = single_df.head(5).copy()
    top_partial = partial_df.head(5).copy()
    configs = build_account_configs()
    rows: list[dict[str, object]] = []

    for _, row in top_single.iterrows():
        combo = SingleCombo(
            max_hold_days=int(row["max_hold_days"]),
            profit_rule=str(row["profit_rule"]),
            profit_param=float(row["profit_param"]),
            protect_rule=str(row["protect_rule"]),
            protect_param=float(row["protect_param"]),
            stop_rule=str(row["stop_rule"]),
            stop_param=float(row["stop_param"]),
        )
        trades = [evaluate_single(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(RESULT_DIR / f"trades_single_{combo.name}.csv", index=False, encoding="utf-8")
        for config in configs:
            metrics = run_account_backtest(trades_df, data_map, config)
            metrics.update({"combo_type": "single", "combo_name": combo.name, **row.to_dict(), **config.__dict__})
            rows.append(metrics)
        print({"focus_single_done": combo.name}, flush=True)

    for _, row in top_partial.iterrows():
        combo = PartialCombo(
            max_hold_days=int(row["max_hold_days"]),
            first_rule=str(row["first_rule"]),
            first_param=float(row["first_param"]),
            second_rule=str(row["second_rule"]),
            second_param=float(row["second_param"]),
            stop_rule=str(row["stop_rule"]),
            stop_param=float(row["stop_param"]),
        )
        trades = [evaluate_partial(signal, data_map[str(signal["code"])], combo) for _, signal in signal_df.iterrows()]
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(RESULT_DIR / f"trades_partial_{combo.name}.csv", index=False, encoding="utf-8")
        for config in configs:
            metrics = run_account_backtest(trades_df, data_map, config)
            metrics.update({"combo_type": "partial", "combo_name": combo.name, **row.to_dict(), **config.__dict__})
            rows.append(metrics)
        print({"focus_partial_done": combo.name}, flush=True)

    out = pd.DataFrame(rows).sort_values(
        ["annual_return", "avg_return", "success_rate", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    out.to_csv(RESULT_DIR / "account_focus_results.csv", index=False, encoding="utf-8")
    summary = {
        "top_account": out.head(20).to_dict("records"),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
