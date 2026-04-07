from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brickfilter_case_rank_lgbm_top20 as case_rank_filter


RESULT_ROOT = ROOT / "results"
DATA_DIR = ROOT / "data" / "20260324"
TOP_N = 20
MAX_WORKERS = 10
THRESHOLD_QUANTILES = [0.10, 0.05, 0.00]


def update_progress(result_dir: Path, stage: str, **extra) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_scored_training_df() -> tuple[pd.DataFrame, dict]:
    bundle = case_rank_filter._load_runtime_bundle()
    model_result_dir = Path(bundle["model_search_result_dir"])
    df = pd.read_csv(model_result_dir / "candidate_dataset.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = case_rank_filter.rank_model._prepare_features(df)
    df["model_score"] = case_rank_filter.phase0._score_with_model(bundle["model"], df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df, bundle


def build_thresholds(scored_train: pd.DataFrame) -> pd.DataFrame:
    pos = scored_train[scored_train["label"] == 1].copy()
    rows = []
    for q in THRESHOLD_QUANTILES:
        rows.append(
            {
                "quantile": q,
                "threshold": float(pd.to_numeric(pos["model_score"], errors="coerce").quantile(q)),
            }
        )
    return pd.DataFrame(rows)


def select_daily_topn(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    current = df[pd.to_numeric(df["model_score"], errors="coerce").fillna(-1.0) >= threshold].copy()
    if current.empty:
        return current
    return (
        current.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True])
        .groupby("signal_date", group_keys=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )


def summarize_case_layer(name: str, threshold: float, selected: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    total_positive = int(full_df["label"].sum())
    selected_positive = int(selected["label"].sum())
    selected_count = int(len(selected))
    date_hits = selected.groupby("signal_date")["label"].max().fillna(0).astype(int) if not selected.empty else pd.Series(dtype=int)
    full_date_hits = full_df.groupby("signal_date")["label"].max().fillna(0).astype(int)
    hit_dates = int(date_hits.sum()) if not date_hits.empty else 0
    total_positive_dates = int(full_date_hits.sum())
    return {
        "variant": name,
        "threshold": threshold,
        "selected_count": selected_count,
        "selected_positive": selected_positive,
        "precision": float(selected_positive / selected_count) if selected_count else 0.0,
        "case_recall": float(selected_positive / total_positive) if total_positive else 0.0,
        "hit_dates": hit_dates,
        "date_recall": float(hit_dates / total_positive_dates) if total_positive_dates else 0.0,
    }


def load_prebuilt_recent_df() -> pd.DataFrame:
    prebuilt = Path(case_rank_filter._resolve_phase0_result_dir()) / "daily_scored_candidates.csv"
    df = pd.read_csv(prebuilt, parse_dates=["signal_date"])
    max_date = pd.Timestamp(df["signal_date"].max())
    window_start = max_date - pd.Timedelta(days=60)
    df = df[df["signal_date"].between(window_start, max_date)].copy()
    df["code6"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    return df


def build_code_path_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in DATA_DIR.glob("*.txt"):
        code6 = "".join(ch for ch in path.stem if ch.isdigit())[-6:]
        if code6:
            mapping[code6] = str(path)
    normal_dir = DATA_DIR / "normal"
    if normal_dir.exists():
        for path in normal_dir.glob("*.txt"):
            code6 = "".join(ch for ch in path.stem if ch.isdigit())[-6:]
            if code6 and code6 not in mapping:
                mapping[code6] = str(path)
    return mapping


def calc_trade_return(task: tuple[str, str, str]) -> dict | None:
    code6, path_str, signal_date_str = task
    sim = case_rank_filter.case_first.sim
    df = sim.load_stock_data(path_str)
    if df is None or df.empty:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    signal_date = pd.Timestamp(signal_date_str)
    match = df.index[df["date"] == signal_date]
    if len(match) == 0:
        return None
    idx = int(match[-1])
    if idx + 3 >= len(df):
        return None
    entry = df.iloc[idx + 1]
    exit_row = df.iloc[idx + 3]
    entry_open = float(entry["open"])
    exit_close = float(exit_row["close"])
    if entry_open <= 0:
        return None
    return {
        "code": code6,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "entry_date": pd.Timestamp(entry["date"]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(exit_row["date"]).strftime("%Y-%m-%d"),
        "entry_open": entry_open,
        "exit_close": exit_close,
        "return_h3": exit_close / entry_open - 1.0,
    }


def summarize_recent_returns(name: str, threshold: float, selected_recent: pd.DataFrame, code_map: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    if selected_recent.empty:
        return {
            "variant": name,
            "threshold": threshold,
            "trade_count": 0,
            "avg_return_h3": 0.0,
            "win_rate": 0.0,
            "avg_daily_basket_return": 0.0,
        }, pd.DataFrame()

    tasks = []
    for row in selected_recent.itertuples(index=False):
        code6 = str(getattr(row, "code6"))
        path_str = code_map.get(code6)
        if not path_str:
            continue
        tasks.append((code6, path_str, pd.Timestamp(getattr(row, "signal_date")).strftime("%Y-%m-%d")))

    rows = []
    ctx = get_context("fork")
    with ctx.Pool(processes=MAX_WORKERS) as pool:
        for item in pool.imap_unordered(calc_trade_return, tasks, chunksize=16):
            if item is not None:
                rows.append(item)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return {
            "variant": name,
            "threshold": threshold,
            "trade_count": 0,
            "avg_return_h3": 0.0,
            "win_rate": 0.0,
            "avg_daily_basket_return": 0.0,
        }, trades

    trades["return_h3"] = pd.to_numeric(trades["return_h3"], errors="coerce")
    daily = trades.groupby("signal_date")["return_h3"].mean().reset_index(name="basket_return_h3")
    summary = {
        "variant": name,
        "threshold": threshold,
        "trade_count": int(len(trades)),
        "avg_return_h3": float(trades["return_h3"].mean()),
        "win_rate": float((trades["return_h3"] > 0).mean()),
        "avg_daily_basket_return": float(daily["basket_return_h3"].mean()),
    }
    return summary, trades


def main() -> None:
    result_dir = RESULT_ROOT / f"brick_case_rank_threshold_compare_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_progress(result_dir, "loading")
        scored_train, bundle = load_scored_training_df()
        threshold_df = build_thresholds(scored_train)
        recent_df = load_prebuilt_recent_df()
        code_map = build_code_path_map()

        case_rows = []
        return_rows = []
        selected_map: dict[str, pd.DataFrame] = {}

        total = len(threshold_df)
        for idx, row in enumerate(threshold_df.itertuples(index=False), start=1):
            variant = f"q{int(round(float(row.quantile) * 100)):02d}"
            threshold = float(row.threshold)
            selected_train = select_daily_topn(scored_train, threshold)
            case_rows.append(summarize_case_layer(variant, threshold, selected_train, scored_train))

            selected_recent = select_daily_topn(recent_df, threshold)
            ret_summary, trades = summarize_recent_returns(variant, threshold, selected_recent, code_map)
            return_rows.append(ret_summary)
            selected_map[variant] = trades
            update_progress(result_dir, "running", done=idx, total=total, variant=variant, threshold=threshold)

        case_df = pd.DataFrame(case_rows)
        return_df = pd.DataFrame(return_rows)
        merged = case_df.merge(return_df, on=["variant", "threshold"], how="left")

        threshold_df.to_csv(result_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
        case_df.to_csv(result_dir / "case_summary.csv", index=False, encoding="utf-8-sig")
        return_df.to_csv(result_dir / "recent_h3_summary.csv", index=False, encoding="utf-8-sig")
        merged.to_csv(result_dir / "merged_summary.csv", index=False, encoding="utf-8-sig")
        for variant, trades in selected_map.items():
            trades.to_csv(result_dir / f"{variant}_recent_h3_trades.csv", index=False, encoding="utf-8-sig")

        payload = {
            "model_result_dir": bundle["model_search_result_dir"],
            "phase0_result_dir": bundle["phase0_result_dir"],
            "top_n": TOP_N,
            "threshold_quantiles": THRESHOLD_QUANTILES,
            "recent_window_start": str(pd.Timestamp(recent_df["signal_date"].min()).date()) if not recent_df.empty else None,
            "recent_window_end": str(pd.Timestamp(recent_df["signal_date"].max()).date()) if not recent_df.empty else None,
            "variants": merged.to_dict("records"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (result_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        update_progress(result_dir, "finished", output_dir=str(result_dir))
        print(result_dir)
        print(merged.to_string(index=False))
    except Exception as exc:
        (result_dir / "error.json").write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))
        raise


if __name__ == "__main__":
    main()
