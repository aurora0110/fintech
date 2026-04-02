from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
BASE_SCRIPT = ROOT / "utils" / "tmp" / "run_brick_case_rank_final_spec_search_v1_20260327.py"
CASE_SOURCE = ROOT / "results" / "brick_case_rank_model_search_v1_full_20260327_r1" / "best_model_top20_candidates.csv"
R3_SOURCE = ROOT / "results" / "brick_relaxed_seq_model_search_v1_full_20260326_r3" / "relaxed_selected_candidates.csv"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(BASE_SCRIPT, "brick_case_rank_r3_backup_base")

DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
BUY_GAP_LIMIT = 0.04
STOP_CFG = {"stop_base": "signal_low", "stop_exec_mode": "same_day_close"}
FIXED_PROFILE = {
    "name": "atr20x2_h4",
    "family": "fixed_tp",
    "tp_pct": None,
    "hold_days": 4,
    "exit_mode": "next_day_open",
}


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_error(result_dir: Path, exc: BaseException) -> None:
    payload = {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))


def _resolve_daily_stem(code: Any) -> str:
    return base._resolve_daily_stem(code)


def _load_case_source(file_limit_codes: int, date_limit: int) -> pd.DataFrame:
    df = pd.read_csv(CASE_SOURCE, parse_dates=["signal_date", "entry_date", "exit_date"])
    df = df[(df["signal_date"] < base.EXCLUDE_START) | (df["signal_date"] > base.EXCLUDE_END)].copy()
    df["code"] = df["code"].map(_resolve_daily_stem)
    cols = ["code", "signal_date", "entry_date", "sort_score"]
    df = df[cols].copy()
    df["source_kind"] = "case_rank"
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    if date_limit > 0:
        keep_dates = sorted(df["signal_date"].dt.strftime("%Y-%m-%d").unique())[:date_limit]
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    return df.reset_index(drop=True)


def _load_r3_source(file_limit_codes: int, date_limit: int) -> pd.DataFrame:
    df = pd.read_csv(R3_SOURCE, parse_dates=["signal_date", "entry_date", "exit_date"])
    df["code"] = df["code"].map(_resolve_daily_stem)
    df = df[(df["signal_date"] < base.EXCLUDE_START) | (df["signal_date"] > base.EXCLUDE_END)].copy()
    df["sort_score"] = pd.to_numeric(df["rank_score"], errors="coerce")
    cols = ["code", "signal_date", "entry_date", "sort_score"]
    df = df[cols].copy()
    df["source_kind"] = "r3"
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    if date_limit > 0:
        keep_dates = sorted(df["signal_date"].dt.strftime("%Y-%m-%d").unique())[:date_limit]
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    return df.reset_index(drop=True)


def _policy_defs() -> list[dict[str, Any]]:
    return [
        {"name": "case_only", "trigger": "never", "threshold": 0, "topn": 0, "fill_to": 0},
        {"name": "backup_if_zero_top3", "trigger": "eq", "threshold": 0, "topn": 3, "fill_to": 0},
        {"name": "backup_if_zero_top5", "trigger": "eq", "threshold": 0, "topn": 5, "fill_to": 0},
        {"name": "backup_if_zero_top10", "trigger": "eq", "threshold": 0, "topn": 10, "fill_to": 0},
        {"name": "backup_if_lt3_fill3", "trigger": "lt", "threshold": 3, "topn": 0, "fill_to": 3},
        {"name": "backup_if_lt5_fill5", "trigger": "lt", "threshold": 5, "topn": 0, "fill_to": 5},
        {"name": "backup_if_lt10_fill10", "trigger": "lt", "threshold": 10, "topn": 0, "fill_to": 10},
    ]


def _need_backup(primary_count: int, policy: dict[str, Any]) -> bool:
    trigger = policy["trigger"]
    if trigger == "never":
        return False
    if trigger == "eq":
        return primary_count == int(policy["threshold"])
    if trigger == "lt":
        return primary_count < int(policy["threshold"])
    raise ValueError(trigger)


def _build_policy_candidates(case_df: pd.DataFrame, r3_df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_dates = sorted(set(case_df["signal_date"]).union(set(r3_df["signal_date"])))
    for signal_date in all_dates:
        primary = case_df[case_df["signal_date"] == signal_date].sort_values("sort_score", ascending=False).copy()
        primary["source_priority"] = 0
        picked = primary.to_dict("records")
        if _need_backup(len(primary), policy):
            backup = r3_df[r3_df["signal_date"] == signal_date].sort_values("sort_score", ascending=False).copy()
            backup["source_priority"] = 1
            exist_codes = {str(x["code"]) for x in picked}
            if int(policy["fill_to"]) > 0:
                need = max(0, int(policy["fill_to"]) - len(primary))
                backup = backup[~backup["code"].astype(str).isin(exist_codes)].head(need)
            elif int(policy["topn"]) > 0:
                backup = backup[~backup["code"].astype(str).isin(exist_codes)].head(int(policy["topn"]))
            else:
                backup = backup.iloc[0:0]
            picked.extend(backup.to_dict("records"))
        if picked:
            day_df = pd.DataFrame(picked)
            day_df["policy_name"] = policy["name"]
            rows.extend(day_df.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=["code", "signal_date", "entry_date", "sort_score", "source_kind", "source_priority", "policy_name"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["signal_date", "source_priority", "sort_score", "code"], ascending=[True, True, False, True]).reset_index(drop=True)
    return out


def _compute_atr20_columns(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.sort_values("date").copy()
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_20"] = tr.ewm(alpha=1.0 / 20, adjust=False, min_periods=20).mean()
    return df


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    daily_path = base.DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        return [], [{"code": code, "reason": "missing_daily_series"}]

    daily_df = base.tp_mod.hybrid.base.load_daily_df(daily_path)
    if daily_df is None or daily_df.empty:
        return [], [{"code": code, "reason": "empty_daily_df"}]
    daily_df = daily_df[(daily_df["date"] < base.EXCLUDE_START) | (daily_df["date"] > base.EXCLUDE_END)].copy()
    daily_df = _compute_atr20_columns(daily_df)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_indexed = daily_df.set_index("date")

    min5_df = None
    min5_path = base.MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = base.tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = base.tp_mod._prepare_min5_indicators(loaded) if loaded is not None and not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        signal_date = pd.Timestamp(item["signal_date"])
        if signal_date not in daily_indexed.index:
            skipped.append({"code": code, "signal_date": item["signal_date"], "policy_name": item["policy_name"], "reason": "missing_signal_row"})
            continue
        signal_row = daily_indexed.loc[signal_date]
        signal_close = float(signal_row.get("close", np.nan))
        atr20 = float(signal_row.get("atr_20", np.nan))
        if not np.isfinite(signal_close) or signal_close <= 0 or not np.isfinite(atr20) or atr20 <= 0:
            skipped.append({"code": code, "signal_date": item["signal_date"], "policy_name": item["policy_name"], "reason": "invalid_atr20"})
            continue
        tp_pct = float(2.0 * atr20 / signal_close)
        profile = {
            "name": FIXED_PROFILE["name"],
            "family": FIXED_PROFILE["family"],
            "tp_pct": tp_pct,
            "hold_days": FIXED_PROFILE["hold_days"],
            "exit_mode": FIXED_PROFILE["exit_mode"],
        }
        signal_idx = int(np.where(daily_df["date"].to_numpy() == np.datetime64(signal_date))[0][0])
        sim = base.simulate_one_trade_profile(
            code=code,
            signal_date=signal_date,
            entry_date=pd.Timestamp(item["entry_date"]),
            signal_idx=signal_idx,
            sort_score=float(item["sort_score"]),
            signal_open=float(signal_row["open"]),
            signal_close=signal_close,
            signal_low=float(signal_row["low"]),
            daily_df=daily_df,
            min5_df=min5_df,
            profile=profile,
            stop_cfg=STOP_CFG,
            buy_gap_limit=float(payload["buy_gap_limit"]),
        )
        if sim.skipped or sim.trade is None:
            skipped.append(
                {
                    "code": code,
                    "signal_date": item["signal_date"],
                    "entry_date": item["entry_date"],
                    "policy_name": item["policy_name"],
                    "reason": sim.skip_reason,
                }
            )
            continue
        trade = sim.trade
        trade.update(
            {
                "policy_name": item["policy_name"],
                "source_kind": item["source_kind"],
                "source_priority": item["source_priority"],
                "selected_tp_pct": tp_pct,
                "selected_hold_days": FIXED_PROFILE["hold_days"],
            }
        )
        trades.append(trade)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_name, g in trades.groupby("policy_name", sort=True):
        strategy = f"case_rank_r3_backup|{policy_name}"
        row = base.tp_mod.hybrid.base.summarize_trades(g, strategy)
        row.update(
            {
                "policy_name": policy_name,
                "avg_tp_pct": float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").mean()),
                "hold_days": int(FIXED_PROFILE["hold_days"]),
                "source_case_count": int((g["source_kind"] == "case_rank").sum()),
                "source_r3_count": int((g["source_kind"] == "r3").sum()),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "final_equity_signal_basket", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_account(trades: pd.DataFrame, result_dir: Path, max_workers: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for policy_name, g in trades.groupby("policy_name", sort=True):
        tmp_dir = result_dir / "_account_tmp" / policy_name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        sub = g.copy()
        sub["profile_name"] = policy_name
        sub_summary = base._summarize_account(sub, tmp_dir, max_workers=max_workers)
        if sub_summary.empty:
            continue
        row = sub_summary.iloc[0].to_dict()
        row["policy_name"] = policy_name
        row["avg_tp_pct"] = float(pd.to_numeric(g["selected_tp_pct"], errors="coerce").mean())
        row["hold_days"] = int(g["selected_hold_days"].iloc[0])
        row["source_case_count"] = int((g["source_kind"] == "case_rank").sum())
        row["source_r3_count"] = int((g["source_kind"] == "r3").sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["annual_return", "final_equity", "max_drawdown", "sharpe", "calmar"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_r3_backup_search_v1_{args.mode}_{ts}_r1"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ["error.json", "summary.json", "signal_basket_summary.csv", "account_summary.csv", "trades.csv", "skipped.csv", "progress.json"]:
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()
    update_progress(output_dir, "loading_sources")

    case_df = _load_case_source(args.file_limit_codes, args.date_limit)
    r3_df = _load_r3_source(args.file_limit_codes, args.date_limit)
    policies = _policy_defs()

    candidate_frames = [_build_policy_candidates(case_df, r3_df, p) for p in policies]
    combined = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    combined.to_csv(output_dir / "combined_source_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(policies).to_csv(output_dir / "policy_configs.csv", index=False, encoding="utf-8-sig")

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for _, row in combined.iterrows():
        grouped_rows.setdefault(str(row["code"]), []).append(
            {
                "signal_date": pd.Timestamp(row["signal_date"]),
                "entry_date": pd.Timestamp(row["entry_date"]),
                "sort_score": float(row["sort_score"]),
                "policy_name": str(row["policy_name"]),
                "source_kind": str(row["source_kind"]),
                "source_priority": int(row["source_priority"]),
            }
        )

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    code_items = sorted(grouped_rows.items())
    update_progress(output_dir, "simulating_trades", total_codes=len(code_items), total_policies=len(policies))
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(simulate_code_bundle, {"code": code, "rows": rows, "buy_gap_limit": BUY_GAP_LIMIT}): code
            for code, rows in code_items
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            batch_trades, batch_skipped = fut.result()
            trades.extend(batch_trades)
            skipped.extend(batch_skipped)
            if done % 25 == 0 or done == len(code_items):
                update_progress(output_dir, "simulating_trades", done_codes=done, total_codes=len(code_items), trade_count=len(trades), skipped_count=len(skipped))

    trades_df = pd.DataFrame(trades)
    skipped_df = pd.DataFrame(skipped)
    trades_df.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(output_dir / "skipped.csv", index=False, encoding="utf-8-sig")

    basket_df = _summarize_signal_basket(trades_df)
    basket_df.to_csv(output_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")

    update_progress(output_dir, "summarizing_account", trade_count=len(trades_df))
    account_df = _summarize_account(trades_df, output_dir, max_workers=args.max_workers)
    account_df.to_csv(output_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    best_account = account_df.iloc[0].to_dict() if not account_df.empty else {}
    best_basket = basket_df.iloc[0].to_dict() if not basket_df.empty else {}
    summary = {
        "mode": args.mode,
        "fixed_buy_model": "case_rank_lgbm_top20 + r3 backup",
        "fixed_exit_model": "ATR20 x 2.0 + hold4 + signal_low same_day_close + next_day_open",
        "best_account_policy": best_account,
        "best_signal_basket_policy": best_basket,
        "source_case_rows": int((combined["source_kind"] == "case_rank").sum()) if not combined.empty else 0,
        "source_r3_rows": int((combined["source_kind"] == "r3").sum()) if not combined.empty else 0,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(output_dir, "finished")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="case_rank 主策略 + r3 空缺补位回测")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit-codes", type=int, default=0)
    parser.add_argument("--date-limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        if args.file_limit_codes <= 0:
            args.file_limit_codes = 200
        if args.date_limit <= 0:
            args.date_limit = 12
    try:
        output_dir = run(args)
        print(output_dir)
    except BaseException as exc:  # noqa: BLE001
        out = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_r3_backup_search_v1_{args.mode}_{datetime.now().strftime('%Y%m%d')}_r1"
        out.mkdir(parents=True, exist_ok=True)
        write_error(out, exc)
        raise


if __name__ == "__main__":
    main()
