from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
BUY_MODEL_RESULT_DIR = RESULT_ROOT / "brick_case_rank_model_search_v1_full_20260327_r1"
SOURCE_CANDIDATES = BUY_MODEL_RESULT_DIR / "best_model_top20_candidates.csv"
DAILY_DIR = ROOT / "data" / "20260324"
MIN5_DIR = ROOT / "data" / "202603245min"
TP_SEARCH_PATH = ROOT / "utils" / "tmp" / "run_brick_r3_minute_tp_search_v1_20260327.py"
REAL_ACCOUNT_PATH = ROOT / "utils" / "tmp" / "run_brick_real_account_compare_v1_20260326.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TP_PCT = 0.055
TP_PROFILE = "tp_next_open"
BUY_GAP_LIMIT = 0.04
MAX_HOLD_DAYS = 3
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 4) - 1, 10))
STOP_CONFIGS: list[tuple[str, float]] = [
    ("min_oc", 1.0),
    ("min_oc", 0.9975),
    ("min_oc", 0.9950),
    ("min_oc", 0.9925),
    ("entry_low", 1.0),
    ("entry_low", 0.9975),
    ("entry_low", 0.9950),
    ("entry_low", 0.9925),
]
_DAILY_STEM_MAP: dict[str, str] | None = None


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tp_mod = load_module(TP_SEARCH_PATH, "brick_case_rank_sl_tp_mod")
real_account = load_module(REAL_ACCOUNT_PATH, "brick_case_rank_sl_real_account")


@dataclass
class SimResult:
    trade: dict[str, Any] | None
    skipped: bool = False
    skip_reason: str = ""


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


def _code_key(value: Any) -> str:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return digits
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6)


def _build_stem_map(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        key = _code_key(path.stem)
        if key:
            mapping[key] = path.stem
    return mapping


def _resolve_daily_stem(code: Any) -> str:
    global _DAILY_STEM_MAP
    if _DAILY_STEM_MAP is None:
        _DAILY_STEM_MAP = _build_stem_map(DAILY_DIR)
    text = str(code)
    key = _code_key(text)
    return _DAILY_STEM_MAP.get(key, text)


def load_source_candidates(source_csv: Path, score_col: str, strategy_key: str, file_limit_codes: int, date_limit: int) -> pd.DataFrame:
    sample_cols = pd.read_csv(source_csv, nrows=0).columns.tolist()
    parse_dates = [col for col in ["signal_date", "entry_date", "exit_date"] if col in sample_cols]
    df = pd.read_csv(source_csv, parse_dates=parse_dates)
    sort_cols = ["signal_date", score_col, "code"] if score_col in df.columns else ["signal_date", "code"]
    ascending = [True, False, True] if len(sort_cols) == 3 else [True, True]
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    if file_limit_codes > 0:
        keep_codes = sorted(df["code"].astype(str).unique())[:file_limit_codes]
        df = df[df["code"].astype(str).isin(keep_codes)].copy()
    if date_limit > 0:
        keep_dates = sorted(pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d").unique())[:date_limit]
        df = df[df["signal_date"].dt.strftime("%Y-%m-%d").isin(keep_dates)].copy()
    if score_col not in df.columns:
        raise RuntimeError(f"源候选缺少排序列: {score_col}")
    df["code"] = df["code"].map(_resolve_daily_stem)
    df["sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df["strategy_key"] = strategy_key
    return df.reset_index(drop=True)


def simulate_code_bundle(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(payload["code"])
    rows = payload["rows"]
    tp_profile = str(payload["tp_profile"])
    tp_pct = float(payload["tp_pct"])
    buy_gap_limit = float(payload["buy_gap_limit"])
    max_hold_days = int(payload["max_hold_days"])
    strategy_key_prefix = str(payload["strategy_key_prefix"])
    daily_path = DAILY_DIR / f"{code}.txt"
    if not daily_path.exists():
        skipped = []
        for item in rows:
            for stop_base, stop_multiplier in payload["stop_configs"]:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "stop_base": stop_base,
                        "stop_multiplier": stop_multiplier,
                        "reason": "missing_daily_series",
                    }
                )
        return [], skipped

    daily_df = tp_mod.hybrid.base.load_daily_df(daily_path)
    min5_df = None
    min5_path = MIN5_DIR / f"{code}.txt"
    if min5_path.exists():
        loaded = tp_mod.hybrid.base.load_minute_df(min5_path)
        min5_df = tp_mod._prepare_min5_indicators(loaded) if not loaded.empty else None

    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in rows:
        for stop_base, stop_multiplier in payload["stop_configs"]:
            sim = tp_mod.simulate_one_trade_profile(
                code=code,
                signal_date=pd.Timestamp(item["signal_date"]),
                entry_date=pd.Timestamp(item["entry_date"]),
                signal_idx=int(item.get("signal_idx", -1)),
                daily_df=daily_df,
                min5_df=min5_df,
                profile_name=tp_profile,
                stop_base=stop_base,
                stop_multiplier=float(stop_multiplier),
                tp_pct=tp_pct,
                buy_gap_limit=buy_gap_limit,
                max_hold_days=max_hold_days,
                strategy_key_prefix=f"{strategy_key_prefix}|{stop_base}_x_{float(stop_multiplier):.4f}",
            )
            if sim.skipped or sim.trade is None:
                skipped.append(
                    {
                        "code": code,
                        "signal_date": item["signal_date"],
                        "entry_date": item["entry_date"],
                        "stop_base": stop_base,
                        "stop_multiplier": stop_multiplier,
                        "reason": sim.skip_reason,
                    }
                )
                continue
            tr = sim.trade
            tr["sort_score"] = float(item["sort_score"])
            tr["stop_base"] = stop_base
            tr["stop_multiplier"] = float(stop_multiplier)
            tr["stop_key"] = f"{stop_base}_x_{float(stop_multiplier):.4f}"
            trades.append(tr)
    return trades, skipped


def _summarize_signal_basket(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stop_base, stop_multiplier), g in trades.groupby(["stop_base", "stop_multiplier"], sort=True):
        strategy = f"case_rank|{stop_base}_x_{float(stop_multiplier):.4f}|{TP_PROFILE}"
        row = tp_mod.hybrid.base.summarize_trades(g, strategy)
        row["stop_base"] = stop_base
        row["stop_multiplier"] = float(stop_multiplier)
        row["stop_key"] = f"{stop_base}_x_{float(stop_multiplier):.4f}"
        row["trigger_source_5min_ratio"] = float((g["trigger_source"] == "5min").mean())
        row["trigger_source_daily_ratio"] = float((g["trigger_source"] == "daily_fallback").mean())
        row["tp_armed_ratio"] = float(g["tp_arm_date"].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return_signal_basket", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _summarize_account(trades: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    all_codes = sorted(trades["code"].astype(str).unique())
    market_dates, close_map = real_account.build_close_map(
        all_codes,
        progress_cb=lambda done, total: update_progress(result_dir, "building_close_map", done_codes=int(done), total_codes=int(total)),
    )
    if len(market_dates) == 0:
        raise RuntimeError("无法构建账户层 close_map")

    rows: list[dict[str, Any]] = []
    config = real_account.AccountConfig()
    for (stop_base, stop_multiplier), g in trades.groupby(["stop_base", "stop_multiplier"], sort=True):
        use = g.copy()
        use["strategy_key"] = f"case_rank|{stop_base}_x_{float(stop_multiplier):.4f}|{TP_PROFILE}"
        equity_df, executed_df, summary = real_account.simulate_real_account(use, close_map, market_dates, config)
        stem = f"{stop_base}_x_{float(stop_multiplier):.4f}".replace(".", "p")
        equity_df.to_csv(result_dir / f"equity_{stem}.csv", index=False, encoding="utf-8-sig")
        executed_df.to_csv(result_dir / f"executed_{stem}.csv", index=False, encoding="utf-8-sig")
        row = {"stop_base": stop_base, "stop_multiplier": float(stop_multiplier), "stop_key": f"{stop_base}_x_{float(stop_multiplier):.4f}", **summary}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["annual_return", "avg_trade_return", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_search(
    result_dir: Path,
    source_csv: Path,
    score_col: str,
    strategy_key: str,
    file_limit_codes: int,
    date_limit: int,
    max_workers: int,
    tp_pct: float,
    buy_gap_limit: float,
    max_hold_days: int,
    stop_configs: list[tuple[str, float]],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    update_progress(
        result_dir,
        "loading_source",
        source_csv=str(source_csv),
        score_col=score_col,
        strategy_key=strategy_key,
        file_limit_codes=file_limit_codes,
        date_limit=date_limit,
        max_workers=max_workers,
        tp_pct=tp_pct,
        stop_configs=stop_configs,
    )
    candidates = load_source_candidates(source_csv, score_col, strategy_key, file_limit_codes, date_limit)
    candidates.to_csv(result_dir / "source_candidates.csv", index=False, encoding="utf-8-sig")
    if candidates.empty:
        raise RuntimeError("新买点源候选为空")

    if "signal_idx" not in candidates.columns:
        candidates["signal_idx"] = -1
    grouped_payloads = []
    for code, g in candidates.groupby("code", sort=True):
        grouped_payloads.append(
            {
                "code": str(code),
                "rows": g[["signal_date", "entry_date", "signal_idx", "sort_score"]].to_dict("records"),
                "tp_profile": TP_PROFILE,
                "tp_pct": float(tp_pct),
                "buy_gap_limit": float(buy_gap_limit),
                "max_hold_days": int(max_hold_days),
                "strategy_key_prefix": str(strategy_key),
                "stop_configs": [(str(base), float(mult)) for base, mult in stop_configs],
            }
        )

    total_codes = len(grouped_payloads)
    total_jobs = len(candidates) * len(stop_configs)
    trade_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(simulate_code_bundle, payload): payload["code"] for payload in grouped_payloads}
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            code = future_map[future]
            trades_part, skipped_part = future.result()
            trade_rows.extend(trades_part)
            skipped_rows.extend(skipped_part)
            if completed == 1 or completed % 25 == 0 or completed == total_codes:
                update_progress(
                    result_dir,
                    "simulating_trades",
                    done_codes=completed,
                    total_codes=total_codes,
                    done_jobs=len(trade_rows) + len(skipped_rows),
                    total_jobs=total_jobs,
                    last_code=code,
                )

    trades = pd.DataFrame(trade_rows).sort_values(["stop_base", "stop_multiplier", "signal_date", "code"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows).sort_values(["stop_base", "stop_multiplier", "signal_date", "code"], na_position="last").reset_index(drop=True) if skipped_rows else pd.DataFrame()
    trades.to_csv(result_dir / "stop_profile_trades.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(result_dir / "stop_profile_skipped.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "trades_ready", trade_count=int(len(trades)), skipped_count=int(len(skipped)))

    signal_summary = _summarize_signal_basket(trades)
    signal_summary.to_csv(result_dir / "signal_basket_summary.csv", index=False, encoding="utf-8-sig")
    update_progress(result_dir, "signal_basket_ready", profile_count=int(len(signal_summary)))

    account_summary = _summarize_account(trades, result_dir)
    account_summary.to_csv(result_dir / "account_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "source_candidates": str(source_csv),
            "fixed_buy_model": strategy_key,
            "fixed_tp_profile": TP_PROFILE,
            "tp_pct": tp_pct,
            "buy_gap_limit": buy_gap_limit,
            "max_hold_days": max_hold_days,
            "minute_priority": "5min_then_daily_fallback",
            "same_bar_priority": "stop_first",
            "stop_configs": [{"stop_base": base, "stop_multiplier": float(mult)} for base, mult in stop_configs],
        },
        "best_signal_basket_stop": signal_summary.iloc[0].to_dict() if not signal_summary.empty else {},
        "best_account_stop": account_summary.iloc[0].to_dict() if not account_summary.empty else {},
        "signal_stop_count": int(len(signal_summary)),
        "account_stop_count": int(len(account_summary)),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    update_progress(result_dir, "finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定新买点和 tp_next_open，搜索真实失效型止损")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-csv", type=str, default=str(SOURCE_CANDIDATES))
    parser.add_argument("--score-col", type=str, default="sort_score")
    parser.add_argument("--strategy-key", type=str, default="case_rank_lgbm_top20")
    parser.add_argument("--tp-pct", type=float, default=TP_PCT)
    parser.add_argument("--buy-gap-limit", type=float, default=BUY_GAP_LIMIT)
    parser.add_argument("--max-hold-days", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--file-limit-codes", type=int, default=120)
    parser.add_argument("--date-limit", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--stop-configs", type=str, default="")
    return parser.parse_args()


def _parse_stop_configs(text: str) -> list[tuple[str, float]]:
    if not text.strip():
        return STOP_CONFIGS.copy()
    configs: list[tuple[str, float]] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        base, mult = item.split(":")
        configs.append((base.strip(), float(mult.strip())))
    return configs


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_case_rank_sl_search_v1_{args.mode}_{timestamp}"
    file_limit_codes = int(args.file_limit_codes)
    date_limit = int(args.date_limit)
    if args.mode == "full":
        file_limit_codes = 0
        date_limit = 0
    try:
        run_search(
            result_dir=output_dir,
            source_csv=Path(args.source_csv),
            score_col=str(args.score_col),
            strategy_key=str(args.strategy_key),
            file_limit_codes=file_limit_codes,
            date_limit=date_limit,
            max_workers=int(args.max_workers),
            tp_pct=float(args.tp_pct),
            buy_gap_limit=float(args.buy_gap_limit),
            max_hold_days=int(args.max_hold_days),
            stop_configs=_parse_stop_configs(str(args.stop_configs)),
        )
    except Exception as exc:
        write_error(output_dir, exc)
        raise


if __name__ == "__main__":
    main()
