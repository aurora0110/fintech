from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brickfilter_relaxed_fusion as relaxed

DEFAULT_DATA_SNAPSHOT_DIR = ROOT / "data" / "20260324"
CASE_DIR = ROOT / "data" / "完美图" / "砖型图"
RESULT_ROOT = ROOT / "results"
DEFAULT_MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
EXCLUDE_START = pd.Timestamp("2015-06-01")
EXCLUDE_END = pd.Timestamp("2024-09-30")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(result_dir / "progress.json", payload)


def code_key(value: Any) -> str:
    text = str(value)
    m = re.search(r"(\d{6})", text)
    return m.group(1) if m else text


def resolve_daily_data_dir(base_dir: Path) -> Path:
    direct_txt = list(base_dir.glob("*.txt"))
    if direct_txt:
        return base_dir
    normal_dir = base_dir / "normal"
    if normal_dir.exists() and list(normal_dir.glob("*.txt")):
        return normal_dir
    raise FileNotFoundError(f"未找到可用日线目录: {base_dir}")


def parse_cases(daily_data_dir: Path) -> pd.DataFrame:
    pat = re.compile(r"(.+?)(\d{8})\.png$")
    rows: list[dict[str, Any]] = []
    for path in sorted(CASE_DIR.glob("*.png")):
        if "反例" in path.name or path.stem == "案例图":
            continue
        m = pat.match(path.name)
        if not m:
            continue
        stock_name, ds = m.groups()
        signal_date = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
        if pd.isna(signal_date):
            continue
        rows.append(
            {
                "stock_name": stock_name,
                "signal_date": signal_date,
                "case_file": str(path),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["code"] = out["stock_name"].map(relaxed._build_name_code_map(str(daily_data_dir)))
    out["code_key"] = out["code"].map(code_key)
    return out.dropna(subset=["code"]).copy().reset_index(drop=True)


def _record_for_date(file_path_str: str, target_date_str: str) -> dict[str, Any] | None:
    target_date = pd.Timestamp(target_date_str)
    df = relaxed.sim.load_stock_data(file_path_str)
    if df is None or df.empty:
        return None
    x = relaxed.sim.compute_relaxed_brick_features(df).reset_index(drop=True)
    if x.empty:
        return None
    x["signal_vs_ma5_proxy"] = relaxed._compute_signal_vs_ma5_proxy(x)
    match_idx = x.index[x["date"] == target_date]
    if len(match_idx) == 0:
        return None
    signal_idx = int(match_idx[-1])
    latest = x.iloc[signal_idx]
    if signal_idx < max(relaxed.sim.SEQUENCE_LENS):
        return {
            "code": str(latest["code"]),
            "signal_date": target_date,
            "stage_reason": "insufficient_sequence_bars",
            "signal_relaxed": False,
            "pattern_a_relaxed": bool(latest.get("pattern_a_relaxed", False)),
            "pattern_b_relaxed": bool(latest.get("pattern_b_relaxed", False)),
            "prev_green_streak": float(latest.get("prev_green_streak", 0.0) or 0.0),
            "rebound_ratio": float(latest.get("rebound_ratio", 0.0) or 0.0),
        }
    if not bool(latest.get("signal_relaxed", False)):
        return {
            "code": str(latest["code"]),
            "signal_date": target_date,
            "stage_reason": "signal_relaxed_false",
            "signal_relaxed": False,
            "pattern_a_relaxed": bool(latest.get("pattern_a_relaxed", False)),
            "pattern_b_relaxed": bool(latest.get("pattern_b_relaxed", False)),
            "prev_green_streak": float(latest.get("prev_green_streak", 0.0) or 0.0),
            "rebound_ratio": float(latest.get("rebound_ratio", 0.0) or 0.0),
        }

    seq_map: dict[int, dict[str, Any]] = {}
    for seq_len in relaxed.sim.SEQUENCE_LENS:
        seq_map[seq_len] = relaxed.sim.extract_sequence(x.iloc[signal_idx - seq_len + 1 : signal_idx + 1], seq_len)

    prev_green_streak = float(latest.get("prev_green_streak", 0.0) or 0.0)
    prev_red_streak = float(latest.get("prev_red_streak", 0.0) or 0.0)
    green4_flag = prev_green_streak == 4
    red4_flag = prev_red_streak == 4
    trend_layer = "high" if bool(latest.get("trend_ok", False)) else "low"

    return {
        "code": str(latest["code"]),
        "signal_date": target_date,
        "stage_reason": "candidate_ok",
        "signal_idx": int(signal_idx),
        "entry_date": target_date + pd.Timedelta(days=1),
        "exit_date": target_date + pd.Timedelta(days=3),
        "entry_price": float(latest["close"]),
        "signal_low": float(latest["low"]),
        "signal_open": float(latest["open"]),
        "signal_close": float(latest["close"]),
        "label": 0,
        "result": "pending",
        "ret": 0.0,
        "ret1": float(latest.get("ret1", 0.0) or 0.0),
        "ret5": float(latest.get("ret5", 0.0) or 0.0),
        "ret10": float(latest.get("ret10", 0.0) or 0.0),
        "signal_ret": float(latest.get("signal_ret", 0.0) or 0.0),
        "trend_spread": float(latest.get("trend_spread", 0.0) or 0.0),
        "close_to_trend": float(latest.get("close_to_trend", 0.0) or 0.0),
        "close_to_long": float(latest.get("close_to_long", 0.0) or 0.0),
        "ma10_slope_5": float(latest.get("ma10_slope_5", 0.0) or 0.0),
        "ma20_slope_5": float(latest.get("ma20_slope_5", 0.0) or 0.0),
        "brick_red_len": float(latest.get("brick_red_len", 0.0) or 0.0),
        "brick_green_len_prev": float(x["brick_green_len"].shift(1).iloc[signal_idx] or 0.0),
        "rebound_ratio": float(latest.get("rebound_ratio", 0.0) or 0.0),
        "RSI14": float(latest.get("RSI14", 0.0) or 0.0),
        "MACD_hist": float(latest.get("MACD_hist", 0.0) or 0.0),
        "KDJ_J": float(latest.get("KDJ_J", 0.0) or 0.0),
        "body_ratio": float(latest.get("body_ratio", 0.0) or 0.0),
        "upper_shadow_pct": float(latest.get("upper_shadow_pct", 0.0) or 0.0),
        "lower_shadow_pct": float(latest.get("lower_shadow_pct", 0.0) or 0.0),
        "signal_vs_ma5_proxy": float(latest.get("signal_vs_ma5_proxy", 0.0) or 0.0),
        "prev_green_streak": prev_green_streak,
        "prev_red_streak": prev_red_streak,
        "trend_ok": bool(latest.get("trend_ok", False)),
        "green4_flag": bool(green4_flag),
        "red4_flag": bool(red4_flag),
        "green4_flag_num": float(green4_flag),
        "red4_flag_num": float(red4_flag),
        "trend_layer": trend_layer,
        "trend_layer_num": 1.0 if trend_layer == "high" else 0.0,
        "green4_low_flag": bool(green4_flag and trend_layer == "low"),
        "green4_low_flag_num": float(green4_flag and trend_layer == "low"),
        "candidate_pool": "brick.relaxed_base",
        "pool_bonus": 0.0,
        "seq_map": seq_map,
        "signal_relaxed": True,
        "pattern_a_relaxed": bool(latest.get("pattern_a_relaxed", False)),
        "pattern_b_relaxed": bool(latest.get("pattern_b_relaxed", False)),
    }


def build_candidates_for_date(target_date: pd.Timestamp, max_workers: int, daily_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(daily_data_dir.glob("*.txt"))
    raw_rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for path in files:
            item = _record_for_date(str(path), str(target_date.date()))
            if item is not None:
                raw_rows.append(item)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for item in executor.map(
                    _record_for_date,
                    [str(p) for p in files],
                    [str(target_date.date())] * len(files),
                    chunksize=16,
                ):
                    if item is not None:
                        raw_rows.append(item)
        except Exception:
            for path in files:
                item = _record_for_date(str(path), str(target_date.date()))
                if item is not None:
                    raw_rows.append(item)
    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df["code_key"] = raw_df["code"].map(code_key)
    candidate_df = raw_df[raw_df["stage_reason"] == "candidate_ok"].copy().reset_index(drop=True) if not raw_df.empty else pd.DataFrame()
    return raw_df, candidate_df


def score_candidates_for_date(target_date: pd.Timestamp, candidate_df: pd.DataFrame, daily_data_dir: Path) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    best, hist_df = relaxed._load_best_and_history()
    train_df, val_df = relaxed._build_train_val_frames(hist_df, target_date)
    if train_df.empty or val_df.empty:
        return pd.DataFrame()

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    q1, q2 = relaxed._turn_layer_thresholds(trainval_df)
    current_df = relaxed._apply_turn_strength_features(candidate_df, q1, q2)
    if "turn_strength_layer_num" not in trainval_df.columns:
        trainval_df = relaxed._apply_turn_strength_features(trainval_df, q1, q2)

    factor_model = relaxed.rolling.build_factor_model(trainval_df)
    trainval_with_factor = relaxed.rolling.apply_factor_model(trainval_df, factor_model)
    current_with_factor = relaxed.rolling.apply_factor_model(current_df, factor_model)

    stage_records = relaxed._build_stage_records(current_with_factor)
    sim_cfg = relaxed.sim.BaseConfig(
        builder=str(best["builder"]),
        seq_len=int(best["seq_len"]),
        rep=str(best["rep"]),
        scorer=str(best["scorer"]),
    )
    templates = relaxed._build_similarity_templates(trainval_df, best)
    sim_df = relaxed.sim.build_scored_df_normal(stage_records, templates, sim_cfg).rename(
        columns={"date": "signal_date", "score": "sim_score"}
    )
    current_with_factor = current_with_factor.merge(
        sim_df[["code", "signal_date", "sim_score"]], on=["code", "signal_date"], how="left"
    )
    current_with_factor["sim_score"] = pd.to_numeric(current_with_factor["sim_score"], errors="coerce").fillna(-1.0)

    perfect_templates = relaxed._build_perfect_case_templates(str(daily_data_dir), target_date, sim_cfg.seq_len, sim_cfg.rep)
    if perfect_templates:
        perfect_sim_df = relaxed.sim.build_scored_df_normal(stage_records, perfect_templates, sim_cfg).rename(
            columns={"date": "signal_date", "score": "perfect_case_sim_score"}
        )
        current_with_factor = current_with_factor.merge(
            perfect_sim_df[["code", "signal_date", "perfect_case_sim_score"]], on=["code", "signal_date"], how="left"
        )
    current_with_factor["perfect_case_sim_score"] = pd.to_numeric(
        current_with_factor.get("perfect_case_sim_score"), errors="coerce"
    ).fillna(-1.0)

    rf_model = relaxed.rolling.fit_rf_model(trainval_with_factor)
    current_prob = relaxed.rolling.predict_rf_prob(current_with_factor, rf_model)
    current_with_factor["ml_score_raw"] = current_prob
    current_with_factor["ml_score"] = relaxed.rolling.normalize_rank(current_prob)

    base_rank_score = (
        relaxed.rolling.normalize_rank(current_with_factor["sim_score"]) * float(best["sim_weight"])
        + pd.to_numeric(current_with_factor["factor_score"], errors="coerce").fillna(0.0) * float(best["factor_weight"])
        + pd.to_numeric(current_with_factor["ml_score"], errors="coerce").fillna(0.0) * float(best["ml_weight"])
        + pd.to_numeric(current_with_factor["pool_bonus"], errors="coerce").fillna(0.0)
    )
    current_with_factor["perfect_case_rank"] = relaxed.rolling.normalize_rank(current_with_factor["perfect_case_sim_score"])
    current_with_factor["rank_score"] = (
        (1.0 - relaxed.PERFECT_CASE_WEIGHT) * base_rank_score
        + relaxed.PERFECT_CASE_WEIGHT * current_with_factor["perfect_case_rank"]
    )
    current_with_factor["sim_gate_pass"] = current_with_factor["sim_score"] >= float(best["sim_gate"])

    gated = current_with_factor[current_with_factor["sim_gate_pass"]].copy()
    if gated.empty:
        current_with_factor["daily_rank"] = pd.NA
        current_with_factor["selected_top10"] = False
        return current_with_factor
    gated = gated.sort_values(
        ["signal_date", "perfect_case_rank", "rank_score", "code"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    gated["daily_rank"] = gated.index + 1
    gated["selected_top10"] = gated["daily_rank"] <= int(best["daily_topn"])
    out = current_with_factor.merge(
        gated[["code", "signal_date", "daily_rank", "selected_top10"]],
        on=["code", "signal_date"],
        how="left",
    )
    out["code_key"] = out["code"].map(code_key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_SNAPSHOT_DIR))
    args = parser.parse_args()

    result_dir = Path(args.output_dir) if args.output_dir else RESULT_ROOT / f"brick_relaxed_perfect_case_coverage_check_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        daily_data_dir = resolve_daily_data_dir(Path(args.data_dir))
        case_df = parse_cases(daily_data_dir)
        if case_df.empty:
            raise RuntimeError("未找到可用完美砖型案例")
        update_progress(result_dir, "cases_ready", total_cases=int(len(case_df)), daily_data_dir=str(daily_data_dir))

        results: list[dict[str, Any]] = []
        for idx, target_date in enumerate(sorted(case_df["signal_date"].drop_duplicates()), 1):
            update_progress(result_dir, "processing_date", done_dates=idx - 1, total_dates=int(case_df["signal_date"].nunique()), current_date=str(target_date.date()))
            day_cases = case_df[case_df["signal_date"] == target_date].copy()
            raw_df, candidate_df = build_candidates_for_date(target_date, args.max_workers, daily_data_dir)
            scored_df = score_candidates_for_date(target_date, candidate_df, daily_data_dir)

            for row in day_cases.itertuples(index=False):
                rec = {
                    "stock_name": row.stock_name,
                    "code": row.code,
                    "code_key": row.code_key,
                    "signal_date": pd.Timestamp(row.signal_date),
                    "case_file": row.case_file,
                    "excluded_period": bool(EXCLUDE_START <= pd.Timestamp(row.signal_date) <= EXCLUDE_END),
                }
                raw_hit = raw_df[raw_df["code_key"] == str(row.code_key)] if not raw_df.empty else pd.DataFrame()
                score_hit = scored_df[scored_df["code_key"] == str(row.code_key)] if not scored_df.empty else pd.DataFrame()
                if rec["excluded_period"]:
                    rec["stage"] = "excluded_period"
                    rec["reason"] = "灾难期默认剔除"
                elif raw_hit.empty:
                    rec["stage"] = "date_or_file_missing"
                    rec["reason"] = "标准化数据中无该日期或文件不可读"
                else:
                    raw_row = raw_hit.iloc[-1]
                    if raw_row["stage_reason"] != "candidate_ok":
                        rec["stage"] = "candidate_pool"
                        rec["reason"] = str(raw_row["stage_reason"])
                    elif score_hit.empty:
                        rec["stage"] = "scoring"
                        rec["reason"] = "训练/验证窗口不足或评分失败"
                    else:
                        score_row = score_hit.iloc[-1]
                        if not bool(score_row.get("sim_gate_pass", False)):
                            rec["stage"] = "sim_gate"
                            rec["reason"] = f"sim_score={float(score_row.get('sim_score', float('nan'))):.4f} < gate"
                        elif not bool(score_row.get("selected_top10", False)):
                            rec["stage"] = "topn"
                            rec["reason"] = f"daily_rank={int(score_row.get('daily_rank'))}"
                        else:
                            rec["stage"] = "selected"
                            rec["reason"] = "进入最终top10"
                        rec["sim_score"] = float(score_row.get("sim_score", float("nan")))
                        rec["perfect_case_sim_score"] = float(score_row.get("perfect_case_sim_score", float("nan")))
                        rec["factor_score"] = float(score_row.get("factor_score", float("nan")))
                        rec["ml_score"] = float(score_row.get("ml_score", float("nan")))
                        rec["rank_score"] = float(score_row.get("rank_score", float("nan")))
                        rec["daily_rank"] = score_row.get("daily_rank")
                    rec["signal_relaxed"] = bool(raw_row.get("signal_relaxed", False))
                    rec["pattern_a_relaxed"] = bool(raw_row.get("pattern_a_relaxed", False))
                    rec["pattern_b_relaxed"] = bool(raw_row.get("pattern_b_relaxed", False))
                    rec["prev_green_streak"] = float(raw_row.get("prev_green_streak", 0.0) or 0.0)
                    rec["rebound_ratio"] = float(raw_row.get("rebound_ratio", 0.0) or 0.0)
                results.append(rec)

        result_df = pd.DataFrame(results).sort_values(["signal_date", "stock_name"]).reset_index(drop=True)
        result_df.to_csv(result_dir / "perfect_case_coverage_results.csv", index=False, encoding="utf-8-sig")
        summary = {
            "total_cases": int(len(result_df)),
            "selected_count": int((result_df["stage"] == "selected").sum()),
            "topn_count": int((result_df["stage"] == "topn").sum()),
            "sim_gate_count": int((result_df["stage"] == "sim_gate").sum()),
            "candidate_pool_count": int((result_df["stage"] == "candidate_pool").sum()),
            "excluded_period_count": int((result_df["stage"] == "excluded_period").sum()),
            "date_or_file_missing_count": int((result_df["stage"] == "date_or_file_missing").sum()),
        }
        write_json(result_dir / "summary.json", summary)
        update_progress(result_dir, "finished", **summary)
        print(result_df.to_string(index=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        write_json(
            result_dir / "error.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        update_progress(result_dir, "error", error_type=type(exc).__name__, message=str(exc))
        raise


if __name__ == "__main__":
    main()
