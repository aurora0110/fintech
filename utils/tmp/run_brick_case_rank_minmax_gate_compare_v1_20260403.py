from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
RESULT_ROOT = ROOT / "results"
TOP_N = 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import brickfilter_case_rank_lgbm_top20 as case_rank_filter

# Focus on the momentum / position / shape features discussed in the 600487 diagnosis.
GATE_FEATURES = [
    "KDJ_J",
    "RSI14",
    "MACD_hist",
    "signal_vs_ma5_proxy",
    "upper_shadow_pct",
    "body_ratio",
    "close_to_trend",
    "signal_ret",
    "rebound_ratio",
    "close_to_long",
]


def update_progress(result_dir: Path, stage: str, **extra) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    (result_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_dataset() -> tuple[pd.DataFrame, dict]:
    bundle = case_rank_filter._load_runtime_bundle()
    model_result_dir = Path(bundle["model_search_result_dir"])
    df = pd.read_csv(model_result_dir / "candidate_dataset.csv")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = case_rank_filter.rank_model._prepare_features(df)
    df["model_score"] = case_rank_filter.phase0._score_with_model(bundle["model"], df)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df, bundle


def build_gate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["label"] == 1].copy()
    rows = []
    for feature in GATE_FEATURES:
        series = pd.to_numeric(pos[feature], errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "min_value": float(series.min()) if not series.empty else None,
                "max_value": float(series.max()) if not series.empty else None,
                "median_value": float(series.median()) if not series.empty else None,
                "count": int(series.size),
            }
        )
    return pd.DataFrame(rows)


def apply_minmax_gate(df: pd.DataFrame, gate_ranges: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for row in gate_ranges.itertuples(index=False):
        feature = str(row.feature)
        values = pd.to_numeric(df[feature], errors="coerce")
        mask &= values.between(float(row.min_value), float(row.max_value), inclusive="both")
    return mask


def select_daily_topn(df: pd.DataFrame, topn: int = TOP_N) -> pd.DataFrame:
    return (
        df.sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True])
        .groupby("signal_date", group_keys=False)
        .head(topn)
        .reset_index(drop=True)
    )


def summarize_variant(name: str, selected: pd.DataFrame, full_df: pd.DataFrame) -> dict:
    total_positive = int(full_df["label"].sum())
    selected_positive = int(selected["label"].sum())
    selected_count = int(len(selected))
    date_hits = (
        selected.groupby("signal_date")["label"].max().fillna(0).astype(int)
        if not selected.empty
        else pd.Series(dtype=int)
    )
    full_date_hits = full_df.groupby("signal_date")["label"].max().fillna(0).astype(int)
    hit_dates = int(date_hits.sum()) if not date_hits.empty else 0
    total_positive_dates = int(full_date_hits.sum())
    return {
        "variant": name,
        "selected_count": selected_count,
        "selected_positive": selected_positive,
        "precision": float(selected_positive / selected_count) if selected_count else 0.0,
        "case_recall": float(selected_positive / total_positive) if total_positive else 0.0,
        "hit_dates": hit_dates,
        "date_recall": float(hit_dates / total_positive_dates) if total_positive_dates else 0.0,
        "avg_daily_selected": float(selected_count / max(1, full_df["signal_date"].nunique())),
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULT_ROOT / f"brick_case_rank_minmax_gate_compare_v1_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_progress(result_dir, "loading")
        df, bundle = load_dataset()
        threshold = float(bundle["threshold"])
        update_progress(result_dir, "building_gate_ranges", rows=len(df), threshold=threshold)

        gate_ranges = build_gate_ranges(df)
        gate_mask = apply_minmax_gate(df, gate_ranges)

        original_pool = df[pd.to_numeric(df["model_score"], errors="coerce").fillna(-1.0) >= threshold].copy()
        gated_pool = df[gate_mask].copy()
        gated_plus_threshold_pool = gated_pool[pd.to_numeric(gated_pool["model_score"], errors="coerce").fillna(-1.0) >= threshold].copy()

        original_selected = select_daily_topn(original_pool)
        gated_selected = select_daily_topn(gated_pool)
        gated_plus_threshold_selected = select_daily_topn(gated_plus_threshold_pool)

        summary_rows = [
            summarize_variant("original_threshold_top20", original_selected, df),
            summarize_variant("minmax_only_top20", gated_selected, df),
            summarize_variant("minmax_plus_threshold_top20", gated_plus_threshold_selected, df),
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values(["case_recall", "precision"], ascending=[False, False])

        gate_ranges.to_csv(result_dir / "gate_ranges.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(result_dir / "summary_table.csv", index=False, encoding="utf-8-sig")
        original_selected.to_csv(result_dir / "original_selected.csv", index=False, encoding="utf-8-sig")
        gated_selected.to_csv(result_dir / "minmax_only_selected.csv", index=False, encoding="utf-8-sig")
        gated_plus_threshold_selected.to_csv(result_dir / "minmax_plus_threshold_selected.csv", index=False, encoding="utf-8-sig")

        top = summary_rows[0]
        payload = {
            "model_result_dir": bundle["model_search_result_dir"],
            "phase0_result_dir": bundle["phase0_result_dir"],
            "threshold": threshold,
            "gate_features": GATE_FEATURES,
            "top_n": TOP_N,
            "dataset_rows": int(len(df)),
            "positive_rows": int(df["label"].sum()),
            "date_count": int(df["signal_date"].nunique()),
            "variants": summary_rows,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (result_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        update_progress(result_dir, "finished", output_dir=str(result_dir))
        print(result_dir)
        print(summary_df.to_string(index=False))
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
