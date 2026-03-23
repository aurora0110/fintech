from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
V6_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_full_factor_experiment_v6_20260320.py"
V4_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v4_20260320.py"
V5_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_similarity_ml_experiment_v5_20260320.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v6_mod = load_module(V6_SCRIPT, "b1_cluster_v6")
v4_mod = load_module(V4_SCRIPT, "b1_cluster_v4")
v5_mod = load_module(V5_SCRIPT, "b1_cluster_v5")

HAS_SKLEARN = bool(getattr(v6_mod, "HAS_SKLEARN", False))
if HAS_SKLEARN:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 分簇子类型实验")
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--topn-list", type=str, default="3,5,8,10")
    parser.add_argument("--result-dir", type=Path, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def update_progress(result_dir: Path, stage: str, **kwargs: Any) -> None:
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds")}
    payload.update(kwargs)
    write_json(result_dir / "progress.json", payload)


def assign_clusters(pos_df: pd.DataFrame, cand_df: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not HAS_SKLEARN:
        pos_df = pos_df.copy()
        cand_df = cand_df.copy()
        pos_df["subtype_cluster"] = 0
        cand_df["subtype_cluster"] = 0
        return pos_df, cand_df, 1

    feature_cols = v6_mod.available_feature_cols(pos_df, cand_df)
    pos_arr = v6_mod.sanitize_numeric_matrix(pos_df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    cand_arr = v6_mod.sanitize_numeric_matrix(cand_df[feature_cols].fillna(0.0).to_numpy(dtype=float))

    scaler = StandardScaler()
    pos_scaled = v6_mod.sanitize_numeric_matrix(scaler.fit_transform(pos_arr))
    cand_scaled = v6_mod.sanitize_numeric_matrix(scaler.transform(cand_arr))

    k_candidates = [k for k in [2, 3, 4, 5, 6] if k < len(pos_df)]
    best_k = k_candidates[0]
    best_model = None
    best_inertia = None
    for k in k_candidates:
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        model.fit(pos_scaled)
        if best_inertia is None or model.inertia_ < best_inertia:
            best_inertia = float(model.inertia_)
            best_model = model
            best_k = k
    assert best_model is not None

    pos_df = pos_df.copy()
    cand_df = cand_df.copy()
    pos_df["subtype_cluster"] = best_model.labels_
    cand_df["subtype_cluster"] = best_model.predict(cand_scaled)

    cluster_df = (
        pos_df.groupby("subtype_cluster")
        .size()
        .rename("positive_count")
        .reset_index()
        .sort_values("subtype_cluster")
        .reset_index(drop=True)
    )
    cluster_df["best_k"] = best_k
    cluster_df.to_csv(result_dir / "subtype_cluster_summary.csv", index=False, encoding="utf-8-sig")
    return pos_df, cand_df, best_k


def add_subtype_scores(pos_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    cand_df = cand_df.copy()
    cand_df["subtype_corr_score"] = 0.0
    cand_df["subtype_cosine_score"] = 0.0
    cand_df["subtype_cluster_fusion_score"] = 0.0

    for cluster_id, cluster_pos in pos_df.groupby("subtype_cluster"):
        mask = cand_df["subtype_cluster"] == cluster_id
        if not mask.any():
            continue
        seq_maps = [v4_mod.derive_rep_map(r["seq_map"]) for _, r in cluster_pos.iterrows()]
        cand_maps = [v4_mod.derive_rep_map(r["seq_map"]) for _, r in cand_df.loc[mask].iterrows()]
        cand_close_vol = [m["close_vol_concat"] for m in cand_maps]
        pos_close_vol = [m["close_vol_concat"] for m in seq_maps]
        cand_close = [m["close_norm"] for m in cand_maps]
        pos_close = [m["close_norm"] for m in seq_maps]
        corr = v4_mod.compute_similarity_column(cand_close_vol, pos_close_vol, "corr")
        cosine = v4_mod.compute_similarity_column(cand_close, pos_close, "cosine")
        cand_df.loc[mask, "subtype_corr_score"] = corr
        cand_df.loc[mask, "subtype_cosine_score"] = cosine

    cand_df["rank_subtype_corr_score"] = cand_df.groupby("signal_date")["subtype_corr_score"].rank(pct=True, ascending=True)
    cand_df["rank_subtype_cosine_score"] = cand_df.groupby("signal_date")["subtype_cosine_score"].rank(pct=True, ascending=True)
    cand_df["subtype_cluster_fusion_score"] = (
        0.6 * cand_df["rank_subtype_corr_score"].fillna(0.0)
        + 0.4 * cand_df["rank_subtype_cosine_score"].fillna(0.0)
    )
    return cand_df


def build_leaderboard(cand_df: pd.DataFrame, topn_list: List[int], result_dir: Path) -> pd.DataFrame:
    pool_cols = [c for c in cand_df.columns if c.startswith("pool_")]
    score_specs = [
        ("subtype", "corr", "subtype_corr_score"),
        ("subtype", "cosine", "subtype_cosine_score"),
        ("subtype", "fusion", "subtype_cluster_fusion_score"),
    ]
    rows: List[Dict[str, Any]] = []
    for pool_name in pool_cols:
        base = cand_df[cand_df[pool_name].fillna(False)].copy()
        if base.empty:
            continue
        for family, variant, score_col in score_specs:
            tmp = base.sort_values(["signal_date", score_col], ascending=[True, False]).copy()
            for topn in topn_list:
                picked = tmp.groupby("signal_date").head(topn).copy()
                if picked.empty:
                    continue
                for split_name, g in picked.groupby("split"):
                    rows.append(
                        {
                            "family": family,
                            "variant": variant,
                            "pool": pool_name,
                            "topn": int(topn),
                            "split": split_name,
                            "sample_count": int(len(g)),
                            "date_count": int(g["signal_date"].nunique()),
                            "ret_3d_mean": float(g["ret_3d"].mean()),
                            "up_3d_rate": float((g["ret_3d"] > 0).mean()),
                            "ret_5d_mean": float(g["ret_5d"].mean()),
                            "up_5d_rate": float((g["ret_5d"] > 0).mean()),
                            "ret_10d_mean": float(g["ret_10d"].mean()),
                            "up_10d_rate": float((g["ret_10d"] > 0).mean()),
                            "ret_20d_mean": float(g["ret_20d"].mean()),
                            "up_20d_rate": float((g["ret_20d"] > 0).mean()),
                            "ret_30d_mean": float(g["ret_30d"] .mean()),
                            "up_30d_rate": float((g["ret_30d"] > 0).mean()),
                        }
                    )
    out = pd.DataFrame(rows).sort_values(["split", "ret_20d_mean", "up_20d_rate"], ascending=[True, False, False])
    out.to_csv(result_dir / "subtype_signal_layer_leaderboard.csv", index=False, encoding="utf-8-sig")
    final_report = out[out["split"] == "final_test"].copy()
    final_report.to_csv(result_dir / "subtype_final_test_report.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.result_dir or (ROOT / "results" / f"b1_cluster_subtype_signal_v1_{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    topn_list = [int(x) for x in args.topn_list.split(",") if x.strip()]
    update_progress(result_dir, "starting", file_limit=args.file_limit, topn_list=topn_list)

    cand_df = v6_mod.build_candidate_df(result_dir, file_limit=args.file_limit)
    pos_df = v6_mod.build_positive_df(result_dir)
    update_progress(result_dir, "candidate_ready", candidate_count=len(cand_df), positive_count=len(pos_df))

    pos_df = pos_df[pos_df["status"] == "ok"].copy().reset_index(drop=True)
    cutoffs = v5_mod.split_three_way_by_positive_dates(pos_df["signal_date"].tolist())
    cand_df["split"] = v5_mod.assign_split(cand_df, cutoffs)
    pos_df["split"] = v5_mod.assign_split(pos_df, cutoffs)
    write_json(result_dir / "split_cutoffs.json", {k: str(v.date()) for k, v in cutoffs.items()})
    pos_df, cand_df, best_k = assign_clusters(pos_df, cand_df, result_dir)
    update_progress(result_dir, "cluster_ready", best_k=best_k)

    cand_df = v6_mod.add_pool_flags(cand_df)
    cand_df = add_subtype_scores(pos_df, cand_df)
    cand_df.to_csv(result_dir / "subtype_candidate_enriched.csv", index=False, encoding="utf-8-sig")

    leader = build_leaderboard(cand_df, topn_list, result_dir)
    validation_best = (
        leader[leader["split"] == "validation"]
        .sort_values(["family", "ret_20d_mean", "up_20d_rate", "sample_count"], ascending=[True, False, False, False])
        .drop_duplicates(subset=["family"], keep="first")
        .reset_index(drop=True)
    )
    validation_best.to_csv(result_dir / "validation_family_best.csv", index=False, encoding="utf-8-sig")

    final_df = cand_df[cand_df["split"] == "final_test"].copy()
    selected_rows: List[pd.DataFrame] = []
    for _, best in validation_best.iterrows():
        family = str(best["family"])
        variant = str(best["variant"])
        pool_name = str(best["pool"])
        topn = int(best["topn"])
        part = final_df[final_df[pool_name].fillna(False)].copy()
        if part.empty:
            continue
        score_col = {
            "corr": "subtype_corr_score",
            "cosine": "subtype_cosine_score",
            "fusion": "subtype_cluster_fusion_score",
        }[variant]
        selected = part.sort_values(["signal_date", score_col], ascending=[True, False]).groupby("signal_date").head(topn).copy()
        if selected.empty:
            continue
        selected["strategy_tag"] = f"{family}_{variant}_{pool_name}_top{topn}"
        selected_rows.append(selected)
    if selected_rows:
        pd.concat(selected_rows, ignore_index=True).to_csv(
            result_dir / "final_test_selected_rows.csv", index=False, encoding="utf-8-sig"
        )

    update_progress(result_dir, "finished", leaderboard_rows=len(leader), candidate_count=len(cand_df), positive_count=len(pos_df), best_k=best_k)
    write_json(
        result_dir / "summary.json",
        {
            "candidate_count": int(len(cand_df)),
            "positive_count": int(len(pos_df)),
            "best_k": int(best_k),
            "leaderboard_rows": int(len(leader)),
            "validation_best_rows": int(len(validation_best)),
            "final_selected_strategy_count": int(len(selected_rows)),
        },
    )


if __name__ == "__main__":
    main()
