from __future__ import annotations

from typing import Dict, List
import pandas as pd

from config import RunConfig, save_json
from contribution import marginal_contribution, shapley_contribution, standalone_factor_performance
from data_loader import load_and_prepare_data
from factor_search import search_single_factor_threshold, search_threshold_params_walkforward
from normalization import normalize_factors
from stability import parameter_sensitivity, regime_stability, rolling_walkforward_eval
from weight_optimizer import bayes_like_weight_search, grid_weight_search, random_weight_search
from backtester import backtest_topk
from metrics import objective_value


def _default_weights(factors: List[str]) -> Dict[str, float]:
    if not factors:
        return {}
    w = 1.0 / len(factors)
    return {f: w for f in factors}


def _drop_high_corr_factors(df: pd.DataFrame, factor_names: List[str], threshold: float) -> List[str]:
    score_cols = [f"{x}_score" for x in factor_names if f"{x}_score" in df.columns]
    if len(score_cols) <= 1:
        return factor_names
    corr = df[score_cols].corr().abs()
    to_drop = set()
    for i, c1 in enumerate(score_cols):
        for j in range(i + 1, len(score_cols)):
            c2 = score_cols[j]
            if corr.loc[c1, c2] > threshold:
                # Drop later column.
                to_drop.add(c2.replace("_score", ""))
    return [f for f in factor_names if f not in to_drop]


def run_full_experiment(cfg: RunConfig) -> Dict:
    df = load_and_prepare_data(cfg)
    df = normalize_factors(df, cfg, cfg.factors)
    all_factors = [f.name for f in cfg.factors if f"{f.name}_score" in df.columns]
    if not all_factors:
        raise ValueError("No factor score columns available. Check factor names and input data.")

    # 1) Single-factor threshold search (J/RSI-like).
    single_factor_tables = {}
    for f in cfg.factors:
        if f.grid and f.kind in {"continuous", "threshold"}:
            tab = search_single_factor_threshold(df, cfg, f, f.grid)
            if not tab.empty:
                single_factor_tables[f.name] = tab
                tab.to_csv(f"{cfg.output_dir}/single_factor_{f.name}.csv", index=False)

    # 2) Correlation filter.
    selected_factors = _drop_high_corr_factors(df, all_factors, cfg.corr_drop_threshold)
    base_weights = _default_weights(selected_factors)

    # 3) Threshold params walk-forward search.
    best_params, threshold_table = search_threshold_params_walkforward(df, cfg, base_weights)
    threshold_table.to_csv(f"{cfg.output_dir}/threshold_search.csv", index=False)

    # 4) Weight optimization.
    if len(selected_factors) <= 5:
        w_grid, table_grid = grid_weight_search(
            df, cfg, selected_factors, best_params, step=0.1
        )
        table_grid.head(200).to_csv(f"{cfg.output_dir}/weight_grid_top200.csv", index=False)
    else:
        w_grid = base_weights

    w_rand, table_rand = random_weight_search(
        df,
        cfg,
        selected_factors,
        best_params,
        n_samples=cfg.random_weight_samples,
        seed=cfg.random_seed,
    )
    table_rand.head(500).to_csv(f"{cfg.output_dir}/weight_random_top500.csv", index=False)

    w_bayes, table_bayes = bayes_like_weight_search(
        df, cfg, selected_factors, best_params, init_samples=500, local_rounds=300, seed=cfg.random_seed
    )
    table_bayes.head(500).to_csv(f"{cfg.output_dir}/weight_bayes_top500.csv", index=False)

    # choose best among available methods
    candidates = [("random", w_rand), ("bayes_like", w_bayes)]
    if len(selected_factors) <= 5:
        candidates.append(("grid", w_grid))
    best_method, best_weights = max(
        candidates,
        key=lambda x: objective_value(
            backtest_topk(df, cfg, x[1], best_params).metrics,
            cfg.objective_name,
            cfg.objective_weights,
        ),
    )

    # 5) Contribution analysis.
    marginal_df = marginal_contribution(df, cfg, best_weights, best_params)
    marginal_df.to_csv(f"{cfg.output_dir}/factor_marginal_contribution.csv", index=False)
    standalone_df = standalone_factor_performance(df, cfg, list(best_weights.keys()), best_params)
    standalone_df.to_csv(f"{cfg.output_dir}/factor_standalone_performance.csv", index=False)
    shapley_df = pd.DataFrame()
    if len(best_weights) <= 8:
        shapley_df = shapley_contribution(df, cfg, list(best_weights.keys()), best_params)
        shapley_df.to_csv(f"{cfg.output_dir}/factor_shapley.csv", index=False)

    # 6) Stability validation.
    wf_df = rolling_walkforward_eval(df, cfg, best_weights, best_params)
    wf_df.to_csv(f"{cfg.output_dir}/walkforward_eval.csv", index=False)
    regime_df = regime_stability(df, cfg, best_weights, best_params)
    regime_df.to_csv(f"{cfg.output_dir}/regime_stability.csv", index=False)
    sens_df = parameter_sensitivity(df, cfg, best_weights, best_params)
    sens_df.to_csv(f"{cfg.output_dir}/parameter_sensitivity.csv", index=False)

    # 7) Final full-period result.
    final_bt = backtest_topk(df, cfg, best_weights, best_params)
    summary = {
        "best_threshold_params": best_params,
        "best_weights": best_weights,
        "best_weight_method": best_method,
        "selected_factors": selected_factors,
        "final_metrics": final_bt.metrics,
        "trade_stats": final_bt.trade_stats,
        "top_k_pct": cfg.top_k_pct,
    }
    save_json(f"{cfg.output_dir}/final_summary.json", summary)

    result = {
        "summary": summary,
        "single_factor_tables": list(single_factor_tables.keys()),
        "files": {
            "summary": f"{cfg.output_dir}/final_summary.json",
            "threshold_search": f"{cfg.output_dir}/threshold_search.csv",
            "marginal": f"{cfg.output_dir}/factor_marginal_contribution.csv",
            "standalone": f"{cfg.output_dir}/factor_standalone_performance.csv",
            "shapley": f"{cfg.output_dir}/factor_shapley.csv" if not shapley_df.empty else "",
            "walkforward": f"{cfg.output_dir}/walkforward_eval.csv",
            "regime": f"{cfg.output_dir}/regime_stability.csv",
            "sensitivity": f"{cfg.output_dir}/parameter_sensitivity.csv",
        },
    }
    return result

