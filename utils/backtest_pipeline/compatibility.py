from __future__ import annotations

from typing import Any


def _tp_variants(values: list[float]) -> list[dict[str, Any]]:
    return [{"name": "exit.fixed_tp", "params": {"take_profit_pct": value}} for value in values]


def _model_only_variants(models: list[str]) -> list[dict[str, Any]]:
    return [{"name": "exit.model_only", "params": {"score_model": model}} for model in models]


def _model_plus_tp_variants(models: list[str], values: list[float]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for model in models:
        for value in values:
            variants.append(
                {
                    "name": "exit.model_plus_tp",
                    "params": {
                        "score_model": model,
                        "take_profit_pct": value,
                    },
                }
            )
    return variants


def _partial_tp_variants(pairs: list[tuple[float, float]]) -> list[dict[str, Any]]:
    return [
        {
            "name": "exit.partial_tp",
            "params": {
                "partial_take_profit_pct": first,
                "final_take_profit_pct": second,
            },
        }
        for first, second in pairs
    ]


FAMILY_MATRIX_SPECS: dict[str, dict[str, Any]] = {
    "b1": {
        "candidate_pools": {
            "b1.low_cross": {
                "confirmers": [None, "b1.semantic_bonus"],
                "rankers": [
                    {"name": "ranker.factor_discovery", "top_n_values": [3, 5, 8], "params": {"score_col": "factor_discovery"}},
                    {"name": "ranker.similarity", "top_n_values": [3, 5, 8], "params": {"score_col": "sim_corr_close_vol_concat"}},
                    {"name": "ranker.xgboost", "top_n_values": [3, 5], "params": {"score_col": "xgboost"}},
                    {"name": "ranker.lightgbm", "top_n_values": [3, 5], "params": {"score_col": "lightgbm"}},
                    {"name": "ranker.naive_bayes", "top_n_values": [3, 5], "params": {"score_col": "naive_bayes"}},
                    {
                        "name": "ranker.fusion",
                        "top_n_values": [3, 5],
                        "params": {"components": [["factor_discovery", 0.5], ["xgboost", 0.5]]},
                    },
                ],
            },
            "b1.txt_confirmed": {
                "confirmers": [None, "b1.semantic_bonus"],
                "rankers": [
                    {"name": "ranker.similarity", "top_n_values": [3, 5, 8], "params": {"score_col": "template_similarity_score"}},
                    {"name": "ranker.factor_discovery", "top_n_values": [3, 5, 8], "params": {"score_col": "factor_discovery"}},
                    {"name": "ranker.xgboost", "top_n_values": [3, 5], "params": {"score_col": "xgboost"}},
                    {"name": "ranker.lightgbm", "top_n_values": [3, 5], "params": {"score_col": "lightgbm"}},
                    {"name": "ranker.naive_bayes", "top_n_values": [3, 5], "params": {"score_col": "naive_bayes"}},
                    {
                        "name": "ranker.fusion",
                        "top_n_values": [3, 5],
                        "params": {"components": [["template_full_fusion", 0.6], ["xgboost", 0.4]]},
                    },
                ],
            },
        },
        "exits": (
            _tp_variants([0.1, 0.2, 0.3, 0.5])
            + _model_only_variants(["xgb_score_v2", "lgb_score_v2", "rf_score_v2", "et_score_v2"])
            + _model_plus_tp_variants(["xgb_score_v2", "lgb_score_v2"], [0.2, 0.3])
            + _partial_tp_variants([(0.1, 0.2), (0.2, 0.3)])
        ),
    },
    "b2": {
        "candidate_pools": {
            "b2.main": {
                "confirmers": [None, "b2.startup_quality"],
                "rankers": [{"name": "ranker.factor_discovery", "top_n_values": [3, 5, 8], "params": {"score_col": "base_score"}}],
            }
        },
        "exits": _tp_variants([0.1, 0.2, 0.3]) + _model_plus_tp_variants(["xgb_score_v2"], [0.2, 0.3]),
    },
    "b3": {
        "candidate_pools": {
            "b3.follow_through": {
                "confirmers": [None, "b3.follow_through_quality"],
                "rankers": [{"name": "ranker.factor_discovery", "top_n_values": [3, 5], "params": {"score_col": "base_score"}}],
            }
        },
        "exits": _tp_variants([0.1, 0.2, 0.3]) + _model_plus_tp_variants(["xgb_score_v2"], [0.2]),
    },
    "pin": {
        "candidate_pools": {
            "pin.trend_wash": {
                "confirmers": [None, "pin.needle_quality"],
                "rankers": [{"name": "ranker.factor_discovery", "top_n_values": [3, 5], "params": {"score_col": "base_score"}}],
            }
        },
        "exits": _tp_variants([0.1, 0.2, 0.3]) + _model_plus_tp_variants(["xgb_score_v2"], [0.2]),
    },
    "brick": {
        "candidate_pools": {
            "brick.main": {
                "confirmers": [None, "brick.turn_quality"],
                "rankers": [{"name": "ranker.factor_discovery", "top_n_values": [3, 5, 8], "params": {"score_col": "base_score"}}],
            },
            "brick.formal_best": {
                "confirmers": [None],
                "rankers": [{"name": "ranker.factor_discovery", "top_n_values": [10], "params": {"score_col": "base_score"}}],
            },
            "brick.relaxed_base": {
                "confirmers": [None],
                "rankers": [
                    {"name": "ranker.brick_similarity_champion", "top_n_values": [8, 10, 12], "params": {"score_col": "sim_score"}},
                    {"name": "ranker.brick_factor_score", "top_n_values": [8, 10, 12], "params": {"score_col": "factor_score"}},
                    {"name": "ranker.brick_similarity_plus_factor", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_similarity_plus_ml", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_full_fusion", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                ],
            },
            "brick.green4_enhance": {
                "confirmers": [None],
                "rankers": [
                    {"name": "ranker.brick_similarity_champion", "top_n_values": [8, 10, 12], "params": {"score_col": "sim_score"}},
                    {"name": "ranker.brick_similarity_plus_factor", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_similarity_plus_ml", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_full_fusion", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                ],
            },
            "brick.green4_low_enhance": {
                "confirmers": [None],
                "rankers": [
                    {"name": "ranker.brick_similarity_champion", "top_n_values": [8, 10, 12], "params": {"score_col": "sim_score"}},
                    {"name": "ranker.brick_similarity_plus_factor", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_similarity_plus_ml", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_full_fusion", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                ],
            },
            "brick.red4_filter": {
                "confirmers": [None],
                "rankers": [
                    {"name": "ranker.brick_similarity_champion", "top_n_values": [8, 10, 12], "params": {"score_col": "sim_score"}},
                    {"name": "ranker.brick_similarity_plus_factor", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_similarity_plus_ml", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_full_fusion", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                ],
            },
            "brick.green4_low_hardfilter": {
                "confirmers": [None],
                "rankers": [
                    {"name": "ranker.brick_similarity_champion", "top_n_values": [8, 10, 12], "params": {"score_col": "sim_score"}},
                    {"name": "ranker.brick_similarity_plus_factor", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_similarity_plus_ml", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                    {"name": "ranker.brick_full_fusion", "top_n_values": [8, 10, 12], "params": {"score_col": "rank_score"}},
                ],
            }
        },
        "exits": _tp_variants([0.03, 0.1, 0.2, 0.3])
        + _model_plus_tp_variants(["xgb_score_v2"], [0.2, 0.3])
        + [{"name": "exit.brick_half_tp_then_green", "params": {"first_take_profit_pct": 0.035}}]
        + [{"name": "exit.fixed_tp_grid", "params": {}}]
        + [{"name": "exit.partial_tp_grid", "params": {}}],
    },
}


def supported_families() -> list[str]:
    return sorted(FAMILY_MATRIX_SPECS.keys())


def family_spec(family: str) -> dict[str, Any]:
    if family not in FAMILY_MATRIX_SPECS:
        raise KeyError(f"未定义实验矩阵规范: {family}")
    return FAMILY_MATRIX_SPECS[family]
