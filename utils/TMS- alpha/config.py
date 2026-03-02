from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass
class FactorConfig:
    name: str
    category: str  # momentum / structure / volume / risk
    kind: str  # continuous / threshold
    direction: str = "higher_better"  # higher_better / lower_better
    comparator: str = ">="  # >= / <= / between
    default_weight: float = 0.0
    grid: List[float] = field(default_factory=list)


@dataclass
class RunConfig:
    data_path: str
    output_dir: str = "utils/TMS- alpha/output"
    date_col: str = "date"
    code_col: str = "code"
    close_col: str = "close"
    next_return_col: str = "next_return"
    board_col: str = "board"
    name_col: str = "name"
    st_flag_col: str = "is_st"
    exclude_boards: List[str] = field(default_factory=lambda: ["科创", "STAR"])
    top_k_pct: float = 0.1
    min_holdings: int = 10
    cost_rate: float = 0.001  # all-in commission + tax + transfer
    slippage_rate: float = 0.0005
    risk_free_rate: float = 0.0
    normalize_method: str = "rank"  # rank / zscore
    objective_name: str = "composite"  # composite / return_dd
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {"sharpe": 1.0, "calmar": 0.5, "max_drawdown": -0.3}
    )
    walk_forward: Dict[str, int] = field(
        default_factory=lambda: {"train_years": 3, "test_years": 1, "step_years": 1}
    )
    corr_drop_threshold: float = 0.7
    random_weight_samples: int = 5000
    random_seed: int = 42
    factors: List[FactorConfig] = field(default_factory=list)
    # Strategy-level threshold parameters (searched independently).
    threshold_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Stability perturbation ranges around best params.
    sensitivity_ranges: Dict[str, float] = field(default_factory=dict)


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    factors = [FactorConfig(**x) for x in raw.get("factors", [])]
    raw["factors"] = factors
    cfg = RunConfig(**raw)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

