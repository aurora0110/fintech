from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("QSTRATEGY_DATA_DIR", str(ROOT / "data")))
RESULTS_DIR = Path(os.environ.get("QSTRATEGY_RESULTS_DIR", str(ROOT / "results")))
CONFIG_DIR = Path(os.environ.get("QSTRATEGY_CONFIG_DIR", str(ROOT / "config")))
UTILS_DIR = ROOT / "utils"


def root_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def results_path(*parts: str) -> Path:
    return RESULTS_DIR.joinpath(*parts)


def config_path(*parts: str) -> Path:
    return CONFIG_DIR.joinpath(*parts)
