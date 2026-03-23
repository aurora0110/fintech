from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExitModuleConfig:
    name: str
    params: Dict[str, object]


class BaseExitModule:
    strategy_family: str = "generic"
    name: str = "base"
