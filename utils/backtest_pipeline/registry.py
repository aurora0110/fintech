from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class RegistryItem:
    name: str
    family: str
    builder: Callable[..., Any]
    description: str


class ModuleRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, RegistryItem] = {}

    def register(self, name: str, family: str, builder: Callable[..., Any], description: str) -> None:
        if name in self._items:
            raise ValueError(f"模块已存在: {name}")
        self._items[name] = RegistryItem(name=name, family=family, builder=builder, description=description)

    def get(self, name: str) -> RegistryItem:
        if name not in self._items:
            raise KeyError(f"未注册模块: {name}")
        return self._items[name]

    def by_family(self, family: str) -> Dict[str, RegistryItem]:
        return {k: v for k, v in self._items.items() if v.family == family}

    def names(self) -> list[str]:
        return sorted(self._items.keys())


DATA_INPUT_REGISTRY = ModuleRegistry()
CANDIDATE_POOL_REGISTRY = ModuleRegistry()
CONFIRMER_REGISTRY = ModuleRegistry()
RANKER_REGISTRY = ModuleRegistry()
EXIT_REGISTRY = ModuleRegistry()
ACCOUNT_POLICY_REGISTRY = ModuleRegistry()
VALIDATOR_REGISTRY = ModuleRegistry()
