from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar, cast, get_type_hints

import yaml

from kgllm_pharm.tariff.schedule import ExperimentConfig

T = TypeVar("T")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def resolve_extends(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    raw = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {target}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (target.parent / str(parent)).resolve()
    return _deep_merge(resolve_extends(parent_path), raw)


def _build(cls: type[T], data: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(cast(Any, cls))}
    for name, value in data.items():
        if name not in known:
            raise KeyError(f"unknown configuration key '{name}' for {cls.__name__}")
        field_type = hints[name]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(cast(type[Any], field_type), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def build_experiment(data: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data)


def load_experiment(path: str | Path) -> ExperimentConfig:
    return build_experiment(resolve_extends(path))
