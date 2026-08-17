"""Lazy registry for pluggable evaluation model backends."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from models.base import ModelBackend


@dataclass(frozen=True)
class BackendSpec:
    name: str
    target: str

    def load(self) -> ModelBackend:
        module_name, class_name = self.target.rsplit(":", 1)
        backend_class = getattr(importlib.import_module(module_name), class_name)
        backend = backend_class()
        if backend.name != self.name:
            raise RuntimeError(
                f"Backend {self.target} declares name={backend.name!r}, "
                f"expected {self.name!r}."
            )
        return backend


_BACKENDS: dict[str, BackendSpec] = {}


def register_model_backend(spec: BackendSpec, *, replace: bool = False) -> None:
    if not spec.name:
        raise ValueError("A model backend requires a non-empty name.")
    if spec.name in _BACKENDS and not replace:
        raise ValueError(f"Model backend already registered: {spec.name}")
    _BACKENDS[spec.name] = spec


def available_model_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_model_backend(name: str) -> ModelBackend:
    try:
        spec = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model backend {name!r}; "
            f"expected one of {list(available_model_backends())}"
        ) from exc
    return spec.load()


register_model_backend(BackendSpec("vllm", "models.vllm:VllmBackend"))
