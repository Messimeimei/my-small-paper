"""Registry for evaluation inference methods."""

from __future__ import annotations

from evaluation.inference.base import InferenceMethod

_METHODS: dict[str, InferenceMethod] = {}


def register_inference_method(
    method: InferenceMethod,
    *,
    replace: bool = False,
) -> None:
    if not method.name:
        raise ValueError("An evaluation method requires a non-empty name.")
    if method.name in _METHODS and not replace:
        raise ValueError(f"Evaluation method already registered: {method.name}")
    _METHODS[method.name] = method


def available_inference_modes() -> tuple[str, ...]:
    return tuple(_METHODS)


def get_inference_method(name: str) -> InferenceMethod:
    try:
        return _METHODS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown inference mode {name!r}; "
            f"expected one of {list(available_inference_modes())}"
        ) from exc


def _register_builtin_methods() -> None:
    from evaluation.inference.greedy import GreedyMethod
    from evaluation.inference.rail import CotRailMethod, RailMethod

    for method in (GreedyMethod(), RailMethod(), CotRailMethod()):
        register_inference_method(method)


_register_builtin_methods()
del _register_builtin_methods
