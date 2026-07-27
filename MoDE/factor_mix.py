"""Pure adapter validation and factor-mixing helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file


COMPATIBILITY_FIELDS = (
    "base_model_name_or_path",
    "bias",
    "fan_in_fan_out",
    "lora_alpha",
    "r",
    "task_type",
    "use_dora",
    "use_rslora",
)


def adapter_weight_file(adapter_dir: Path) -> Path:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / filename
        if candidate.is_file():
            return candidate
    raise ValueError(f"No adapter weights found under {adapter_dir}")


def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise ValueError(f"Missing adapter config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config must be a JSON object: {path}")
    return payload


def load_adapter_state(
    adapter_dir: Path, *, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    weight_path = adapter_weight_file(adapter_dir)
    if weight_path.suffix == ".safetensors":
        state = load_file(str(weight_path), device=str(device))
    else:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Adapter state is empty or invalid: {weight_path}")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError(f"Adapter state contains non-tensor entries: {weight_path}")
    return state


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    result = {field: config.get(field) for field in COMPATIBILITY_FIELDS}
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("Adapter target_modules must be a non-empty list")
    result["target_modules"] = sorted(str(module) for module in target_modules)
    result["rank_pattern"] = config.get("rank_pattern") or {}
    result["alpha_pattern"] = config.get("alpha_pattern") or {}
    result["modules_to_save"] = config.get("modules_to_save") or []
    return result


def validate_adapter_compatibility(
    adapter_dirs: Sequence[Path],
    states: Sequence[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if not adapter_dirs or len(adapter_dirs) != len(states):
        raise ValueError("adapter_dirs and states must have the same non-zero length")

    configs = [load_adapter_config(path) for path in adapter_dirs]
    expected_config = normalized_config(configs[0])
    for adapter_dir, config in zip(adapter_dirs[1:], configs[1:], strict=True):
        actual = normalized_config(config)
        if actual != expected_config:
            raise ValueError(
                f"Adapter config is incompatible with {adapter_dirs[0]}: {adapter_dir}"
            )

    expected_keys = set(states[0])
    if not expected_keys:
        raise ValueError("Adapter state has no tensor keys")
    invalid_keys = sorted(
        key
        for key in expected_keys
        if not key.endswith((".lora_A.weight", ".lora_B.weight"))
    )
    if invalid_keys:
        raise ValueError(f"Unsupported non-LoRA tensor keys: {invalid_keys[:3]}")

    expected_specs = {
        key: (tuple(tensor.shape), tensor.dtype) for key, tensor in states[0].items()
    }
    for adapter_dir, state in zip(adapter_dirs[1:], states[1:], strict=True):
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))[:3]
            extra = sorted(set(state) - expected_keys)[:3]
            raise ValueError(
                f"Adapter keys differ for {adapter_dir}; missing={missing}, extra={extra}"
            )
        for key, tensor in state.items():
            expected_shape, expected_dtype = expected_specs[key]
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Adapter tensor mismatch for {adapter_dir}:{key}; "
                    f"expected shape={expected_shape}, dtype={expected_dtype}, "
                    f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )

    return {
        "adapter_count": len(adapter_dirs),
        "tensor_count": len(expected_keys),
        "tensor_dtype": str(next(iter(states[0].values())).dtype),
        "config": expected_config,
    }


@torch.inference_mode()
def factor_mix_state_dict(
    states: Sequence[dict[str, torch.Tensor]], weights: Sequence[float]
) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must have the same non-zero length")
    numeric_weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in numeric_weights):
        raise ValueError(f"All weights must be finite: {numeric_weights}")

    expected_keys = set(states[0])
    for state in states[1:]:
        if set(state) != expected_keys:
            raise ValueError("All adapter states must contain identical keys")

    mixed: dict[str, torch.Tensor] = {}
    for key, first_tensor in states[0].items():
        result = torch.zeros_like(first_tensor)
        for state, weight in zip(states, numeric_weights, strict=True):
            tensor = state[key]
            if tensor.shape != first_tensor.shape or tensor.dtype != first_tensor.dtype:
                raise ValueError(f"Tensor spec mismatch for {key}")
            result.add_(tensor, alpha=weight)
        mixed[key] = result
    return mixed


def l1_penalty(
    weights: Sequence[float], *, alpha: float = 0.05, reduction: str = "mean"
) -> float:
    values = [abs(float(weight)) for weight in weights]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("weights must contain finite values")
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite non-negative number")
    if reduction == "mean":
        reduced = sum(values) / len(values)
    elif reduction == "sum":
        reduced = sum(values)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return alpha * reduced
