"""vLLM backend initialization and runtime metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.io import read_json, resolve_path, utc_now, write_json


def import_vllm() -> tuple[Any, Any]:
    """Import lazily so configuration and --help work without vLLM installed."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "vLLM is required for evaluation. Install the evaluation dependencies."
        ) from exc
    return LLM, SamplingParams
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_vllm(
    model_path: Path,
    *,
    max_model_len: int,
    max_tokens: int,
    seed: int,
    gpu_memory_utilization: float,
) -> tuple[Any, Any]:
    llm_class, sampling_params_class = import_vllm()
    llm = llm_class(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    params = sampling_params_class(
        max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=seed
    )
    return llm, params
def gpu_time_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(0)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(0)),
            }
        )
    return info


from models.base import ModelBackend


class VllmBackend(ModelBackend):
    """Local deterministic inference through vLLM."""

    name = "vllm"

    def initialize(
        self,
        model_path: Path,
        *,
        max_model_len: int,
        max_tokens: int,
        seed: int,
        gpu_memory_utilization: float,
    ) -> tuple[Any, Any]:
        return init_vllm(
            model_path,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def set_seed(self, seed: int) -> None:
        set_seed(seed)

    def gpu_snapshot(self) -> dict[str, Any]:
        return gpu_time_snapshot()
