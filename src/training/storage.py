"""Storage helpers for completed training artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_identical_adapter_weight(
    adapter_directory: Path,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    """Deduplicate identical exported/checkpoint adapter weights without losing paths."""
    exported = next(
        (adapter_directory / name for name in _WEIGHT_NAMES if (adapter_directory / name).is_file()),
        None,
    )
    checkpoint = next(
        (checkpoint_directory / name for name in _WEIGHT_NAMES if (checkpoint_directory / name).is_file()),
        None,
    )
    if exported is None or checkpoint is None:
        raise RuntimeError(
            "Cannot deduplicate adapter weights: "
            f"exported={exported} checkpoint={checkpoint}"
        )
    exported_stat = exported.stat()
    checkpoint_stat = checkpoint.stat()
    if exported_stat.st_dev != checkpoint_stat.st_dev:
        return {
            "mode": "separate_filesystems",
            "exported": str(exported),
            "checkpoint": str(checkpoint),
        }
    exported_hash = _sha256(exported)
    checkpoint_hash = _sha256(checkpoint)
    if exported_hash != checkpoint_hash:
        raise RuntimeError(
            "Exported and checkpoint adapter weights differ; refusing to hardlink: "
            f"{exported} != {checkpoint}"
        )
    if os.path.samefile(exported, checkpoint):
        return {
            "mode": "hardlink",
            "sha256": exported_hash,
            "bytes": exported_stat.st_size,
            "exported": str(exported),
            "checkpoint": str(checkpoint),
        }

    backup = exported.with_name(exported.name + ".dedup-backup")
    if backup.exists():
        backup.unlink()
    exported.rename(backup)
    try:
        os.link(checkpoint, exported)
    except BaseException:
        if not exported.exists() and backup.exists():
            backup.rename(exported)
        raise
    backup.unlink()
    return {
        "mode": "hardlink",
        "sha256": exported_hash,
        "bytes": exported_stat.st_size,
        "exported": str(exported),
        "checkpoint": str(checkpoint),
    }
