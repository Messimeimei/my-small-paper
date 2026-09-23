#!/usr/bin/env python3
"""Summarize chat prompt-role variants by evaluation task and dataset view."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(path: Path) -> dict[str, object]:
    rows = 0
    tasks: set[str] = set()
    aspects: set[str] = set()
    versions: set[str] = set()
    modes: set[str] = set()
    systems: set[str] = set()
    users: set[str] = set()
    prompt_sequences: set[str] = set()
    assistants: set[str] = set()
    assistant_roles: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if row.get("task") is not None:
                tasks.add(str(row["task"]))
            if row.get("aspect") is not None:
                aspects.add(str(row["aspect"]))
            if row.get("prompt_version") is not None:
                versions.add(str(row["prompt_version"]))
            if row.get("evaluation_mode") is not None:
                modes.add(str(row["evaluation_mode"]))
            prompt = row.get("prompt") or []
            prompt_sequences.add(digest(prompt))
            systems.update(str(m.get("content", "")) for m in prompt if m.get("role") == "system")
            users.update(str(m.get("content", "")) for m in prompt if m.get("role") == "user")
            completion = row.get("completion")
            if completion is not None:
                assistants.add(digest(completion))
                if isinstance(completion, list):
                    assistant_roles.update(str(m.get("role")) for m in completion if isinstance(m, dict))
    return {
        "file": str(path),
        "rows": rows,
        "task": ",".join(sorted(tasks)),
        "aspect": ",".join(sorted(aspects)),
        "modes": ",".join(sorted(modes)),
        "prompt_versions": ",".join(sorted(versions)),
        "system_variants": len(systems),
        "user_variants": len(users),
        "prompt_sequence_variants": len(prompt_sequences),
        "assistant_completion_variants": len(assistants),
        "assistant_roles": ",".join(sorted(assistant_roles)),
    }


def default_files(root: Path) -> list[Path]:
    patterns = [
        "data/origin_data/*.jsonl",
        "data/*/cot/test_cot.jsonl",
        "data/*/label_only/test_label_only.jsonl",
        "data/*/cot/train_cot.jsonl",
        "data/*/label_only/train_label_only.jsonl",
    ]
    return sorted({p for pattern in patterns for p in root.glob(pattern)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [summarize(path) for path in default_files(args.root)]
    fields = list(records[0]) if records else []
    target = args.output.open("w", encoding="utf-8", newline="") if args.output else None
    writer = csv.DictWriter(target or __import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    if target:
        target.close()


if __name__ == "__main__":
    main()
