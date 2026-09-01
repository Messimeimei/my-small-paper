#!/usr/bin/env python3
"""Rescore fixed corrected rationales with their original Qwen3-4B adapters.

Only the assistant-side rationale prefix changes. The system message, user
message, model, adapter, chat template, and greedy sampling configuration are
loaded from each record's original resolved evaluation configuration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/analysis/interface_switch_harmful_samples/corrected_harmful_samples.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("rescored_harmful_samples.jsonl")
DEFAULT_WORK_DIR = DEFAULT_INPUT.parent / "qwen_rescore"
PROMPT_VERSION = "fixed_corrected_rationale_v1"
ASSISTANT_PREFIX_TEMPLATE = "<reasoning>{corrected_rationale}</reasoning>\n<score>"
_JSONL_INDEX_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate records, configs, and prompt messages without loading vLLM.",
    )
    parser.add_argument(
        "--worker-group",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    resolved = path.resolve()
    cached = _JSONL_INDEX_CACHE.get(resolved)
    if cached is not None:
        return cached
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(resolved):
        row_id = str(row.get("id", row.get("record_id", ""))).strip()
        if not row_id:
            raise ValueError(f"{resolved}: row missing id")
        if row_id in indexed:
            raise ValueError(f"{resolved}: duplicate id {row_id}")
        indexed[row_id] = row
    _JSONL_INDEX_CACHE[resolved] = indexed
    return indexed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "group"


def scorer_config(item: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    scorer = item.get("qwen3_4b_scorer")
    if not isinstance(scorer, dict):
        raise ValueError(f"{item.get('record_id')}: missing qwen3_4b_scorer")
    config_path = resolve_path(str(scorer.get("resolved_config_path", "")))
    if not config_path.is_file():
        raise ValueError(
            f"{item.get('record_id')}: missing resolved config {config_path}"
        )
    return read_json(config_path), config_path


def corrected_rationale(item: dict[str, Any]) -> str:
    workflow = item.get("third_layer_workflow") or {}
    edit = workflow.get("rationale_edit") or {}
    review = workflow.get("minimax_review") or {}
    rescore = workflow.get("qwen_rescore") or {}
    rationale = str(edit.get("corrected_rationale", "")).strip()
    record_id = item.get("record_id")
    if edit.get("status") != "completed" or not rationale:
        raise ValueError(f"{record_id}: corrected rationale is not completed")
    if review.get("status") != "approved" or review.get("is_reasonable") is not True:
        raise ValueError(f"{record_id}: MiniMax review is not approved")
    if rescore.get("status") not in {"ready", "completed"}:
        raise ValueError(f"{record_id}: qwen_rescore is not ready")
    lowered = rationale.lower()
    if any(tag in lowered for tag in ("<reasoning", "</reasoning", "<score")):
        raise ValueError(f"{record_id}: corrected rationale contains control tags")
    return rationale


def assistant_prefix(item: dict[str, Any]) -> str:
    scorer = item["qwen3_4b_scorer"]
    template = str(scorer.get("assistant_prefix_template", ""))
    if template != ASSISTANT_PREFIX_TEMPLATE:
        raise ValueError(
            f"{item.get('record_id')}: unexpected assistant prefix template {template!r}"
        )
    return template.format(corrected_rationale=corrected_rationale(item))


def validate_record(item: dict[str, Any]) -> dict[str, Any]:
    record_id = str(item.get("record_id", "")).strip()
    if not record_id:
        raise ValueError("record missing record_id")
    prompt = (item.get("cot_result") or {}).get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) < 2
        or any(not isinstance(message, dict) for message in prompt)
    ):
        raise ValueError(f"{record_id}: cot_result.prompt is invalid")
    if any(message.get("role") == "assistant" for message in prompt):
        raise ValueError(f"{record_id}: original prompt already contains assistant data")

    scorer = item["qwen3_4b_scorer"]
    resolved, config_path = scorer_config(item)
    expected_pairs = {
        "model_name": (resolve_path(str(scorer["model_name"])), resolve_path(str(resolved["model_name"]))),
        "adapter": (resolve_path(str(scorer["adapter"])), resolve_path(str(resolved["adapter"]))),
        "max_model_len": (int(scorer["max_model_len"]), int(resolved["max_model_len"])),
        "max_tokens": (int(scorer["original_max_tokens"]), int(resolved["max_tokens"])),
        "temperature": (float(scorer["temperature"]), float(resolved["temp"])),
        "top_p": (float(scorer["top_p"]), float(resolved["top_p"])),
        "enable_thinking": (bool(scorer["enable_thinking"]), bool(resolved["enable_thinking"])),
    }
    mismatches = {
        key: [str(left), str(right)]
        for key, (left, right) in expected_pairs.items()
        if left != right
    }
    if mismatches:
        raise ValueError(f"{record_id}: scorer/resolved config mismatch: {mismatches}")
    model_path = expected_pairs["model_name"][0]
    adapter_path = expected_pairs["adapter"][0]
    if not model_path.is_dir():
        raise ValueError(f"{record_id}: model directory does not exist: {model_path}")
    if not adapter_path.is_dir():
        raise ValueError(f"{record_id}: adapter directory does not exist: {adapter_path}")
    if not any(
        (adapter_path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError(f"{record_id}: adapter weights are missing: {adapter_path}")
    if resolved.get("inference_mode", "greedy") != "greedy":
        raise ValueError(f"{record_id}: original inference mode is not greedy")
    if float(resolved["temp"]) != 0.0 or float(resolved["top_p"]) != 1.0:
        raise ValueError(f"{record_id}: original inference is not deterministic")
    if int(resolved.get("rollout", 1)) != 1:
        raise ValueError(f"{record_id}: expected original rollout=1")

    score_sets = item.get("score_sets")
    if (
        not isinstance(score_sets, list)
        or not score_sets
        or any(isinstance(score, bool) or not isinstance(score, int) for score in score_sets)
    ):
        raise ValueError(f"{record_id}: invalid score_sets")
    if item.get("gold_score") not in score_sets:
        raise ValueError(f"{record_id}: gold_score is outside score_sets")
    sample_id = str(item.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError(f"{record_id}: sample_id is missing")
    dataset_path = resolve_path(str(resolved["dataset_file"]))
    dataset_row = jsonl_index(dataset_path).get(sample_id)
    if dataset_row is None:
        raise ValueError(f"{record_id}: sample missing from {dataset_path}")
    if dataset_row.get("prompt") != prompt:
        raise ValueError(f"{record_id}: embedded prompt differs from original dataset")
    source_label = dataset_row.get("labels", dataset_row.get("label"))
    if int(source_label) != int(item["gold_score"]):
        raise ValueError(f"{record_id}: gold score differs from original dataset")
    if list(dataset_row.get("score_sets", score_sets)) != list(score_sets):
        raise ValueError(f"{record_id}: score_sets differ from original dataset")

    prediction_path = resolve_path(str(item["cot_result"]["prediction_path"]))
    prediction_row = jsonl_index(prediction_path).get(sample_id)
    if prediction_row is None:
        raise ValueError(f"{record_id}: sample missing from {prediction_path}")
    rollout_predictions = prediction_row.get("rollout_predictions") or []
    raw_outputs = prediction_row.get("raw_outputs") or []
    outputs = prediction_row.get("outputs") or []
    if not rollout_predictions or int(rollout_predictions[0]) != int(
        item["cot_result"]["prediction"]
    ):
        raise ValueError(f"{record_id}: original prediction provenance mismatch")
    if not raw_outputs or raw_outputs[0] != item["cot_result"]["raw_output"]:
        raise ValueError(f"{record_id}: original raw output provenance mismatch")
    if not outputs or outputs[0] != item["cot_result"]["output"]:
        raise ValueError(f"{record_id}: original output provenance mismatch")
    prefix = assistant_prefix(item)

    return {
        "record_id": record_id,
        "task": str(item["task"]),
        "training_method": str(item["training_method"]),
        "seed": int(item["seed"]),
        "model_name": str(resolve_path(str(resolved["model_name"]))),
        "adapter": str(resolve_path(str(resolved["adapter"]))),
        "resolved_config_path": str(config_path),
        "dataset_file": str(dataset_path),
        "prediction_path": str(prediction_path),
        "source_prompt_verified": True,
        "source_prediction_verified": True,
        "score_sets": list(score_sets),
        "assistant_prefix_sha256": text_sha256(prefix),
        "original_prompt_sha256": text_sha256(
            json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        ),
    }


def group_signature(item: dict[str, Any]) -> dict[str, Any]:
    resolved, config_path = scorer_config(item)
    return {
        "model_name": str(resolve_path(str(resolved["model_name"]))),
        "adapter": str(resolve_path(str(resolved["adapter"]))),
        "resolved_config_path": str(config_path),
        "max_model_len": int(resolved["max_model_len"]),
        "max_tokens": int(resolved["max_tokens"]),
        "temperature": float(resolved["temp"]),
        "top_p": float(resolved["top_p"]),
        "seed": int(resolved["seed"]),
        "rollout": int(resolved["rollout"]),
        "batch_size": int(resolved["batch_size"]),
        "gpu_memory_utilization": float(resolved["gpu_memory_utilization"]),
        "merge_cache": str(resolve_path(str(resolved["merge_cache"]))),
        "merge_retention_days": float(resolved.get("merge_retention_days", 0.0)),
        "enable_thinking": bool(resolved["enable_thinking"]),
        "inference_mode": str(resolved.get("inference_mode", "greedy")),
    }


def build_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for item in rows:
        value = validate_record(item)
        record_id = value["record_id"]
        if record_id in seen:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        normalized.append(value)
        signature = group_signature(item)
        signature_text = json.dumps(signature, sort_keys=True)
        key = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:12]
        group = grouped.setdefault(
            key,
            {
                "key": key,
                "signature": signature,
                "record_ids": [],
                "task": value["task"],
                "training_method": value["training_method"],
                "train_seed": value["seed"],
            },
        )
        group["record_ids"].append(record_id)

    groups: list[dict[str, Any]] = []
    for index, group in enumerate(
        sorted(
            grouped.values(),
            key=lambda value: (
                value["task"],
                value["training_method"],
                value["train_seed"],
                value["key"],
            ),
        ),
        1,
    ):
        group = copy.deepcopy(group)
        group["group_id"] = (
            f"{index:03d}_{safe_name(group['task'])}_"
            f"{safe_name(group['training_method'])}_seed{group['train_seed']}"
        )
        group["record_ids"] = sorted(group["record_ids"])
        group["samples"] = len(group["record_ids"])
        groups.append(group)
    return {
        "prompt_version": PROMPT_VERSION,
        "assistant_prefix_template": ASSISTANT_PREFIX_TEMPLATE,
        "records": len(rows),
        "groups": groups,
        "record_validation": normalized,
    }


def chat_template_kwargs(tokenizer: Any, enable_thinking: bool) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None) or ""
    return {"enable_thinking": enable_thinking} if "enable_thinking" in template else {}


def render_prompts(tokenizer: Any, item: dict[str, Any]) -> tuple[str, str]:
    prompt = item["cot_result"]["prompt"]
    resolved, _ = scorer_config(item)
    kwargs = chat_template_kwargs(tokenizer, bool(resolved["enable_thinking"]))
    original = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    prefix = assistant_prefix(item)
    intervention = original + prefix
    if intervention[: len(original)] != original:
        raise ValueError(
            f"{item['record_id']}: original rendered prompt changed before intervention"
        )
    if intervention[len(original) :] != prefix:
        raise ValueError(
            f"{item['record_id']}: intervention differs from exact assistant prefix"
        )
    return original, intervention


def validate_prompt_messages(rows: list[dict[str, Any]]) -> dict[str, Any]:
    models: set[str] = set()
    for item in rows:
        resolved, _ = scorer_config(item)
        models.add(str(resolve_path(str(resolved["model_name"]))))
        prompt_before = copy.deepcopy(item["cot_result"]["prompt"])
        assistant_prefix(item)
        if item["cot_result"]["prompt"] != prompt_before:
            raise ValueError(f"{item['record_id']}: prompt mutated during validation")
    return {
        "models": sorted(models),
        "message_records_verified": len(rows),
        "original_system_and_user_messages_reused_verbatim": True,
        "gold_in_model_prompt": False,
        "only_added_message": "assistant corrected rationale + <score> prefix",
        "rendered_boundary_verification": "worker runtime before generation",
    }


def plan_paths(work_dir: Path) -> tuple[Path, Path]:
    return work_dir / "manifest.json", work_dir / "plan.json"


def prepare(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"{input_path} is empty")
    plan = build_plan(rows)
    prompt_verification = validate_prompt_messages(rows)
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "input": {
            "path": portable_path(input_path),
            "sha256": sha256(input_path),
            "records": len(rows),
        },
        "output": portable_path(output_path),
        "groups": len(plan["groups"]),
        "execution": "serial subprocess per adapter group",
        "sampling": "exact original resolved greedy configuration",
        "gold_in_model_prompt": False,
        "prompt_verification": prompt_verification,
    }
    manifest_path, plan_path = plan_paths(work_dir)
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise ValueError(
            f"configuration differs from existing {manifest_path}; use a new --work-dir"
        )
    if plan_path.is_file() and read_json(plan_path) != plan:
        raise ValueError(
            f"plan differs from existing {plan_path}; use a new --work-dir"
        )
    write_json(manifest_path, manifest)
    write_json(plan_path, plan)
    return rows, plan


def group_result_path(work_dir: Path, group_id: str) -> Path:
    return work_dir / "groups" / group_id / "predictions.jsonl"


def group_is_complete(work_dir: Path, group: dict[str, Any]) -> bool:
    path = group_result_path(work_dir, group["group_id"])
    if not path.is_file():
        return False
    rows = read_jsonl(path)
    return (
        len(rows) == len(group["record_ids"])
        and {row.get("record_id") for row in rows} == set(group["record_ids"])
        and all(row.get("status") == "completed" for row in rows)
    )


def run_worker_process(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    group: dict[str, Any],
) -> None:
    group_dir = work_dir / "groups" / group["group_id"]
    group_dir.mkdir(parents=True, exist_ok=True)
    log_path = group_dir / "run.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--work-dir",
        str(work_dir),
        "--worker-group",
        group["group_id"],
    ]
    print(
        f"[{group['group_id']}] start samples={group['samples']} "
        f"adapter={group['signature']['adapter']}",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"worker {group['group_id']} failed with {process.returncode}:\n"
            + "\n".join(tail)
        )
    if not group_is_complete(work_dir, group):
        raise RuntimeError(f"worker {group['group_id']} exited without complete output")
    print(f"[{group['group_id']}] completed", flush=True)


def run_worker(
    input_path: Path,
    work_dir: Path,
    group_id: str,
) -> None:
    from evaluation.inference.rail_scoring import chat_template_supports_thinking
    from models.adapters import (
        adapter_weight_file,
        cleanup_merged_cache,
        ensure_merged_model,
        remove_merged_model,
    )
    from models.vllm import init_vllm
    from utils.metrics import extract_reasoning, extract_score

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    _, plan_path = plan_paths(work_dir)
    plan = read_json(plan_path)
    groups = {group["group_id"]: group for group in plan["groups"]}
    if group_id not in groups:
        raise ValueError(f"unknown worker group {group_id}")
    group = groups[group_id]
    source_by_id = {row["record_id"]: row for row in read_jsonl(input_path)}
    rows = [source_by_id[record_id] for record_id in group["record_ids"]]
    signature = group["signature"]
    model_name = Path(signature["model_name"])
    adapter = Path(signature["adapter"])
    merge_cache = Path(signature["merge_cache"])
    adapter_weight_file(adapter)
    cleanup_merged_cache(merge_cache, float(signature["merge_retention_days"]))

    merged_model: Path | None = None
    llm = None
    try:
        merged_model = ensure_merged_model(model_name, adapter, merge_cache)
        llm, sampling_params = init_vllm(
            merged_model,
            max_model_len=int(signature["max_model_len"]),
            max_tokens=int(signature["max_tokens"]),
            seed=int(signature["seed"]),
            gpu_memory_utilization=float(signature["gpu_memory_utilization"]),
        )
        tokenizer = llm.get_tokenizer()
        if bool(signature["enable_thinking"]) and not chat_template_supports_thinking(
            tokenizer
        ):
            raise ValueError("resolved config enables thinking but template does not")

        rendered: list[str] = []
        original_rendered: list[str] = []
        for item in rows:
            original_text, intervention_text = render_prompts(tokenizer, item)
            original_rendered.append(original_text)
            rendered.append(intervention_text)
        completions = llm.generate(rendered, sampling_params, use_tqdm=False)

        predictions: list[dict[str, Any]] = []
        for item, original_text, model_input, completion in zip(
            rows, original_rendered, rendered, completions, strict=True
        ):
            generated = completion.outputs[0]
            completion_text = str(generated.text)
            prefix = assistant_prefix(item)
            raw_output = prefix + completion_text
            score_sets = list(item["score_sets"])
            prediction = extract_score(raw_output, set(score_sets))
            fixed_reasoning = extract_reasoning(raw_output)
            if fixed_reasoning != corrected_rationale(item):
                raise RuntimeError(
                    f"{item['record_id']}: raw output does not preserve fixed rationale"
                )
            predictions.append(
                {
                    "status": "completed",
                    "record_id": item["record_id"],
                    "prediction": prediction,
                    "format_valid": prediction is not None,
                    "completion": completion_text,
                    "raw_output": raw_output,
                    "output_token_ids": list(generated.token_ids),
                    "finish_reason": getattr(generated, "finish_reason", None),
                    "stop_reason": getattr(generated, "stop_reason", None),
                    "model_name": str(model_name),
                    "adapter": str(adapter),
                    "resolved_config_path": signature["resolved_config_path"],
                    "prompt_version": PROMPT_VERSION,
                    "input_prompt_field": "cot_result.prompt",
                    "input_rationale_field": (
                        "third_layer_workflow.rationale_edit.corrected_rationale"
                    ),
                    "assistant_prefix": prefix,
                    "original_rendered_prompt_sha256": text_sha256(original_text),
                    "model_input_sha256": text_sha256(model_input),
                    "sampling": {
                        "max_model_len": signature["max_model_len"],
                        "max_tokens": signature["max_tokens"],
                        "temperature": signature["temperature"],
                        "top_p": signature["top_p"],
                        "seed": signature["seed"],
                        "rollout": signature["rollout"],
                        "enable_thinking": signature["enable_thinking"],
                    },
                }
            )
        write_jsonl(group_result_path(work_dir, group_id), predictions)
        write_json(
            work_dir / "groups" / group_id / "summary.json",
            {
                "group_id": group_id,
                "samples": len(predictions),
                "format_valid": sum(row["format_valid"] for row in predictions),
                "model_name": str(model_name),
                "adapter": str(adapter),
                "resolved_config_path": signature["resolved_config_path"],
            },
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        remove_merged_model(merged_model)


def collect_results(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: Path,
    work_dir: Path,
) -> None:
    results: dict[str, dict[str, Any]] = {}
    for group in plan["groups"]:
        path = group_result_path(work_dir, group["group_id"])
        if not path.is_file():
            raise RuntimeError(f"missing group output: {path}")
        for result in read_jsonl(path):
            record_id = str(result["record_id"])
            if record_id in results:
                raise RuntimeError(f"duplicate worker result: {record_id}")
            results[record_id] = result
    expected_ids = {row["record_id"] for row in rows}
    if set(results) != expected_ids:
        missing = sorted(expected_ids - set(results))
        extra = sorted(set(results) - expected_ids)
        raise RuntimeError(f"worker result ID mismatch: missing={missing}, extra={extra}")

    output_rows: list[dict[str, Any]] = []
    compact_predictions: list[dict[str, Any]] = []
    for item in rows:
        result = results[item["record_id"]]
        value = copy.deepcopy(item)
        workflow = value["third_layer_workflow"]
        qwen_rescore = workflow["qwen_rescore"]
        qwen_rescore.update(
            {
                "status": "completed",
                "model_name": result["model_name"],
                "adapter": result["adapter"],
                "resolved_config_path": result["resolved_config_path"],
                "prompt_version": result["prompt_version"],
                "input_prompt_field": result["input_prompt_field"],
                "input_rationale_field": result["input_rationale_field"],
                "assistant_prefix": result["assistant_prefix"],
                "prediction": result["prediction"],
                "completion": result["completion"],
                "raw_output": result["raw_output"],
                "format_valid": result["format_valid"],
                "output_token_ids": result["output_token_ids"],
                "finish_reason": result["finish_reason"],
                "stop_reason": result["stop_reason"],
                "original_rendered_prompt_sha256": result[
                    "original_rendered_prompt_sha256"
                ],
                "model_input_sha256": result["model_input_sha256"],
                "sampling": result["sampling"],
            }
        )
        causal = workflow["causal_comparison"]
        prediction = result["prediction"]
        original_prediction = causal["original_wrong_prediction"]
        gold_score = causal["gold_score"]
        causal.update(
            {
                "corrected_rationale_prediction": prediction,
                "score_changed": (
                    prediction != original_prediction if prediction is not None else None
                ),
                "changed_to_gold": (
                    prediction == gold_score if prediction is not None else None
                ),
            }
        )
        output_rows.append(value)
        compact_predictions.append(
            {
                "record_id": item["record_id"],
                "task": item["task"],
                "training_method": item["training_method"],
                "seed": item["seed"],
                "original_prediction": original_prediction,
                "corrected_rationale_prediction": prediction,
                "gold_score": gold_score,
                "format_valid": result["format_valid"],
                "score_changed": causal["score_changed"],
                "changed_to_gold": causal["changed_to_gold"],
            }
        )

    write_jsonl(output_path, output_rows)
    write_jsonl(work_dir / "predictions.jsonl", compact_predictions)
    valid = [row for row in compact_predictions if row["format_valid"]]
    summary = {
        "prompt_version": PROMPT_VERSION,
        "samples": len(compact_predictions),
        "format_valid": len(valid),
        "format_invalid": len(compact_predictions) - len(valid),
        "score_changed": sum(row["score_changed"] is True for row in valid),
        "score_unchanged": sum(row["score_changed"] is False for row in valid),
        "changed_to_gold": sum(row["changed_to_gold"] is True for row in valid),
        "still_incorrect": sum(row["changed_to_gold"] is False for row in valid),
        "by_task": dict(Counter(row["task"] for row in compact_predictions)),
        "output": portable_path(output_path),
    }
    write_json(work_dir / "summary.json", summary)
    print(
        f"finished: samples={summary['samples']} valid={summary['format_valid']} "
        f"changed={summary['score_changed']} changed_to_gold={summary['changed_to_gold']} "
        f"output={output_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    work_dir = resolve_path(args.work_dir)
    if input_path == output_path:
        raise SystemExit("--output must differ from --input")
    rows, plan = prepare(input_path, output_path, work_dir)
    if args.prepare_only:
        print(
            f"validated {len(rows)} records across {len(plan['groups'])} adapter groups; "
            f"wrote {work_dir / 'manifest.json'}",
            flush=True,
        )
        return
    if args.worker_group:
        run_worker(input_path, work_dir, args.worker_group)
        return

    for index, group in enumerate(plan["groups"], 1):
        if group_is_complete(work_dir, group):
            print(
                f"[{index}/{len(plan['groups'])}] reuse completed {group['group_id']}",
                flush=True,
            )
            continue
        print(f"[{index}/{len(plan['groups'])}] dispatch {group['group_id']}", flush=True)
        run_worker_process(input_path, output_path, work_dir, group)
    collect_results(rows, plan, output_path, work_dir)


if __name__ == "__main__":
    main()
