"""Evaluation defaults and stable RAIL metadata values."""

from pathlib import Path

from shared.project_io import PROJECT_ROOT

DEFAULT_DATASET = PROJECT_ROOT / "data/rw_gen_coherence/cot/test_cot.jsonl"
_AUTODL_TMP = Path("/root/autodl-tmp")
DEFAULT_MERGE_CACHE = (
    _AUTODL_TMP / "merged" if _AUTODL_TMP.is_dir() else PROJECT_ROOT / "merged"
)
DEFAULT_EVAL_OUTPUT_ROOT = PROJECT_ROOT / "eval_output"
DEFAULT_MERGE_RETENTION_DAYS = 0

RAIL_PROBABILITY_NORMALIZATION = "full_vocab_raw"
RAIL_IMPLEMENTATION = "tract_official_release"
RAIL_EXPECTATION_FORMULA = "sum(score * p_full_vocab(score))"
RAIL_DISCRETE_DECODING = "nearest_legal_score_tie_low"

CONFIG_KEYS = {
    "exp_name",
    "model_name",
    "adapter",
    "train_seed",
    "dataset_file",
    "inference_mode",
    "method_options",
    "output_path",
    "max_model_len",
    "max_tokens",
    "temp",
    "top_p",
    "seed",
    "rollout",
    "batch_size",
    "gpu_memory_utilization",
    "merge_cache",
    "merge_retention_days",
    "enable_thinking",
    "train_config",
}
