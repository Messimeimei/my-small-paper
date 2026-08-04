"""Legacy four-condition comparison table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.metrics import format_float, format_pct
from evaluation.condition_labels import infer_eval_condition

COMPARISON_MODES = (
    "base_label_only",
    "base_cot",
    "ft_label_only",
    "ft_cot",
)

# Markdown column titles (ft modes keep short names as requested).
COMPARISON_MODE_LABELS = {
    "base_label_only": "base-Label-only",
    "base_cot": "base-cot",
    "ft_label_only": "Label-only",
    "ft_cot": "cot",
}


def comparison_field_prefix(mode: str) -> str:
    """Map comparison mode -> JSON field prefix (base_label / base_cot / ft_label / ft_cot)."""
    mapping = {
        "base_label_only": "base_label",
        "base_cot": "base_cot",
        "ft_label_only": "ft_label",
        "ft_cot": "ft_cot",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown comparison mode: {mode}")
    return mapping[mode]


def infer_comparison_mode(
    *,
    supervision_mode: str,
    adapter: Path | str | None,
    exp_name: str | None = None,
) -> str:
    """Return one of base_label_only / base_cot / ft_label_only / ft_cot."""
    mode = supervision_mode if supervision_mode in {"cot", "label_only", "score_only"} else "cot"
    if mode == "score_only":
        mode = "label_only"
    text = (exp_name or "").lower()
    if "#ft#" in text or text.endswith("#ft") or "/ft#" in text:
        setting = "ft"
    elif "#base#" in text or text.endswith("#base"):
        setting = "base"
    elif adapter is None or str(adapter).strip().lower() in {"", "none"}:
        setting = "base"
    else:
        setting = "ft"
    return f"{setting}_{mode}"




def render_comparison_table(rows: list[dict[str, Any]]) -> str:
    """Plain GFM Markdown table (Cursor preview strips HTML styles).

    Only fine-tuned columns are marked:
      🟦 Label-only   🟧 cot
    Base columns stay unmarked.
    """
    metric_specs = (
        ("Acc", "acc", format_pct),
        ("F1", "f1", format_pct),
        ("samp/s", "samples_per_sec", lambda v: format_float(v, digits=1)),
    )

    def header_for(mode: str, suffix: str) -> str:
        label = COMPARISON_MODE_LABELS[mode]
        if mode == "ft_label_only":
            return f"🟦 {label} {suffix}"
        if mode == "ft_cot":
            return f"🟧 {label} {suffix}"
        return f"{label} {suffix}"

    def cell_for(mode: str, text: str) -> str:
        if not text:
            return ""
        if mode == "ft_label_only":
            return f"**🟦 {text}**"
        if mode == "ft_cot":
            return f"**🟧 {text}**"
        return text

    headers = ["Criterion"]
    for suffix, _field, _fmt in metric_specs:
        for mode in COMPARISON_MODES:
            headers.append(header_for(mode, suffix))
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "> **🟦 Label-only**　**🟧 cot**　｜　base 列不加标记\n\n",
        "| " + " | ".join(headers) + " |\n",
        "| " + " | ".join(align) + " |\n",
    ]
    for row in rows:
        cells = [str(row.get("criterion", ""))]
        for _suffix, field, fmt in metric_specs:
            for mode in COMPARISON_MODES:
                prefix = comparison_field_prefix(mode)
                cells.append(cell_for(mode, fmt(row.get(f"{prefix}_{field}"))))
        lines.append("| " + " | ".join(cells) + " |\n")
    return "".join(lines)


def _refresh_token_ratios(row: dict[str, Any]) -> None:
    base_label_tok = row.get("base_label_avg_output_tokens")
    base_cot_tok = row.get("base_cot_avg_output_tokens")
    ft_label_tok = row.get("ft_label_avg_output_tokens")
    ft_cot_tok = row.get("ft_cot_avg_output_tokens")
    if base_label_tok and base_cot_tok:
        row["token_ratio_base"] = base_cot_tok / base_label_tok
    if ft_label_tok and ft_cot_tok:
        row["token_ratio_ft"] = ft_cot_tok / ft_label_tok


def compute_efficiency_metrics(
    *,
    n_samples: int | None,
    avg_output_tokens: float | None,
    gpu_time_sec: float | None,
    samples_per_sec: float | None = None,
) -> dict[str, float | None]:
    """Derive throughput metrics from eval timing (prefer GPU generate time)."""
    sps = samples_per_sec
    if sps is None and n_samples and gpu_time_sec and gpu_time_sec > 0:
        sps = n_samples / gpu_time_sec
    tps = None
    if (
        n_samples
        and avg_output_tokens is not None
        and gpu_time_sec
        and gpu_time_sec > 0
    ):
        tps = (n_samples * avg_output_tokens) / gpu_time_sec
    return {
        "gpu_time_sec": gpu_time_sec,
        "samples_per_sec": sps,
        "tokens_per_sec": tps,
    }


def migrate_legacy_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrade old score_*/cot_* fields into base_*/ft_* using source paths when possible."""
    out = dict(row)
    sources = out.get("sources") or {}
    # Prefer explicit new fields; only migrate when missing.
    for source_mode, legacy_prefix, new_mode in (
        ("score_only", "score", "label_only"),
        ("label_only", "label", "label_only"),
        ("cot", "cot", "cot"),
    ):
        source = str(sources.get(source_mode) or sources.get(new_mode) or "")
        if "#base#" in source.lower():
            target_mode = f"base_{new_mode}"
        elif "#ft#" in source.lower():
            target_mode = f"ft_{new_mode}"
        else:
            # Historical tables mostly stored fine-tuned runs under score/cot.
            target_mode = f"ft_{new_mode}"
        target_prefix = comparison_field_prefix(target_mode)
        for metric in ("acc", "f1", "avg_output_tokens"):
            legacy_key = f"{legacy_prefix}_{metric}"
            new_key = f"{target_prefix}_{metric}"
            if legacy_key in out and new_key not in out:
                out[new_key] = out[legacy_key]
        for mode_key in (source_mode, new_mode):
            if mode_key in sources and target_mode not in sources:
                sources = {**sources, target_mode: sources[mode_key]}
    out["sources"] = sources
    _refresh_token_ratios(out)
    return out


def merge_comparison_row(
    existing: dict[str, Any] | None,
    *,
    criterion: str,
    mode: str,
    accuracy: float,
    macro_f1: float,
    avg_output_tokens: float | None,
    gpu_time_sec: float | None = None,
    samples_per_sec: float | None = None,
    tokens_per_sec: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Merge one of the four eval settings into a criterion row."""
    if mode in {"label_only", "cot"}:
        # Backward-compatible callers: treat bare modes as fine-tuned.
        mode = f"ft_{mode}"
    if mode not in COMPARISON_MODES:
        raise ValueError(
            f"mode must be one of {COMPARISON_MODES} (or legacy label_only/cot), got {mode!r}"
        )
    row = migrate_legacy_comparison_row(dict(existing or {"criterion": criterion}))
    row["criterion"] = criterion
    prefix = comparison_field_prefix(mode)
    row[f"{prefix}_acc"] = accuracy
    row[f"{prefix}_f1"] = macro_f1
    row[f"{prefix}_avg_output_tokens"] = avg_output_tokens
    eff = compute_efficiency_metrics(
        n_samples=n_samples,
        avg_output_tokens=avg_output_tokens,
        gpu_time_sec=gpu_time_sec,
        samples_per_sec=samples_per_sec,
    )
    if tokens_per_sec is not None:
        eff["tokens_per_sec"] = tokens_per_sec
    if eff["gpu_time_sec"] is not None:
        row[f"{prefix}_gpu_time_sec"] = eff["gpu_time_sec"]
    if eff["samples_per_sec"] is not None:
        row[f"{prefix}_samples_per_sec"] = eff["samples_per_sec"]
    if eff["tokens_per_sec"] is not None:
        row[f"{prefix}_tokens_per_sec"] = eff["tokens_per_sec"]
    if n_samples is not None:
        row[f"{prefix}_n_samples"] = n_samples
    _refresh_token_ratios(row)
    return row
