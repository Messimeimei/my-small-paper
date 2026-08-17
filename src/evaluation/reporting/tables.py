"""Shared TRACT-style Markdown result-table rendering."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Hashable, Mapping, Sequence


BEST = "best"
SECOND = "second"
MISSING = "\u2014"


@dataclass(frozen=True)
class PaperResultRow:
    """One configuration row in a task-by-metric result matrix."""

    group: str
    config: tuple[str, ...]
    values: Mapping[str, float | None]
    display_values: Mapping[str, str] = field(default_factory=dict)
    rank_eligible: bool = True


@dataclass(frozen=True)
class PaperMetricColumn:
    """A metric column in a TRACT-style grouped result table."""

    header: str
    key: str
    formatter: Callable[[float], str]
    higher_is_better: bool = True


@dataclass(frozen=True)
class PaperMetricRow:
    """A row for tables whose metrics are columns rather than tasks."""

    group: str
    config: tuple[str, ...]
    values: Mapping[str, float | None]
    display_values: Mapping[str, str] = field(default_factory=dict)
    rank_eligible: bool = True


def ranked_positions(
    values: Mapping[Hashable, float | None],
    *,
    higher_is_better: bool,
) -> dict[Hashable, str]:
    """Return best/second labels, assigning ties to the same rank."""

    present = [float(value) for value in values.values() if value is not None]
    distinct = sorted(set(present), reverse=higher_is_better)
    if not distinct:
        return {}
    best = distinct[0]
    second = distinct[1] if len(distinct) > 1 else None
    ranks: dict[Hashable, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        number = float(value)
        if math.isclose(number, best, rel_tol=0.0, abs_tol=1e-12):
            ranks[key] = BEST
        elif second is not None and math.isclose(
            number,
            second,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            ranks[key] = SECOND
    return ranks


def decorate_rank(text: str, rank: str | None) -> str:
    if text == MISSING or not text:
        return text or MISSING
    if rank == BEST:
        return f"**{text}**"
    if rank == SECOND:
        return f"<u>{text}</u>"
    return text


def render_grouped_metric_table(
    rows: Sequence[PaperMetricRow],
    *,
    config_headers: Sequence[str],
    metric_columns: Sequence[PaperMetricColumn],
) -> str:
    """Render a paper-style table with explicit section/group rows."""

    if any(len(row.config) != len(config_headers) for row in rows):
        raise ValueError("each row config must match config_headers")

    ranks = {
        column.key: ranked_positions(
            {
                index: row.values.get(column.key)
                for index, row in enumerate(rows)
                if row.rank_eligible
            },
            higher_is_better=column.higher_is_better,
        )
        for column in metric_columns
    }
    headers = [*config_headers, *(column.header for column in metric_columns)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(config_headers) + ["---:"] * len(metric_columns)
        ) + " |",
    ]
    previous_group = None
    for index, row in enumerate(rows):
        if row.group != previous_group:
            lines.append(
                "| " + " | ".join(
                    [f"**{row.group}**"] + [""] * (len(headers) - 1)
                ) + " |"
            )
            previous_group = row.group
        cells: list[str] = []
        for column in metric_columns:
            value = row.values.get(column.key)
            text = (
                row.display_values.get(column.key)
                or (column.formatter(float(value)) if value is not None else MISSING)
            )
            cells.append(decorate_rank(text, ranks[column.key].get(index)))
        lines.append("| " + " | ".join([*row.config, *cells]) + " |")
    return "\n".join(lines)


def render_result_matrix(
    rows: Sequence[PaperResultRow],
    *,
    config_headers: Sequence[str],
    task_keys: Sequence[str],
    task_labels: Mapping[str, str],
    applicable_tasks: Sequence[str],
    value_formatter: Callable[[float], str],
    higher_is_better: bool = True,
    average_label: str = "Average",
) -> str:
    """Render grouped configurations against tasks, followed by a strict average.

    The average is shown only when every applicable task is present. Rankings use
    raw values from eligible rows: best is bold and second-best is underlined.
    """

    if any(len(row.config) != len(config_headers) for row in rows):
        raise ValueError("each row config must match config_headers")

    applicable = tuple(applicable_tasks)
    average_values: dict[int, float | None] = {}
    for index, row in enumerate(rows):
        present = [
            float(row.values[task])
            for task in applicable
            if row.values.get(task) is not None
        ]
        average_values[index] = (
            statistics.fmean(present) if len(present) == len(applicable) else None
        )

    task_ranks = {
        task: ranked_positions(
            {
                index: row.values.get(task)
                for index, row in enumerate(rows)
                if row.rank_eligible
            },
            higher_is_better=higher_is_better,
        )
        for task in task_keys
    }
    average_ranks = ranked_positions(
        {
            index: average_values[index]
            for index, row in enumerate(rows)
            if row.rank_eligible
        },
        higher_is_better=higher_is_better,
    )

    headers = [
        *config_headers,
        "Coverage",
        *(task_labels[task] for task in task_keys),
        average_label,
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] * len(config_headers)
            + ["---:"] * (len(task_keys) + 2)
        )
        + " |",
    ]

    previous_group = None
    for index, row in enumerate(rows):
        if row.group != previous_group:
            lines.append(
                "| "
                + " | ".join(
                    [f"**{row.group}**"] + [""] * (len(headers) - 1)
                )
                + " |"
            )
            previous_group = row.group

        coverage = sum(row.values.get(task) is not None for task in applicable)
        cells = []
        for task in task_keys:
            value = row.values.get(task)
            rendered = (
                row.display_values.get(task)
                or (value_formatter(float(value)) if value is not None else MISSING)
            )
            cells.append(decorate_rank(rendered, task_ranks[task].get(index)))
        average = average_values[index]
        average_text = (
            value_formatter(average) if average is not None else MISSING
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    *row.config,
                    f"{coverage}/{len(applicable)}",
                    *cells,
                    decorate_rank(average_text, average_ranks.get(index)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)
