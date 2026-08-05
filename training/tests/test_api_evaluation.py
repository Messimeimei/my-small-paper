from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from api_evaluation.pipeline import (
    ApiRun,
    build_artifacts,
    expand_runs,
    read_completed_responses,
)
from api_evaluation.report import update_api_reports


class ApiEvaluationTests(unittest.TestCase):
    def test_matrix_expands_models_tasks_and_modes(self) -> None:
        config = {
            "models": [
                {"name": "model-a-20250101", "slug": "model_a_20250101"},
                {"name": "model-b-20250101", "slug": "model_b_20250101"},
            ],
            "tasks": [
                {
                    "name": "task-a",
                    "datasets": {"label_only": "a.jsonl", "cot": "b.jsonl"},
                }
            ],
        }
        runs = expand_runs(config)
        self.assertEqual(len(runs), 4)
        self.assertEqual({run.mode for run in runs}, {"label_only", "cot"})

    def test_matrix_preserves_explicit_response_model_aliases(self) -> None:
        config = {
            "models": [
                {
                    "name": "dated-model-20250101",
                    "slug": "dated_model_20250101",
                    "response_model_aliases": ["normalized-model"],
                }
            ],
            "tasks": [
                {
                    "name": "task-a",
                    "datasets": {"label_only": "a.jsonl", "cot": "b.jsonl"},
                }
            ],
        }
        runs = expand_runs(config)
        self.assertEqual(runs[0].response_model_aliases, ("normalized-model",))

    def test_resume_only_accepts_matching_request_hash(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "responses.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "response",
                                "sample_id": "a",
                                "request_hash": "same",
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "response",
                                "sample_id": "b",
                                "request_hash": "old",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                set(read_completed_responses(path, request_hash="same")), {"a"}
            )

    def test_artifacts_use_shared_score_parser_and_metrics(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "test_cot.jsonl"
            run = ApiRun("fixed-model", "fixed_model", "rev_util_actionability", "cot", dataset)
            rows = [
                {"id": "a", "label": 1, "task": "rev_util", "aspect": "actionability"},
                {"id": "b", "label": 2, "task": "rev_util", "aspect": "actionability"},
            ]
            responses = {
                "a": {"content": "<score>1</score>", "response_model": "fixed-model"},
                "b": {"content": "invalid", "response_model": "fixed-model"},
            }
            resolved = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
            build_artifacts(
                run=run,
                rows=rows,
                score_sets=[1, 2],
                responses=responses,
                resolved=resolved,
                output_dir=root,
                elapsed_sec=1.0,
                complete_dataset=True,
            )
            metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["test_accuracy"], 0.5)
            self.assertEqual(metrics["format_valid_rate"], 0.5)

    def test_combined_report_does_not_modify_local_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            api = root / "api"
            local.mkdir()
            marker = local / "evaluation_analysis.md"
            marker.write_text("unchanged", encoding="utf-8")
            api_report, comparison = update_api_reports(api, local)
            self.assertTrue(api_report.is_file())
            self.assertTrue(comparison.is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
