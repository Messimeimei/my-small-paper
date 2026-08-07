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
from evaluation.experiment_report import update_experiment_report


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

    def test_reports_show_incomplete_run_progress(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            api = root / "api"
            run_dir = (
                api
                / "rev_util_helpfulness"
                / "rev_util_helpfulness#fixed_model#api#greedy#on_label_only#snapshot"
            )
            local.mkdir()
            run_dir.mkdir(parents=True)
            (run_dir / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "model_name": "fixed-model",
                        "task": "rev_util_helpfulness",
                        "mode": "label_only",
                        "dataset_samples": 2,
                        "request_hash": "request",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "api_responses.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "response",
                                "sample_id": "a",
                                "request_hash": "request",
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "error",
                                "sample_id": "b",
                                "request_hash": "request",
                                "error_type": "RateLimitError",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            api_report, comparison = update_api_reports(api, local)
            for report in (api_report, comparison):
                content = report.read_text(encoding="utf-8")
                self.assertIn("1/2", content)
                self.assertIn("中断（RateLimitError）", content)


    def test_comparison_marks_best_values_and_includes_supplemental_summary(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            api = root / "api"
            supplemental = root / "supplemental"
            local.mkdir()
            for model, qwk, mae in (("model-a", 0.5, 0.4), ("model-b", 0.6, 0.5)):
                run_dir = api / model
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "backend": "openai-compatible-api",
                            "complete_dataset": True,
                            "task": "rev_util_actionability",
                            "supervision_mode": "cot",
                            "model_name": model,
                            "aggregate": {
                                "samples": 2,
                                "accuracy": qwk,
                                "macro_f1": qwk,
                                "qwk": qwk,
                                "mae": mae,
                                "pearson": qwk,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            supplemental_run = supplemental / "rev_util_verifiability" / "run"
            supplemental_run.mkdir(parents=True)
            (supplemental_run / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "model_name": "deepseek-v4-pro",
                        "task": "rev_util_verifiability",
                        "mode": "cot",
                        "dataset_samples": 788,
                        "request_hash": "request",
                    }
                ),
                encoding="utf-8",
            )
            _, comparison = update_api_reports(
                api,
                local,
                supplemental_api_roots=((supplemental, " (CoT 2048)"),),
            )
            content = comparison.read_text(encoding="utf-8")
            self.assertIn("| API | model-b | cot | 2 | **0.600**", content)
            self.assertIn("| API | model-a | cot | 2 | <u>0.500</u>", content)
            self.assertIn("**0.400**", content)
            self.assertIn("deepseek-v4-pro (CoT 2048)", content)
            self.assertIn("## 分指标七任务汇总", content)
            for metric_heading in (
                "### QWK 汇总",
                "### Accuracy (%) 汇总",
                "### Pearson 汇总",
                "### Macro-F1 汇总",
                "### MAE 汇总",
            ):
                self.assertIn(metric_heading, content)
            for task in (
                "Actionability",
                "Grounding Specificity",
                "Helpfulness",
                "Verifiability",
                "Coherence",
                "Positioning Check",
                "Positioning Type",
            ):
                self.assertIn(task, content)
            self.assertIn(
                "| API | deepseek-v4-pro (CoT 2048) | cot | 0/7 |",
                content,
            )


    def test_shared_api_root_disambiguates_cot_2048_and_updates_full_ledger(self) -> None:
        with TemporaryDirectory() as temporary:
            eval_root = Path(temporary) / "eval_output"
            local = eval_root / "results"
            api = eval_root / "api_results"
            local.mkdir(parents=True)
            for slug, max_tokens in (("deepseek_v4_pro", 512), ("deepseek_v4_pro_cot2048", 2048)):
                run_dir = (
                    api
                    / "rev_util_verifiability"
                    / f"rev_util_verifiability#{slug}#api#greedy#on_cot#snapshot"
                )
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "backend": "openai-compatible-api",
                            "complete_dataset": True,
                            "task": "rev_util_verifiability",
                            "supervision_mode": "cot",
                            "model_name": "deepseek-v4-pro",
                            "model_slug": slug,
                            "max_tokens": max_tokens,
                            "aggregate": {
                                "samples": 2,
                                "accuracy": 0.5,
                                "macro_f1": 0.4,
                                "qwk": 0.6,
                                "mae": 0.5,
                                "pearson": 0.7,
                                "format_valid_rate": 1.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            api_report, comparison = update_api_reports(api, local)
            for report in (api_report, comparison):
                content = report.read_text(encoding="utf-8")
                self.assertIn("deepseek-v4-pro (CoT 2048)", content)
                self.assertIn("deepseek-v4-pro", content)

            ledger = update_experiment_report(eval_root).read_text(encoding="utf-8")
            self.assertIn("Local records: 0; API records: 2; total: 2", ledger)
            self.assertIn("deepseek_v4_pro_cot2048", ledger)
            self.assertIn("| 2048 | complete |", ledger)


if __name__ == "__main__":
    unittest.main()
