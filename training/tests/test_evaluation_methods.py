from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from evaluation.cli_config import parse_args
from evaluation.condition_labels import infer_eval_condition, resolve_eval_condition
from evaluation.methods import available_inference_modes, get_evaluation_method
from evaluation.report_generation import (
    render_method_average_table,
    update_evaluation_analysis,
)


def args(**overrides):
    values = {
        "max_tokens": 512,
        "enable_thinking": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class EvaluationMethodTests(unittest.TestCase):
    def test_builtin_methods_are_registered_without_torch(self) -> None:
        self.assertEqual(
            available_inference_modes(),
            ("greedy", "rail", "cot_rail"),
        )
        self.assertNotIn("torch", sys.modules)

    def test_method_metadata_is_centralized(self) -> None:
        greedy = get_evaluation_method("greedy")
        rail = get_evaluation_method("rail")
        cot_rail = get_evaluation_method("cot_rail")

        self.assertEqual(
            greedy.resolved_config_metadata(args())["effective_max_tokens"],
            512,
        )
        self.assertEqual(
            rail.resolved_config_metadata(args())["effective_max_tokens"],
            1,
        )
        self.assertEqual(
            cot_rail.resolved_config_metadata(args())["reasoning_max_tokens"],
            512,
        )
        with self.assertRaisesRegex(SystemExit, "requires --disable_thinking"):
            rail.validate(args(enable_thinking=True))

    def test_analysis_output_directory_is_created(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "new-results"
            analysis_path = update_evaluation_analysis(output_root)
            self.assertEqual(analysis_path, output_root / "evaluation_analysis.md")
            self.assertTrue(analysis_path.is_file())
            self.assertTrue(
                (output_root / "evaluation_analysis_records.json").is_file()
            )

    def test_analysis_keeps_qwen_base_and_scirm_rl_in_separate_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            task = "rev_util_actionability"
            for model_slug, model_name, accuracy in (
                ("qwen3_4b", "Qwen3-4B", 0.4),
                ("scirm_7b", "SciRM-7B", 0.6),
            ):
                exp_name = (
                    f"{task}#{model_slug}#base#greedy#on_label_only#seed_base"
                )
                result_dir = output_root / task / exp_name
                result_dir.mkdir(parents=True)
                (result_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "exp_name": exp_name,
                            "task": task,
                            "supervision_mode": "label_only",
                            "eval_condition": "B-L",
                            "model_name": model_name,
                            "test_accuracy": accuracy,
                            "test_qwk": accuracy,
                            "finished_at_utc": "2026-08-05T00:00:00+00:00",
                            "aggregate": {"samples": 10},
                        }
                    ),
                    encoding="utf-8",
                )

            analysis_path = update_evaluation_analysis(output_root)
            cache = json.loads(
                (output_root / "evaluation_analysis_records.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(cache["schema_version"], 4)
            self.assertEqual(
                [record["condition"] for record in cache["records"]],
                ["B-L", "SciRM-L"],
            )
            analysis = analysis_path.read_text(encoding="utf-8")
            self.assertIn("| B-L | Qwen3-4B Base |", analysis)
            self.assertIn("| SciRM-L | SciRM-7B RL |", analysis)
            self.assertIn("## 10. 各方法跨任务平均指标", analysis)

    def test_method_averages_weight_tasks_equally_after_averaging_seeds(self) -> None:
        records = {
            ("rev_util_actionability", "LL", "42"): {
                "train_seed": "42",
                "qwk": 0.2,
                "accuracy": 0.4,
                "pearson": 0.1,
                "macro_f1": 0.3,
                "mae": 0.8,
            },
            ("rev_util_actionability", "LL", "43"): {
                "train_seed": "43",
                "qwk": 0.4,
                "accuracy": 0.6,
                "pearson": 0.3,
                "macro_f1": 0.5,
                "mae": 0.6,
            },
            ("rev_util_grounding_specificity", "LL", "42"): {
                "train_seed": "42",
                "qwk": 0.7,
                "accuracy": 0.8,
                "pearson": 0.5,
                "macro_f1": 0.6,
                "mae": 0.2,
            },
            ("rw_gen_coherence", "LL", "42"): {
                "train_seed": "42",
                "accuracy": 0.9,
                "pearson": 0.7,
                "macro_f1": 0.8,
            },
        }

        table = render_method_average_table(records)

        self.assertIn(
            "| LL | Label-only SFT | Greedy | Label-only | 3/7 | "
            "0.500 (n=2) | 73.3 (n=3) | 0.467 (n=3) | "
            "0.600 (n=3) | 0.450 (n=2) |",
            table,
        )

    def test_method_options_are_forward_compatible(self) -> None:
        parsed = parse_args(
            [
                "--exp_name",
                "test",
                "--model_name",
                "/model",
                "--dataset_file",
                "/dataset.jsonl",
                "--method_options",
                '{"candidate_count": 4}',
            ]
        )
        self.assertEqual(parsed.method_options, {"candidate_count": 4})

    def test_greedy_prediction_record(self) -> None:
        method = get_evaluation_method("greedy")
        row = {"id": "x", "label": 1, "task": "task", "aspect": "aspect"}
        record = method.build_prediction_record(
            row,
            [
                {"prediction": 1, "output": "<score>1</score>"},
                {"prediction": None, "output": "invalid"},
            ],
        )
        self.assertEqual(record["rollout_correct"], [True, False])
        self.assertEqual(record["mean_correct"], 0.5)

    def test_rail_prediction_record(self) -> None:
        method = get_evaluation_method("rail")
        row = {"id": "x", "label": 1}
        prediction = {
            "prediction": 1,
            "output": "<score>0.8</score>",
            "raw_output": "1",
            "expected_score": 0.8,
            "score_probability_mass": 0.9,
            "score_probabilities": {"0": 0.1, "1": 0.8},
            "score_logprobs": {"0": -2.3, "1": -0.2},
        }
        record = method.build_prediction_record(row, [prediction])
        self.assertEqual(record["expected_score"], 0.8)
        self.assertFalse(record["candidate_renormalization"])

    def test_condition_resolution(self) -> None:
        self.assertEqual(
            infer_eval_condition(
                exp_name="task#qwen#ft#cot#greedy#on_label_only#seed_43",
                supervision_mode="label_only",
                adapter="/adapter",
            ),
            "CL",
        )
        self.assertEqual(
            infer_eval_condition(
                exp_name="task#scirm_7b#base#greedy#on_label_only#seed_base",
                supervision_mode="label_only",
                adapter=None,
            ),
            "SciRM-L",
        )
        self.assertEqual(
            infer_eval_condition(
                exp_name="task#scirm_7b#base#greedy#on_cot#seed_base",
                supervision_mode="cot",
                adapter=None,
            ),
            "SciRM-C",
        )
        self.assertEqual(
            resolve_eval_condition(
                exp_name="task#qwen#ft#cot_raft#rail#on_cot#seed_43",
                supervision_mode="cot",
                adapter="/adapter",
                train_config=None,
                training_method="cot_raft",
                inference_mode="cot_rail",
            ),
            "COT-RAFT-R",
        )


if __name__ == "__main__":
    unittest.main()
