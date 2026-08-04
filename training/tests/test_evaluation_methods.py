from __future__ import annotations

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
from evaluation.report_generation import update_evaluation_analysis


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
