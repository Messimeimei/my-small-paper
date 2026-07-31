from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from eval_analysis import extract_run_record, render_evaluation_analysis


def make_metrics(method: str, mode: str, accuracy: float) -> dict:
    exp_name = f"rev_util_actionability#qwen3_4b#ft#{method}#on_{mode}"
    return {
        "exp_name": exp_name,
        "task": "rev_util_actionability",
        "supervision_mode": mode,
        "adapter": f"train_outputs/rev_util_actionability/{method}/run/adapter",
        "finished_at_utc": "2026-07-31T00:00:00+00:00",
        "aggregate": {
            "samples": 10,
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
        },
        "full_config": {
            "train_config": {
                "experiment_name": f"rev_util_actionability#qwen3_4b#{method}"
            }
        },
    }


class EvalAnalysisPaperAlignTest(unittest.TestCase):
    def test_legacy_and_paper_align_records_use_distinct_conditions(self):
        inputs = (
            ("align", "cot", 0.51),
            ("paper_align", "cot", 0.61),
            ("paper_align", "label_only", 0.62),
        )
        records = {}
        for method, mode, accuracy in inputs:
            exp_name = f"rev_util_actionability#qwen3_4b#ft#{method}#on_{mode}"
            record = extract_run_record(
                make_metrics(method, mode, accuracy),
                Path("eval_output")
                / "rev_util_actionability"
                / exp_name
                / "metrics.json",
            )
            self.assertIsNotNone(record)
            assert record is not None
            records[(record["task"], record["condition"])] = record

        self.assertEqual(records[("rev_util_actionability", "AC")]["accuracy"], 0.51)
        self.assertEqual(records[("rev_util_actionability", "PAC")]["accuracy"], 0.61)
        self.assertEqual(records[("rev_util_actionability", "PAL")]["accuracy"], 0.62)

        rendered = render_evaluation_analysis(records)
        self.assertIn("| PAL | Paper Align SFT | Label-only |", rendered)
        self.assertIn("| PAC | Paper Align SFT | CoT |", rendered)
        self.assertIn(
            "| 任务 | N | B-L | B-C | LL | LC | CL | CC | AL | AC | PAL | PAC |",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
