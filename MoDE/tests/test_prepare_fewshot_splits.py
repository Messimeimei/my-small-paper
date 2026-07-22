from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from MoDE.prepare_fewshot_splits import create_splits


def row(sample_id: str, label: int, prompt_text: str) -> dict:
    return {
        "id": sample_id,
        "task": "rw_gen",
        "aspect": "toy",
        "labels": label,
        "score_sets": [0, 1],
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt_text},
        ],
    }


class FewshotSplitTests(unittest.TestCase):
    def test_nested_balanced_splits_and_duplicate_exclusion(self) -> None:
        rows = []
        for label in (0, 1):
            rows.extend(
                row(f"label{label}_{index}", label, f"p-{label}-{index}")
                for index in range(7)
            )
        rows.extend(
            [
                row("duplicate_a", 0, "duplicate prompt"),
                row("duplicate_b", 0, "duplicate prompt"),
                row("leaked_test", 1, "training overlap"),
            ]
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps({"train": [], "test": rows}, ensure_ascii=False),
                encoding="utf-8",
            )
            training_source = root / "training.jsonl"
            training_row = row("train_leak", 1, "training overlap")
            training_source.write_text(
                json.dumps(training_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = root / "out"
            manifest = create_splits(
                sources={"toy": source},
                training_sources={"toy": training_source},
                output_dir=output,
                seed=42,
                shots_per_class=[1, 3, 5],
            )

            task = manifest["tasks"]["toy"]
            selected_sets = {}
            for shots in (1, 3, 5):
                split_meta = task["splits"][str(shots)]
                split_path = Path(split_meta["file"])
                if not split_path.is_absolute():
                    split_path = Path(__file__).resolve().parents[1] / split_path
                # Temporary outputs are outside MoDE, so the manifest stores absolute paths.
                payload = json.loads(split_path.read_text(encoding="utf-8"))
                self.assertEqual(len(payload["train"]), 2 * shots)
                clean_pool_count = len(rows) - 2
                self.assertEqual(len(payload["test"]), clean_pool_count - 2 * shots)
                calibration_counts = {
                    label: sum(item["labels"] == label for item in payload["train"])
                    for label in (0, 1)
                }
                self.assertEqual(calibration_counts, {0: shots, 1: shots})
                selected = {item["id"] for item in payload["train"]}
                selected_sets[shots] = selected
                self.assertFalse({"duplicate_a", "duplicate_b"} & selected)
                test_ids = {item["id"] for item in payload["test"]}
                self.assertIn("duplicate_a", test_ids)
                self.assertNotIn("duplicate_b", test_ids)
                self.assertNotIn("leaked_test", test_ids)
                self.assertIn(
                    f"cal{2 * shots}_test{clean_pool_count - 2 * shots}",
                    split_path.name,
                )

            self.assertLessEqual(selected_sets[1], selected_sets[3])
            self.assertLessEqual(selected_sets[3], selected_sets[5])
            self.assertEqual(
                task["excluded_duplicate_prompt_groups"],
                [["duplicate_a", "duplicate_b"]],
            )
            self.assertEqual(task["removed_duplicate_test_ids"], ["duplicate_b"])
            self.assertEqual(task["removed_training_overlap_ids"], ["leaked_test"])
            self.assertEqual(
                task["training_overlap_groups"][0]["training_ids"], ["train_leak"]
            )


if __name__ == "__main__":
    unittest.main()
