from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from MoDE.prepare_fewshot_splits import create_splits, sha256_file


def row(sample_id: str, label: int, prompt_text: str, *, completion: bool) -> dict:
    result = {
        "id": sample_id,
        "task": "rw_gen",
        "aspect": "toy",
        "label": label,
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": prompt_text},
        ],
    }
    if completion:
        result["completion"] = [
            {
                "role": "assistant",
                "content": f"<reasoning>teacher</reasoning><score>{label}</score>",
            }
        ]
    else:
        result["labels"] = result.pop("label")
        result["score_sets"] = [0, 1]
    return result


class FewshotSplitTests(unittest.TestCase):
    def test_validation_calibration_and_independent_clean_test(self) -> None:
        train_rows = [
            row("train_base", 0, "train base", completion=True),
            row("train_test_leak", 0, "leaks into test", completion=True),
            row("train_validation_leak", 1, "leaks into validation", completion=True),
        ]
        validation_rows = [
            row(f"validation_{label}_{index}", label, f"v-{label}-{index}", completion=True)
            for label in (0, 1)
            for index in range(7)
        ]
        validation_rows.extend(
            [
                row(
                    "validation_train_overlap",
                    1,
                    "leaks into validation",
                    completion=True,
                ),
                row(
                    "validation_test_overlap",
                    0,
                    "also present in test",
                    completion=True,
                ),
            ]
        )
        dataset_rows = train_rows + validation_rows
        test_rows = [
            row("test_clean", 1, "clean test", completion=False),
            row("test_duplicate_a", 0, "duplicate test", completion=False),
            row("test_duplicate_b", 0, "duplicate test", completion=False),
            row("test_train_leak", 0, "leaks into test", completion=False),
            row("test_validation_overlap", 0, "also present in test", completion=False),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                "".join(json.dumps(item) + "\n" for item in dataset_rows),
                encoding="utf-8",
            )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_sha256": sha256_file(dataset),
                        "split_seed": 7,
                        "validation_ratio": 0.5,
                        "train_ids": [item["id"] for item in train_rows],
                        "validation_ids": [item["id"] for item in validation_rows],
                    }
                ),
                encoding="utf-8",
            )
            test_path = root / "test.json"
            test_path.write_text(
                json.dumps({"train": [], "test": test_rows}), encoding="utf-8"
            )
            output = root / "out"
            manifest = create_splits(
                task_sources={
                    "toy": {
                        "dataset": dataset,
                        "split": split_path,
                        "test": test_path,
                    }
                },
                output_dir=output,
                seed=42,
                shots_per_class=[1, 3, 5],
            )

            task = manifest["tasks"]["toy"]
            self.assertEqual(
                task["excluded_validation_ids"],
                ["validation_test_overlap", "validation_train_overlap"],
            )
            clean_test_path = Path(task["clean_test"]["file"])
            clean_test = json.loads(clean_test_path.read_text(encoding="utf-8"))
            clean_test_ids = {item["id"] for item in clean_test["test"]}
            self.assertEqual(
                clean_test_ids,
                {"test_clean", "test_duplicate_a", "test_validation_overlap"},
            )
            self.assertEqual(
                task["clean_test"]["removed_training_overlap_ids"],
                ["test_train_leak"],
            )
            self.assertEqual(
                task["clean_test"]["removed_duplicate_test_ids"],
                ["test_duplicate_b"],
            )

            selected_sets = {}
            for shots in (1, 3, 5):
                split_meta = task["splits"][str(shots)]
                payload = json.loads(
                    Path(split_meta["file"]).read_text(encoding="utf-8")
                )
                calibration = payload["train"]
                self.assertNotIn("test", payload)
                self.assertEqual(len(calibration), 2 * shots)
                self.assertTrue(all(item.get("completion") for item in calibration))
                self.assertEqual(
                    {
                        label: sum(item["label"] == label for item in calibration)
                        for label in (0, 1)
                    },
                    {0: shots, 1: shots},
                )
                selected_sets[shots] = {item["id"] for item in calibration}
                self.assertNotIn("validation_train_overlap", selected_sets[shots])
                self.assertNotIn("validation_test_overlap", selected_sets[shots])
                self.assertEqual(split_meta["test_file"], task["clean_test"]["file"])
                self.assertIn(
                    f"validation_k{shots}_per_class_cal{2 * shots}",
                    Path(split_meta["file"]).name,
                )

            self.assertLessEqual(selected_sets[1], selected_sets[3])
            self.assertLessEqual(selected_sets[3], selected_sets[5])


if __name__ == "__main__":
    unittest.main()
