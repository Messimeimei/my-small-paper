from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

TRAINING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training_methods import available_training_methods, resolve_training_method
from training_methods.sft_config import build_sft_config_kwargs
from training_workflow.run_lifecycle import PROJECT_ROOT as RESOLVED_PROJECT_ROOT
from training_workflow.training_pipeline import prepare_run_context


class ProjectLayoutTests(unittest.TestCase):
    def test_runtime_resolves_repository_root(self) -> None:
        self.assertEqual(RESOLVED_PROJECT_ROOT, PROJECT_ROOT)

    def test_directory_names_explain_their_responsibility(self) -> None:
        for directory in (
            "shared",
            "training_workflow",
            "training_methods",
            "custom_trainers",
            "evaluation",
        ):
            self.assertTrue((TRAINING_DIR / directory).is_dir(), directory)
        for obsolete in ("common", "runtime", "supervision", "trainers"):
            self.assertFalse((TRAINING_DIR / obsolete).exists(), obsolete)

    def test_training_method_registry_is_lightweight(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("trl", sys.modules)

    def test_builtin_training_methods_are_registered(self) -> None:
        self.assertEqual(
            set(available_training_methods()),
            {
                "standard",
                "legacy_align",
                "paper_align",
                "raft_without_cot",
                "cot_raft",
            },
        )

    def test_checkpoint_policy_is_shared_and_final_only(self) -> None:
        context = SimpleNamespace(
            config={
                "training": {
                    "bf16": True,
                    "report_to": "none",
                    "save_total_limit": 9,
                }
            },
            run_directory=Path("/run"),
            run_id="run-id",
            seed=43,
        )
        kwargs = build_sft_config_kwargs(context, pretokenized=True)
        self.assertEqual(kwargs["save_strategy"], "epoch")
        self.assertEqual(kwargs["save_total_limit"], 1)
        self.assertFalse(kwargs["load_best_model_at_end"])
        self.assertFalse(kwargs["completion_only_loss"])
        self.assertFalse(kwargs["remove_unused_columns"])
        self.assertEqual(kwargs["report_to"], [])
        self.assertEqual(kwargs["seed"], 43)

    def test_dataset_mode_constraints_are_declarative(self) -> None:
        self.assertEqual(resolve_training_method({}, "cot"), "standard")
        self.assertEqual(
            resolve_training_method(
                {"supervision": {"method": "cot_raft"}},
                "cot",
            ),
            "cot_raft",
        )
        with self.assertRaisesRegex(ValueError, "dataset supervision mode"):
            resolve_training_method(
                {"supervision": {"method": "cot_raft"}},
                "label_only",
            )

    def test_each_builtin_training_method_can_prepare_without_gpu(self) -> None:
        config_root = TRAINING_DIR / "configs" / "rw_gen_coherence"
        expected = {
            "cot.yaml": "standard",
            "label_only.yaml": "standard",
            "legacy_align.yaml": "legacy_align",
            "paper_align.yaml": "paper_align",
            "raft_without_cot.yaml": "raft_without_cot",
            "cot_raft.yaml": "cot_raft",
        }
        for filename, method in expected.items():
            with self.subTest(config=filename):
                context = prepare_run_context(
                    argparse.Namespace(
                        config=config_root / filename,
                        seed=42,
                        resume=None,
                        fresh=False,
                        dry_run=True,
                    )
                )
                self.assertEqual(context["training_method"], method)


if __name__ == "__main__":
    unittest.main()
