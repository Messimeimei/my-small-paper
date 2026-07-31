from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from supervision.paper_align import (
    PaperAlignPairCollator,
    build_paired_align_dataset,
    validate_and_pair_rows,
)
from trainers.paper_align_trainer import PaperAlignGenerativeEvalSFTTrainer


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(
        self,
        prompt,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return "\n".join(message["content"] for message in prompt) + "\nASSISTANT:"

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [2 + (ord(char) % 89) for char in text]}


def make_rows():
    user = {"role": "user", "content": "shared context"}
    cot = {
        "id": "sample-1",
        "label": 5,
        "prompt": [
            {"role": "system", "content": "reason first, then score"},
            user,
        ],
        "completion": [
            {
                "role": "assistant",
                "content": (
                    "<reasoning>full rationale</reasoning>\n"
                    "<score>5</score>"
                ),
            }
        ],
    }
    label = {
        "id": "sample-1",
        "label": 5,
        "prompt": [
            {"role": "system", "content": "score only"},
            user,
        ],
        "completion": [
            {"role": "assistant", "content": "<score>5</score>"}
        ],
    }
    return cot, label


class PaperAlignPairingTest(unittest.TestCase):
    def test_pair_contract_uses_direct_and_full_reason_views(self):
        cot, label = make_rows()

        pairs = validate_and_pair_rows([cot], [label])

        self.assertEqual(pairs, [(label, cot)])
        self.assertNotEqual(label["prompt"][0], cot["prompt"][0])
        self.assertIn("<score>", label["completion"][0]["content"])
        self.assertNotIn("<reasoning>", label["completion"][0]["content"])
        self.assertIn("<reasoning>", cot["completion"][0]["content"])
        self.assertIn("<score>", cot["completion"][0]["content"])

    def test_one_source_item_collates_to_one_complete_view_pair(self):
        cot, label = make_rows()
        dataset = build_paired_align_dataset(
            [cot],
            [label],
            FakeTokenizer(),
            max_length=4096,
        )

        self.assertEqual(len(dataset), 1)
        batch = PaperAlignPairCollator(pad_token_id=0)([dataset[0]])

        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertGreater(batch["label_loss_mask"][0].sum().item(), 0)
        self.assertEqual(batch["rationale_loss_mask"][0].sum().item(), 0)
        self.assertEqual(batch["label_loss_mask"][1].sum().item(), 0)
        self.assertGreater(batch["rationale_loss_mask"][1].sum().item(), 0)
        self.assertGreater(
            batch["rationale_loss_mask"][1].sum().item(),
            batch["label_loss_mask"][0].sum().item(),
        )

    def test_collator_supports_unpaired_direct_validation_rows(self):
        features = [
            {
                "input_ids": [2, 3, 4],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 3, 4],
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [-100, 6],
            },
        ]

        batch = PaperAlignPairCollator(pad_token_id=0)(features)

        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertNotIn("label_loss_mask", batch)
        self.assertNotIn("rationale_loss_mask", batch)
        self.assertEqual(batch["input_ids"][1].tolist(), [5, 6, 0])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 6, -100])

    def test_loss_averages_each_view_before_weighting(self):
        vocab_size = 7
        labels = torch.tensor([[-100, 1, 2], [-100, 2, 3]])
        logits = torch.zeros((2, 3, vocab_size))
        logits[0, 0, 1] = 4
        logits[0, 1, 2] = 4
        logits[1, 0, 2] = 1
        logits[1, 1, 3] = 1

        class DummyModel:
            training = True

            def __call__(self, **inputs):
                return SimpleNamespace(logits=logits)

        trainer = object.__new__(PaperAlignGenerativeEvalSFTTrainer)
        trainer.label_coeff = 0.25
        trainer.rationale_coeff = 0.75
        trainer.log = lambda payload: None
        inputs = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "labels": labels,
            "label_loss_mask": torch.tensor([[0, 1, 1], [0, 0, 0]]),
            "rationale_loss_mask": torch.tensor([[0, 0, 0], [0, 1, 1]]),
        }

        actual = trainer.compute_loss(DummyModel(), inputs)
        direct = F.cross_entropy(logits[0, :2], labels[0, 1:])
        reason = F.cross_entropy(logits[1, :2], labels[1, 1:])
        expected = 0.25 * direct + 0.75 * reason

        torch.testing.assert_close(actual, expected)

    def test_rejects_mismatched_user_context(self):
        cot, label = make_rows()
        label["prompt"][1] = {"role": "user", "content": "different"}

        with self.assertRaisesRegex(ValueError, "User/context prompt mismatch"):
            validate_and_pair_rows([cot], [label])


if __name__ == "__main__":
    unittest.main()
