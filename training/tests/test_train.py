from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from training.train import generate_validation


class FakeBatch(dict):
    def to(self, device: torch.device) -> "FakeBatch":
        return self


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "prompt"

    def __call__(self, texts, **kwargs) -> FakeBatch:  # noqa: ANN001, ANN003
        return FakeBatch(
            input_ids=torch.tensor([[10, 11]] * len(texts)),
            attention_mask=torch.tensor([[1, 1]] * len(texts)),
        )

    def batch_decode(self, generated, **kwargs) -> list[str]:  # noqa: ANN001, ANN003
        return ["<score>1</score>" for _ in generated]


class FakeModel(torch.nn.Module):
    def __init__(self, *, fail_generation: bool = False) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(use_cache=False)
        self.gradient_checkpointing_enabled = True
        self.fail_generation = fail_generation

    def gradient_checkpointing_disable(self) -> None:
        raise AssertionError("generation validation must not disable gradient checkpointing")

    def generate(self, input_ids, **kwargs):  # noqa: ANN001, ANN003
        self.assert_generation_state()
        if self.fail_generation:
            raise RuntimeError("generation failed")
        generated = torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return torch.cat((input_ids, generated), dim=1)

    def assert_generation_state(self) -> None:
        if self.training:
            raise AssertionError("model must be in eval mode during generation")
        if not self.config.use_cache:
            raise AssertionError("KV cache must be enabled during generation")
        if not self.gradient_checkpointing_enabled:
            raise AssertionError("gradient checkpointing state must be preserved")


class GenerateValidationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [{"id": "sample-1", "label": 1, "prompt": [{"role": "user"}]}]

    def assert_training_state_restored(
        self, model: FakeModel, tokenizer: FakeTokenizer
    ) -> None:
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.gradient_checkpointing_enabled)
        self.assertEqual(tokenizer.padding_side, "right")

    def test_restores_training_state_after_generation(self) -> None:
        model = FakeModel()
        tokenizer = FakeTokenizer()

        metrics, predictions = generate_validation(
            model,
            tokenizer,
            self.rows,
            batch_size=1,
            max_length=32,
            max_new_tokens=4,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(predictions[0]["prediction"], 1)
        self.assert_training_state_restored(model, tokenizer)

    def test_restores_training_state_when_generation_fails(self) -> None:
        model = FakeModel(fail_generation=True)
        tokenizer = FakeTokenizer()

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            generate_validation(
                model,
                tokenizer,
                self.rows,
                batch_size=1,
                max_length=32,
                max_new_tokens=4,
            )

        self.assert_training_state_restored(model, tokenizer)


if __name__ == "__main__":
    unittest.main()
