from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from MoDE.factor_mix import (
    factor_mix_state_dict,
    l1_penalty,
    validate_adapter_compatibility,
)


def adapter_config() -> dict:
    return {
        "base_model_name_or_path": "base",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": 4,
        "r": 2,
        "task_type": "CAUSAL_LM",
        "target_modules": ["v_proj", "q_proj"],
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
    }


def toy_state(offset: float) -> dict[str, torch.Tensor]:
    return {
        "base.layer.q_proj.lora_A.weight": torch.tensor(
            [[1.0 + offset, 2.0], [3.0, 4.0 + offset]], dtype=torch.float32
        ),
        "base.layer.q_proj.lora_B.weight": torch.tensor(
            [[5.0, 6.0 + offset], [7.0 + offset, 8.0]], dtype=torch.float32
        ),
    }


class FactorMixTests(unittest.TestCase):
    def test_zero_one_hot_and_linear_mix(self) -> None:
        states = [toy_state(0.0), toy_state(10.0), toy_state(-2.0)]
        snapshots = [{key: value.clone() for key, value in state.items()} for state in states]

        zero = factor_mix_state_dict(states, [0.0, 0.0, 0.0])
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in zero.values()))

        one_hot = factor_mix_state_dict(states, [0.0, 1.0, 0.0])
        for key in states[1]:
            self.assertTrue(torch.equal(one_hot[key], states[1][key]))

        weights = [0.5, -0.25, 1.5]
        mixed = factor_mix_state_dict(states, weights)
        for key in mixed:
            expected = sum(weight * state[key] for weight, state in zip(weights, states))
            self.assertTrue(torch.allclose(mixed[key], expected))
        for state, snapshot in zip(states, snapshots):
            for key in state:
                self.assertTrue(torch.equal(state[key], snapshot[key]))

    def test_factor_mix_contains_cross_terms(self) -> None:
        states = [toy_state(0.0), toy_state(1.0)]
        weights = [0.4, 0.8]
        mixed = factor_mix_state_dict(states, weights)
        actual = (
            mixed["base.layer.q_proj.lora_B.weight"]
            @ mixed["base.layer.q_proj.lora_A.weight"]
        )
        delta_sum = sum(
            weight
            * state["base.layer.q_proj.lora_B.weight"]
            @ state["base.layer.q_proj.lora_A.weight"]
            for weight, state in zip(weights, states)
        )
        self.assertFalse(torch.allclose(actual, delta_sum))

    def test_invalid_weights_and_l1(self) -> None:
        with self.assertRaises(ValueError):
            factor_mix_state_dict([toy_state(0.0)], [math.nan])
        with self.assertRaises(ValueError):
            factor_mix_state_dict([toy_state(0.0)], [1.0, 2.0])
        self.assertAlmostEqual(l1_penalty([1.0, -2.0], alpha=0.05), 0.075)
        self.assertAlmostEqual(
            l1_penalty([1.0, -2.0], alpha=0.05, reduction="sum"), 0.15
        )

    def test_compatibility_ignores_target_module_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dirs = [root / "a", root / "b"]
            for index, directory in enumerate(dirs):
                directory.mkdir()
                config = adapter_config()
                if index == 1:
                    config["target_modules"] = list(reversed(config["target_modules"]))
                (directory / "adapter_config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
            result = validate_adapter_compatibility(dirs, [toy_state(0.0), toy_state(1.0)])
            self.assertEqual(result["adapter_count"], 2)
            self.assertEqual(result["tensor_count"], 2)


if __name__ == "__main__":
    unittest.main()
