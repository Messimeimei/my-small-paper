from __future__ import annotations

import math
import unittest

from MoDE.optimize_factor_mix import weights_from_record


class OptimizeFactorMixTests(unittest.TestCase):
    def test_weights_from_record_follows_expert_order(self) -> None:
        record = {"weights": {"second": -0.5, "first": 1.25}}
        self.assertEqual(
            weights_from_record(record, ["first", "second"]), [1.25, -0.5]
        )

        with self.assertRaises(ValueError):
            weights_from_record(record, ["first", "missing"])
        with self.assertRaises(ValueError):
            weights_from_record({"weights": {"first": math.nan}}, ["first"])


if __name__ == "__main__":
    unittest.main()
