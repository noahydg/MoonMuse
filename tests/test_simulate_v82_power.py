from __future__ import annotations

import unittest

import numpy as np

from scripts.simulate_v82_power import (
    SimulationAssumptions,
    form_system_matrix,
    lower_t_bound,
    simulate,
)


class V82PowerSimulationTest(unittest.TestCase):
    def test_form_matrix_is_balanced_and_reversed(self):
        matrix = form_system_matrix()
        self.assertEqual(matrix.shape, (2, 24))
        self.assertTrue(np.all(matrix.sum(axis=1) == 12))
        self.assertTrue(np.all(matrix.sum(axis=0) == 1))
        for start in range(0, 24, 2):
            self.assertEqual(int(matrix[0, start : start + 2].sum()), 1)
            self.assertEqual(int(matrix[1, start : start + 2].sum()), 1)

    def test_lower_bound_moves_with_constant_shift(self):
        values = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(
            lower_t_bound(values + 0.5), lower_t_bound(values) + 0.5
        )

    def test_small_deterministic_simulation_has_complete_schema(self):
        report = simulate([48], 100, 123, SimulationAssumptions())
        power = report["estimated_power"]["48"]
        self.assertIn("overall_claim_pass", power)
        self.assertTrue(all(0 <= value <= 1 for value in power.values()))


if __name__ == "__main__":
    unittest.main()
