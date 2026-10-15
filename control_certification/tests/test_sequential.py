import unittest

import numpy as np

from certification.sequential import mixture_test
from examples.finite_power_study import finite_horizon_support
from certification.finite_power import MatrixCalibration


class SequentialTests(unittest.TestCase):
    def test_stopping_matches_direct_products_at_inspections(self):
        values = np.full(400, .1)
        answer = mixture_test(values, .2, .05, 400, 16)
        stakes = 2. ** -np.arange(8)
        weights = np.array([.5] + [1 / 14] * 7)
        expected = next(n for n in range(16, 401, 16)
                        if np.dot(weights, (1 + .5 * stakes) ** n) >= 20)
        self.assertEqual(answer['pairs'], expected)
        self.assertTrue(answer['adopted'])

    def test_failure_paths_and_no_lookahead_past_adoption(self):
        self.assertFalse(mixture_test(np.zeros(100), 1, .05, 100)['adopted'])
        self.assertFalse(mixture_test(-np.ones(100), 1, .05, 100)['adopted'])
        bad = np.array([0., 2., *([1.] * 98)])
        self.assertEqual(mixture_test(bad, 1, .05, 100)['status'], 'support_violation')
        self.assertEqual(mixture_test(bad, 1, .05, 100)['pairs'], 2)
        early = np.ones(100)
        early[-1] = 2
        self.assertTrue(mixture_test(early, 1, .05, 100)['adopted'])

    def test_finite_horizon_envelope_has_exact_scalar_solution(self):
        fit = MatrixCalibration(np.array([[.5]]), .1, .05, .01, 100, (1.,), 'isotropic')
        h, d, gamma = 10, .2, .95
        distance = 0.; expected = 0.
        for t in range(h):
            expected += gamma ** t * distance
            distance = .65 * distance + 2 * d
        bound = finite_horizon_support(fit, d, h, gamma, np.array([[2.]]))
        self.assertAlmostEqual(bound, expected)


if __name__ == '__main__':
    unittest.main()
