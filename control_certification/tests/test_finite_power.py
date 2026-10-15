"""Regression, independent formula checks, and failure paths for new code."""
import unittest

import numpy as np
from scipy.stats import f

from certification.finite_power import (
    block_certificate, calibrate_matrix, calibration_budget, discounted_length,
    fixed_pair_test, perturbation_majorant, return_support, validation_budget,
)


class FinitePowerTests(unittest.TestCase):
    def test_scalar_linear_known_exact_envelope(self):
        # For x'=a*x+u, the l-infinity induced gain is exactly 1/(1-a).
        a = .7
        cert = block_certificate([[a]], 0, 0, 12, action_gain=2)
        self.assertAlmostEqual(cert.gain_per_displacement, 2 / (1 - a))
        z = 0.
        for _ in range(300):
            z = a * z + 2
            self.assertLessEqual(z, cert.gain_per_displacement + 1e-12)

    def test_nonnormal_baseline_and_analysis_constants(self):
        matrix = np.array([[.5, 1], [0, .5]])
        self.assertGreater(np.linalg.norm(matrix, 2), 1)
        self.assertIsNone(block_certificate(matrix, .02, .001, 1))
        cert = block_certificate(matrix, .02, .001, 4)
        self.assertIsNotNone(cert)
        self.assertEqual(cert.block_length, 4)
        _, q = perturbation_majorant(matrix, .02, 4)
        self.assertAlmostEqual(q[:4].sum(), 4.178052097866157)
        eta = 1 - q[4] - .021 * q[:4].sum()
        self.assertAlmostEqual(eta, .31956341628312135)
        self.assertLessEqual(cert.gain_per_displacement, q[:4].sum() / eta)
        self.assertEqual(calibration_budget(2, .02, .001, .005, 1, .05), 173)

    def test_search_minimizes_all_feasible_blocks(self):
        matrix = np.array([[.5, 1], [0, .5]])
        actuator = np.array([[0.], [2.]])
        for image in (None, actuator):
            answers = []
            p = np.eye(2)
            s = force = 0.
            for k in range(1, 17):
                s += np.linalg.norm(p, 2)
                force += np.linalg.norm(p @ image, 2) if image is not None else np.linalg.norm(p, 2)
                p = p @ matrix
                eta = 1 - np.linalg.norm(p, 2) - .021 * s
                if eta > 0:
                    answers.append((force / eta, k))
            result = block_certificate(matrix, .02, .001, 16, actuator=image)
            self.assertAlmostEqual(result.gain_per_displacement, min(answers)[0])
            self.assertEqual(result.block_length, min(answers)[1])

    def test_residual_and_roundoff_margin_rejected(self):
        self.assertIsNone(block_certificate(.9 * np.eye(2), 0, .2, 30))
        self.assertIsNone(block_certificate([[1 - 1e-15]], 0, 0, 1))
        self.assertIsNone(block_certificate([[1.1]], 0, 0, 10))

    def test_calibration_against_independent_orthogonal_design_formula(self):
        # Orthogonal, fixed reset design makes each pivot's scale explicit.
        x = np.vstack((np.eye(2), -np.eye(2)))
        matrix = np.array([[.2, .1], [0, .3]])
        v = np.array([[.1, .2], [.2, -.1], [-.3, .1], [.1, -.2]])
        y = x @ matrix.T
        delta, ell = .07, .01
        iso = calibrate_matrix(x, y + v, y - v, ell, delta)
        np.testing.assert_allclose(iso.matrix, matrix, atol=1e-15)
        variance = np.sum((2 * v) ** 2) / 16
        expected = np.sqrt(variance * f.ppf(1 - delta, 4, 8)) + ell * np.sqrt(2)
        self.assertAlmostEqual(iso.error_radius, expected)
        full = calibrate_matrix(x, y + v, y - v, ell, delta, 'full')
        vars_sum = np.sum((2 * v) ** 2) / 8
        expected_full = np.sqrt(2 * f.ppf(1 - delta / 2, 2, 4) * vars_sum / 4) + ell * np.sqrt(2)
        self.assertAlmostEqual(full.error_radius, expected_full)

    def test_return_formula_discount_and_displacement(self):
        cert = block_certificate([[.5]], 0, 0, 2, action_gain=3)
        kwargs = dict(horizon=7, gamma=.95, reward_state_lip=2,
                      reward_action_lip=4, baseline_lip=5)
        h = sum(.95 ** t for t in range(7))
        b = return_support(cert, .1, **kwargs)
        self.assertAlmostEqual(b, .1 * h * (4 + (2 + 4 * 5) * 6))
        self.assertAlmostEqual(return_support(cert, .01, **kwargs), b / 10)
        self.assertAlmostEqual(return_support(cert, .1, 7, .95,
                                              equal_state_reward_lip=2), .1 * h * 2 * 6)
        self.assertEqual(return_support(cert, .1, ordinary_bound=1, **kwargs), 1)
        self.assertAlmostEqual(discounted_length(None, .95), 20)
        self.assertEqual(discounted_length(7, 1), 7)

    def test_fixed_validation_decision_and_support_violation(self):
        result = fixed_pair_test(np.full(500, .1), .2, .05, 500)
        self.assertTrue(result['adopted'])
        self.assertAlmostEqual(result['threshold'], .2 * np.sqrt(2 * np.log(20) / 500))
        self.assertFalse(fixed_pair_test(np.zeros(500), .2, .05, 500)['adopted'])
        self.assertFalse(fixed_pair_test(np.full(500, -.1), .2, .05, 500)['adopted'])
        self.assertFalse(fixed_pair_test(np.zeros(4), 0, .05, 4)['adopted'])
        for data, n in [([.21], 1), ([float('nan')], 1), ([.1, .1], 1)]:
            with self.assertRaises(ValueError):
                fixed_pair_test(data, .2, .05, n)

    def test_validation_power_allocation_is_displacement_independent(self):
        c, m, alpha, beta = 3., .4, .05, .1
        n = validation_budget(c, m, alpha, beta)
        for d in (.1, .01, .001):
            self.assertLessEqual(c * d * (np.sqrt(2 * -np.log(alpha) / n)
                                         + np.sqrt(2 * -np.log(beta) / n)), m * d)
        self.assertGreater(c * (np.sqrt(2 * -np.log(alpha) / (n - 1))
                                + np.sqrt(2 * -np.log(beta) / (n - 1))), m)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            calibrate_matrix(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2)), 0, .05)
        for kwargs in [dict(matrix=[[float('nan')]]), dict(matrix=[[.5]], matrix_error=-1),
                       dict(matrix=[[.5]], cap=0), dict(matrix=[[.5]], cap=1.5)]:
            inputs = dict(matrix=[[.5]], matrix_error=0, residual_lip=0, cap=3)
            inputs.update(kwargs)
            with self.assertRaises(ValueError):
                block_certificate(**inputs)
        with self.assertRaises(ValueError):
            block_certificate([[.5]], 0, 0, 3, action_gain=2, actuator=[[1]])
        with self.assertRaises(ValueError):
            discounted_length(None, 1)
        with self.assertRaises(ValueError):
            return_support(None, .1, 5, .95)
        with self.assertRaises(ValueError):
            calibration_budget(2, .001, .001, .005, 1, .05)


if __name__ == '__main__':
    unittest.main()
