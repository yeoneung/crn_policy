"""Check the example-specific reference against exact deterministic gains."""
import unittest

import numpy as np
from scipy.stats import chi2

from certification.finite_power import MatrixCalibration
from certification.model_reference import horizon_three_gain_interval


class ModelReferenceTests(unittest.TestCase):
    def fit(self, coefficient=1., error=.02, ell=.001, variance=0.):
        return MatrixCalibration(np.array([[.5, coefficient], [0., .5]]),
                                 error, ell, .0125, 173,
                                 (variance, variance), 'isotropic')

    def test_exact_deterministic_gain_inside_interval(self):
        # Actual h(x)=s*x2*e1 ranges over both residual-bound endpoints.
        for coefficient in (-1., 1.):
            for residual in (-.001, .001):
                for action in (-.1, -.0125, 0., .0125, .1):
                    report = horizon_three_gain_interval(self.fit(coefficient), action, .95, .0375)
                    exact = .95 ** 2 * np.tanh((coefficient + residual) * action)
                    self.assertLessEqual(report['gain_lower'], exact)
                    self.assertGreaterEqual(report['gain_upper'], exact)
                    self.assertEqual(report['adopted'], coefficient * action > 0)

    def test_scale_upper_and_second_moment(self):
        fit = self.fit(variance=.005 ** 2)
        report = horizon_three_gain_interval(fit, .05, .95, .0375)
        upper = 346 * .005 ** 2 / chi2.ppf(.0375, 346)
        self.assertAlmostEqual(report['noise_variance_upper'], upper)
        expected = upper * ((np.sqrt(1.25) + .02 + np.sqrt(2) * .001) ** 2 + 1)
        self.assertAlmostEqual(report['baseline_second_moment_upper'], expected)
        self.assertAlmostEqual(report['failure_probability'], .05)
        self.assertEqual(report['validation_pairs'], 0)

    def test_uncertain_sign_or_large_noise_does_not_certify(self):
        for fit in (self.fit(coefficient=0., error=.1), self.fit(variance=10.)):
            report = horizon_three_gain_interval(fit, .05, .95, .0375)
            self.assertFalse(report['adopted'])
            self.assertLessEqual(report['gain_lower'], 0)

    def test_wrong_model_assumptions_rejected(self):
        fit = MatrixCalibration(np.eye(3), .01, .001, .01, 173, (1., 1., 1.), 'full')
        with self.assertRaises(ValueError):
            horizon_three_gain_interval(fit, .05, .95, .04)
        for action, gamma, radius in [(np.nan, .95, 1), (.05, 1.1, 1), (.05, .95, 0)]:
            with self.assertRaises(ValueError):
                horizon_three_gain_interval(self.fit(), action, gamma, .04, radius)


if __name__ == '__main__':
    unittest.main()
