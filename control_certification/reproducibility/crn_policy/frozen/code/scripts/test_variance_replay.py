import unittest
import numpy as np
from variance_replay import log_paths, stop
from return_certificate import log_wealth


class ReplayTests(unittest.TestCase):
    def test_mixture_matches_original_each_prefix_and_endpoint(self):
        d = np.array([-.7, .3, 1, -1, .8, 0.])
        got = log_paths(d, 1, .05, "fixed_mixture")
        for n in range(1, len(d)+1):
            self.assertAlmostEqual(got[n-1], log_wealth(d[:n], 1), places=12)

    def test_bernstein_matches_independent_scalar_recursion(self):
        d = np.random.default_rng(44).uniform(-.3, .3, 2000)
        acc, total, residual, previous_mean, previous_variance = 0., 0., .25, .5, .25
        expected = []
        for i, value in enumerate(d, 1):
            x = (value+.3)/.6
            lam = min(.75, np.sqrt(2*np.log(40)/(previous_variance*i*np.log1p(i))))
            acc += lam*(x-.5) - (x-previous_mean)**2*(-np.log1p(-lam)-lam)
            expected.append(acc)
            total += x
            current_mean = (.5+total)/(i+1)
            residual += (x-current_mean)**2
            previous_mean, previous_variance = current_mean, residual/(i+1)
        np.testing.assert_allclose(log_paths(d, .3, .05, "predictable_empirical_bernstein"), expected, rtol=1e-12, atol=1e-12)

    def test_predictability_and_suffix_invariance(self):
        rng = np.random.default_rng(7)
        a = rng.uniform(-1, 1, 300)
        b = a.copy(); b[100:] = 1
        for method in ("fixed_mixture", "predictable_empirical_bernstein"):
            np.testing.assert_array_equal(log_paths(a, 1, .05, method)[:100], log_paths(b, 1, .05, method)[:100])

    def test_support_violation_is_an_error(self):
        for method in ("fixed_mixture", "predictable_empirical_bernstein"):
            with self.assertRaises(ValueError): log_paths([1.01], 1, .05, method)
            with self.assertRaises(ValueError): log_paths([np.nan], 1, .05, method)

    def test_constant_law_cannot_beat_uniform_range_lower_bound(self):
        for mu in (.02, .1, .5):
            bound = np.log(1/.05)/np.log1p(mu)
            for method in ("fixed_mixture", "predictable_empirical_bernstein"):
                result = stop(np.full(2000, mu), 1, .05, method, 2000, block=1)
                self.assertTrue(result["adopted"])
                self.assertGreaterEqual(result["pairs"], bound)

    def test_stochastic_example_has_actual_noise_cancellation(self):
        w = np.random.default_rng(98).uniform(-1, 1, 10000)
        for d in (.005, .05, .5):
            theta = 2*np.arcsin(d/2)
            b = np.clip(w, -2, 2)
            p = np.clip(w+np.sin(theta), -2, 2)
            np.testing.assert_allclose(p-b, d*np.sqrt(1-d*d/4), rtol=1e-12)
            self.assertGreater(np.var(b), .3)


if __name__ == "__main__": unittest.main()
