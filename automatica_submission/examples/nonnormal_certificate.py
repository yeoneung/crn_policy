"""Fresh calibration and validation for the manuscript's analytic example.

Run from the package root: python -m examples.nonnormal_certificate
This example has a known simulator; the verifier sees calibration data and
public Lipschitz constants only. True-model quantities occur solely in the
Theorem 7 planning calculation and independent diagnostic checks.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from certification.finite_power import (
    block_certificate, calibrate_matrix, calibration_budget, discounted_length,
    fixed_pair_test, perturbation_majorant, return_support, validation_budget,
)


def run(displacement=.05, horizon=3, seed=20261015):
    if not 0 < displacement <= .1 or horizon < 3 or int(horizon) != horizon:
        raise ValueError('Analytic gain proof requires 0<d<=0.1 and integer H>=3')
    matrix = np.array([[.5, 1.], [0., .5]])
    ell, sigma, reset_scale = .001, .005, 1.
    target, cap, gamma = .02, 4, .95
    alpha, beta, delta = .05, .05, .05
    # Fixed allocations before observing calibration or validation data.
    nc = calibration_budget(2, target, ell, sigma, reset_scale, delta)
    _, q = perturbation_majorant(matrix, target, cap)
    total = float(q[:cap].sum())
    margin = float(1 - q[cap] - (ell + target) * total)
    if margin <= 0:
        raise ValueError('Theorem 7 analysis margin is not positive')
    coefficient = discounted_length(horizon, gamma) * total / margin
    p0 = float(2 * norm.cdf((1 - ell) / (1.5 * sigma)) - 1)
    m = float(gamma ** 2 * p0 / np.cosh(1.1) ** 2)
    nv = validation_budget(coefficient, m, alpha, beta)
    calibration_seed, validation_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(calibration_seed)
    x = rng.normal(scale=reset_scale, size=(nc, 2))
    mean_response = x @ matrix.T + ell * np.tanh(x)
    y1 = mean_response + rng.normal(scale=sigma, size=x.shape)
    y2 = mean_response + rng.normal(scale=sigma, size=x.shape)
    # Estimation/calibration gets neither true M nor noise sigma.
    fit = calibrate_matrix(x, y1, y2, ell, delta / 4)
    cert = block_certificate(fit.matrix, fit.error_radius, ell, cap)
    report = {
        'scope': 'Separate new-theorem example; not a rerun of frozen manuscript experiments.',
        'seed': seed, 'displacement': displacement, 'horizon': horizon,
        'discount': gamma, 'alpha': alpha, 'beta': beta, 'delta': delta,
        'planned_resets': nc, 'planned_validation_pairs': nv,
        'analysis': {'S_star': total, 'eta_star': margin,
                     'support_coefficient': coefficient, 'gain_margin': m,
                     'true_matrix_norm': float(np.linalg.norm(matrix, 2))},
        'calibration': {'matrix_estimate': fit.matrix.tolist(),
                        'error_radius': fit.error_radius,
                        'failure_probability': fit.failure_probability,
                        'one_step_bound': fit.one_step_bound},
        'certificate': asdict(cert) if cert else None,
        'used_calibration_transitions': 2 * nc,
        'used_validation_transitions': 0,
    }
    if cert is None:
        report.update(adopted=False, decision='retain baseline: no block certified')
        return report
    support = return_support(cert, displacement, horizon, gamma,
                             equal_state_reward_lip=1.)
    rng = np.random.default_rng(validation_seed)
    base, candidate = np.zeros((nv, 2)), np.zeros((nv, 2))
    differences = np.zeros(nv)
    max_state_ratio = 0.
    # Reward at t precedes transition t; exactly H transitions per trajectory.
    # Each pair has shared noise, and distinct pairs have independent noises.
    for t in range(horizon):
        differences += gamma ** t * (np.tanh(candidate[:, 0]) - np.tanh(base[:, 0]))
        noise = rng.normal(scale=sigma, size=base.shape)
        base = base @ matrix.T + ell * np.tanh(base) + noise
        candidate = candidate @ matrix.T + ell * np.tanh(candidate) + noise
        candidate[:, 1] += displacement
        ratio = np.max(np.linalg.norm(candidate - base, axis=1)) / (displacement * cert.gain_per_displacement)
        max_state_ratio = max(max_state_ratio, float(ratio))
    decision = fixed_pair_test(differences, support, alpha, nv)
    report.update(decision)
    report.update(
        used_validation_transitions=2 * horizon * nv,
        total_used_transitions=2 * nc + 2 * horizon * nv,
        maximum_state_to_bound_ratio=max_state_ratio,
        observed_gain_per_displacement=decision['mean_difference'] / displacement,
        false_adoption_bound=alpha + delta / 4,
        theorem_power_lower_bound=1 - beta - delta,
        diagnostic_true_error_within_radius=bool(np.linalg.norm(fit.matrix - matrix, 2) <= fit.error_radius),
        diagnostic_radius_within_target=fit.error_radius <= target,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--displacement', type=float, default=.05)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--output', type=Path, default=Path('evidence/finite_power_example.json'))
    args = parser.parse_args()
    report = run(args.displacement, args.horizon, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['planned_resets', 'planned_validation_pairs',
                                           'adopted', 'used_validation_transitions']}))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
