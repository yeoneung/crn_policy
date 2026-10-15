"""Numerical stress checks for the finite-power extension, not a proof.

These diagnostics are separate from the frozen experiment records. The
analytical guarantees and their full proofs are in main.tex.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import norm

from certification.finite_power import (
    powers, perturbation_majorant, block_certificate, calibrate_matrix,
    calibration_budget,
)

ROOT = Path(__file__).resolve().parent
SEED = 20261015


def gate(matrix, radius, residual, cap, actuator=None):
    selected = block_certificate(matrix, radius, residual, cap, actuator=actuator)
    return ((selected.gain_per_displacement, selected.block_length, selected.margin)
            if selected is not None else None)


def main(output=None):
    rng = np.random.default_rng(SEED)
    matrix = np.array([[.5, 1.], [0., .5]])
    residual, target, cap = .001, .02, 4
    dimension, error, sigma, reset_scale = 2, .05, .005, 1.
    actual, q = perturbation_majorant(matrix, target, cap)
    total = float(q[:cap].sum())
    eta = float(1 - q[cap] - (residual + target) * total)
    logterm = np.log(8 / error)
    tail = np.sqrt(2 * np.log(4 / error))
    allocation_terms = [
        (12 * sigma * (dimension + np.sqrt(2 * logterm)) /
         (5 * reset_scale * (target - 2 * residual * np.sqrt(dimension)))) ** 2,
        9 * (np.sqrt(dimension) + tail) ** 2,
        16 * logterm / dimension,
    ]
    count = int(np.ceil(max(allocation_terms)))
    assert count == calibration_budget(dimension, target, residual, sigma,
                                       reset_scale, error)
    assert count == 173 and eta > 0
    assert round(actual[1], 4) == 1.2071
    assert round(total, 4) == 4.1781 and round(eta, 4) == .3196
    assert np.max(np.abs(np.linalg.eigvals(matrix))) < 1 < actual[1]

    # A stable matrix alone is insufficient when residual gain is too large.
    assert gate(.9 * np.eye(2), 0., .2, 30) is None

    # Perturb matrices in directions that need not commute with the baseline.
    perturbation_checks = 0
    for n in (2, 3, 5):
        for _ in range(100):
            transform = np.eye(n) + .45 * rng.normal(size=(n, n))
            if np.linalg.cond(transform) > 50:
                continue
            eigenvalues = rng.uniform(-.85, .85, size=n)
            base = transform @ np.diag(eigenvalues) @ np.linalg.inv(transform)
            accuracy = float(rng.uniform(.0001, .025))
            delta = rng.normal(size=(n, n))
            delta *= accuracy * rng.uniform(.2, 1) / np.linalg.norm(delta, 2)
            _, majorant = perturbation_majorant(base, accuracy, 16)
            for j, value in enumerate(powers(base + delta, 16)):
                assert np.linalg.norm(value, 2) <= majorant[j] + 1e-10
                perturbation_checks += 1

    # Stress the envelope with arbitrary time-varying residual directions,
    # rather than only the smooth residual used in the analytic example.
    trajectory_checks = 0
    maximum_fraction = 0.
    for actuator in (np.array([[0.], [1.]]), np.array([[1.], [0.]])):
        certified_gain = gate(matrix, target, residual, 12, actuator)[0]
        for _ in range(100):
            z = np.zeros(2)
            for _ in range(400):
                direction = rng.normal(size=2)
                direction /= np.linalg.norm(direction)
                perturb = (target + residual) * np.linalg.norm(z) * direction
                candidates = [matrix @ z + perturb + sign * actuator[:, 0]
                              for sign in (-1, 1)]
                z = max(candidates, key=np.linalg.norm)
                ratio = np.linalg.norm(z) / certified_gain
                assert ratio <= 1 + 1e-10
                maximum_fraction = max(maximum_fraction, float(ratio))
                trajectory_checks += 1

    coverage = useful = scalar = 0
    radii, selected_blocks = [], []
    repetitions = 1000
    for _ in range(repetitions):
        x = rng.normal(scale=reset_scale, size=(count, dimension))
        response = x @ matrix.T + residual * np.tanh(x)
        y1 = response + rng.normal(scale=sigma, size=response.shape)
        y2 = response + rng.normal(scale=sigma, size=response.shape)
        fit = calibrate_matrix(x, y1, y2, residual, error / 4)
        estimate, radius = fit.matrix, fit.error_radius
        coverage += np.linalg.norm(estimate - matrix, 2) <= radius
        scalar += np.linalg.norm(estimate, 2) + radius + residual < 1
        selected = gate(estimate, radius, residual, cap)
        useful += selected is not None
        if selected is not None:
            selected_blocks.append(selected[1])
        radii.append(float(radius))
    # These fixed-seed assertions check this illustrative family, not a
    # general probabilistic coverage theorem or a new return benchmark.
    assert scalar == 0 and useful == repetitions and coverage == repetitions

    # Check the sample-path time-two identity and its reward lower bound.
    gain_checks = 0
    for displacement in (.0125, .025, .05, .1):
        w0 = rng.normal(scale=sigma, size=(2000, 2))
        w1 = rng.normal(scale=sigma, size=(2000, 2))
        base_two = w0 @ matrix.T + residual * np.tanh(w0) + w1
        update_one = w0 + np.array([0., displacement])
        update_two = (update_one @ matrix.T + residual * np.tanh(update_one)
                      + np.array([0., displacement]) + w1)
        np.testing.assert_allclose(update_two[:, 0] - base_two[:, 0],
                                   displacement, atol=1e-14)
        inside = np.abs(base_two[:, 0]) <= 1
        reward_gain = np.tanh(update_two[inside, 0]) - np.tanh(base_two[inside, 0])
        assert np.all(reward_gain >= displacement / np.cosh(1.1) ** 2 - 1e-14)
        gain_checks += int(inside.sum())
    p0 = float(2 * norm.cdf((1 - residual) / (1.5 * sigma)) - 1)
    report = {
        'seed': SEED,
        'scope': 'New theory diagnostics; separate from frozen experiment records; not a proof.',
        'example': {
            'matrix': matrix.tolist(), 'matrix_norm': float(actual[1]),
            'spectral_radius': .5, 'block_cap': cap, 'error_target': target,
            'residual_lipschitz_bound': residual, 'majorant': q.tolist(),
            'S_star': total, 'eta_star': eta, 'Gamma_star': total / eta,
            'calibration_allocation_terms': [float(t) for t in allocation_terms],
            'replicated_resets': count, 'calibration_transitions': 2 * count,
            'gain_margin_at_discount_0_95': .95 ** 2 * p0 / np.cosh(1.1) ** 2,
        },
        'noncommuting_perturbation_inequalities_checked': perturbation_checks,
        'adversarial_residual_trajectory_steps_checked': trajectory_checks,
        'maximum_state_to_certified_bound_ratio': maximum_fraction,
        'replicated_gaussian_diagnostics': {
            'replications': repetitions, 'matrix_confidence_coverage_count': int(coverage),
            'block_certificate_count': int(useful), 'one_step_certificate_count': int(scalar),
            'selected_block_lengths': sorted(set(selected_blocks)),
            'radius_range': [min(radii), max(radii)],
        },
        'time_two_gain_inequalities_checked': gain_checks,
        'large_residual_false_certificate_rejected': True,
        'all_checks_passed': True,
    }
    output = Path(output) if output else ROOT / 'evidence/matrix_certificate_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + '\n')
    print(f'Finite-power audit passed: {perturbation_checks} perturbation inequalities, '
          f'{trajectory_checks} state steps, {repetitions} calibration replications.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
