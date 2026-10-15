"""Corollary 5, Proposition 6 and Theorem 7, in Euclidean/operator 2-norm.

Assumptions are mathematical inputs, not properties inferred from the data:
F_B(x)=Mx+h(x), h(0)=0, global Lip(h)<=ell, a globally enforced action
displacement <=d, and valid global action/reward constants. Calibration uses
independent Gaussian replicate noises; validation uses fresh independent
trajectory pairs with shared noise *within* each pair and a frozen policy.
The numerical implementation uses double precision, not interval arithmetic.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import f


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _probability(value, name):
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f'{name} must be in (0,1)')
    return value


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _matrix(value):
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or not out.shape[0] or out.shape[0] != out.shape[1]:
        raise ValueError('Expected a nonempty square matrix')
    if not np.all(np.isfinite(out)):
        raise ValueError('Matrix must be finite')
    return out


def powers(matrix, cap):
    matrix = _matrix(matrix)
    cap = _count(cap, 'cap')
    out = [np.eye(len(matrix))]
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(cap):
            product = out[-1] @ matrix
            if not np.all(np.isfinite(product)):
                raise ValueError('Matrix powers overflowed; no certificate computed')
            out.append(product)
    return out


@dataclass(frozen=True)
class MatrixCalibration:
    matrix: np.ndarray
    error_radius: float
    residual_lip: float
    failure_probability: float
    samples: int
    noise_variances: tuple
    noise_model: str

    @property
    def one_step_bound(self):
        return float(np.linalg.norm(self.matrix, 2) + self.error_radius + self.residual_lip)


def calibrate_matrix(x, y1, y2, residual_lip, failure_probability,
                     noise_model='isotropic'):
    """Replicated-reset fit; does not accept the true matrix or noise scale.

    Rows are reset states/next states. The two responses at each reset have
    independent noises, also independent across resets and of reset states.
    'isotropic' assumes unknown positive sigma^2 I (Proposition 6).
    'full' permits unknown positive definite covariance (Appendix A.4);
    its coordinatewise pivots are combined by a union bound.
    """
    ell = _nonnegative(residual_lip, 'residual_lip')
    delta = _probability(failure_probability, 'failure_probability')
    x, y1, y2 = (np.asarray(v, dtype=float) for v in (x, y1, y2))
    if x.ndim != 2 or not all(x.shape) or y1.shape != x.shape or y2.shape != x.shape:
        raise ValueError('Expected three arrays of shape (N,n)')
    if not all(np.all(np.isfinite(v)) for v in (x, y1, y2)):
        raise ValueError('Observations must be finite')
    count, n = x.shape
    if count < n or np.linalg.matrix_rank(x) < n:
        raise ValueError('Full-column-rank reset states are required')
    gram = x.T @ x
    smallest = float(np.linalg.eigvalsh(gram)[0])
    if not smallest > 0:
        raise ValueError('Reset Gram matrix is not numerically positive definite')
    estimate = np.linalg.solve(gram, x.T @ ((y1 + y2) / 2)).T
    variances = np.sum((y1 - y2) ** 2, axis=0) / (2 * count)
    if noise_model == 'isotropic':
        quantile = f.isf(delta, n * n, n * count)
        stochastic = np.sqrt(variances.mean() * n * n * quantile / (2 * smallest))
    elif noise_model == 'full':
        quantile = f.isf(delta / n, n, count)
        stochastic = np.sqrt(n * quantile * variances.sum() / (2 * smallest))
    else:
        raise ValueError("noise_model must be 'isotropic' or 'full'")
    radius = float(stochastic + ell * np.sqrt(np.trace(gram) / smallest))
    if not np.isfinite(radius) or not np.all(np.isfinite(estimate)):
        raise ValueError('Nonfinite fit or confidence radius')
    estimate.setflags(write=False)
    return MatrixCalibration(estimate, radius, ell, delta, count,
                             tuple(float(v) for v in variances), noise_model)


@dataclass(frozen=True)
class PowerCertificate:
    gain_per_displacement: float
    block_length: int
    margin: float
    power_sum: float
    actuator_aware: bool


def block_certificate(matrix, matrix_error, residual_lip, cap,
                      action_gain=1., actuator=None):
    """Uniform state bound d*gain_per_displacement; None means retain B.

    Search all lengths 1..cap on the SAME matrix-confidence event. With a
    known forcing image G, supply actuator=G for forces G*v, ||v||<=d;
    action_gain must then be its default 1 (G already contains its scale).
    Otherwise action_gain is the global action-to-state Lipschitz bound.
    """
    uncertainty = (_nonnegative(matrix_error, 'matrix_error')
                   + _nonnegative(residual_lip, 'residual_lip'))
    action_gain = _nonnegative(action_gain, 'action_gain')
    pp = powers(matrix, cap)
    sizes = np.array([np.linalg.norm(a, 2) for a in pp])
    if actuator is None:
        forcing = action_gain * sizes
    else:
        actuator = np.asarray(actuator, dtype=float)
        if (actuator.ndim != 2 or actuator.shape[0] != len(pp[0])
                or not actuator.shape[1] or not np.all(np.isfinite(actuator))):
            raise ValueError('Actuator must be a finite n-by-m matrix')
        if action_gain != 1.:
            raise ValueError('Supply either action_gain or a scaled actuator image')
        forcing = np.array([np.linalg.norm(a @ actuator, 2) for a in pp])
    eligible = []
    for k in range(1, len(pp)):
        total = float(sizes[:k].sum())
        margin = float(1 - sizes[k] - uncertainty * total)
        # Reject near-zero margins to avoid rounding-triggered acceptance.
        guard = 64 * np.finfo(float).eps * (1 + sizes[k] + uncertainty * total)
        if np.isfinite(margin) and margin > guard:
            gain = float(forcing[:k].sum() / margin)
            if np.isfinite(gain):
                eligible.append(PowerCertificate(gain, k, margin, total,
                                                  actuator is not None))
    return min(eligible, key=lambda v: (v.gain_per_displacement, v.block_length)) if eligible else None


def discounted_length(horizon, gamma):
    gamma = _nonnegative(gamma, 'gamma')
    if gamma > 1 or (horizon is None and gamma == 1):
        raise ValueError('Require gamma<=1, and gamma<1 for infinite horizon')
    if horizon is None:
        return 1 / (1 - gamma)
    horizon = _count(horizon, 'horizon')
    if gamma == 1:
        return float(horizon)
    if gamma == 0:
        return 1.
    return float(-np.expm1(horizon * np.log(gamma)) / (1 - gamma))


def return_support(certificate, displacement, horizon, gamma,
                   reward_state_lip=0., reward_action_lip=0., baseline_lip=0.,
                   equal_state_reward_lip=None, ordinary_bound=None):
    """Corollary 5 support; optional independently valid ordinary bound.

    equal_state_reward_lip asserts the manuscript's pointwise reward
    identity, not merely equality in distribution. Passing None uses the
    general state/action Lipschitz formula. This function never clips data.
    """
    if certificate is None:
        raise ValueError('No finite-power certificate; retain the baseline')
    d = _nonnegative(displacement, 'displacement')
    lx = _nonnegative(reward_state_lip, 'reward_state_lip')
    lu = _nonnegative(reward_action_lip, 'reward_action_lip')
    lb = _nonnegative(baseline_lip, 'baseline_lip')
    if equal_state_reward_lip is None:
        coefficient = lu + (lx + lu * lb) * certificate.gain_per_displacement
    else:
        coefficient = (_nonnegative(equal_state_reward_lip, 'equal_state_reward_lip')
                       * certificate.gain_per_displacement)
    bound = d * discounted_length(horizon, gamma) * coefficient
    if not np.isfinite(bound):
        raise ValueError('Return bound overflowed')
    return min(bound, _nonnegative(ordinary_bound, 'ordinary_bound')) if ordinary_bound is not None else bound


def perturbation_majorant(matrix, target, cap):
    """Theorem 7 analysis tool; true M is not an estimator input."""
    target = _nonnegative(target, 'target')
    actual = np.array([np.linalg.norm(a, 2) for a in powers(matrix, cap)])
    q = np.ones(len(actual))
    for j in range(1, len(q)):
        q[j] = actual[j] + target * np.dot(actual[:j][::-1], q[:j])
    if not np.all(np.isfinite(q)):
        raise ValueError('Perturbation majorant overflowed')
    return actual, q


def calibration_budget(dimension, target, residual_lip, sigma, reset_scale, delta):
    """Explicit Theorem 7 sufficient N_c under isotropic Gaussian resets.

    sigma is a true/externally justified scale for *analysis/planning*;
    the matrix estimator itself does not know sigma. The displayed theorem
    uses 0<delta<1/2 and calibrates at failure probability delta/4.
    """
    n = _count(dimension, 'dimension')
    target = _nonnegative(target, 'target')
    ell = _nonnegative(residual_lip, 'residual_lip')
    sigma = _nonnegative(sigma, 'sigma')
    scale = _nonnegative(reset_scale, 'reset_scale')
    delta = _probability(delta, 'delta')
    if not delta < .5 or sigma == 0 or scale == 0 or target <= 2 * ell * np.sqrt(n):
        raise ValueError('Require delta<1/2, positive scales, target>2*ell*sqrt(n)')
    logterm = np.log(8 / delta)
    terms = ((12 * sigma * (n + np.sqrt(2 * logterm)) /
              (5 * scale * (target - 2 * ell * np.sqrt(n)) )) ** 2,
             9 * (np.sqrt(n) + np.sqrt(2 * np.log(4 / delta))) ** 2,
             16 * logterm / n)
    return math.ceil(max(terms))


def validation_budget(support_coefficient, gain_margin, alpha, beta):
    """Sufficient fixed N_v for b<=C*d and mean D>=m*d, m>0.

    C and m must be justified before validation, not estimated from these
    validation returns. In Theorem 7, C uses S_star/eta_star, not the fitted
    matrix's subsequently observed gain.
    """
    coefficient = _nonnegative(support_coefficient, 'support_coefficient')
    margin = _nonnegative(gain_margin, 'gain_margin')
    alpha, beta = _probability(alpha, 'alpha'), _probability(beta, 'beta')
    if margin == 0 or coefficient < margin:
        raise ValueError('Require 0<gain_margin<=support_coefficient')
    return math.ceil(2 * (coefficient / margin) ** 2
                     * (np.sqrt(-np.log(alpha)) + np.sqrt(-np.log(beta))) ** 2)


def fixed_pair_test(differences, support, alpha, expected_pairs):
    """Fixed-sample Hoeffding rule; no early stopping or data-based N.

    Caller must freeze policy, support and count before observing these
    independent fresh paired returns. Invalid support observations raise
    rather than being clipped. A supplied data array cannot establish
    independence or a global support assumption.
    """
    bound = _nonnegative(support, 'support')
    alpha = _probability(alpha, 'alpha')
    count = _count(expected_pairs, 'expected_pairs')
    values = np.asarray(differences, dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError('Require exactly expected_pairs finite differences')
    if np.any(np.abs(values) > bound):
        raise ValueError('Observed difference violates supplied support')
    threshold = float(bound * np.sqrt(2 * -np.log(alpha) / count))
    mean = float(values.mean())
    return {'adopted': mean > threshold, 'mean_difference': mean,
            'threshold': threshold, 'support': bound, 'pairs': count,
            'conditional_testing_level': alpha}
