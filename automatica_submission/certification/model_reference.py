"""A robust model-based reference for the specific horizon-three example.

Known structure: X'=M X+h(X)+e2*a+W in R^2, h(0)=0,
Lip(h)<=ell, X0=0, B=0, constant candidate a, reward tanh(x1),
H=3, independent isotropic Gaussian W with unknown positive scale.
This is an example-specific expected-gain certificate, not a general
competitor for arbitrary controllers, rewards or horizons.
"""
import numpy as np
from scipy.stats import chi2

from .finite_power import MatrixCalibration, _nonnegative, _probability


def horizon_three_gain_interval(fit: MatrixCalibration, action, gamma,
                               noise_failure_probability, state_radius=1.):
    """Use the SAME reset data as the block method; no validation returns.

    Coverage is at least 1-fit.failure_probability-noise_failure_probability.
    The estimator/checker is supplied neither the true M nor noise scale.
    The residual may be any function satisfying the stated global bound.
    """
    if fit.matrix.shape != (2, 2) or fit.noise_model != 'isotropic':
        raise ValueError('This reference requires the two-state isotropic example')
    action = float(action)
    if not np.isfinite(action):
        raise ValueError('Action must be finite')
    gamma = _nonnegative(gamma, 'gamma')
    radius = _nonnegative(state_radius, 'state_radius')
    delta = _probability(noise_failure_probability, 'noise_failure_probability')
    if gamma > 1 or radius == 0 or fit.failure_probability + delta >= 1:
        raise ValueError('Require gamma<=1, positive radius and total failure<1')
    degrees = 2 * fit.samples
    variance = float(np.mean(fit.noise_variances))
    sigma_squared_upper = float(degrees * variance / chi2.ppf(delta, degrees))
    row_bound = float(np.linalg.norm(fit.matrix[0]) + fit.error_radius)
    # W1 is independent and centered; Minkowski handles h(W0) and M W0.
    second_moment = sigma_squared_upper * ((row_bound + np.sqrt(2) * fit.residual_lip) ** 2 + 1)
    inside_probability = max(0., 1 - second_moment / radius ** 2)
    cminus = float(fit.matrix[0, 1] - fit.error_radius - fit.residual_lip)
    cplus = float(fit.matrix[0, 1] + fit.error_radius + fit.residual_lip)
    lo, hi = sorted((action * cminus, action * cplus))
    # sech^2 evaluated in a form that does not overflow for large arguments.
    argument = radius + max(abs(lo), abs(hi))
    exponential = np.exp(-2 * argument)
    derivative = float(4 * exponential / (1 + exponential) ** 2)
    if lo >= 0:
        lower, upper = lo * derivative * inside_probability, hi
    elif hi <= 0:
        lower, upper = lo, hi * derivative * inside_probability
    else:
        lower, upper = lo, hi
    return {
        'gain_lower': float(gamma ** 2 * lower),
        'gain_upper': float(gamma ** 2 * upper),
        'adopted': bool(lower > 0 and gamma > 0),
        'failure_probability': fit.failure_probability + delta,
        'noise_variance_upper': sigma_squared_upper,
        'baseline_second_moment_upper': float(second_moment),
        'inside_probability_lower': inside_probability,
        'sensitivity_interval': [cminus, cplus],
        'validation_pairs': 0,
        'calibration_transitions': 2 * fit.samples,
    }
