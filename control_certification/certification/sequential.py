"""Fixed betting mixture used for the additional finite-power study.

Stakes/weights, inspection interval and maximum count are declared before
fresh validation. A support violation causes fallback at that observation;
observations after a stopping decision do not change that decision.
"""
import numpy as np
from scipy.special import logsumexp

from .finite_power import _count, _nonnegative, _probability


def mixture_test(differences, support, alpha, cap, inspect_every=16):
    bound = _nonnegative(support, 'support')
    alpha = _probability(alpha, 'alpha')
    cap, interval = _count(cap, 'cap'), _count(inspect_every, 'inspect_every')
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < cap or not np.all(np.isfinite(values[:cap])):
        raise ValueError('Require at least cap finite paired differences')
    if bound == 0:
        return {'adopted': False, 'pairs': 0, 'status': 'zero_support'}
    invalid = np.flatnonzero(np.abs(values[:cap]) > bound)
    usable = min(cap, int(invalid[0])) if len(invalid) else cap
    stakes = 2. ** -np.arange(8)
    weights = np.array([.5] + [1 / 14] * 7)
    with np.errstate(divide='ignore'):
        log_components = np.cumsum(np.log1p(values[:usable, None] / bound * stakes), axis=0)
    log_mixture = logsumexp(log_components + np.log(weights), axis=1)
    counts = np.arange(1, usable + 1)
    eligible = (counts % interval == 0) | (counts == cap)
    crossings = np.flatnonzero(eligible & (log_mixture >= -np.log(alpha)))
    if len(crossings):
        return {'adopted': True, 'pairs': int(crossings[0] + 1), 'status': 'adopted'}
    if len(invalid):
        return {'adopted': False, 'pairs': usable + 1, 'status': 'support_violation'}
    return {'adopted': False, 'pairs': cap, 'status': 'censored'}
