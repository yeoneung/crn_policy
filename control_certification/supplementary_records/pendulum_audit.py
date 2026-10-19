"""Check retained regulation records, analytic bounds and stopping decisions.

Requires NumPy and SciPy. No Gymnasium installation or original experiment
directory layout is needed. The scalar check reconstructs a subset of the
recorded evaluation streams; it does not train or generate a new experiment.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import hashlib
import json

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def matrix(values):
    return [[F(str(v)) for v in row] for row in values]


def transpose(a):
    return list(map(list, zip(*a)))


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def scale(a, value):
    return [[value * x for x in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def determinant(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def positive_definite(a):
    return a[0][0] > 0 and determinant(a) > 0


def exact_geometry(cfg):
    """Repeat the retained exact-rational checks, including input sensitivity."""
    t = matrix(cfg['T'])
    p = multiply(transpose(t), t)
    pi = scale([[p[1][1], -p[0][1]], [-p[1][0], p[0][0]]], 1 / determinant(p))
    k = [[F.from_float(x) for x in cfg['K']]]
    gaps = []
    for gain, damping, cosine in product([F(3, 4), F(19, 20)],
                                         [F(1, 20), F(3, 20)], [F(49, 50), F(1)]):
        row = [F(3, 4) * cosine - F(3, 20) * gain * k[0][0],
               1 - F(1, 20) * damping - F(3, 20) * gain * k[0][1]]
        a = [[1 + F(1, 20) * row[0], F(1, 20) * row[1]], row]
        gap = add(scale(p, F('0.864') ** 2),
                  scale(multiply(multiply(transpose(a), p), a), -1))
        assert positive_definite(gap)
        gaps.append(float(determinant(gap)))
    g = matrix([[.0075], [.15]])
    g2 = multiply(multiply(transpose(g), p), g)[0][0]
    assert g2 < F('.1351') ** 2
    assert F('.95') ** 2 * g2 < F('.12835') ** 2
    assert multiply(multiply(k, pi), transpose(k))[0][0] < F('7.06') ** 2
    q = add(matrix([[1, 0], [0, .1]]), scale(multiply(transpose(k), k), F('.001')))
    assert positive_definite(add(scale(p, F('1.334')), scale(q, -1)))
    assert F('.254') ** 2 * pi[0][0] < F('.2') ** 2
    assert F('.254') ** 2 * pi[1][1] < F('.73') ** 2
    assert F('.864') * F('.24') + F('.1351') * F('.03') < F('.24')
    assert F('.24') + F('.12835') * F('.0125') / (1 - F('.864')) < F('.254')
    assert F('7.06') * F('.254') + F('.0125') < 2
    assert F('.73') + F('.75') * F('.2') + F('.15') * F('.95') * 2 + F('.15') * F('.03') < 8
    return {'exact_rational_vertex_checks': len(gaps),
            'minimum_vertex_determinant': min(gaps),
            'invariant_region_and_clipping_checks': True}


def supports(cfg):
    result = {'global': cfg['H'] * (np.pi ** 2 + .1 * 8 ** 2 + .001 * 2 ** 2)}
    for statecap in [False, True]:
        s, e, paired, marginal = cfg['initial_radius'], 0., 0., 0.
        for _ in range(cfg['H']):
            a = cfg['d'] * min(1., (s + e) / cfg['initial_radius']) if statecap else cfg['d']
            effort = .001 * (2 * cfg['Knorm'] * (s + e) * a + a * a)
            paired += cfg['Qnorm'] * (2 * s + e) * e + effort
            marginal += cfg['Qnorm'] * (s + e) ** 2 + effort
            e, s = (cfg['kappa'] * e + cfg['action_gain'] * a,
                    cfg['kappa'] * s + cfg['noise_gain'] * cfg['width'])
        result['statecap' if statecap else 'constantcap'] = paired * 1.00000001
        if statecap:
            result['regional_marginal'] = marginal * 1.00000001
    for key, value in result.items():
        np.testing.assert_allclose(value, cfg['bounds'][key], rtol=1e-13)
    return result


def log_wealth(d, bound, alpha, method):
    if method == 'fixed_mixture':
        stakes = 2. ** (-np.arange(8))
        weights = np.array([.5] + [.5 / 7] * 7)
        return logsumexp(np.log(weights)[:, None]
                         + np.log1p(stakes[:, None] * d / bound).cumsum(axis=1), axis=0)
    assert method == 'predictable_empirical_bernstein'
    total, squared, previous_mean, previous_var, wealth = 0., 0., .5, .25, 0.
    result = np.empty(len(d))
    for t, value in enumerate(d, 1):
        x = (value / bound + 1) / 2
        stake = min(.75, np.sqrt(2 * np.log(2 / alpha) / (previous_var * t * np.log1p(t))))
        wealth += stake * (x - .5) - (x - previous_mean) ** 2 * (-np.log1p(-stake) - stake)
        result[t - 1] = wealth
        total += x
        mean = (.5 + total) / (t + 1)
        squared += (x - mean) ** 2
        previous_var, previous_mean = (.25 + squared) / (t + 1), mean
    return result


def scalar_reconstruction(folder, row, cfg):
    """Replay 128 held-out episodes per branch using an independent scalar loop."""
    t, k = np.array(cfg['T']), np.array(cfg['K'])
    coef = np.array(read(folder / 'frozen_policy.json')['coef'])
    rng = np.random.default_rng([row['seed'], 4])
    angles = rng.uniform(0, 2 * np.pi, 4096)
    radii = np.sqrt(rng.uniform(size=4096)) * cfg['initial_radius']
    starts = np.column_stack([np.cos(angles), np.sin(angles)]) * radii[:, None] @ np.linalg.inv(t).T
    noise = rng.uniform(-cfg['width'], cfg['width'], (cfg['H'], 4096))
    recovered = np.zeros((2, 128))
    for branch in range(2):
        for episode in range(128):
            theta, velocity = starts[episode]
            for step in range(cfg['H']):
                features = t @ np.array([theta, velocity])
                u = -k @ np.array([theta, velocity])
                if branch:
                    u += cfg['d'] * min(1, np.linalg.norm(features) / cfg['initial_radius']) * np.sign(features @ coef)
                u = min(2, max(-2, u))
                angle = (theta + np.pi) % (2 * np.pi) - np.pi
                recovered[branch, episode] -= angle ** 2 + .1 * velocity ** 2 + .001 * u ** 2
                velocity = min(8, max(-8, velocity + .75 * np.sin(theta)
                                     + .15 * row['gain'] * u - .05 * row['damping'] * velocity
                                     + .15 * noise[step, episode]))
                theta += .05 * velocity
    with np.load(folder / 'evaluation.npz') as bank:
        error = float(np.max(abs(recovered - bank['returns'][:, :128])))
    assert error < 1e-12
    return error


def audit():
    manifest = read(ROOT / 'summaries/pendulum_record_manifest.json')
    base = ROOT / 'data/pendulum_regional_study'
    for name, info in manifest['files'].items():
        data = (base / name).read_bytes()
        assert len(data) == info['bytes']
        assert hashlib.sha256(data).hexdigest() == info['sha256']
    protocol = read(base / 'final/protocol.json')
    cfg = protocol['config']
    result = exact_geometry(cfg)
    result['supports'] = supports(cfg)
    gains, relative, rows, replayed, reconstruction = [], [], [], 0, []
    for seed in protocol['final_seeds']:
        folder = base / 'final' / f'seed{seed}'
        row = read(folder / 'metrics.json')
        for name, checksum in read(folder / 'COMPLETE.json')['outputs'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == checksum
        with np.load(folder / 'certification.npz') as bank:
            assert bank['returns'].shape == (2, protocol['cap'])
            d = bank['returns'][1] - bank['returns'][0]
        for cell in row['results']:
            assert cell['alpha'] == protocol['per_test_alpha']
            assert cell['bound'] == cfg['bounds'][cell['kind']]
            assert abs(d).max() < cell['bound']
            pos = log_wealth(d, cell['bound'], cell['alpha'], cell['method'])
            neg = log_wealth(-d, cell['bound'], cell['alpha'], cell['method'])
            idx = np.arange(15, len(d), 16)
            crossings = idx[(pos[idx] >= np.log(1 / cell['alpha'])) | (neg[idx] >= np.log(1 / cell['alpha']))]
            n, outcome = len(d), 'inconclusive'
            if len(crossings):
                j = crossings[0]
                n, outcome = int(j + 1), 'adopt' if pos[j] >= np.log(1 / cell['alpha']) else 'reject'
            assert cell['pairs'] == n and cell['outcome'] == outcome
            np.testing.assert_allclose(cell['positive_log_wealth'], pos[n - 1], atol=1e-8)
            assert cell['total_used_transitions'] == 43520 + 80 * n
            rows.append(cell)
            replayed += 1
        with np.load(folder / 'evaluation.npz') as bank:
            assert bank['returns'].shape == (2, 4096)
            gain = float((bank['returns'][1] - bank['returns'][0]).mean())
            ratio = gain / -bank['returns'][0].mean()
        np.testing.assert_allclose(gain, row['independent_gain'], atol=1e-14)
        np.testing.assert_allclose(ratio, row['relative_gain'], atol=1e-14)
        assert row['certification_audit']['state_envelope_violations'] == 0
        assert row['evaluation_audit']['state_envelope_violations'] == 0
        gains.append(gain)
        relative.append(ratio)
        reconstruction.append(scalar_reconstruction(folder, row, cfg))
    summary = read(base / 'final/summary.json')
    for group in summary['groups']:
        selected = [r for r in rows if r['kind'] == group['kind'] and r['method'] == group['method']]
        assert len(selected) == 8
        assert sum(r['outcome'] == 'adopt' for r in selected) == group['adoptions']
        assert sum(r['outcome'] == 'reject' for r in selected) == group['rejections']
        assert sum(r['outcome'] == 'inconclusive' for r in selected) == group['censored']
        np.testing.assert_allclose(np.mean([r['pairs'] for r in selected]), group['mean_pairs'])
    np.testing.assert_allclose(np.mean(gains), summary['mean_gain'], atol=1e-14)
    np.testing.assert_allclose(np.mean(relative), summary['mean_relative_gain'], atol=1e-14)
    result.update(passed=True, plants=len(gains), stopping_decisions_replayed=replayed,
                  retained_files_checked=len(manifest['files']), groups=summary['groups'],
                  mean_gain=float(np.mean(gains)), plant_se=float(np.std(gains, ddof=1) / np.sqrt(len(gains))),
                  mean_relative_gain_percent=float(100 * np.mean(relative)),
                  positive_gain_estimates=int(sum(x > 0 for x in gains)),
                  reconstructed_recorded_transitions=8 * 2 * 128 * cfg['H'],
                  maximum_reconstruction_error=max(reconstruction),
                  scope='Retained records only; no new training or experimental observations.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
