"""Independent predeclared study connecting the new block result to cost.

python -m examples.finite_power_study --output-dir evidence/finite_power_study
Original manuscript experiment records are never read or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from certification.finite_power import (
    block_certificate, calibrate_matrix, calibration_budget, discounted_length,
    powers, return_support,
)
from certification.model_reference import horizon_three_gain_interval
from certification.sequential import mixture_test

ROOT = Path(__file__).resolve().parents[1]


def finite_horizon_support(fit, d, horizon, gamma, actuator):
    """Proposition 4, equal-state reward with Lipschitz constant one."""
    pp = powers(fit.matrix, max(1, horizon - 1))
    norms = np.array([np.linalg.norm(p, 2) for p in pp])
    forcing = np.array([np.linalg.norm(p @ actuator, 2) for p in pp])
    envelope = np.zeros(horizon)
    for t in range(1, horizon):
        envelope[t] = (d * forcing[:t].sum()
                       + (fit.error_radius + fit.residual_lip)
                       * np.dot(norms[:t][::-1], envelope[:t]))
    return float(np.dot(gamma ** np.arange(horizon), envelope))


def protocol():
    code = ['examples/finite_power_study.py', 'certification/finite_power.py',
            'certification/model_reference.py', 'certification/sequential.py']
    return {
        'study': 'Finite-power acquisition, validation and model-based reference',
        'scope': 'Independent additional study; no training or replacement of original frozen records.',
        'seed': 20261017, 'replications': 24,
        'development': {'seeds': [20261015, 20261016],
                        'amendments': ['Tighten the ordinary marginal range to 2*(gamma+gamma^2), since the known initial reward is zero.',
                                       'Rename the summary error-level field to false_adoption_bound_per_decision; no test or threshold change.'],
                        'confirmation': 'Final seed is disjoint from development; all final replications are retained.'},
        'actions': [.05, .025, .0125, 0., -.05],
        'matrix': [[.5, 1.], [0., .5]],
        'residual': '.001*tanh(x)', 'residual_lip': .001,
        'actuator': [[0.], [1.]], 'reward': 'tanh(x1)',
        'initial_state': [0., 0.], 'horizon': 3, 'discount': .95,
        'noise_sigma_for_simulation_and_budget_analysis_only': .005,
        'reset_scale': 1., 'matrix_accuracy_target_for_analysis': .02,
        'block_cap': 4, 'allocation_delta': .05,
        'matrix_failure_probability': .0125,
        'conditional_validation_level': .0375,
        'ordinary_validation_level': .05,
        'model_reference_noise_failure_probability': .0375,
        'supported_pair_cap': 2048, 'inspection_interval': 16,
        'family_error_control': '0.05 per adoption decision; no joint family-wise claim across methods, actions or replications.',
        'bank_reuse': 'Methods and actions share validation noises within a replication; independent reset/validation streams and independent replications.',
        'stopping': 'Fixed eight-stake mixture; no futility stopping; retain baseline on censoring or support violation.',
        'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code},
    }


def simulate(cfg, nc):
    matrix = np.array(cfg['matrix'])
    actions, count = cfg['actions'], cfg['replications']
    sigma = cfg['noise_sigma_for_simulation_and_budget_analysis_only']
    maximum = cfg['supported_pair_cap'] + (2 * nc) // (2 * cfg['horizon'])
    xs, ys1, ys2, banks = [], [], [], []
    for seed in np.random.SeedSequence(cfg['seed']).spawn(count):
        calibration_seed, validation_seed = seed.spawn(2)
        rng = np.random.default_rng(calibration_seed)
        x = rng.normal(scale=cfg['reset_scale'], size=(nc, 2))
        mean = x @ matrix.T + cfg['residual_lip'] * np.tanh(x)
        xs.append(x)
        ys1.append(mean + rng.normal(scale=sigma, size=x.shape))
        ys2.append(mean + rng.normal(scale=sigma, size=x.shape))
        rng = np.random.default_rng(validation_seed)
        base = np.zeros((maximum, 2))
        candidates = np.zeros((len(actions), maximum, 2))
        differences = np.zeros((len(actions), maximum))
        for t in range(cfg['horizon']):
            differences += cfg['discount'] ** t * (np.tanh(candidates[:, :, 0]) - np.tanh(base[:, 0]))
            noise = rng.normal(scale=sigma, size=base.shape)
            base = base @ matrix.T + cfg['residual_lip'] * np.tanh(base) + noise
            candidates = candidates @ matrix.T + cfg['residual_lip'] * np.tanh(candidates) + noise
            candidates[:, :, 1] += np.asarray(actions)[:, None]
        banks.append(differences)
    return dict(x=np.array(xs), y1=np.array(ys1), y2=np.array(ys2), differences=np.array(banks))


def analyze(cfg, data, nc):
    actuator = np.array(cfg['actuator'])
    gamma, horizon, cap = cfg['discount'], cfg['horizon'], cfg['supported_pair_cap']
    ordinary_cap = cap + (2 * nc) // (2 * horizon)
    # Known X0=0 gives zero initial reward in each marginal return.
    ordinary_bound = 2 * (discounted_length(horizon, gamma) - 1)
    records, fits = [], []
    for repetition in range(cfg['replications']):
        fit = calibrate_matrix(data['x'][repetition], data['y1'][repetition], data['y2'][repetition],
                               cfg['residual_lip'], cfg['matrix_failure_probability'])
        block = block_certificate(fit.matrix, fit.error_radius, fit.residual_lip,
                                  cfg['block_cap'], actuator=actuator)
        fits.append({'replication': repetition, 'matrix_estimate': fit.matrix.tolist(),
                     'radius': fit.error_radius, 'one_step_bound': fit.one_step_bound,
                     'block_certified': block is not None,
                     'selected_block': block.block_length if block else None})
        for index, action in enumerate(cfg['actions']):
            d = abs(action)
            values = data['differences'][repetition, index]
            supports = {
                'ordinary': ordinary_bound,
                'finite_horizon': min(ordinary_bound, finite_horizon_support(fit, d, horizon, gamma, actuator)),
                'finite_power': return_support(block, d, horizon, gamma, equal_state_reward_lip=1.,
                                               ordinary_bound=ordinary_bound) if block else None,
            }
            for method, bound in supports.items():
                allocation = ordinary_cap if method == 'ordinary' else cap
                alpha = cfg['ordinary_validation_level'] if method == 'ordinary' else cfg['conditional_validation_level']
                outcome = (mixture_test(values, bound, alpha, allocation, cfg['inspection_interval'])
                           if bound is not None else dict(adopted=False, pairs=0, status='no_certificate'))
                calibration = 0 if method == 'ordinary' else 2 * nc
                records.append(dict(replication=repetition, action=action, method=method,
                                    support=bound, pair_cap=allocation, calibration_transitions=calibration,
                                    total_transitions=calibration + 2 * horizon * outcome['pairs'], **outcome))
            reference = horizon_three_gain_interval(fit, action, gamma,
                                                    cfg['model_reference_noise_failure_probability'])
            records.append(dict(replication=repetition, action=action, method='robust_model',
                                support=None, pair_cap=0,
                                total_transitions=2 * nc, pairs=0,
                                status='adopted' if reference['adopted'] else 'retained', **reference))
    rows = []
    for action in cfg['actions']:
        for method in ['ordinary', 'finite_horizon', 'finite_power', 'robust_model']:
            selected = [r for r in records if r['action'] == action and r['method'] == method]
            rows.append(dict(action=action, method=method, replications=len(selected),
                             adopted=sum(r['adopted'] for r in selected),
                             mean_pairs=float(np.mean([r['pairs'] for r in selected])),
                             mean_total_transitions=float(np.mean([r['total_transitions'] for r in selected])),
                             support_violations=sum(r['status'] == 'support_violation' for r in selected)))
    summary = dict(scope=cfg['scope'], seed=cfg['seed'], replicated_resets=nc,
                   calibration_transitions=2 * nc, supported_pair_cap=cap,
                   ordinary_pair_cap=ordinary_cap, false_adoption_bound_per_decision=.05,
                   one_step_uniform_gates=sum(fit['one_step_bound'] < 1 for fit in fits),
                   finite_power_gates=sum(fit['block_certified'] for fit in fits),
                   selected_block_lengths=sorted({fit['selected_block'] for fit in fits if fit['block_certified']}),
                   matrix_radius_range=[min(fit['radius'] for fit in fits), max(fit['radius'] for fit in fits)],
                   rows=rows,
                   controlled_negative_and_zero_action_adoptions=sum(r['adopted'] for r in records if r['action'] <= 0),
                   all_reported_costs_include_calibration=True,
                   robust_model_uses_true_parameters=False)
    return records, fits, summary


def run(output_dir, replay_from=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = protocol()
    # Protocol is saved BEFORE drawing any new calibration or validation bank.
    (output_dir / 'protocol.json').write_text(json.dumps(cfg, indent=2) + '\n')
    nc = calibration_budget(2, cfg['matrix_accuracy_target_for_analysis'], cfg['residual_lip'],
                            cfg['noise_sigma_for_simulation_and_budget_analysis_only'],
                            cfg['reset_scale'], cfg['allocation_delta'])
    if replay_from:
        replay_from = Path(replay_from)
        saved = json.loads((replay_from / 'protocol.json').read_text())
        if saved != cfg:
            raise ValueError('Replay protocol or implementation differs from the recorded study')
        with np.load(replay_from / 'banks.npz') as bank:
            data = {key: bank[key] for key in bank.files}
    else:
        data = simulate(cfg, nc)
    records, fits, summary = analyze(cfg, data, nc)
    if replay_from:
        expected = json.loads((replay_from / 'summary.json').read_text())
        if summary != expected:
            raise ValueError('Recomputed summary differs from recorded study')
    np.savez_compressed(output_dir / 'banks.npz', **data)
    for name, value in [('records.json', records), ('calibrations.json', fits), ('summary.json', summary)]:
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('evidence/finite_power_study'))
    parser.add_argument('--replay-from', type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.replay_from)
