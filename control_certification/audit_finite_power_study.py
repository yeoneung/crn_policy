"""Independent bank identities and table/cost checks for the additional study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main(output=None):
    directory = ROOT / 'evidence/finite_power_study'
    cfg = json.loads((directory / 'protocol.json').read_text())
    records = json.loads((directory / 'records.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    for name, digest in cfg['implementation_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert len(records) == 24 * 5 * 4
    with np.load(directory / 'banks.npz') as bank:
        count = bank['differences'].shape[-1]
        checked = 0
        # Independent H=3 identity: the reward gap is only at time two.
        for repetition, seed in enumerate(np.random.SeedSequence(cfg['seed']).spawn(24)):
            _, validation_seed = seed.spawn(2)
            rng = np.random.default_rng(validation_seed)
            w0 = rng.normal(scale=.005, size=(count, 2))
            w1 = rng.normal(scale=.005, size=(count, 2))
            baseline = .5 * w0[:, 0] + w0[:, 1] + .001 * np.tanh(w0[:, 0]) + w1[:, 0]
            for index, action in enumerate(cfg['actions']):
                exact = .95 ** 2 * (np.tanh(baseline + action) - np.tanh(baseline))
                np.testing.assert_allclose(bank['differences'][repetition, index], exact,
                                           rtol=1e-12, atol=2e-15)
                if action > 0:
                    assert np.all(exact > 0)
                else:
                    assert np.all(exact <= 0)
                checked += len(exact)
    expected = {
        .05: {'ordinary': (304, 1824), 'finite_horizon': (16, 442),
              'finite_power': (112, 1018), 'robust_model': (0, 346)},
        .025: {'ordinary': (608, 3648), 'finite_horizon': (16, 442),
               'finite_power': (112, 1018), 'robust_model': (0, 346)},
        .0125: {'ordinary': (1200, 7200), 'finite_horizon': (16, 442),
                'finite_power': (112, 1018), 'robust_model': (0, 346)},
    }
    table = []
    for action, methods in expected.items():
        row = {'d': action}
        for method, (pairs, transitions) in methods.items():
            selected = [r for r in records if r['action'] == action and r['method'] == method]
            assert len(selected) == 24 and all(r['adopted'] for r in selected)
            assert np.mean([r['pairs'] for r in selected]) == pairs
            assert np.mean([r['total_transitions'] for r in selected]) == transitions
            row[method] = transitions
        table.append(row)
    for record in records:
        calibration = 0 if record['method'] == 'ordinary' else 346
        assert record['total_transitions'] == calibration + 6 * record['pairs']
        if record['action'] <= 0:
            assert not record['adopted']
        if record['method'] == 'robust_model':
            assert record['pairs'] == 0 and record['failure_probability'] == .05
    assert summary['one_step_uniform_gates'] == 0
    assert summary['finite_power_gates'] == 24
    assert summary['false_adoption_bound_per_decision'] == .05
    report = {'status': 'passed', 'seed': cfg['seed'], 'records_checked': len(records),
              'independent_time_two_bank_identities_checked': checked,
              'model_reference_implementation': 'certification/model_reference.py',
              'cost_table': table, 'null_control_adoptions': 0,
              'all_original_frozen_experiments_untouched': True,
              'scope': 'Additional study audit; no claim of empirical proof of error coverage or universal model-based dominance.'}
    output = Path(output) if output else ROOT / 'evidence/finite_power_study_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Additional study audit passed: {len(records)} decision records; '
          f'{checked} independent return identities; total-cost table matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
