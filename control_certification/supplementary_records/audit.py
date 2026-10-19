"""Recompute retained metric summaries and replay verifier comparisons."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def metrics():
    summary = read(ROOT / 'summaries/mechanical_physical_metrics.json')
    banks = sorted((ROOT / 'data/mechanical_physical_metrics').glob('dim*/evaluation.npz'))
    assert len(banks) == 16
    gains, relative = [], []
    for bank in banks:
        record = read(bank.parent / 'COMPLETE')
        assert hashlib.sha256(bank.read_bytes()).hexdigest() == record['raw_sha256']
        with np.load(bank) as saved:
            costs = saved['costs']
        assert costs.shape == (6, 2, 4096)
        np.testing.assert_allclose(costs[3] + costs[4], costs[5], atol=1e-12)
        baseline = costs[:, 0].mean(axis=1)
        gain = (costs[:, 0] - costs[:, 1]).mean(axis=1)
        ratio = 100 * gain / baseline
        np.testing.assert_allclose(gain, record['gain'], atol=1e-12)
        np.testing.assert_allclose(ratio, record['relative_gain_percent'], atol=1e-10)
        gains.append(gain)
        relative.append(ratio)
    gains = np.array(gains)
    relative = np.array(relative)
    for index, row in enumerate(summary['summary']):
        np.testing.assert_allclose(gains[:, index], row['gain']['values'], atol=1e-12)
        np.testing.assert_allclose(relative[:, index], row['relative_gain_percent']['values'], atol=1e-10)
        np.testing.assert_allclose(relative[:, index].mean(), row['relative_gain_percent']['mean'], atol=1e-10)
    assert int((gains[:, 2] < 0).sum()) == 15
    assert int((gains[:, 4] < 0).sum()) == 13
    return {'banks': len(banks), 'raw_metric_streams': 16 * 6 * 2,
            'evaluation_transitions': 16 * 4096 * 2 * 40,
            'velocity_cost_increases': int((gains[:, 2] < 0).sum()),
            'effort_cost_increases': int((gains[:, 4] < 0).sum()),
            'mean_relative_cost_reductions_percent': relative.mean(axis=0).tolist()}


def replay(archive):
    scripts = ROOT / 'original_scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('retained_verifier_replay', scripts / 'variance_replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = read(ROOT / 'summaries/verifier_comparison.json')
    compared = 0
    selected_banks = 0
    with ZipFile(archive) as z:
        members = set(z.namelist())
        for row in expected:
            study = row['study']
            folders = sorted(n.removesuffix('/certification.npz') for n in members
                             if n.startswith('code/results/' + study + '/dim')
                             and n.endswith('/certification.npz'))
            assert len(folders) == row['plants']
            stopped = []
            for folder in folders:
                pre = json.loads(z.read(folder + '/pre_evaluation.json'))
                certificates = json.loads(z.read(folder + '/certificates.json'))
                index = pre['names'].index('Ridge_d0.05_b81920') if study == 'coupled_support_confirmatory' else 1
                with np.load(BytesIO(z.read(folder + '/certification.npz'))) as saved:
                    observations = saved['returns'][index] - saved['returns'][0]
                cap = 4096 if row['scope'] == 'extended' else (
                    512 if row['support'] != 'ordinary_range' or study == 'coupled_support_confirmatory' else 716)
                matches = [c for c in certificates if c['bound_kind'] == row['support'] and c['cap'] == cap
                           and (study != 'coupled_support_confirmatory' or c['name'] == 'Ridge_d0.05_b81920')]
                assert len(matches) == 1
                certificate = matches[0]
                stopped.append(module.stop(observations, certificate['bound'], certificate['alpha'],
                                           row['method'], cap))
                selected_banks += 1
            assert sum(r['adopted'] for r in stopped) == row['adoptions']
            np.testing.assert_allclose(np.mean([r['pairs'] for r in stopped]), row['mean_consumed_pairs'])
            compared += 1

        # Aggregate the original per-plant residual replays at their original levels.
        records = []
        for name in sorted(members):
            if name.startswith('code/results/v2_coupling_sensitivity/dim') and name.endswith('/ell_replay.json'):
                records.extend(json.loads(z.read(name)))
        residual = read(ROOT / 'summaries/coupling_sensitivity.json')
        for row in residual['ell']:
            group = [r for r in records if r['ell_factor'] == row['factor']
                     and r['method'] == row['method'] and r['cap'] == row['cap']]
            assert len(group) == 24
            assert sum(r['adopted'] for r in group) == row['adopted']
            np.testing.assert_allclose(np.mean([r['pairs'] for r in group]), row['consumed_pairs']['mean'])
        factors = [0.5, 1, 2, 5, 10]
        residual_rows = []
        for factor in factors:
            group = [r for r in records if r['ell_factor'] == factor
                     and r['method'] == 'fixed_mixture' and r['cap'] == 512]
            residual_rows.append({'factor': factor,
                                  'contracting': sum(r['structural_support_available'] for r in group),
                                  'betting_adoptions': sum(r['adopted'] for r in group),
                                  'premise_satisfied': sum(r['premise_satisfied'] for r in group)})
    return {'distinct_plants': 64, 'verifier_summary_cells_recomputed': compared,
            'raw_return_bank_replays': selected_banks,
            'residual_summary_cells_recomputed': len(residual['ell']),
            'residual_primary': residual_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded-data', type=Path,
                        help='Original crn_policy_experiment_data.zip for full verifier/residual replay.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'mechanical_metrics': metrics(), 'passed': True,
              'scope': 'Retained observations only; no new training or simulation.'}
    if args.recorded_data:
        result['verifier_and_residual_replay'] = replay(args.recorded_data)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
