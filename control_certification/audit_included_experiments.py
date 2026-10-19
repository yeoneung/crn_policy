"""Replay active calibration, verifier and secondary-cost comparisons."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import argparse
import json
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'reproducibility/crn_policy'
ARCHIVE = LEGACY / 'data/crn_policy_experiment_data.zip'
sys.path.insert(0, str(LEGACY / 'frozen/code/scripts'))
sys.path.insert(0, str(ROOT / 'supplementary_records'))
from stability_calibration import estimate_contraction
from variance_replay import stop
from audit import metrics, replay


def calibration_prefixes():
    rows = {n: {'replicated_resets': n, 'calibration_transitions': 2 * n,
                'supported_pair_cap': 512, 'ordinary_pair_cap': 512 + 2 * n // 80,
                'supported_adoptions': 0, 'ordinary_adoptions': 0,
                'supported_total_used': [], 'ordinary_total_used': []}
            for n in [512, 2048, 8192]}
    checks = 0
    with ZipFile(ARCHIVE) as z:
        folders = sorted(name.rsplit('/', 1)[0] for name in z.namelist()
                         if name.startswith('code/results/stability_calibration_confirmatory/dim')
                         and name.endswith('/certificates.json'))
        assert len(folders) == 24
        for folder in folders:
            plant = folder.rsplit('/', 1)[1]
            stable = 'code/results/coupled_support_confirmatory/' + plant
            choices = json.loads(z.read(stable + '/pre_evaluation.json'))['names']
            index = choices.index('Ridge_d0.05_b81920')
            with np.load(BytesIO(z.read(stable + '/certification.npz'))) as bank:
                differences = bank['returns'][index] - bank['returns'][0]
            with np.load(BytesIO(z.read(folder + '/observations.npz'))) as bank:
                x, y1, y2 = [bank[key].copy() for key in ['x', 'y1', 'y2']]
            assert len(x) == 8192
            stored = json.loads(z.read(folder + '/pre_return_calibration.json'))['fits']
            cells = json.loads(z.read(folder + '/certificates.json'))
            for n, group in rows.items():
                fit = estimate_contraction(x[:n], y1[:n], y2[:n], .005, .025 / (24 * 3))
                np.testing.assert_allclose(fit['kappa'], stored[str(n)]['kappa'], rtol=1e-12)
                assert fit['valid'] == stored[str(n)]['valid']
                for kind, label in [('calibrated_support', 'supported'), ('ordinary_range', 'ordinary')]:
                    chosen = [c for c in cells if c['name'] == 'Ridge_d0.05_b81920'
                              and c['samples'] == n and c['bound_kind'] == kind]
                    assert len(chosen) == 1
                    cell = chosen[0]
                    assert cell['cap'] == group[label + '_pair_cap']
                    assert cell['calibration_transitions'] == (2 * n if label == 'supported' else 0)
                    if cell['outcome'] == 'uncertified_contraction':
                        assert label == 'supported' and not fit['valid'] and cell['episodes'] == 0
                    else:
                        result = stop(differences, cell['bound'], cell['alpha'],
                                      'fixed_mixture', cell['cap'])
                        assert result['adopted'] == cell['adopted']
                        assert result['pairs'] == cell['episodes']
                        assert result['outcome'] == cell['outcome']
                    assert cell['total_used_transitions'] == (
                        81920 + cell['calibration_transitions'] + 80 * cell['episodes'])
                    group[label + '_adoptions'] += int(cell['adopted'])
                    group[label + '_total_used'].append(cell['total_used_transitions'])
                    checks += 1
    for n, expected in zip(rows, [16, 22, 24]):
        assert rows[n]['supported_adoptions'] == expected
        assert rows[n]['ordinary_adoptions'] == 0
        for label in ['supported', 'ordinary']:
            values = rows[n].pop(label + '_total_used')
            assert len(values) == 24
            rows[n][label + '_mean_total_used_transitions'] = float(np.mean(values))
    return {'plants': 24, 'decisions_checked': checks,
            'fits_recomputed_from_raw_replicates': 72, 'rows': list(rows.values()),
            'scope': 'Nested prefixes of the same reset bank and frozen policies. Total allowances differ between prefix sizes; no fixed-total-cost optimum is claimed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'calibration_prefixes': calibration_prefixes(),
              'mechanical_secondary_metrics': metrics(),
              'verifier_and_residual_comparisons': replay(ARCHIVE),
              'passed': True, 'scope': 'Recompute retained observations; no new training or plant selection.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Passed: 72 recalibrations, 144 prefix decisions, 64-plant verifier replay and 16 secondary metric banks.')


if __name__ == '__main__':
    main()
