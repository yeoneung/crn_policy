"""Check recorded numerical summaries directly against frozen per-plant records."""
from pathlib import Path
import argparse
import io
import json
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'reproducibility/crn_policy/data/crn_policy_experiment_data.zip'
NAME = 'Ridge_d0.05_b81920'


def describe(rows, gains):
    gains = np.asarray(gains)
    return {'plants': len(rows), 'adopted': sum(r['adopted'] for r in rows),
            'mean_deployed_gain': float(gains.mean()),
            'deployed_gain_se': float(gains.std(ddof=1) / np.sqrt(len(gains))),
            'mean_transitions': float(np.mean([r['total_used_transitions'] for r in rows])),
            'mean_pairs': float(np.mean([r['episodes'] for r in rows]))}


def main(output=None):
    report = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        group = 'code/results/coupled_support_confirmatory/'
        files = sorted(n for n in names if n.startswith(group) and n.endswith('/certificates.json'))
        assert len(files) == 24
        for kind, label in [('normalized_support', 'known_support'),
                            ('ordinary_range', 'ordinary')]:
            rows, gains = [], []
            for name in files:
                directory = name.rsplit('/', 1)[0]
                choices = json.loads(archive.read(directory + '/pre_evaluation.json'))['names']
                with np.load(io.BytesIO(archive.read(directory + '/evaluation.npz'))) as bank:
                    gain = float((bank['returns'][choices.index(NAME)] - bank['returns'][0]).mean())
                records = json.loads(archive.read(name))
                selected = [r for r in records if r['name'] == NAME
                            and r['bound_kind'] == kind and r['cap'] == 512]
                assert len(selected) == 1
                row = selected[0]
                rows.append(row)
                gains.append(gain if row['adopted'] else 0.)
            report[label] = describe(rows, gains)

        group = 'code/results/stability_calibration_confirmatory/'
        files = sorted(n for n in names if n.startswith(group) and n.endswith('/certificates.json'))
        assert len(files) == 24
        for kind, label in [('calibrated_support', 'calibrated'),
                            ('ordinary_range', 'ordinary_extra_pairs')]:
            rows = []
            for name in files:
                selected = [r for r in json.loads(archive.read(name))
                            if r['name'] == NAME and r['bound_kind'] == kind
                            and r['samples'] == 2048]
                assert len(selected) == 1
                rows.append(selected[0])
            report[label] = describe(rows, [r['deployed_gain'] for r in rows])

        group = 'code/results/actuation_structure_confirmatory/'
        files = sorted(n for n in names if n.startswith(group) and n.endswith('/certificates.json'))
        assert len(files) == 16
        mechanical = {}
        for kind in ['input_structure', 'matrix_power', 'ordinary_range']:
            mechanical[kind] = {}
            for scope in ['primary', 'extended']:
                rows = []
                for name in files:
                    selected = [r for r in json.loads(archive.read(name))
                                if r['bound_kind'] == kind and r['scope'] == scope]
                    assert len(selected) == 1
                    rows.append(selected[0])
                mechanical[kind][scope] = {
                    'plants': len(rows), 'adopted': sum(r['adopted'] for r in rows),
                    'mean_pairs': float(np.mean([r['episodes'] for r in rows]))}
        report['mechanical'] = mechanical

    expected = {'known_support': (24, 4.995, .255, 99947),
                'ordinary': (0, 0., 0., 122880),
                'calibrated': (22, 4.729, .375, 110496),
                'ordinary_extra_pairs': (0, 0., 0., 126960)}
    for label, (adopted, gain, se, transitions) in expected.items():
        row = report[label]
        assert row['adopted'] == adopted
        assert round(1000 * row['mean_deployed_gain'], 3) == gain
        assert round(1000 * row['deployed_gain_se'], 3) == se
        assert round(row['mean_transitions']) == transitions
    for kind, primary, mean in [('input_structure', 16, 295),
                                ('matrix_power', 0, 587),
                                ('ordinary_range', 0, 2582)]:
        assert mechanical[kind]['primary']['adopted'] == primary
        assert mechanical[kind]['extended']['adopted'] == 16
        assert round(mechanical[kind]['extended']['mean_pairs']) == mean
    report['all_summary_and_mechanical_checks_passed'] = True
    output = Path(output) if output else ROOT / 'evidence/numerical_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Primary and mechanical tables match frozen per-plant records.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
