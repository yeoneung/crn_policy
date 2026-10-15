"""Draw a comparison figure from frozen records; no simulations or fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'reproducibility/crn_policy/data/crn_policy_experiment_data.zip'
RECORD = 'code/results/v2_coupling_sensitivity/summary.json'


def main(output_dir=None):
    output_dir = Path(output_dir) if output_dir else ROOT
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw = archive.read(RECORD)
    source = json.loads(raw)
    assert source['plants'] == 24
    ablation = [r for r in source['ablation'] if r['method'] == 'fixed_mixture']
    sweep = [r for r in source['sweep']
             if r['method'] == 'fixed_mixture' and r['cap'] == 4096]
    evidence = output_dir / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    data = {k: source[k] for k in ['plants', 'radii', 'gain', 'variance_ratio',
                                  'crn_variance', 'independent_variance']}
    data.update(ablation=ablation, sweep=sweep, source_record=RECORD,
                source_record_sha256=hashlib.sha256(raw).hexdigest())
    (evidence / 'figure_data.json').write_text(json.dumps(data, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.85), layout='constrained')
    blue, orange, gray = '#2166ac', '#c35a13', '#666666'

    ax = axes[0, 0]
    values = [source['independent_variance']['mean'], source['crn_variance']['mean']]
    ax.bar([0, 1], values, color=[gray, blue], width=.55)
    ax.set_yscale('log')
    ax.set_ylim(4e-6, .12)
    ax.set_xticks([0, 1], ['Independent noise', 'Common noise (CRN)'])
    ax.set_ylabel('Mean contrast variance')
    ax.set_title('(a) Noise coupling')
    ax.text(.98, .94, 'Mean per-plant\nvariance ratio: 1,796', transform=ax.transAxes,
            ha='right', va='top', fontsize=8)
    for x, value in enumerate(values):
        ax.text(x, value * 1.3, f'{value:.2g}', ha='center', fontsize=7)

    ax = axes[0, 1]
    conditions = [('independent', 'ordinary'), ('CRN', 'ordinary'), ('CRN', 'paired')]
    labels = ['Independent\nordinary', 'CRN\nordinary', 'CRN\npaired']
    x = np.arange(3)
    for offset, cap, color in [(-.18, 512, blue), (.18, 4096, orange)]:
        counts = [next(r['adopted'] for r in ablation if r['cap'] == cap
                       and (r['coupling'], r['support']) == c) for c in conditions]
        ax.bar(x + offset, counts, width=.34, color=color, label=f'{cap:,} pairs')
        for px, count in zip(x + offset, counts):
            ax.text(px, count + .4, str(count), ha='center', fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 29)
    ax.set_yticks([0, 8, 16, 24])
    ax.set_ylabel('Adoptions / 24 plants')
    ax.set_title('(b) Information supplied to the same test')
    ax.legend(loc='upper left', frameon=False)

    ax = axes[1, 0]
    ds = source['radii']
    for support, color, marker, label in [('paired', blue, 'o', 'Paired support'),
                                        ('ordinary', orange, '^', 'Ordinary range')]:
        rows = sorted((r for r in sweep if r['support'] == support), key=lambda r: r['d'])
        assert [r['d'] for r in rows] == ds
        ys = [r['consumed_pairs']['mean'] for r in rows]
        ax.plot(ds, ys, color=color, lw=1.3, label=label)
        for r in rows:
            ax.plot(r['d'], r['consumed_pairs']['mean'], marker=marker,
                    color=color, markersize=4,
                    markerfacecolor='white' if r['censored'] else color)
    ax.axhline(4096, color=gray, lw=.7, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(ds, [str(d) for d in ds])
    ax.set_yticks([128, 512, 2048, 4096], ['128', '512', '2,048', '4,096'])
    ax.set_ylim(100, 6200)
    ax.set_xlabel('Enforced displacement d')
    ax.set_ylabel('Mean consumed pairs')
    ax.set_title('(c) Frozen-direction displacement sweep')
    ax.legend(frameon=False, loc='lower right')

    ax = axes[1, 1]
    gains = sorted(source['gain'], key=lambda r: r['d'])
    ax.errorbar(ds, [r['gain_over_d']['mean'] for r in gains],
                yerr=[r['gain_over_d']['se'] for r in gains],
                fmt='o-', color=blue, markersize=4, capsize=2, lw=1.3)
    ax.axhline(0, color=gray, lw=.7, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_xticks(ds, [str(d) for d in ds])
    ax.set_ylim(-.01, .155)
    ax.set_xlabel('Enforced displacement d')
    ax.set_ylabel('Independent mean gain / d')
    ax.set_title('(d) Gain margin and its limit')
    ax.annotate('8/24 negative gains', xy=(.2, gains[-1]['gain_over_d']['mean']),
                xytext=(.043, .043), fontsize=7,
                arrowprops={'arrowstyle': '-', 'color': gray, 'lw': .7})
    fig.savefig(output_dir / 'certification_mechanism.pdf', metadata={'Title': 'Frozen-policy certification mechanism'})
    plt.close(fig)

    primary = [next(r['adopted'] for r in ablation if r['cap'] == 512
                    and (r['coupling'], r['support']) == c) for c in conditions]
    assert primary == [0, 0, 24]
    assert round(source['variance_ratio']['mean']) == 1796
    assert gains[-1]['positive_plants'] == 16
    checks = {'source_record_sha256': hashlib.sha256(raw).hexdigest(),
              'primary_ablation_adoptions': primary,
              'mean_per_plant_variance_ratio': source['variance_ratio']['mean'],
              'negative_estimated_gains_at_d_0_2': 8,
              'plot_is_reaggregation_only': True,
              'all_checks_passed': True}
    (evidence / 'figure_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Figure generated from frozen summary; numerical checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    main(parser.parse_args().output_dir)
