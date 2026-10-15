"""Tables/figure from frozen final banks; no simulation or selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'code/results/pendulum_regional_study'

def main():
    s=json.loads((DATA/'final/summary.json').read_text())
    labels=['Global range','Regional marginal','Paired, constant cap','Paired, state cap']
    kinds=['global','regional_marginal','constantcap','statecap']
    methods=['fixed_mixture','predictable_empirical_bernstein']
    rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Support & Bound & Betting A/8 & Mean pairs & PM-EB A/8 \\',r'\midrule']
    for k,l in zip(kinds,labels):
        a=next(r for r in s['groups'] if r['kind']==k and r['method']==methods[0])
        b=next(r for r in s['groups'] if r['kind']==k and r['method']==methods[1])
        rows.append(f'{l} & {s["bounds"][k]:.5g} & {a["adoptions"]} & {a["mean_pairs"]:,.0f} & {b["adoptions"]} '+r'\\')
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'manuscript/tables/v2_pendulum_regional.tex').write_text('\n'.join(rows)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    for j,method in enumerate(methods):
        vals=[next(r for r in s['groups'] if r['kind']==k and r['method']==method) for k in kinds]
        axes[0].bar(np.arange(4)+(j-.5)*.34,[v['adoptions'] for v in vals],width=.32,label=['Betting','PM-EB'][j])
    axes[0].set(xticks=np.arange(4),xticklabels=['Global','Regional\nmarginal','Paired\nconstant','Paired\nstate cap'],ylabel='Adoptions / 8',ylim=(0,9))
    axes[0].legend(frameon=False,fontsize=7)
    records=[json.loads(p.read_text()) for p in sorted((DATA/'final').glob('seed*/metrics.json'))]
    axes[1].errorbar([r['gain'] for r in records],[1e5*r['independent_gain'] for r in records],
        yerr=[1.96e5*r['independent_gain_se'] for r in records],fmt='o',color='#146b76',capsize=3)
    axes[1].set(xlabel='Actuator gain',ylabel='Independent return gain (x 1e-5)',ylim=(0,2.2))
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ['pdf','png']:fig.savefig(ROOT/f'manuscript/figs_revision/v2_pendulum_regional.{ext}',dpi=200)
    plt.close(fig)
    report=dict(final_mean_gain=s['mean_gain'],plant_se=float(np.std([r['independent_gain'] for r in records],ddof=1)/np.sqrt(8)),
                relative_gain_percent=100*s['mean_relative_gain'],
                evaluation_pair_se_max=max(r['independent_gain_se'] for r in records),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DATA/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
