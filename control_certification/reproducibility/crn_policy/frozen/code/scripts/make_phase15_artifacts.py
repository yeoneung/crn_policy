"""Scientific figures and paper tables, computed from frozen study summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/'results/phase15_summary'
FIG=ROOT.parent/'manuscript/figs_revision'
TABLE=ROOT.parent/'manuscript/tables'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#555555']


def read(name):return json.loads((SUMMARY/f'{name}.json').read_text())
def latex(path,lines):path.write_text('\n'.join(lines)+'\n')
def pm(stat,scale=1):return f"${scale*stat['mean']:.3f}\\pm{scale*stat['se']:.3f}$"
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def support():
    s=read('support');cal=read('stability');rows=s['rows']
    files=sorted((ROOT/'results/coupled_support_confirmatory').glob('dim*/COMPLETE'))
    records=[json.loads(p.read_text()) for p in files]
    fig,ax=plt.subplots(1,3,figsize=(6.6,2.0),layout='constrained')
    means=[];errors=[]
    for n in [4,8,16]:
        values=[100*r['raw_gains'][r['names'].index('Ridge_d0.05_b81920')]/abs(r['mean_returns'][0]) for r in records if r['n']==n]
        means.append(np.mean(values));errors.append(np.std(values,ddof=1)/np.sqrt(8))
    ax[0].bar([0,1,2],means,yerr=errors,color=COLORS[0],width=.6,capsize=3)
    ax[0].set(xticks=[0,1,2],xticklabels=['4','8','16'],xlabel='State/action dimension',ylabel='Return gain / |baseline| (%)',title='Independent return')
    points=np.arange(16,4097,16)
    for color,kind,label in [(COLORS[0],'normalized_support','Paired support'),(COLORS[1],'ordinary_range','Ordinary range')]:
        gates=[]
        for p in files:
            rec=json.loads((p.parent/'certificates.json').read_text())
            gates.append(next(g for g in rec if g['name']=='Ridge_d0.05_b81920' and g['cap']==4096 and g['bound_kind']==kind))
        curve=[sum(g['adopted'] and g['episodes']<=t for g in gates) for t in points]
        ax[1].step(points,curve,where='post',label=label,color=color)
    ax[1].axvline(512,color='.6',linestyle=':',linewidth=1)
    ax[1].set(xscale='log',xlim=(64,4096),ylim=(-.5,25),yticks=[0,8,16,24],xlabel='Paired episodes',ylabel='Adopted plants / 24',title='Same policy, same data')
    ax[1].legend(fontsize=6,loc='lower right',frameon=False)
    for kind,label,color in [('calibrated_support','Calibrated support',COLORS[2]),('ordinary_range','Extra return samples',COLORS[1])]:
        chosen=[next(r for r in cal['rows'] if r['method']=='Ridge' and r['d']==.05 and r['training_budget']==81920 and r['samples']==n and r['bound_kind']==kind) for n in [512,2048,8192]]
        ax[2].plot([1024,4096,16384],[r['adopted'] for r in chosen],marker='o',color=color,label=label)
    ax[2].set(xscale='log',ylim=(-.5,25),yticks=[0,8,16,24],xlabel='Calibration allocation (steps)',ylabel='Adopted plants / 24',title='Charge calibration cost')
    ax[2].legend(fontsize=6,frameon=False,loc='center right')
    save(fig,'phase15_support')
    lines=[r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Certificate & Adopt & Gain ($\times10^3$) & Steps\\',r'\midrule']
    for kind,label in [('normalized_support','Known support'),('ordinary_range','Ordinary')]:
        r=next(r for r in rows if r['method']=='Ridge' and r['d']==.05 and r['training_budget']==81920 and r['cap']==512 and r['bound_kind']==kind)
        lines.append(f"{label} & {r['adopted']}/24 & {pm(r['deployed'],1000)} & {r['mean_used_transitions']:,.0f}"+r'\\')
    lines+=[r'\midrule']
    for kind,label in [('calibrated_support','Calibrated'),('ordinary_range','Ordinary (+data)')]:
        r=next(r for r in cal['rows'] if r['method']=='Ridge' and r['d']==.05 and r['training_budget']==81920 and r['samples']==2048 and r['bound_kind']==kind)
        lines.append(f"{label} & {r['adopted']}/24 & {pm(r['deployed'],1000)} & {r['mean_used_transitions']:,.0f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    latex(TABLE/'phase15_support.tex',lines)


def practical():
    s=read('practical');lines=[r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Proposal & Raw & Deployed & Adopt\\',r'\midrule']
    labels={'UniformResidual':'Uniform','DecisionResidual':'Decision','ScalarResidual':'Scalar NN','VectorResidual':'Vector NN','DecisionFull':'No prior','ARS':'ARS'}
    for r in s['rows']:
        if r['training_budget']!=20480:continue
        lines.append(f"{labels[r['method']]} & {pm(r['raw'])} & {pm(r['deployed'])} & {r['adopted']}/32"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];latex(TABLE/'phase15_practical.tex',lines)
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.15),layout='constrained')
    for j,m in enumerate(['DecisionResidual','UniformResidual','ScalarResidual','ARS']):
        r=[r for r in s['rows'] if r['method']==m]
        for a,field,title in [(ax[0],'raw','Frozen proposals'),(ax[1],'deployed','After independent certification')]:
            a.plot([v['training_budget'] for v in r],[v[field]['mean'] for v in r],marker='o',color=COLORS[j],label=labels[m])
            a.set(xscale='log',xlabel='Training transitions',ylabel='Mean return gain',title=title)
            a.axhline(0,color='.7',linewidth=.6)
    ax[1].legend(fontsize=7,frameon=False);save(fig,'phase15_practical')


def scalar():
    s=read('scalar');lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Rule & Exact gain & Valid lower bound\\',r'\midrule']
    for r in s['rows']:
        if r['sigma']!=.25 or r['n']!=64 or r['method'] in ['B','Cylinder']:continue
        lower=pm(r['lower']) if r['method'] not in ['Plugin','FixedAngle'] else r'--'
        lines.append(f"{r['method']} & {pm(r['gain'])} & {lower}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];latex(TABLE/'phase15_scalar.tex',lines)


def information():
    from scalar_information import thresholds
    angles=np.geomspace(.025,.4,100);e=[];b=[]
    for rho in angles:
        th=thresholds(rho,.1,1.,.5,.2);e.append(th['ellipse']);b.append(th['norm'])
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.0),layout='constrained')
    ax[0].loglog(angles,e,color=COLORS[0],label='Full covariance: decision')
    ax[0].loglog(angles,b,color=COLORS[1],label='Enclosing norm ball')
    ax[0].set(xlabel='Permitted angle (radians)',ylabel='Queries for 90% power',title='Exact scalar Gaussian calculation')
    ax[0].legend(fontsize=7,frameon=False)
    from return_certificate import FRACTIONS,WEIGHTS,log_wealth
    mu=np.geomspace(.0001,.1,100);alpha=.001
    ordinary=np.log(1/alpha)/np.log1p(mu)
    paired=np.log(1/alpha)/np.log1p(np.ones_like(mu)/10)
    ax[1].loglog(mu,ordinary,color=COLORS[1],label='Only ordinary range $B=1$')
    ax[1].loglog(mu,paired,color=COLORS[0],label=r'Proved paired range $B=10\mu$')
    ax[1].set(xlabel='Constant positive return gain',ylabel='Necessary observations',title='Range-only barrier, zero variance')
    ax[1].legend(fontsize=7,frameon=False)
    save(fig,'phase15_information')


def reacher():
    s=read('reacher');lines=[r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Proposal & Raw & Deployed & Adopt\\',r'\midrule']
    labels={'uniform':'Uniform','decision':'Decision','global':'Global angle','ars':'ARS','SACFT':'SAC fine-tune'}
    for r in s['rows']:
        if r['training_budget']!=80000:continue
        lines.append(f"{labels[r['method']]} & {pm(r['raw'])} & {pm(r['deployed'])} & {r['adopted']}/8"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];latex(TABLE/'phase15_reacher.tex',lines)


def step():
    s=read('step');fig,ax=plt.subplots(1,3,figsize=(6.6,2.0),layout='constrained')
    selected=[r for r in s['rows'] if r['bound_kind']=='normalized_support'];h=np.array([r['h'] for r in selected])
    mean=np.array([r['gain_over_displacement']['mean'] for r in selected])
    sd=np.sqrt([r['variance_over_displacement_squared']['mean'] for r in selected])
    ax[0].plot(h,mean,'o-',color=COLORS[0],label='Mean gain / d')
    ax[0].plot(h,sd,'s--',color=COLORS[2],label='RMS deviation / d')
    ax[0].set(ylabel='Scaled return difference',title='Scaled signal and noise')
    ax[0].legend(fontsize=6,frameon=False)
    for kind,label,color in [('normalized_support','Paired support',COLORS[0]),('ordinary_range','Ordinary (capped)',COLORS[1])]:
        r=[r for r in s['rows'] if r['bound_kind']==kind]
        ax[1].plot(h,[v['mean_pairs'] for v in r],marker='o',color=color,label=label)
        ax[2].plot(h,[v['mean_certification_transitions'] for v in r],marker='o',color=color,label=label)
        if kind=='ordinary_range':
            ax[1].plot(h[1:],[4096]*3,'^',color=color,markersize=7,markerfacecolor='white')
            ax[2].plot(h[1:],[v['mean_certification_transitions'] for v in r][1:],'^',color=color,markersize=7,markerfacecolor='white')
    ax[1].set(yscale='log',ylabel='Mean paired episodes',title='Validation episodes')
    ax[1].legend(fontsize=6,frameon=False,loc='center left')
    ax[2].set(yscale='log',ylabel='Mean simulator transitions',title='Simulator transitions')
    for a in ax:
        a.set(xscale='log',xlabel='Control interval h',xlim=(.056,.0056))
        a.set_xticks([.05,.0125,.00625],['.05','.0125','.00625']);a.minorticks_off()
    save(fig,'phase15_step')


if __name__=='__main__':
    FIG.mkdir(exist_ok=True);TABLE.mkdir(exist_ok=True)
    support();practical();scalar();information()
    if (SUMMARY/'reacher.json').exists():reacher()
    if (SUMMARY/'step.json').exists():step()
