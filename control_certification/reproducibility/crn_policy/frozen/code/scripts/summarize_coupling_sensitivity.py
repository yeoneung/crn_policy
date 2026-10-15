"""Complete-case summaries/plots of the frozen three-experiment extension."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coupling_sensitivity import OUT, ROOT, DS, METHODS, CAPS, ELL_FACTORS


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/np.sqrt(len(a))),
                min=float(a.min()),max=float(a.max()),median=float(np.median(a)))
def stopstats(rows):
    return dict(n=len(rows),adopted=sum(r['adopted'] for r in rows),
                censored=sum(r['outcome']=='inconclusive' for r in rows),
                rejected=sum(r['outcome']=='reject' for r in rows),
                consumed_pairs=stats([r['pairs'] for r in rows]),
                deployed_gain=stats([r['deployed_gain'] for r in rows]))


def summarize():
    folders=sorted(OUT.glob('dim*_seed*'))
    assert len(folders)==24 and all((p/'COMPLETE.json').exists() for p in folders),"No partial final inference"
    for relative,expected in json.loads((OUT/'input_hashes.json').read_text()).items():assert sha(ROOT/relative)==expected,relative
    for relative,expected in json.loads((OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/relative)==expected,relative
    rows=[];metrics=[];ell=[];abl=[];complete=[]
    for p in folders:
        end=json.loads((p/'COMPLETE.json').read_text());complete.append(end)
        for name,expected in end['outputs'].items():assert sha(p/name)==expected,(p,name)
        for name,target in [('stopping.json',rows),('metrics.json',metrics),('ell_replay.json',ell)]:
            target.extend(dict(plant=p.name,n=end['n'],seed=end['seed'],**r) for r in json.loads((p/name).read_text()))
        abl.append(dict(plant=p.name,**json.loads((p/'ablation.json').read_text())))
    sweep=[];gain=[];ablation=[];sensitivity=[]
    for d in DS:
        m=[r for r in metrics if r['d']==d]
        gain.append(dict(d=float(d),gain=stats([r['gain'] for r in m]),gain_over_d=stats([r['gain_over_d'] for r in m]),
                         variance_over_d_squared=stats([r['variance_over_d_squared'] for r in m]),
                         positive_plants=sum(r['gain']>0 for r in m),
                         positive_mc_lower=sum(r['gain']>1.96*r['gain_mc_se'] for r in m),
                         angle_cap_fraction=stats([r['angle_cap_fraction'] for r in m]),
                         mean_displacement_over_d=stats([r['mean_displacement']/d for r in m])))
        for support in ['paired','ordinary']:
            for method in METHODS:
                for cap in CAPS:
                    group=[r for r in rows if r['d']==d and r['coupling']=='CRN' and r['support']==support and r['method']==method and r['cap']==cap]
                    sweep.append(dict(d=float(d),support=support,method=method,cap=cap,**stopstats(group)))
    for coupling,support in [('independent','ordinary'),('CRN','ordinary'),('CRN','paired')]:
        for method in METHODS:
            for cap in CAPS:
                group=[r for r in rows if r['d']==.05 and r['coupling']==coupling and r['support']==support and r['method']==method and r['cap']==cap]
                ablation.append(dict(coupling=coupling,support=support,method=method,cap=cap,**stopstats(group)))
    ell_audits=[]
    for factor in ELL_FACTORS:
        group=[r for r in ell if r['ell_factor']==factor and r['method']==METHODS[0] and r['cap']==512]
        ell_audits.append(dict(factor=factor,premise_satisfied_plants=sum(r['premise_satisfied'] for r in group),
            structural_support_available=sum(r['structural_support_available'] for r in group),
            covers_analytic_upper=sum(r['covers_analytic_upper'] for r in group),
            contradicted_by_origin_witness=sum(r['contradicted_by_origin_witness'] for r in group),
            bound=stats([r['bound'] for r in group]),kappa=stats([r['kappa'] for r in group]),
            old_return_violations=sum(r['old_return_violations'] for r in group),old_return_pairs=sum(r['old_return_pairs'] for r in group),
            fresh_return_violations=sum(r['fresh_return_violations'] for r in group),fresh_return_pairs=sum(r['fresh_return_pairs'] for r in group),
            old_max_state_time_violations=sum(r['old_max_state_time_violations'] for r in group),
            fresh_state_time_violations=sum(r['fresh_state_time_violations'] for r in group),
            fresh_episodes_with_state_violation=sum(r['fresh_episodes_with_state_violation'] for r in group)))
        for method in METHODS:
            for cap in CAPS:
                group=[r for r in ell if r['ell_factor']==factor and r['method']==method and r['cap']==cap]
                sensitivity.append(dict(factor=factor,method=method,cap=cap,**stopstats(group)))
    slopes=[]
    for method in METHODS:
        for support in ['paired','ordinary']:
            group=[r for r in sweep if r['method']==method and r['support']==support and r['cap']==4096]
            slopes.append(dict(method=method,support=support,
                log_log_slope_of_cap_truncated_mean=float(np.polyfit(np.log([r['d'] for r in group]),np.log([r['consumed_pairs']['mean'] for r in group]),1)[0]),
                censor_counts=[r['censored'] for r in group],rejection_counts=[r['rejected'] for r in group],
                note="Cap-truncated consumption includes negative/futility stops. This is not an uncensored positive-certification complexity exponent."))
    result=dict(plants=24,radii=DS.tolist(),sweep=sweep,gain=gain,ablation=ablation,ell=sensitivity,ell_audits=ell_audits,
                variance_ratio=stats([r['variance_ratio'] for r in abl]),crn_variance=stats([r['crn_variance'] for r in abl]),
                independent_variance=stats([r['independent_variance'] for r in abl]),slopes=slopes,
                actual_simulator_transitions=sum(r['actual_simulator_transitions'] for r in complete),
                logical_comparison_transitions=sum(r['logical_comparison_transitions'] for r in complete),
                new_training_transitions=0,total_worker_seconds=sum(r['seconds'] for r in complete),
                fresh_stopping_cells=len(rows),ell_stopping_cells=len(ell),analysis_sha256=sha(Path(__file__)))
    dump(OUT/'summary.json',result)
    dump(OUT/'complete_records.json',dict(stopping=rows,metrics=metrics,ell=ell,ablation=abl))
    print(json.dumps({k:result[k] for k in ['variance_ratio','slopes','gain','ell_audits','actual_simulator_transitions']},indent=2))
    return result,rows,metrics,ell


def save(fig,name):
    target=ROOT/'manuscript/figs_revision'
    target.mkdir(parents=True,exist_ok=True)
    fig.savefig(target/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(target/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def plots(s,rows,metrics,ell):
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.85),layout='constrained')
    colors={'paired':'#146b76','ordinary':'#b25a21'}
    labels={'fixed_mixture':'Betting','predictable_empirical_bernstein':'PM-EB'}
    for method,ls in zip(METHODS,['-','--']):
        for support in ['paired','ordinary']:
            g=[r for r in s['sweep'] if r['method']==method and r['support']==support and r['cap']==4096]
            means=[r['consumed_pairs']['mean'] for r in g]
            axes[0].plot(DS,means,ls,color=colors[support],lw=1.7,label=f'{labels[method]}, {support}')
            for r in g:
                axes[0].plot(r['d'],r['consumed_pairs']['mean'],'D' if r['rejected'] else 'o',ms=4,color=colors[support],mfc='white' if r['censored'] else colors[support])
    axes[0].axhline(4096,color='.6',ls=':',lw=.8)
    guide=np.array([.016,.04])
    axes[0].plot(guide,40/guide,':',color='.25',lw=1,label='Slope -1 guide (not a fit)')
    axes[0].set(xscale='log',yscale='log',xlabel='Displacement bound d',ylabel='Mean consumed pairs',ylim=(100,6000))
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,fontsize=7,frameon=False,ncol=3,loc='outside upper center',handlelength=2.8)
    axes[0].set_title('Fixed learned directions; 4,096-pair cap')
    for plant in sorted(set(r['plant'] for r in metrics)):
        g=sorted([r for r in metrics if r['plant']==plant],key=lambda r:r['d'])
        axes[1].plot(DS,[r['gain_over_d'] for r in g],color='.75',lw=.5,alpha=.6)
    axes[1].errorbar(DS,[r['gain_over_d']['mean'] for r in s['gain']],
        yerr=[r['gain_over_d']['se'] for r in s['gain']],color='#146b76',marker='o',capsize=2,label='Mean +/- plant SE')
    axes[1].axhline(0,color='.5',lw=.7)
    axes[1].set(xscale='log',xlabel='Displacement bound d',ylabel='Independent return gain / d',title='Separate evaluation bank')
    axes[1].legend(fontsize=7,frameon=False,loc='lower left')
    for a in axes:
        a.set_xticks(DS,[str(x) for x in DS]);a.minorticks_off()
    save(fig,'v2_displacement_sweep')

    fig,axes=plt.subplots(1,3,figsize=(7,2.5),layout='constrained')
    for i,key in enumerate(['independent_variance','crn_variance']):
        v=s[key]
        axes[0].bar(i,v['mean'],color=['#b25a21','#146b76'][i])
    axes[0].set(xticks=[0,1],xticklabels=['Independent','CRN'],yscale='log',ylabel='Contrast variance',title='Same frozen policy')
    conditions=[('independent','ordinary'),('CRN','ordinary'),('CRN','paired')]
    for ax,cap in zip(axes[1:],CAPS):
        for j,method in enumerate(METHODS):
            vals=[next(r['adopted'] for r in s['ablation'] if r['coupling']==co and r['support']==su and r['method']==method and r['cap']==cap) for co,su in conditions]
            ax.bar(np.arange(3)+(j-.5)*.32,vals,width=.3,label=labels[method],color=['#146b76','#638ebb'][j])
        ax.set(xticks=[0,1,2],xticklabels=['Ind.\nglobal','CRN\nglobal','CRN\npaired'],ylim=(0,27),yticks=[0,8,16,24],title=f'{cap:,}-pair cap',ylabel='Adoptions / 24')
        ax.legend(fontsize=6.5,frameon=False,loc='upper left')
    save(fig,'v2_coupling_ablation')

    fig,axes=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
    x=np.asarray(ELL_FACTORS)
    for ax in axes:
        ax.axvspan(.45,.71,color='#f8e6e0',zorder=0)
        ax.set(xscale='log',xlabel='Assumed / true residual Lipschitz bound',xlim=(.45,11))
        ax.set_xticks(x,['0.5','1','2','5','10']);ax.minorticks_off()
    for plant in sorted(set(r['plant'] for r in ell)):
        g=sorted([r for r in ell if r['plant']==plant and r['method']==METHODS[0] and r['cap']==512],key=lambda r:r['ell_factor'])
        axes[0].plot(x,[r['bound'] for r in g],color='.65',alpha=.5,lw=.6)
    axes[0].plot(x,[r['bound']['mean'] for r in s['ell_audits']],color='#146b76',marker='o',label='Mean supplied support')
    axes[0].set(yscale='log',ylabel='Return-difference support bound',title='Same 2,048 reset pairs')
    axes[0].legend(fontsize=7,frameon=False,loc='upper left')
    for method,col in zip(METHODS,['#146b76','#638ebb']):
        for cap,ls in zip(CAPS,['-','--']):
            vals=[next(r['adopted'] for r in s['ell'] if r['factor']==factor and r['method']==method and r['cap']==cap) for factor in x]
            axes[1].plot(x,vals,ls,color=col,marker='o',ms=3,label=f'{labels[method]}, {cap:,}')
    axes[1].set(ylabel='Adoptions / 24',ylim=(-.5,27),yticks=[0,8,16,24],title='Original ordered return bank')
    handles,names=axes[1].get_legend_handles_labels()
    fig.legend(handles,names,fontsize=7,frameon=False,ncol=4,loc='outside upper center',handlelength=2.8)
    save(fig,'v2_ell_sensitivity')


def tables(s):
    target=ROOT/'manuscript/tables'
    target.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{tabular}{lrccc}',r'\toprule',r'Test & Cap & Ind./global & CRN/global & CRN/paired\\',r'\midrule']
    for method,label in zip(METHODS,['Betting','PM-EB']):
        for cap in CAPS:
            vals=[next(r['adopted'] for r in s['ablation'] if r['coupling']==co and r['support']==su and r['method']==method and r['cap']==cap) for co,su in [('independent','ordinary'),('CRN','ordinary'),('CRN','paired')]]
            lines.append(f'{label} & {cap:,} & {vals[0]} & {vals[1]} & {vals[2]}'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (target/'v2_noise_ablation.tex').write_text('\n'.join(lines)+'\n')
    def arc(row): return f"{row['adopted']}/{row['rejected']}/{row['censored']}"
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'$d$ & Test & Cap & Paired A/R/C & Paired mean $N$ & Ordinary A/R/C & Ordinary mean $N$\\',r'\midrule']
    for d in DS:
        for method,label in zip(METHODS,['Betting','PM-EB']):
            for cap in CAPS:
                pair,ordinary=[next(r for r in s['sweep'] if r['d']==d and r['method']==method and r['support']==support and r['cap']==cap) for support in ['paired','ordinary']]
                lines.append(f"{d:g} & {label} & {cap:,} & {arc(pair)} & {pair['consumed_pairs']['mean']:.1f} & {arc(ordinary)} & {ordinary['consumed_pairs']['mean']:.1f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (target/'v2_displacement_full.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Factor & Contracting & Mean $B_*$ & Betting, 512 & PM-EB, 512 & Betting, 4,096 & PM-EB, 4,096\\',r'\midrule']
    for audit in s['ell_audits']:
        factor=audit['factor'];vals=[]
        for cap in CAPS:
            for method in METHODS:
                row=next(r for r in s['ell'] if r['factor']==factor and r['cap']==cap and r['method']==method)
                vals.append(f"{row['adopted']} ({row['consumed_pairs']['mean']:.0f})")
        lines.append(f"{factor:g} & {audit['structural_support_available']}/24 & {audit['bound']['mean']:.3f} & "+' & '.join(vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (target/'v2_ell_full.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    result,rows,metrics,ell=summarize();plots(result,rows,metrics,ell);tables(result)
