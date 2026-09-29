"""Recompute plant-level statistics from complete frozen final studies.

No missing final plant is silently dropped. Tests use independent plant/critic
seeds as units; confidence intervals across plants are approximate and are
separate from the conditional finite-sample adoption guarantees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t,ttest_1samp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phase15_summary'


def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def stats(values):
    v=np.asarray(values,dtype=float);n=len(v);mean=float(v.mean())
    se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    error=float(student_t.ppf(.975,n-1))*se if n>1 else 0.
    variable=np.ptp(v)>1e-12*max(1.,float(np.max(abs(v))))
    p=float(ttest_1samp(v,0).pvalue) if variable else (1. if mean==0 else None)
    return dict(n=n,mean=mean,se=se,t_interval=[mean-error,mean+error],p=p,
        positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),values=v.tolist())


def bootstrap(values,groups,bank):
    v=np.asarray(values);g=np.asarray(groups);random=np.random.default_rng(bank);parts=[]
    for name in sorted(set(groups)):
        values=v[g==name]
        parts.append(values[random.integers(0,len(values),(20000,len(values)))].mean(1)*len(values)/len(v))
    means=np.sum(parts,axis=0)
    return np.quantile(means,[.025,.975]).tolist()


def holm(rows):
    order=np.argsort([r['p'] if r['p'] is not None else 1. for r in rows]);running=0.
    for i,j in enumerate(order):
        running=max(running,min(1.,(len(rows)-i)*(rows[j]['p'] if rows[j]['p'] is not None else 1.)))
        rows[j]['holm_p']=running


def frozen_records(name,pattern,expected):
    folder=ROOT/'results'/name;files=sorted(folder.glob(pattern+'/COMPLETE'))
    if len(files)!=expected:raise RuntimeError(f'{name}: {len(files)}/{expected} complete; no partial final inference')
    manifest=json.loads((folder/'source_hashes.json').read_text())
    for rel,digest in manifest.items():
        assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==digest,rel
    records=[]
    for file in files:
        result=json.loads(file.read_text());result['folder']=file.parent;result['id']=file.parent.name;records.append(result)
    return records


def practical():
    records=frozen_records('practical_confirmatory','*_seed*',32)
    methods=['UniformResidual','DecisionResidual','ScalarResidual','VectorResidual','DecisionFull','ARS']
    groups=[r['kind'] for r in records];rows=[];primary={}
    for r in records:
        r['certificates']=json.loads((r['folder']/'certificates.json').read_text())
    for b in [10240,20480,40960,81920]:
        for m in methods:
            name=f'{m}_b{b}';raw=[];deployed=[];cost=[];outcomes=[]
            for r in records:
                j=r['names'].index(name);raw.append(r['raw_gains'][j]);deployed.append(r['deployed_gains'][j])
                gate=next(c for c in r['certificates'] if c['name']==name)
                cost.append(gate['total_used_transitions']);outcomes.append(gate['outcome'])
            row=dict(method=m,training_budget=b,total_allowed=4*b,raw=stats(raw),deployed=stats(deployed),
                average_used_transitions=float(np.mean(cost)),adopted=outcomes.count('adopt'),
                rejected=outcomes.count('reject'),inconclusive=outcomes.count('inconclusive'),
                family_raw={g:float(np.mean(np.array(raw)[np.array(groups)==g])) for g in sorted(set(groups))},
                family_deployed={g:float(np.mean(np.array(deployed)[np.array(groups)==g])) for g in sorted(set(groups))})
            rows.append(row)
            if b==20480:primary[m]=np.array(deployed)
    comparisons=[]
    for m in ['B','UniformResidual','ScalarResidual','ARS']:
        difference=primary['DecisionResidual']-(np.zeros(32) if m=='B' else primary[m])
        s=stats(difference);s.update(comparison=f'DecisionResidual minus {m}',
            stratified_bootstrap_interval=bootstrap(difference,groups,840900))
        comparisons.append(s)
    holm(comparisons)
    result=dict(study='practical_confirmatory',plants=32,plant_ids=[r['id'] for r in records],rows=rows,
        primary_comparisons=comparisons,general_method_success=all(c['mean']>0 and c['holm_p']<.05 for c in comparisons),
        source_shift=stats([r['target_baseline_return']-r['source_baseline_return'] for r in records]),
        research_costs={k:sum(r[k] for r in records) for k in ['certification_research_transitions','final_evaluation_transitions','source_audit_transitions']},
        final_training_transitions=32*6*81920)
    dump(OUT/'practical.json',result);return result


def support():
    records=frozen_records('coupled_support_confirmatory','dim*_seed*',24)
    groups=[r['n'] for r in records];rows=[]
    for r in records:r['certificates']=json.loads((r['folder']/'certificates.json').read_text())
    for method in ['Ridge','ARS']:
        for d in [.05,.15]:
            for b in [5120,20480,81920]:
                name=f'{method}_d{d:g}_b{b}'
                raw=np.array([r['raw_gains'][r['names'].index(name)] for r in records])
                for cap in [128,512,2048,4096]:
                    for kind in ['normalized_support','ordinary_range']:
                        gates=[next(c for c in r['certificates'] if c['name']==name and c['cap']==cap and c['bound_kind']==kind) for r in records]
                        deployed=raw*np.array([c['adopted'] for c in gates])
                        rows.append(dict(method=method,d=d,training_budget=b,cap=cap,bound_kind=kind,
                            raw=stats(raw),deployed=stats(deployed),adopted=sum(c['adopted'] for c in gates),
                            rejected=sum(c['outcome']=='reject' for c in gates),inconclusive=sum(c['outcome']=='inconclusive' for c in gates),
                            mean_pair_count=float(np.mean([c['episodes'] for c in gates])),
                            mean_used_transitions=float(np.mean([c['total_used_transitions'] for c in gates])),
                            mean_bound=float(np.mean([c['bound'] for c in gates])),
                            dimension_gains={str(g):float(deployed[np.array(groups)==g].mean()) for g in sorted(set(groups))},
                            dimension_adoptions={str(g):sum(c['adopted'] for c,group in zip(gates,groups) if group==g) for g in sorted(set(groups))}))
    selected=[r for r in rows if r['method']=='Ridge' and r['d']==.05 and r['training_budget']==81920 and r['cap']==512]
    difference=np.array(selected[0]['deployed']['values'])-selected[1]['deployed']['values']
    primary=stats(difference);primary.update(stratified_bootstrap_interval=bootstrap(difference,groups,851900),
        comparison='normalized support minus ordinary range, same frozen ridge policy and 512 paired episodes')
    result=dict(study='coupled_support_confirmatory',plants=24,plant_ids=[r['id'] for r in records],rows=rows,primary=primary,
        primary_success=primary['mean']>0 and primary['p'] is not None and primary['p']<.05,
        research_costs={k:sum(r[k] for r in records) for k in ['certification_research_transitions','final_evaluation_transitions']},
        final_training_transitions=24*2*2*81920)
    dump(OUT/'support.json',result);return result


def stability():
    records=frozen_records('stability_calibration_confirmatory','dim*_seed*',24)
    groups=[r['n'] for r in records];rows=[];calibration=[]
    for r in records:
        r['certificates']=json.loads((r['folder']/'certificates.json').read_text())
        f=json.loads((r['folder']/'pre_return_calibration.json').read_text())['fits']
        for size,fit in f.items():calibration.append(dict(n=r['n'],seed=r['seed'],samples=int(size),**{k:v for k,v in fit.items() if k!='samples'}))
    for method in ['Ridge','ARS']:
        for d in [.05,.15]:
            for b in [5120,20480,81920]:
                name=f'{method}_d{d:g}_b{b}'
                for size in [512,2048,8192]:
                    for kind in ['calibrated_support','ordinary_range']:
                        gates=[next(c for c in r['certificates'] if c['name']==name and c['samples']==size and c['bound_kind']==kind) for r in records]
                        deployed=[c['deployed_gain'] for c in gates]
                        rows.append(dict(method=method,d=d,training_budget=b,samples=size,bound_kind=kind,
                            deployed=stats(deployed),adopted=sum(c['adopted'] for c in gates),
                            mean_pair_count=float(np.mean([c['episodes'] for c in gates])),
                            mean_used_transitions=float(np.mean([c['total_used_transitions'] for c in gates])),
                            total_allowed_transitions=gates[0]['total_allowed_transitions'],
                            dimension_adoptions={str(g):sum(c['adopted'] for c,group in zip(gates,groups) if group==g) for g in sorted(set(groups))}))
    selected=[r for r in rows if r['method']=='Ridge' and r['d']==.05 and r['training_budget']==81920 and r['samples']==2048]
    difference=np.array(selected[0]['deployed']['values'])-selected[1]['deployed']['values']
    primary=stats(difference);primary['stratified_bootstrap_interval']=bootstrap(difference,groups,853900)
    result=dict(study='stability_calibration_confirmatory',plants=24,plant_ids=[r['id'] for r in records],
        independent_return_replication=False,rows=rows,calibration=calibration,primary=primary,
        all_cover=all(r['all_calibrations_cover'] for r in records),all_contractive=all(r['all_contractions_certified'] for r in records),
        calibration_research_transitions=sum(r['calibration_research_transitions'] for r in records))
    dump(OUT/'stability.json',result);return result


def scalar():
    records=frozen_records('scalar_reward_confirmatory','*_seed*',32)
    allrows=[c for r in records for c in json.loads((r['folder']/'cells.json').read_text())];rows=[]
    for sigma in [.25,1.,4.]:
        for n in [16,64,256]:
            cells=[c for c in allrows if c['sigma']==sigma and c['n']==n];assert len(cells)==32
            for j,method in enumerate(['B','Plugin','FixedAngle','Angular','Norm','Cylinder','Ellipsoid']):
                rows.append(dict(method=method,sigma=sigma,n=n,transitions=16*n,
                    gain=stats([c['exact_gains'][j] for c in cells]),
                    lower=stats([c['lower_returns'][j] for c in cells]),
                    negative=int(sum(c['exact_gains'][j]<-1e-10 for c in cells))))
    violations=[c for c in allrows if any(c['exact_gains'][j]<c['lower_returns'][j]-1e-10 for j in [3,4,5,6])]
    result=dict(study='scalar_reward_confirmatory',plants=32,cells=288,rows=rows,
        covered=sum(c.get('covered',True) for c in allrows),rank_deficient=sum(c['rank_deficient'] for c in allrows),
        bound_violation_cells=len(violations),confidence_policy_outputs=288*4,
        calibration_research_transitions=sum(r['calibration_research_transitions'] for r in records))
    dump(OUT/'scalar.json',result);return result


def reacher():
    records=frozen_records('reacher_confirmatory','seed*',8);rows=[]
    for r in records:r['certificates']=json.loads((r['folder']/'certificates.json').read_text())
    for b in [10000,20000,40000,80000]:
        for m in ['uniform','decision','global','ars','SACFT']:
            name=f'{m}_b{b}'
            raw=[r['raw_gains'][r['names'].index(name)] for r in records]
            deployed=[r['deployed_gains'][r['names'].index(name)] for r in records]
            gates=[next(c for c in r['certificates'] if c['name']==name) for r in records]
            rows.append(dict(method=m,training_budget=b,raw=stats(raw),deployed=stats(deployed),
                adopted=sum(c['adopted'] for c in gates),rejected=sum(c['outcome']=='reject' for c in gates),
                mean_used_transitions=float(np.mean([c['total_used_transitions'] for c in gates]))))
    chosen={r['method']:r for r in rows if r['training_budget']==80000}
    primary=stats(np.array(chosen['uniform']['raw']['values'])-chosen['global']['raw']['values'])
    result=dict(study='reacher_confirmatory',plants=8,plant_ids=[r['id'] for r in records],rows=rows,primary=primary,
        source_shift=stats([r['target_baseline_return']-r['source_baseline_return'] for r in records]),
        stage_cap_events=sum(sum(r['stage_cap_events']) for r in records),
        research_costs={k:sum(r[k] for r in records) for k in ['certification_research_transitions','final_evaluation_transitions','source_audit_transitions']},
        final_training_transitions=8*5*80000)
    dump(OUT/'reacher.json',result);return result


def step():
    records=frozen_records('step_certification','dim*_seed*',12)
    cells=[r for rec in records for r in json.loads((rec['folder']/'cells.json').read_text())]
    rows=[]
    for h in [.05,.025,.0125,.00625]:
        selected=[r for r in cells if r['h']==h];assert len(selected)==12
        for kind in ['normalized_support','ordinary_range']:
            gates=[next(g for g in r['certificates'] if g['bound_kind']==kind) for r in selected]
            rows.append(dict(h=h,bound_kind=kind,gain=stats([r['gain'] for r in selected]),
                gain_over_displacement=stats([r['gain_over_displacement'] for r in selected]),
                variance_over_displacement_squared=stats([r['variance_over_displacement_squared'] for r in selected]),
                support_over_displacement=stats([r['support_over_displacement'] for r in selected]),
                adopted=sum(g['adopted'] for g in gates),rejected=sum(g['outcome']=='reject' for g in gates),
                inconclusive=sum(g['outcome']=='inconclusive' for g in gates),
                mean_pairs=float(np.mean([g['episodes'] for g in gates])),
                mean_certification_transitions=float(np.mean([g['certification_transitions'] for g in gates])),
                dimension_gain_over_displacement={str(n):float(np.mean([r['gain_over_displacement'] for r in selected if r['n']==n])) for n in [4,8,16]}))
    slopes=[]
    for n in [4,8,16]:
        for seed in range(300,304):
            part=sorted([r for r in cells if r['n']==n and r['seed']==seed],key=lambda x:x['h'])
            slope=dict(n=n,seed=seed,positive_all=all(r['gain']>0 for r in part))
            if slope['positive_all']:
                slope['gain_slope']=float(np.polyfit(np.log([r['h'] for r in part]),np.log([r['gain'] for r in part]),1)[0])
            slope['variance_slope']=float(np.polyfit(np.log([r['h'] for r in part]),np.log([r['variance'] for r in part]),1)[0])
            slopes.append(slope)
    result=dict(study='step_certification',plants=12,cells=48,rows=rows,slopes=slopes,
        training_transitions=12*81920,
        research_certification_transitions=sum(r['research_certification_transitions'] for r in cells),
        final_evaluation_transitions=sum(r['final_evaluation_transitions'] for r in cells))
    dump(OUT/'step.json',result);return result


def nonnormal():
    records=frozen_records('nonnormal_confirmatory','dim*_seed*',24)
    groups=[r['n'] for r in records];rows=[]
    for r in records:r['certificates']=json.loads((r['folder']/'certificates.json').read_text())
    for scope in ['primary','extended']:
        for kind in ['matrix_power','ordinary_range']:
            gates=[next(g for g in r['certificates'] if g['scope']==scope and g['bound_kind']==kind) for r in records]
            deployed=[r['raw_gain'] if g['adopted'] else 0. for r,g in zip(records,gates)]
            rows.append(dict(scope=scope,bound_kind=kind,deployed=stats(deployed),
                adopted=sum(g['adopted'] for g in gates),rejected=sum(g['outcome']=='reject' for g in gates),
                mean_pairs=float(np.mean([g['episodes'] for g in gates])),
                mean_used_transitions=float(np.mean([g['total_used_transitions'] for g in gates])),
                dimension_adoptions={str(n):sum(g['adopted'] for g,r in zip(gates,records) if r['n']==n) for n in [4,8,16]}))
    difference=np.array(rows[0]['deployed']['values'])-rows[1]['deployed']['values']
    primary=stats(difference);primary['stratified_bootstrap_interval']=bootstrap(difference,groups,857900)
    result=dict(study='nonnormal_confirmatory',plants=24,plant_ids=[r['id'] for r in records],rows=rows,
        primary=primary,raw_gain=stats([r['raw_gain'] for r in records]),
        relative_gain_percent=stats([100*r['raw_gain']/abs(r['baseline_return']) for r in records]),
        all_matrix_bounds_cover=all(r['matrix_covered'] for r in records),
        all_linear_norms_above_one=all(r['true_linear_norm']>1 for r in records),
        all_scalar_bounds_above_one=all(r['scalar_lipschitz_upper']>1 for r in records),
        scalar_upper_range=[min(r['scalar_lipschitz_upper'] for r in records),max(r['scalar_lipschitz_upper'] for r in records)],
        no_observed_support_violations=all(not r['observed_return_support_violation'] and not r['observed_trajectory_envelope_violation'] for r in records),
        dimension_relative_gains={str(n):float(np.mean([100*r['raw_gain']/abs(r['baseline_return']) for r in records if r['n']==n])) for n in [4,8,16]},
        research_costs={k:sum(r[k] for r in records) for k in ['training_transitions','calibration_transitions','research_certification_transitions','final_evaluation_transitions']})
    dump(OUT/'nonnormal.json',result);return result


if __name__=='__main__':
    studies=['practical','support','stability','scalar','reacher','step','nonnormal']
    ap=argparse.ArgumentParser();ap.add_argument('study',choices=studies+['all']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in (studies if args.study=='all' else [args.study]):
        result=globals()[name]();print(name,json.dumps({k:v for k,v in result.items() if k in ['primary','primary_comparisons','general_method_success','primary_success','covered','bound_violation_cells','all_cover','all_contractive']},indent=2))
