"""Post-hoc descriptive nominal-reference evaluation; no new adoption tests.

Run the frozen baseline on its nominal gain=1, damping=0 plant using the
same initial states and torque noises as each original held-out bank.
Preserve all original policies, protocols, and return banks.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'code/results/pendulum_regional_study/final'
OUT=ROOT/'code/results/pendulum_nominal_reference'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def initial_bank(cfg,seed):
    n=4096;rng=np.random.default_rng([seed,4]);ang=rng.uniform(0,2*np.pi,n)
    rad=np.sqrt(rng.uniform(size=n))*cfg['initial_radius']
    x=np.column_stack([np.cos(ang),np.sin(ang)])*rad[:,None]@np.linalg.inv(np.array(cfg['T'])).T
    return x,rng.uniform(-cfg['width'],cfg['width'],(cfg['H'],n))

def baseline(x,noise,K,gain=1.,damping=0.):
    x=x.copy();ret=np.zeros(len(x))
    for w in noise:
        u=np.clip(-x@K,-2,2);theta=x[:,0];v=x[:,1]
        ret-=((theta+np.pi)%(2*np.pi)-np.pi)**2+.1*v*v+.001*u*u
        v=np.clip(v+.05*(15*np.sin(theta)+3*gain*u-damping*v+3*w),-8,8)
        x=np.column_stack([theta+.05*v,v])
    return ret

def main():
    threadpool_limits(1);OUT.mkdir(parents=True,exist_ok=True)
    inputs=[INPUT/'protocol.json']
    for p in sorted(INPUT.glob('seed*')):inputs.extend([p/'metrics.json',p/'evaluation.npz',p/'frozen_policy.json'])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    source=sha(Path(__file__));cfg=json.loads((INPUT/'protocol.json').read_text())['config']
    if (OUT/'COMPLETE.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['source_sha256']==source and p['input_hashes']==hashes
        for k,v in json.loads((OUT/'COMPLETE.json').read_text())['outputs'].items():assert sha(OUT/k)==v
        print((OUT/'summary.json').read_text());return
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve unfinished reference evaluation')
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,input_hashes=hashes,
        purpose='Post-hoc reference metric requested after the original study; descriptive, no new significance/adoption claim.',
        nominal_gain=1.,nominal_damping=0.,seeds=list(range(1000,1008)),episodes_per_seed=4096,
        reference='Same baseline and initial/noise banks; only plant gain/damping set to nominal.',
        estimand='sum(target repaired - target baseline) / sum(nominal baseline - target baseline)',
        caveat='Across-environment nominal-baseline gap, not the target optimal-policy regret or a recoverable upper bound.',
        denominator_rule='Report every signed per-plant gap; label nonpositive denominators rather than calling them lost return.',
        bootstrap='20000 paired plant resamples; ratio of means; fixed seed 2026092807; descriptive percentile interval')
    dump(OUT/'protocol.json',protocol)
    records=[];arrays={};audit_error=0
    for seed in protocol['seeds']:
        folder=INPUT/f'seed{seed}';metrics=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'evaluation.npz') as old:target=old['returns']
        x,w=initial_bank(cfg,seed);K=np.array(cfg['K'])
        nominal=baseline(x,w,K)
        # Independently reconstruct a target prefix to check matching bank/semantics.
        target_prefix=baseline(x[:128],w[:,:128],K,metrics['gain'],metrics['damping'])
        err=float(np.max(abs(target_prefix-target[0,:128])));assert err<1e-12;audit_error=max(audit_error,err)
        improvement=target[1]-target[0];gap=nominal-target[0];mu=float(improvement.mean());loss=float(gap.mean())
        records.append(dict(seed=seed,actuator_gain=metrics['gain'],damping=metrics['damping'],
            nominal_baseline_return=float(nominal.mean()),target_baseline_return=float(target[0].mean()),
            target_repaired_return=float(target[1].mean()),mean_gain=mu,nominal_gap=loss,
            gap_se=float(gap.std(ddof=1)/np.sqrt(len(gap))),positive_gap=loss>0,
            signed_recovery_fraction=None if loss==0 else mu/loss))
        arrays[f'nominal_seed{seed}']=nominal
    np.savez_compressed(OUT/'nominal_returns.npz',**arrays)
    gains=np.array([r['mean_gain'] for r in records]);losses=np.array([r['nominal_gap'] for r in records])
    rng=np.random.default_rng(2026092807);idx=rng.integers(0,len(records),(20000,len(records)))
    den=losses[idx].mean(axis=1);num=gains[idx].mean(axis=1)
    ratio=float(gains.mean()/losses.mean()) if losses.mean()!=0 else None
    intervals=np.quantile(num/den,[.025,.975]).tolist() if np.all(den>0) else None
    tube=(cfg['R']-cfg['rho'])*(1-cfg['kappa'])/cfg['action_gain']
    summary=dict(records=records,mean_gain=float(gains.mean()),mean_nominal_gap=float(losses.mean()),
        ratio_of_means=ratio,mean_plant_ratio=float(np.mean([r['signed_recovery_fraction'] for r in records])),
        positive_gap_plants=int(np.sum(losses>0)),bootstrap_ratio_interval=intervals,
        bootstrap_nonpositive_denominator_count=int(np.sum(den<=0)),
        constant_displacement_tube_sufficient_upper=tube,chosen_d=cfg['d'],chosen_over_tube_upper=cfg['d']/tube,
        interpretation='d is a prespecified conservative cap within the sufficient tube limit, not its maximum or a learned optimum.',
        additional_reference_transitions=8*4096*cfg['H'],additional_audit_transitions=8*128*cfg['H'],
        target_prefix_max_absolute_error=audit_error)
    dump(OUT/'summary.json',summary)
    dump(OUT/'COMPLETE.json',dict(outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
