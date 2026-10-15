"""Post-hoc parameter-informed LQR references on unchanged held-out banks.

Primary: clip the target-LQR residual through the identical state-dependent cap.
Secondary: cap times residual sign (matches learned action magnitude); target
LQR without the repair cap (retains native torque clip) isolates its cap loss.
All definitions are fixed before any new return is generated. No new tests.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from scipy.linalg import solve_discrete_are
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'code/results/pendulum_regional_study/final'
NOM=ROOT/'code/results/pendulum_nominal_reference'
OUT=ROOT/'code/results/pendulum_oracle_reference'
KINDS=['capped_residual','capped_direction','uncapped_target_lqr']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def bank(cfg,seed):
    rng=np.random.default_rng([seed,4]);n=4096
    ang=rng.uniform(0,2*np.pi,n);rad=cfg['initial_radius']*np.sqrt(rng.uniform(size=n))
    x=np.column_stack([np.cos(ang),np.sin(ang)])*rad[:,None]@np.linalg.inv(cfg['T']).T
    return x,rng.uniform(-cfg['width'],cfg['width'],(cfg['H'],n))

def lqr(g,c):
    A=np.array([[1.0375,.05*(1-.05*c)],[.75,1-.05*c]])
    G=g*np.array([[.0075],[.15]]);Q=np.diag([1.,.1]);R=np.array([[.001]])
    S=solve_discrete_are(A,G,Q,R);K=np.linalg.solve(R+G.T@S@G,G.T@S@A).ravel()
    residual=S-(Q+A.T@S@A-A.T@S@G@np.linalg.solve(R+G.T@S@G,G.T@S@A))
    return K,dict(riccati_residual=float(np.max(abs(residual))),closed_loop_spectral_radius=float(max(abs(np.linalg.eigvals(A-G@K[None,:])))))

def rollout(g,c,cfg,Kt,x0,noise,kind):
    K=np.array(cfg['K']);T=np.array(cfg['T']);x=x0.copy();ret=np.zeros(len(x));maximum_displacement=0.;violations=0;saturations=0;cap_active=0
    for w in noise:
        b=np.clip(-x@K,-2,2);delta=x@(K-Kt)
        h=cfg['d']*np.minimum(1,np.linalg.norm(x@T.T,axis=1)/cfg['initial_radius'])
        if kind=='capped_residual':action=b+np.clip(delta,-h,h)
        elif kind=='capped_direction':action=b+h*np.sign(delta)
        elif kind=='uncapped_target_lqr':action=-x@Kt
        else:raise ValueError(kind)
        cap_active+=int(np.sum(abs(delta)>h));saturations+=int(np.sum(abs(action)>2))
        u=np.clip(action,-2,2);disp=abs(u-b);maximum_displacement=max(maximum_displacement,float(disp.max()))
        violations+=int(np.sum(disp>h+1e-12))
        theta=x[:,0];v=x[:,1]
        ret-=((theta+np.pi)%(2*np.pi)-np.pi)**2+.1*v*v+.001*u*u
        v=np.clip(v+.05*(15*np.sin(theta)+3*g*u-c*v+3*w),-8,8)
        x=np.column_stack([theta+.05*v,v])
    if kind.startswith('capped'):assert violations==0
    return ret,dict(max_action_displacement=maximum_displacement,statecap_violations=violations,
        torque_saturation_steps=saturations,raw_lqr_residual_above_cap_steps=cap_active,steps=len(x)*cfg['H'])

def main():
    threadpool_limits(1);OUT.mkdir(parents=True,exist_ok=True)
    files=[OLD/'protocol.json',NOM/'nominal_returns.npz',NOM/'summary.json']
    for p in sorted(OLD.glob('seed*')):files.extend([p/'metrics.json',p/'evaluation.npz',p/'frozen_policy.json'])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files};source=sha(Path(__file__))
    if (OUT/'COMPLETE.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['input_hashes']==hashes and p['source_sha256']==source
        for k,v in json.loads((OUT/'COMPLETE.json').read_text())['outputs'].items():assert sha(OUT/k)==v
        print((OUT/'summary.json').read_text());return
    if (OUT/'protocol.json').exists():raise RuntimeError('Preserve unfinished oracle study')
    cfg=json.loads((OLD/'protocol.json').read_text())['config']
    dump(OUT/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=source,input_hashes=hashes,
        status='Post-hoc descriptive reference; original policies and adoption decisions frozen.',
        seeds=list(range(1000,1008)),episodes_per_seed=4096,config=cfg,kinds=KINDS,
        primary='B(x)+clip((K_nom-K_target)x, -h_d(||Tx||), h_d(||Tx||)), then native torque clip.',
        secondary_direction='B(x)+h_d(||Tx||) sign((K_nom-K_target)x), then native torque clip.',
        secondary_uncapped='-K_target x, native torque clip only; outside displacement-constrained class.',
        LQR='Discrete infinite-horizon LQR for true target linearization at origin, Q=diag(1,.1), R=.001.',
        same_initial_states_and_torque_noises_as_original_evaluation=True,
        interpretation='Oracle denotes access to target parameters, not a nonlinear finite-horizon constrained optimum.',
        summaries='All per-plant outcomes; ratio of summed gain to summed nominal gap; summed learned/oracle gains.',
        bootstrap='20000 paired plant resamples, seed 2026092808; descriptive percentile intervals only.'))
    records=[];arrays={}
    with np.load(NOM/'nominal_returns.npz') as nominal:
        for seed in range(1000,1008):
            cell=OLD/f'seed{seed}';m=json.loads((cell/'metrics.json').read_text());g,c=m['gain'],m['damping']
            with np.load(cell/'evaluation.npz') as data:old=data['returns']
            x,w=bank(cfg,seed);Kt,diagnostic=lqr(g,c)
            gap=float(np.mean(nominal[f'nominal_seed{seed}']-old[0]));learn=float(np.mean(old[1]-old[0]));comparisons={}
            for kind in KINDS:
                ret,audit=rollout(g,c,cfg,Kt,x,w,kind);arrays[f'{kind}_seed{seed}']=ret
                gain=ret-old[0];contrast=old[1]-ret
                comparisons[kind]=dict(mean_gain=float(gain.mean()),recovery_fraction=float(gain.mean()/gap),
                    learned_over_reference_gain=float(learn/gain.mean()),learned_minus_reference=float(contrast.mean()),
                    learned_minus_reference_mc_se=float(contrast.std(ddof=1)/np.sqrt(len(contrast))),audit=audit)
            records.append(dict(seed=seed,gain=g,damping=c,target_K=Kt.tolist(),lqr_diagnostic=diagnostic,
                                nominal_gap=gap,learned_gain=learn,references=comparisons))
    np.savez_compressed(OUT/'oracle_returns.npz',**arrays)
    rng=np.random.default_rng(2026092808);idx=rng.integers(0,8,(20000,8));learn=np.array([r['learned_gain'] for r in records]);gap=np.array([r['nominal_gap'] for r in records])
    aggregates={}
    for kind in KINDS:
        gains=np.array([r['references'][kind]['mean_gain'] for r in records]);ratio=learn.sum()/gains.sum()
        bootgain=gains[idx].sum(axis=1);bootlearn=learn[idx].sum(axis=1);bootgap=gap[idx].sum(axis=1)
        aggregates[kind]=dict(mean_gain=float(gains.mean()),recovery_fraction=float(gains.sum()/gap.sum()),
            learned_over_reference_gain=float(ratio),learned_ratio_bootstrap_interval=np.quantile(bootlearn/bootgain,[.025,.975]).tolist() if np.all(bootgain>0) else None,
            recovery_bootstrap_interval=np.quantile(bootgain/bootgap,[.025,.975]).tolist() if np.all(bootgap>0) else None,
            learned_better_plants=int(np.sum(learn>gains)),reference_positive_plants=int(np.sum(gains>0)),
            mean_learned_minus_reference=float(np.mean(learn-gains)))
    result=dict(records=records,aggregates=aggregates,learned_recovery_fraction=float(learn.sum()/gap.sum()),
                actual_new_transitions=8*4096*cfg['H']*len(KINDS),original_adoption_tests_unchanged=True)
    dump(OUT/'summary.json',result);dump(OUT/'COMPLETE.json',dict(outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
