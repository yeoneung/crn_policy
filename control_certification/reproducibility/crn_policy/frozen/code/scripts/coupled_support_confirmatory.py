"""Frozen same-policy/same-data comparison of two justified return supports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from coupled_support_core import *
from return_certificate import TwoSidedAdoption

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/coupled_support_confirmatory'
BUDGETS=[5120,20480,81920]
CAPS=[128,512,2048,4096]
ALPHA=.05/(24*2*3*2*2)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_support(env,d):
    c=env.constants
    _,distance=env.support(d)
    lipschitz=c['reward_state_lipschitz']+c['reward_action_lipschitz']*c['baseline_lipschitz']
    return min(c['return_range'],float((GAMMA**np.arange(H))@(lipschitz*distance)))


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('coupled_support_core.py'),
        Path(__file__).with_name('return_certificate.py'),ROOT.parent/'COUPLED_SUPPORT_CONFIRMATORY_PLAN.md']
    hashes={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in files}
    path=OUT/'source_hashes.json'
    if path.exists():assert json.loads(path.read_text())==hashes,'Frozen source changed'
    else:
        dump(path,hashes)
        for p in files:
            target=OUT/'source_snapshot'/p.relative_to(ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        dump(OUT/'protocol.json',dict(dimensions=DIMENSIONS,seeds=list(range(200,208)),
            radii=[.05,.15],budgets=BUDGETS,caps=CAPS,alpha=ALPHA,ridge=.1,
            ars_step=.01,ars_noise=.1,ars_directions=4,ars_episodes=4,
            final_episodes=4096,certification_episodes=4096,primary=dict(method='Ridge',d=.05,budget=81920,cap=512)))


def replay(d,bound,cap):
    test=TwoSidedAdoption(bound,ALPHA)
    for i in range(0,cap,16):
        if test.update(d[i:i+16]):break
    return test.result()


def run(n,seed):
    assert n in DIMENSIONS and seed in range(200,208)
    out=OUT/f'dim{n}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    if (out/'STARTED').exists():raise RuntimeError('Preserve partial attempt; audit before recovery')
    dump(out/'STARTED',dict(n=n,seed=seed,source_hash=sha(OUT/'source_hashes.json')))
    started=time.time();env=StablePlant(n,seed);dump(out/'public_constants.json',env.constants)
    np.savez_compressed(out/'plant.npz',A=env.A,B0=env.B0,B=env.B,K=env.K,C=env.C,Q=env.Q)
    policies=[None];names=['B'];limits=[0.];costs=[0]
    for di,d in enumerate([.05,.15]):
        current=None;xx=[];yy=[];spent=0
        for stage,budget in enumerate(BUDGETS):
            data,c=collect_pool(env,current,16,gen(851100,n,seed,di,stage));spent+=c
            actions=candidates(env.baseline(data['x']),d)
            pairs,ix=select(data,actions,budget-spent,gen(851200,n,seed,di,stage))
            values,c=query(env,data,actions,pairs,ix,gen(851300,n,seed,di,stage));spent+=c
            assert spent==budget
            si,pi=ix.T;chosen=np.stack([actions[si,0],actions[si,pairs[pi,0]],actions[si,pairs[pi,1]]],1)
            ff=features(data['x'][si],data['remaining'][si],chosen,d)
            x=ff[:,2]-ff[:,1];y=np.diff(values,axis=1).ravel()/d;xx.append(x);yy.append(y)
            xa=np.concatenate(xx);ya=np.concatenate(yy)
            coef=np.linalg.solve(xa.T@xa+.1*np.eye(xa.shape[1]),xa.T@ya)
            current=RidgePatch(coef,d,n);name=f'Ridge_d{d:g}_b{budget}'
            policies.append(current);names.append(name);limits.append(d);costs.append(spent)
            np.savez_compressed(out/f'{name}.npz',**data,actions=actions,pairs=pairs,selected=ix,
                returns=values,fit_x=x,fit_target=y,coef=coef)
        weight=np.zeros((n,n+1));spent=0;iteration=0;random=gen(851500,n,seed,di)
        for budget in BUDGETS:
            while spent<budget:
                weight,trace=ars_update(env,weight,d,random,step=.01,noise=.1,directions=4,episodes=4)
                spent+=int(trace['transitions']);np.savez_compressed(out/f'ARS_d{d:g}_iteration{iteration}.npz',**trace);iteration+=1
            assert spent==budget
            policies.append(ARSPatch(weight.copy(),d));names.append(f'ARS_d{d:g}_b{budget}');limits.append(d);costs.append(spent)
    dump(out/'pre_evaluation.json',dict(names=names,limits=limits,costs=costs,
        hashes={p.name:sha(p) for p in out.glob('*.npz')}))
    cert=evaluate(env,policies,4096,gen(851600,n,seed),limits)
    np.savez_compressed(out/'certification.npz',**cert)
    gates=[]
    for j in range(1,len(names)):
        dif=cert['returns'][j]-cert['returns'][0];bound=normalized_support(env,limits[j])
        assert np.max(abs(dif))<=bound
        for kind,b in [('normalized_support',bound),('ordinary_range',env.constants['return_range'])]:
            for cap in CAPS:
                record=replay(dif,b,cap)
                record.update(name=names[j],bound_kind=kind,cap=cap,training_transitions=costs[j],
                    certification_transitions=2*H*record['episodes'],
                    total_used_transitions=costs[j]+2*H*record['episodes'],
                    total_allowed_transitions=costs[j]+2*H*cap,
                    general_paired_support=env.support(limits[j])[0])
                gates.append(record)
    dump(out/'certificates.json',gates)
    final=evaluate(env,policies,4096,gen(851700,n,seed),limits)
    np.savez_compressed(out/'evaluation.npz',**final)
    for j in range(1,len(names)):
        assert np.max(abs(final['returns'][j]-final['returns'][0]))<=normalized_support(env,limits[j])
    result=dict(confirmatory=True,n=n,seed=seed,names=names,limits=limits,costs=costs,
        raw_gains=(final['returns']-final['returns'][0]).mean(1).tolist(),
        mean_returns=final['returns'].mean(1).tolist(),seconds=time.time()-started,
        certification_research_transitions=int(cert['transitions']),final_evaluation_transitions=int(final['transitions']))
    dump(out/'COMPLETE',result)
    print(n,seed,'complete',round(result['seconds'],1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dimension',type=int);ap.add_argument('--seed',type=int)
    ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
    threadpool_limits(limits=1);freeze()
    if not args.freeze_only:run(args.dimension,args.seed)
