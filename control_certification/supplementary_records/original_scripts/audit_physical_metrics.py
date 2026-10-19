"""Independently replay every secondary mechanical metric bank and its hashes."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from mechanical_repair_core import MechanicalPlant
from stable_sphere_candidates import RidgePatch
from coupled_support_core import gen,H,DT,GAMMA
from summarize_phase15 import stats
from audit_phase15 import close

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/mechanical_physical_metrics'


def read(p):return json.loads(p.read_text())


def replay(n,seed,coef,d):
    env=MechanicalPlant(n,seed);policy=RidgePatch(coef,d,env.m);random=gen(872700,n,seed)
    outputs=[];p=n//2
    for start in range(0,4096,256):
        x=env.initial(256,random);states=np.concatenate([x,x]);totals=np.zeros((6,512))
        for t in range(H):
            actions=np.concatenate([env.baseline(states[:256]),
                policy.action(states[256:],np.full(256,H-t),env.baseline)])
            q2=np.einsum('bi,bi->b',states[:,:p],states[:,:p])
            v2=np.einsum('bi,bi->b',states[:,p:],states[:,p:])
            effort=.5*np.einsum('bi,bi->b',actions,actions)
            state_cost=2*q2+.3*v2
            states,reward=env.step(states,actions,np.tile(random.normal(size=(256,n)),(2,1)))
            totals[0]-=GAMMA**t*reward
            totals[1:]+=DT*GAMMA**t*np.stack([q2/p,v2/p,state_cost,effort,state_cost+effort])
        outputs.append(totals.reshape(6,2,256))
    return np.concatenate(outputs,axis=2)


def main():
    start=time.time();hashes=read(STUDY/'source_hashes.json')
    for name,digest in hashes.items():assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==digest,name
    summary=read(ROOT/'results/phase15_summary/physical_metrics.json')
    files=sorted(STUDY.glob('dim*/COMPLETE'));assert len(files)==16
    rows=[];maximum=0.
    for path in files:
        record=read(path);raw=path.parent/'evaluation.npz'
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==record['raw_sha256']
        costs=np.load(raw)['costs'];assert costs.shape==(6,2,4096)
        close(costs[3]+costs[4],costs[5])
        original=ROOT/'results/actuation_structure_confirmatory'/record['plant']
        coef=np.load(original/'train_b81920.npz')['coef'];d=read(original/'COMPLETE')['d']
        regenerated=replay(record['n'],record['seed'],coef,d)
        close(costs,regenerated)
        maximum=max(maximum,float(np.max(abs(costs-regenerated))))
        gain=(costs[:,0]-costs[:,1]).mean(1);baseline=costs[:,0].mean(1)
        close(gain,record['gain']);close(baseline,record['baseline_cost'])
        close(100*gain/baseline,record['relative_gain_percent']);rows.append(record)
    for j,row in enumerate(summary['summary']):
        for name,values in [('gain',[r['gain'][j] for r in rows]),
                            ('relative_gain_percent',[r['relative_gain_percent'][j] for r in rows])]:
            derived=stats(values)
            for key in ['values','mean','se','t_interval']:close(derived[key],row[name][key])
    result=dict(passed=True,source_and_policy_hashes=len(hashes),raw_banks=16,
                independently_replayed_metric_streams=16*6*2,
                maximum_absolute_difference=maximum,transitions=16*4096*2*H,
                seconds=time.time()-start,
                scope='Reproducibility audit of secondary physical metrics; no new statistical units or safety guarantee.')
    (ROOT/'results/phase15_summary/physical_metrics_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
