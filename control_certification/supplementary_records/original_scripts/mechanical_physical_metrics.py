"""New secondary metric banks for fixed repairs; no new return certificate."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from mechanical_repair_core_v2 import MechanicalPlant
from stable_sphere_candidates import RidgePatch
import coupled_support_core as c
from summarize_phase15 import frozen_records,stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/mechanical_physical_metrics'
METRICS=['bounded_cost','position_energy_per_coordinate','velocity_energy_per_coordinate',
         'quadratic_state_cost','quadratic_effort_cost','quadratic_lqr_cost']


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def evaluate(env,policy,random,count=4096):
    banks=[];max_displacement=max_radius_error=0.
    for start in range(0,count,256):
        size=min(256,count-start);x=np.tile(env.initial(size,random),(2,1))
        costs=np.zeros((len(METRICS),2,size));p=env.n//2
        for t in range(c.H):
            baseline=env.baseline(x[:size]);reference=env.baseline(x[size:])
            repair=policy.action(x[size:],np.full(size,c.H-t),env.baseline)
            max_displacement=max(max_displacement,float(np.linalg.norm(repair-reference,axis=1).max()))
            max_radius_error=max(max_radius_error,float(abs(np.linalg.norm(repair,axis=1)-np.linalg.norm(reference,axis=1)).max()))
            action=np.concatenate([baseline,repair])
            position=c.DT*np.sum(x[:,:p]**2,axis=1)/p
            velocity=c.DT*np.sum(x[:,p:]**2,axis=1)/p
            quadratic=c.DT*np.einsum('bi,ij,bj->b',x,env.Q,x)
            effort=c.DT*.5*np.sum(action**2,axis=1)
            x,reward=env.step(x,action,np.tile(random.normal(size=(size,env.n)),(2,1)))
            costs+=c.GAMMA**t*np.stack([-reward,position,velocity,quadratic,effort,quadratic+effort]).reshape(len(METRICS),2,size)
        banks.append(costs)
    assert max_displacement<=policy.d+1e-10 and max_radius_error<=1e-10
    return np.concatenate(banks,axis=2),max_displacement,max_radius_error


def main():
    start=time.time();records=frozen_records('actuation_structure_confirmatory','dim*',16)
    if OUT.exists():raise RuntimeError('Preserve existing secondary metric attempt')
    OUT.mkdir()
    sources=[Path(__file__),Path(__file__).with_name('mechanical_repair_core.py'),
             Path(__file__).with_name('mechanical_repair_core_v2.py'),
             Path(__file__).with_name('stable_sphere_candidates.py'),
             Path(__file__).with_name('coupled_support_core.py'),ROOT.parent/'MECHANICAL_PHYSICAL_METRICS_PLAN.md']
    hashes={p.relative_to(ROOT.parent).as_posix():digest(p) for p in sources}
    for r in records:
        path=r['folder']/'train_b81920.npz'
        assert digest(path)==read(r['folder']/'pre_evaluation.json')['hashes'][path.name]
        hashes[path.relative_to(ROOT.parent).as_posix()]=digest(path)
    dump(OUT/'source_hashes.json',hashes)
    dump(OUT/'protocol.json',dict(metrics=METRICS,plants=16,existing_plants=True,bank=872700,
                                 pairs=4096,horizon=c.H,dt=c.DT,gamma=c.GAMMA,
                                 policy='Frozen original Ridge_d0.15_b81920',new_certificate=False))
    rows=[]
    for r in records:
        env=MechanicalPlant(r['n'],r['seed']);coef=np.load(r['folder']/'train_b81920.npz')['coef']
        policy=RidgePatch(coef,r['d'],env.m);folder=OUT/r['id'];folder.mkdir()
        dump(folder/'STARTED',dict(source_hash=digest(OUT/'source_hashes.json')))
        costs,displacement,radius=evaluate(env,policy,c.gen(872700,r['n'],r['seed']))
        np.savez_compressed(folder/'evaluation.npz',costs=costs)
        gain=(costs[:,0]-costs[:,1]).mean(1)
        result=dict(plant=r['id'],n=r['n'],seed=r['seed'],baseline_cost=costs[:,0].mean(1).tolist(),
                    repair_cost=costs[:,1].mean(1).tolist(),gain=gain.tolist(),
                    relative_gain_percent=(100*gain/costs[:,0].mean(1)).tolist(),
                    max_action_displacement=displacement,max_same_state_radius_error=radius,
                    transitions=2*4096*c.H,raw_sha256=digest(folder/'evaluation.npz'))
        dump(folder/'COMPLETE',result);rows.append(result)
        print(r['id'],'complete',flush=True)
    summary=[]
    for j,metric in enumerate(METRICS):
        summary.append(dict(metric=metric,gain=stats([r['gain'][j] for r in rows]),
                            relative_gain_percent=stats([r['relative_gain_percent'][j] for r in rows]),
                            mean_baseline_cost=float(np.mean([r['baseline_cost'][j] for r in rows]))))
    result=dict(scope='Post-primary secondary evaluation on new episode banks of 16 existing plants. Unbounded metrics have no adoption guarantee; no new primary test or independent plant replication.',
                summary=summary,rows=rows,research_evaluation_transitions=sum(r['transitions'] for r in rows),
                seconds=time.time()-start)
    dump(ROOT/'results/phase15_summary/physical_metrics.json',result)
    print(json.dumps(dict(summary=[dict(metric=r['metric'],mean_gain=r['gain']['mean'],se=r['gain']['se'],
                                        relative_percent=r['relative_gain_percent']['mean'],positive=r['gain']['positive']) for r in summary],
                         research_transitions=result['research_evaluation_transitions']),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
