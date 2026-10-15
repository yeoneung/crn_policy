"""Frozen learned-policy study in a bounded-noise Pendulum regulation variant.

Run --development first (four corners, no final seeds). --final freezes
the full protocol and hashes before drawing eight new target plants.
Never infer an envelope from observed sample extrema.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from pendulum_statecap_development import design,bank,transition
from variance_replay import stop

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_regional_study'
METHODS=['fixed_mixture','predictable_empirical_bernstein']
KINDS=['global','regional_marginal','constantcap','statecap']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def setup():
    cfg=design();cfg['H']=40
    for statecap in [False,True]:
        s=cfg['initial_radius'];e=0.;b=m=0.;trace=[]
        for t in range(cfg['H']):
            a=cfg['d']*min(1,(s+e)/cfg['initial_radius']) if statecap else cfg['d']
            br=cfg['Qnorm']*(2*s+e)*e+.001*(2*cfg['Knorm']*(s+e)*a+a*a)
            mr=cfg['Qnorm']*(s+e)**2+.001*(2*cfg['Knorm']*(s+e)*a+a*a)
            trace.append([s,e,a,br,mr]);b+=br;m+=mr
            e=cfg['kappa']*e+cfg['action_gain']*a;s=cfg['kappa']*s+cfg['noise_gain']*cfg['width']
        cfg['statecap' if statecap else 'constantcap']=dict(paired=float(b*1.00000001),marginal=float(m*1.00000001),trace=trace)
    cfg['bounds']=dict(global_bound=cfg['H']*(np.pi**2+.1*8**2+.001*2**2),
                       regional_marginal=cfg['statecap']['marginal'],
                       constantcap=cfg['constantcap']['paired'],statecap=cfg['statecap']['paired'])
    cfg['bounds']['global']=cfg['bounds'].pop('global_bound')
    return cfg

def cap(x,cfg):
    return cfg['d']*np.minimum(1,np.linalg.norm(x@np.array(cfg['T']).T,axis=1)/cfg['initial_radius'])

def train(g,c,seed,cfg):
    K=np.array(cfg['K']);T=np.array(cfg['T'])
    x,w=bank(cfg,64,[seed,1]);anchors=[]
    for t,noise in enumerate(w):
        if t in [0,1,2,4,8,12,20,30]:anchors.append(x.copy())
        x,_=transition(x,-x@K,noise,g,c)
    x0=np.concatenate(anchors);rng=np.random.default_rng([seed,2])
    noise=rng.uniform(-cfg['width'],cfg['width'],(cfg['H'],len(x0)))
    delta=cap(x0,cfg);returns=[]
    for sign in [-1,1]:
        x=x0.copy();ret=np.zeros(len(x))
        for t,w in enumerate(noise):
            a=-x@K
            if t==0:a+=sign*delta
            x,r=transition(x,a,w,g,c);ret+=r
        returns.append(ret)
    y=(returns[1]-returns[0])/(2*delta)
    f=x0@T.T
    coef=np.linalg.solve(f.T@f+1e-6*np.eye(2),f.T@y)
    return dict(coef=coef.tolist(),training_transitions=64*cfg['H']+2*len(x0)*cfg['H'],
                training_rmse=float(np.sqrt(np.mean((f@coef-y)**2)))),dict(anchors=x0,labels=y)

def pair_rollout(g,c,policy,bk,cfg):
    x0,noise=bk;K=np.array(cfg['K']);T=np.array(cfg['T']);coef=np.array(policy['coef'])
    allret=[];traj=[]
    for repaired in [False,True]:
        x=x0.copy();ret=np.zeros(len(x));maxnorm=np.linalg.norm(x@T.T,axis=1)
        maxact=np.zeros(len(x));states=[x.copy()]
        for w in noise:
            a=-x@K
            if repaired:a+=cap(x,cfg)*np.sign(x@T.T@coef)
            maxact=np.maximum(maxact,abs(a));x,r=transition(x,a,w,g,c)
            ret+=r;maxnorm=np.maximum(maxnorm,np.linalg.norm(x@T.T,axis=1));states.append(x.copy())
        allret.append(ret);traj.append(np.stack(states))
    diffnorm=np.linalg.norm((traj[1]-traj[0])@T.T,axis=2)
    maxima=dict(max_baseline_norm=float(np.linalg.norm(traj[0]@T.T,axis=2).max()),
                max_repair_norm=float(np.linalg.norm(traj[1]@T.T,axis=2).max()),
                max_state_difference=float(diffnorm.max()),
                state_envelope_violations=int(np.sum(diffnorm[:-1]>np.array(cfg['statecap']['trace'])[:,1,None]+1e-12)))
    return np.stack(allret),maxima

def run_cell(g,c,seed,cfg,n,path,alpha):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'COMPLETE.json').exists():return json.loads((path/'metrics.json').read_text())
    policy,training=train(g,c,seed,cfg)
    dump(path/'frozen_policy.json',policy)
    np.savez_compressed(path/'training.npz',**training)
    # Policy written before certification/evaluation; no access to true g,c in train's feature regression.
    cert,ca=pair_rollout(g,c,policy,bank(cfg,n,[seed,3]),cfg)
    ev,ea=pair_rollout(g,c,policy,bank(cfg,4096,[seed,4]),cfg)
    np.savez_compressed(path/'certification.npz',returns=cert)
    np.savez_compressed(path/'evaluation.npz',returns=ev)
    d=cert[1]-cert[0];gain=ev[1]-ev[0]
    rows=[]
    for kind in KINDS:
        for method in METHODS:
            res=stop(d,cfg['bounds'][kind],alpha,method,n)
            rows.append(dict(kind=kind,method=method,bound=cfg['bounds'][kind],alpha=alpha,
                             total_used_transitions=policy['training_transitions']+2*cfg['H']*res['pairs'],**res))
    result=dict(gain=g,damping=c,seed=seed,independent_gain=float(gain.mean()),
                independent_gain_se=float(gain.std(ddof=1)/np.sqrt(len(gain))),
                baseline_mean_return=float(ev[0].mean()),relative_gain=float(gain.mean()/-ev[0].mean()),
                max_abs_return_difference=float(abs(d).max()),certification_audit=ca,evaluation_audit=ea,
                training_transitions=policy['training_transitions'],actual_transitions=policy['training_transitions']+2*cfg['H']*(n+4096),
                results=rows)
    dump(path/'metrics.json',result)
    dump(path/'COMPLETE.json',dict(outputs={p.name:digest(p) for p in path.iterdir() if p.is_file()}))
    print(json.dumps(dict(seed=seed,g=g,c=c,mean_gain=result['independent_gain'],
                         statecap=[r for r in rows if r['kind']=='statecap'])),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');parser.add_argument('--final',action='store_true')
    args=parser.parse_args();assert args.development != args.final
    threadpool_limits(1);cfg=setup();mode='development' if args.development else 'final';dest=OUT/mode;dest.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'code/scripts/pendulum_statecap_development.py',ROOT/'code/scripts/pendulum_regional_development.py',
           ROOT/'code/scripts/review_theory_pendulum.py',ROOT/'code/scripts/variance_replay.py',ROOT/'code/scripts/return_certificate.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    if (dest/'protocol.json').exists():
        protocol=json.loads((dest/'protocol.json').read_text());assert protocol['source_hashes']==hashes
    else:
        protocol=dict(utc=datetime.now(timezone.utc).isoformat(),mode=mode,source_hashes=hashes,config=cfg,
                      final_seeds=list(range(1000,1008)),target_gain_uniform=[.75,.95],target_damping_uniform=[.05,.15],
                      cap=32768,evaluation_pairs=4096,methods=METHODS,kinds=KINDS,positive_family_alpha=.05,
                      per_test_alpha=.05/(8*len(METHODS)*len(KINDS)),all_outcomes_retained=True,
                      statement='Upright regulation variant using native Gymnasium dynamics with explicit actuator gain, damping and bounded torque noise; not swing-up benchmark.',
                      development_choices='Original Gaussian rectangle rejected; optimized metric, bounded-noise disk and state-proportional cap developed on corners; H=40 assessed on four corners before final freeze.')
        dump(dest/'protocol.json',protocol)
    if args.development:
        cells=[(.75,.05,900),(.75,.15,901),(.95,.05,902),(.95,.15,903)]
    else:
        cells=[]
        for seed in protocol['final_seeds']:
            rng=np.random.default_rng([seed,0]);cells.append((rng.uniform(.75,.95),rng.uniform(.05,.15),seed))
    rows=[run_cell(float(g),float(c),seed,cfg,protocol['cap'],dest/f'seed{seed}',protocol['per_test_alpha']) for g,c,seed in cells]
    summary=dict(mode=mode,plants=len(rows),bounds=cfg['bounds'],mean_gain=float(np.mean([r['independent_gain'] for r in rows])),
                 mean_relative_gain=float(np.mean([r['relative_gain'] for r in rows])),positive_plants=sum(r['independent_gain']>0 for r in rows),
                 actual_transitions=sum(r['actual_transitions'] for r in rows),groups=[])
    for k in KINDS:
        for m in METHODS:
            group=[s for r in rows for s in r['results'] if s['kind']==k and s['method']==m]
            summary['groups'].append(dict(kind=k,method=m,adoptions=sum(s['adopted'] for s in group),
                     rejections=sum(s['outcome']=='reject' for s in group),censored=sum(s['outcome']=='inconclusive' for s in group),
                     mean_pairs=float(np.mean([s['pairs'] for s in group]))))
    dump(dest/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
