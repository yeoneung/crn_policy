"""Frozen correlated-noise, noncontractive finite-horizon validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import coupled_support_core as c
from nonnormal_development import NonnormalPlant,train
from finite_horizon_envelope import calibrate_matrix,return_support
from return_certificate import TwoSidedAdoption

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/nonnormal_confirmatory'
CAL_ALPHA=.025/24
TEST_ALPHA=.025/(24*2)


def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__).with_name(n) for n in [Path(__file__).name,'nonnormal_development.py',
        'coupled_support_core.py','finite_horizon_envelope.py','return_certificate.py']]
    files+=[ROOT.parent/'NONNORMAL_CONFIRMATORY_PLAN.md']
    hashes={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in files}
    path=OUT/'source_hashes.json'
    if path.exists():assert json.loads(path.read_text())==hashes
    else:
        dump(path,hashes)
        for f in files:
            target=OUT/'source_snapshot'/f.relative_to(ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
        dump(OUT/'protocol.json',dict(dimensions=[4,8,16],seeds=list(range(400,408)),ridge=.1,d=.05,
            training_transitions=81920,calibration_pairs=8192,calibration_alpha=CAL_ALPHA,test_alpha=TEST_ALPHA,
            primary_caps=dict(matrix_power=512,ordinary_range=716),total_budget=139264,
            final_pairs=4096,research_certification_pairs=4096,noise_covariance_supplied=False))


def replay(dif,bound,cap):
    test=TwoSidedAdoption(bound,TEST_ALPHA)
    for i in range(0,cap,16):
        block=dif[i:min(i+16,cap)]
        if np.max(abs(block))>bound:
            record=test.result();record.update(adopted=False,outcome='support_violation',
                deployed_gain_lower=0.,proposal_gain_lower=None,episodes=min(i+16,cap))
            return record
        if test.update(block):break
    return test.result()


def run(n,seed):
    assert n in [4,8,16] and seed in range(400,408)
    out=OUT/f'dim{n}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    if (out/'STARTED').exists():raise RuntimeError('Preserve partial attempt')
    dump(out/'STARTED',dict(n=n,seed=seed,source_hash=sha(OUT/'source_hashes.json')))
    started=time.time();env=NonnormalPlant(n,seed);policy=train(env,n,seed,857100,out)
    random=c.gen(857400,n,seed);x=env.initial(8192,random);a=env.baseline(x)
    y1,_=env.step(x,a,random.normal(size=x.shape));y2,_=env.step(x,a,random.normal(size=x.shape))
    np.savez_compressed(out/'calibration.npz',x=x,y1=y1,y2=y2)
    fit=calibrate_matrix(x,y1,y2,.005,CAL_ALPHA);bound,distance,powers=return_support(fit,env.constants,.05)
    dump(out/'calibration_fit.json',fit)
    np.savez_compressed(out/'envelope.npz',distance=distance,power_norms=powers,bound=np.array(bound))
    dump(out/'pre_evaluation.json',dict(hashes={p.name:sha(p) for p in out.glob('*.npz')},
        calibration_hash=sha(out/'calibration_fit.json')))
    cert=c.evaluate(env,[None,policy],4096,c.gen(857600,n,seed),[0,.05]);np.savez_compressed(out/'certification.npz',**cert)
    dif=cert['returns'][1]-cert['returns'][0];gates=[]
    for kind,b,cap,calibration in [('matrix_power',bound,512,16384),('ordinary_range',env.constants['return_range'],716,0)]:
        for scope,limit in [('primary',cap),('extended',4096)]:
            record=replay(dif,b,limit)
            record.update(bound_kind=kind,scope=scope,cap=limit,training_transitions=81920,
                calibration_transitions=calibration,certification_transitions=80*record['episodes'],
                total_used_transitions=81920+calibration+80*record['episodes'],
                total_allowed_transitions=139264 if scope=='primary' else 81920+calibration+80*limit)
            gates.append(record)
    dump(out/'certificates.json',gates)
    final=c.evaluate(env,[None,policy],4096,c.gen(857700,n,seed),[0,.05]);np.savez_compressed(out/'evaluation.npz',**final)
    gain=float((final['returns'][1]-final['returns'][0]).mean())
    error=float(np.linalg.norm(np.array(fit['matrix'])-env.linear_baseline,2))
    result=dict(confirmatory=True,n=n,seed=seed,raw_gain=gain,baseline_return=float(final['returns'][0].mean()),
        deployed={r['bound_kind']:gain if r['adopted'] else 0. for r in gates if r['scope']=='primary'},
        true_linear_norm=float(np.linalg.norm(env.linear_baseline,2)),scalar_lipschitz_upper=fit['scalar_lipschitz_upper'],
        matrix_error=error,matrix_error_bound=fit['matrix_error'],matrix_covered=error<=fit['matrix_error'],
        paired_bound=bound,ordinary_bound=env.constants['return_range'],
        observed_return_support_violation=bool(np.max(abs(dif))>bound),
        observed_trajectory_envelope_violation=bool(np.max(cert['max_state_distance'][1]-distance)>1e-10),
        seconds=time.time()-started,training_transitions=81920,calibration_transitions=16384,
        research_certification_transitions=int(cert['transitions']),final_evaluation_transitions=int(final['transitions']))
    dump(out/'COMPLETE',result);print(n,seed,'complete',round(result['seconds'],1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dimension',type=int);ap.add_argument('--seed',type=int)
    args=ap.parse_args();threadpool_limits(limits=1);freeze();run(args.dimension,args.seed)
