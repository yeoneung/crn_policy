"""Data-calibrated contraction extension, charging every calibration step."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from coupled_support_core import StablePlant,gen,H
from stability_calibration import estimate_contraction,paired_support
from return_certificate import TwoSidedAdoption

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stability_calibration_confirmatory'
CAL_ALPHA=.025/(24*3)
TEST_ALPHA=.025/(24*3*2*3*2*2)


def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('stability_calibration.py'),
        Path(__file__).with_name('coupled_support_core.py'),Path(__file__).with_name('return_certificate.py'),
        ROOT.parent/'STABILITY_CALIBRATION_PLAN.md',ROOT/'results/coupled_support_confirmatory/source_hashes.json']
    hashes={str(p.relative_to(ROOT.parent)).replace('\\','/'):sha(p) for p in files}
    manifest=OUT/'source_hashes.json'
    if manifest.exists():assert json.loads(manifest.read_text())==hashes
    else:
        dump(manifest,hashes)
        for p in files:
            target=OUT/'source_snapshot'/p.relative_to(ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        dump(OUT/'protocol.json',dict(samples=[512,2048,8192],calibration_alpha=CAL_ALPHA,
            return_alpha=TEST_ALPHA,learned_cap=512,noise_variance_known=False,
            primary=dict(method='Ridge',d=.05,training=81920,calibration=2048,total_budget=126976)))


def replay(values,bound,cap):
    test=TwoSidedAdoption(bound,TEST_ALPHA)
    for i in range(0,cap,16):
        if test.update(values[i:min(i+16,cap)]):break
    return test.result()


def run(n,seed):
    folder=ROOT/'results/coupled_support_confirmatory'/f'dim{n}_seed{seed}'
    if not (folder/'COMPLETE').exists():return False
    out=OUT/f'dim{n}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return True
    if (out/'STARTED').exists():raise RuntimeError('Preserve partial calibration attempt')
    dump(out/'STARTED',dict(n=n,seed=seed,protocol_sha256=sha(OUT/'source_hashes.json')))
    env=StablePlant(n,seed);random=gen(853100,n,seed);x=env.initial(8192,random);a=env.baseline(x)
    y1,_=env.step(x,a,random.normal(size=x.shape));y2,_=env.step(x,a,random.normal(size=x.shape))
    np.savez_compressed(out/'observations.npz',x=x,y1=y1,y2=y2)
    fits={size:estimate_contraction(x[:size],y1[:size],y2[:size],.005,CAL_ALPHA) for size in [512,2048,8192]}
    # Freeze estimates before opening policy return data or hidden-matrix audits.
    dump(out/'pre_return_calibration.json',dict(fits=fits,observation_sha256=sha(out/'observations.npz')))
    prior=json.loads((folder/'COMPLETE').read_text());returns=np.load(folder/'certification.npz')['returns']
    rows=[];audit=[]
    for size,fit in fits.items():
        true_upper=float(np.linalg.norm(env.A+env.B@env.K,2)+.005)
        audit.append(dict(samples=size,true_analytic_upper=true_upper,covered=true_upper<=fit['kappa']))
        for j,name in enumerate(prior['names'][1:],1):
            values=returns[j]-returns[0];d=prior['limits'][j];training=prior['costs'][j]
            for kind in ['calibrated_support','ordinary_range']:
                cap=512 if kind=='calibrated_support' else 512+2*size//(2*H)
                if kind=='calibrated_support' and not fit['valid']:
                    record=dict(adopted=False,outcome='uncertified_contraction',episodes=0,
                        deployed_gain_lower=0.,proposal_gain_lower=None,bound=None,alpha=TEST_ALPHA)
                else:
                    bound=paired_support(env.constants,d,fit['kappa']) if kind=='calibrated_support' else env.constants['return_range']
                    record=replay(values,bound,cap)
                calibration=2*size if kind=='calibrated_support' else 0
                gain=prior['raw_gains'][j]
                record.update(name=name,bound_kind=kind,samples=size,cap=cap,
                    training_transitions=training,calibration_transitions=calibration,
                    certification_transitions=2*H*record['episodes'],
                    total_used_transitions=training+calibration+2*H*record['episodes'],
                    total_allowed_transitions=training+2*size+512*2*H,
                    raw_gain=gain,deployed_gain=gain if record['adopted'] else 0.)
                rows.append(record)
    dump(out/'certificates.json',rows);dump(out/'coverage_audit.json',audit)
    dump(out/'COMPLETE',dict(n=n,seed=seed,confirmatory_extension=True,
        independent_return_replication=False,calibration_research_transitions=16384,
        return_data_hash=sha(folder/'certification.npz'),final_data_hash=sha(folder/'evaluation.npz'),
        all_contractions_certified=all(f['valid'] for f in fits.values()),all_calibrations_cover=all(a['covered'] for a in audit)))
    return True


if __name__=='__main__':
    threadpool_limits(limits=1);freeze()
    for n in [4,8,16]:
        for seed in range(200,208):print(n,seed,run(n,seed),flush=True)
