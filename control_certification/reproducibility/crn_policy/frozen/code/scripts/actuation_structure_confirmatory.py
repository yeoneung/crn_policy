"""Fresh 16-plant confirmation of input-geometry-aware return support."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import coupled_support_core as c
from mechanical_repair_core_v2 import MechanicalPlant, train
from finite_horizon_envelope import calibrate_matrix, return_support
from actuation_envelope import actuation_return_support
from return_certificate import TwoSidedAdoption

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/actuation_structure_confirmatory'
CAL_ALPHA = .025/16
TEST_ALPHA = .025/(16*3)


def dump(path, value):path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(exist_ok=True)
    choice_path=ROOT/'results/mechanical_repair_development_v2/summary.json'
    choice=json.loads(choice_path.read_text())['chosen_displacement']
    files=[Path(__file__).with_name(name) for name in [Path(__file__).name,'mechanical_repair_core_v2.py',
            'mechanical_repair_core.py','stable_sphere_candidates.py','actuation_envelope.py',
            'coupled_support_core.py','finite_horizon_envelope.py','return_certificate.py']]
    files += [ROOT.parent/'MECHANICAL_REPAIR_PLAN.md',ROOT.parent/'MECHANICAL_REPAIR_AMENDMENT.md',
              ROOT.parent/'ACTUATION_STRUCTURE_PLAN.md',choice_path]
    hashes={p.relative_to(ROOT.parent).as_posix():sha(p) for p in files}
    if (OUT/'source_hashes.json').exists():assert hashes==json.loads((OUT/'source_hashes.json').read_text())
    else:
        dump(OUT/'source_hashes.json',hashes)
        for p in files:
            target=OUT/'source_snapshot'/p.relative_to(ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
        dump(OUT/'protocol.json',dict(dimensions=[8,16],seeds=list(range(700,708)),d=choice,
            training_transitions=81920,calibration_pairs=8192,calibration_alpha=CAL_ALPHA,
            test_alpha=TEST_ALPHA,primary_caps=dict(input_structure=512,matrix_power=512,ordinary_range=716),
            total_budget=139264,final_pairs=4096,research_certification_pairs=4096,
            source_baseline='Known nominal-model discrete LQR',external_benchmark=False))
    return choice


def run(n,seed,d):
    out=OUT/f'dim{n}_seed{seed}';out.mkdir(exist_ok=True)
    if (out/'COMPLETE').exists():return
    if (out/'STARTED').exists():raise RuntimeError('Preserve partial final attempt')
    dump(out/'STARTED',dict(n=n,seed=seed,d=d,source_hash=sha(OUT/'source_hashes.json')))
    start=time.time();env=MechanicalPlant(n,seed);policy=train(env,seed,d,870100,out)
    random=c.gen(870400,n,seed);x=.5*random.normal(size=(8192,n));a=env.baseline(x)
    y1,_=env.step(x,a,random.normal(size=x.shape));y2,_=env.step(x,a,random.normal(size=x.shape))
    np.savez_compressed(out/'calibration.npz',x=x,y1=y1,y2=y2)
    fit=calibrate_matrix(x,y1,y2,env.nonlinear_lip,CAL_ALPHA)
    bound,distance,powers=return_support(fit,env.constants,d)
    structured,structured_distance,_,forcing=actuation_return_support(fit,env.constants,env.B0,d)
    assert structured<=bound+1e-12
    dump(out/'calibration_fit.json',fit)
    np.savez_compressed(out/'envelope.npz',distance=distance,power_norms=powers,bound=np.array(bound))
    np.savez_compressed(out/'actuation_envelope.npz',distance=structured_distance,forcing=forcing,bound=np.array(structured))
    np.savez_compressed(out/'public_action_factor.npz',matrix=env.B0)
    dump(out/'pre_evaluation.json',dict(hashes={p.name:sha(p) for p in out.glob('*.npz')},
                                       calibration_hash=sha(out/'calibration_fit.json')))
    cert=c.evaluate(env,[None,policy],4096,c.gen(870600,n,seed),[0,d])
    np.savez_compressed(out/'certification.npz',**cert);dif=cert['returns'][1]-cert['returns'][0]
    gates=[]
    for kind,support,cap,calibration in [('input_structure',structured,512,16384),
        ('matrix_power',bound,512,16384),('ordinary_range',env.constants['return_range'],716,0)]:
        for scope,limit in [('primary',cap),('extended',4096)]:
            test=TwoSidedAdoption(support,TEST_ALPHA)
            for first in range(0,limit,16):
                if test.update(dif[first:min(first+16,limit)]):break
            g=test.result();g.update(bound_kind=kind,scope=scope,cap=limit,training_transitions=81920,
                calibration_transitions=calibration,certification_transitions=80*g['episodes'],
                total_used_transitions=81920+calibration+80*g['episodes'],
                total_allowed_transitions=139264 if scope=='primary' else 81920+calibration+80*limit)
            gates.append(g)
    dump(out/'certificates.json',gates)
    final=c.evaluate(env,[None,policy],4096,c.gen(870700,n,seed),[0,d]);np.savez_compressed(out/'evaluation.npz',**final)
    gain=float((final['returns'][1]-final['returns'][0]).mean())
    error=float(np.linalg.norm(np.array(fit['matrix'])-env.linear_baseline,2))
    result=dict(confirmatory=True,structured_extension=True,n=n,m=env.m,seed=seed,d=d,
        raw_gain=gain,baseline_return=float(final['returns'][0].mean()),
        deployed={g['bound_kind']:gain if g['adopted'] else 0. for g in gates if g['scope']=='primary'},
        true_linear_norm=float(np.linalg.norm(env.linear_baseline,2)),scalar_lipschitz_upper=fit['scalar_lipschitz_upper'],
        matrix_error=error,matrix_error_bound=fit['matrix_error'],matrix_covered=error<=fit['matrix_error'],
        paired_bound=bound,structured_bound=structured,ordinary_bound=env.constants['return_range'],
        observed_return_support_violation=bool(np.max(abs(dif))>bound),
        observed_trajectory_envelope_violation=bool(np.max(cert['max_state_distance'][1]-distance)>1e-10),
        structured_return_support_violation=bool(np.max(abs(dif))>structured),
        structured_trajectory_envelope_violation=bool(np.max(cert['max_state_distance'][1]-structured_distance)>1e-10),
        seconds=time.time()-start,training_transitions=81920,calibration_transitions=16384,
        research_certification_transitions=int(cert['transitions']),final_evaluation_transitions=int(final['transitions']))
    dump(out/'COMPLETE',result);print(n,seed,'complete',round(result['seconds'],1),flush=True)


if __name__=='__main__':
    threadpool_limits(limits=1);choice=freeze()
    for n in [8,16]:
        for seed in range(700,708):run(n,seed,choice)
