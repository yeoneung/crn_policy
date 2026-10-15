"""Frozen d-sweep, independent-noise ablation, and residual-bound sensitivity.

Fresh certification/evaluation banks on the original 24 stable plants.
No refitting, plant selection, return-order selection, or stopping-rule tuning.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
import coupled_support_core as c
from coupled_support_confirmatory import normalized_support
from stability_calibration import estimate_contraction, paired_support
from variance_replay import stop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/results/v2_coupling_sensitivity"
DS = np.array([.0125, .025, .05, .1, .2])
D0 = .05
ELL_FACTORS = [.5, 1., 2., 5., 10.]
ALPHA = .05 / (24 * (5 * 2 * 2 + 2))  # 528 distinct positive-tail tests
CAL_ALPHA = .025 / (24 * 3)
ELL_ALPHA = .025 / (24 * 3 * 2 * 3 * 2 * 2)  # original replay level
METHODS = ["fixed_mixture", "predictable_empirical_bernstein"]
CAPS = [512, 4096]
BATCH = 128


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class FrozenSphereFamily:
    """Scale the original policy's chord, following its fixed great-circle direction.

    At each queried state the ORIGINAL d=.05 policy chooses a direction once;
    the new displacement never reoptimizes its candidate scores. Cap rotation
    at the original .8 radians. This allows small baseline radii and no-change
    decisions, preserves action norms, and exactly recovers the original at d0.
    """
    def __init__(self, coef, dimension):
        self.original = c.RidgePatch(coef, D0, dimension)

    def action(self, x, remaining, baseline, displacements):
        base = baseline(x)
        reference = self.original.action(x, remaining, baseline)
        eta = np.linalg.norm(base, axis=1)
        u = c.unit(base)
        chord0 = np.linalg.norm(reference - base, axis=1)
        requested = chord0 * np.asarray(displacements) / D0
        cap = 2 * eta * np.sin(.4)
        target = np.minimum(requested, cap)
        ratio = np.divide(target, 2*eta, out=np.zeros_like(eta), where=eta>1e-12)
        theta = 2*np.arcsin(np.clip(ratio, 0, 1))
        tangent = reference - u*np.sum(reference*u, axis=1)[:, None]
        length = np.linalg.norm(tangent, axis=1)
        tangent = np.divide(tangent, length[:, None], out=np.zeros_like(tangent), where=length[:, None]>1e-12)
        action = eta[:, None]*(np.cos(theta)[:, None]*u + np.sin(theta)[:, None]*tangent)
        active = (eta>1e-12) & (chord0>1e-12) & (length>1e-12)
        action = np.where(active[:, None], action, base)
        at_reference = np.asarray(displacements) == D0
        action = np.where(at_reference[:, None], reference, action)
        return action, (requested > cap+1e-12) & active


def simulate(env, family, count, bank, batch=BATCH):
    common = c.gen(bank, env.n, env.seed)
    independent = c.gen(bank+1, env.n, env.seed)
    all_returns = []
    reference_distances = []
    max_state = np.zeros((5,c.H)); max_displacement = np.zeros(6)
    sum_displacement = np.zeros(6); max_radius_error = np.zeros(6)
    capped = np.zeros(6, dtype=int); points = 0
    limits = np.r_[DS, D0]
    for start in range(0,count,batch):
        b = min(batch, count-start)
        shared_initial = env.initial(b, common)
        independent_initial = env.initial(b, independent)
        states = np.concatenate([np.tile(shared_initial[None],(6,1,1)), independent_initial[None]])
        returns = np.zeros((7,b)); reference_trace = np.zeros((b,c.H))
        for t in range(c.H):
            distances = np.linalg.norm(states[1:6]-states[0],axis=2)
            max_state = np.maximum(max_state, np.pad(distances.max(axis=1)[:,None],((0,0),(t,c.H-t-1))))
            reference_trace[:,t] = distances[2]
            flat = states[1:].reshape(-1,env.n)
            base = env.baseline(flat)
            actions, saturation = family.action(flat,np.full(len(flat),c.H-t),env.baseline,np.repeat(limits,b))
            displacement = np.linalg.norm(actions-base,axis=1).reshape(6,b)
            max_displacement = np.maximum(max_displacement,displacement.max(axis=1))
            sum_displacement += displacement.sum(axis=1)
            max_radius_error = np.maximum(max_radius_error,abs(np.linalg.norm(actions,axis=1)-np.linalg.norm(base,axis=1)).reshape(6,b).max(axis=1))
            capped += saturation.reshape(6,b).sum(axis=1); points += b
            action = np.concatenate([env.baseline(states[0]),actions])
            noise = np.concatenate([np.tile(common.normal(size=(b,env.n)),(6,1)),independent.normal(size=(b,env.n))])
            states,reward = env.step(states.reshape(-1,env.n),action,noise)
            states = states.reshape(7,b,env.n)
            returns += c.GAMMA**t*reward.reshape(7,b)
        all_returns.append(returns); reference_distances.append(reference_trace)
    returns = np.concatenate(all_returns,axis=1)
    assert np.all(max_displacement <= limits+1e-10), max_displacement
    assert max_radius_error.max()<1e-10
    for j,d in enumerate(DS):
        _,envelope = env.support(d)
        assert np.all(max_state[j] <= envelope+1e-10)
        assert np.max(abs(returns[j+1]-returns[0])) <= normalized_support(env,d)+1e-10
    return dict(returns=returns, reference_state_distances=np.concatenate(reference_distances),
                max_state_distance=max_state,max_action_displacement=max_displacement,
                mean_action_displacement=sum_displacement/points,max_same_state_radius_error=max_radius_error,
                angle_cap_fraction=capped/points,actual_simulator_transitions=np.array(7*count*c.H),
                logical_pair_comparison_transitions=np.array(6*2*count*c.H))


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    source_files = [Path(__file__),Path(__file__).with_name("test_coupling_sensitivity.py")]
    source_files += [Path(__file__).with_name(n) for n in
                    ["coupled_support_core.py","coupled_support_confirmatory.py","stability_calibration.py","return_certificate.py","variance_replay.py"]]
    sources = {p.relative_to(ROOT).as_posix():sha(p) for p in source_files}
    if (OUT/"source_hashes.json").exists():
        assert json.loads((OUT/"source_hashes.json").read_text()) == sources,"Frozen execution source changed"
        return
    inputs = {}
    for n in c.DIMENSIONS:
        for seed in range(200,208):
            plant = f"dim{n}_seed{seed}"
            old = ROOT.parent/"aistats2027/code/results/coupled_support_confirmatory"/plant
            checkpoint = old/"Ridge_d0.05_b81920.npz"
            expected = json.loads((ROOT/"code/results/coupled_support_confirmatory"/plant/"pre_evaluation.json").read_text())["hashes"][checkpoint.name]
            assert sha(checkpoint)==expected
            target=OUT/"frozen_policies"/plant/checkpoint.name
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(checkpoint,target)
            inputs[target.relative_to(ROOT).as_posix()]=expected
            for dataset,names in [("coupled_support_confirmatory",["certification.npz","evaluation.npz","pre_evaluation.json","certificates.json"]),
                                  ("stability_calibration_confirmatory",["observations.npz","pre_return_calibration.json","certificates.json"])]:
                for name in names:
                    p=ROOT/"code/results"/dataset/plant/name
                    inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    protocol = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        status="Prospective new return streams on previously studied fixed plants/policies; ell analysis reuses prior calibration and returns",
        dimensions=c.DIMENSIONS,seeds=list(range(200,208)),displacements=DS.tolist(),reference_displacement=D0,
        policies="Original Ridge_d0.05_b81920 checkpoint, unmodified; fixed direction field with chord-scaled spherical interpolation/extrapolation and original 0.8-radian cap",
        no_change="Original baseline choices remain baseline; do not force full d where the source policy uses less",
        fresh_banks={"certification":960100,"evaluation":960200,"independent_branch":"bank+1"},episodes_per_bank=4096,batch=BATCH,
        independent_condition="Independent initial state AND process disturbance for replacement, same marginal laws; reuse the common branch's baseline for computational efficiency",
        allocation="Every evaluated comparison is charged 2*H=80 transitions per pair; unique simulator work also reported separately",
        tests=METHODS,caps=CAPS,check_every=16,positive_tail_alpha=ALPHA,
        simultaneous_scope="0.05 positive-tail union bound over 528 distinct fresh-stream tests: 24*(5 radii*2 supports*2 tests + independent ordinary*2 tests); caps are nested optional stopping, not new tests. Negative-label error separate. This does not cover the entire historical manuscript.",
        slopes="log-log fit for mean consumed pairs over all plants, explicitly cap-truncated; never fit successful plants alone as an uncensored rate. Show cap fractions and gain/d separately. Reference slopes 0 and -1 are guides, not guaranteed finite-sample outcomes.",
        evaluation="Gain/d and gain/d-squared variance scaling from independent evaluation bank, not the certification bank",
        ell=dict(factors=ELL_FACTORS,true_residual_lipschitz=.005,calibration_prefix=2048,calibration_alpha=CAL_ALPHA,return_alpha=ELL_ALPHA,
                 return_bank="Original certification stream; original independent evaluation for gain. Fresh reference-policy state traces provide an additional pathwise audit only.",
                 unavailable_contraction="Use ordinary range, retaining calibration cost; no invented contraction bound",
                 checks="Assumption violation (true residual norm is exactly .005), bound versus analytic closed-loop upper bound and origin Jacobian lower witness, old/fresh return violation counts, old maximal state-time violations, fresh per-episode/per-time violations",
                 interpretation="ell<true invalidates the stated premise but need not produce any empirical envelope violation; ell>=true conservatism preserves the premise. No empirical count proves global validity."))
    dump(OUT/"protocol.json",protocol); dump(OUT/"source_hashes.json",sources); dump(OUT/"input_hashes.json",inputs)
    for p in source_files:
        target=OUT/"source_snapshot"/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    print("FROZEN",protocol["frozen_utc"],len(inputs),"inputs",flush=True)


def analyze_ell(env, cert, evaluation):
    folder=ROOT/"code/results/coupled_support_confirmatory"/f"dim{env.n}_seed{env.seed}"
    cal=ROOT/"code/results/stability_calibration_confirmatory"/folder.name
    with np.load(cal/"observations.npz") as z: x,y1,y2=[z[k][:2048].copy() for k in ["x","y1","y2"]]
    index=json.loads((folder/"pre_evaluation.json").read_text())["names"].index("Ridge_d0.05_b81920")
    with np.load(folder/"certification.npz") as z:
        old_d=z["returns"][index]-z["returns"][0];old_state=z["max_state_distance"][index].copy()
    with np.load(folder/"evaluation.npz") as z:
        old_eval=z["returns"][index]-z["returns"][0]
        old_eval_state=z["max_state_distance"][index].copy()
    fresh_d=np.concatenate([z["returns"][3]-z["returns"][0] for z in [cert,evaluation]])
    fresh_state=np.concatenate([z["reference_state_distances"] for z in [cert,evaluation]])
    linear=env.A+env.B@env.K
    upper=float(np.linalg.norm(linear,2)+.005)
    witness=float(np.linalg.norm(linear+.005*env.C,2))
    rows=[]
    for factor in ELL_FACTORS:
        fit=estimate_contraction(x,y1,y2,.005*factor,CAL_ALPHA)
        if factor==1:
            original=json.loads((cal/"pre_return_calibration.json").read_text())["fits"]["2048"]
            assert abs(fit["kappa"]-original["kappa"])<1e-12
        structural=fit["valid"]
        bound=paired_support(env.constants,D0,fit["kappa"]) if structural else env.constants["return_range"]
        # A finite-horizon diagnostic envelope exists even if kappa>=1. It is
        # not used to change the original scalar-contraction verifier.
        e=np.zeros(c.H)
        for t in range(1,c.H): e[t]=fit["kappa"]*e[t-1]+env.constants["action_gain"]*D0
        old_viol=int(np.sum(abs(old_d)>bound+1e-12)+np.sum(abs(old_eval)>bound+1e-12))
        fresh_viol=int(np.sum(abs(fresh_d)>bound+1e-12))
        audit=dict(ell_factor=factor,ell=.005*factor,kappa=fit["kappa"],structural_support_available=structural,
                   bound=bound,true_residual_lipschitz=.005,premise_satisfied=factor>=1,
                   analytic_closed_loop_upper=upper,origin_jacobian_lower_witness=witness,
                   covers_analytic_upper=fit["kappa"]>=upper,contradicted_by_origin_witness=fit["kappa"]<witness,
                   old_return_violations=old_viol,old_return_pairs=len(old_d)+len(old_eval),
                   fresh_return_violations=fresh_viol,fresh_return_pairs=len(fresh_d),
                   old_max_state_time_violations=int(np.sum(old_state>e+1e-12)+np.sum(old_eval_state>e+1e-12)),
                   fresh_state_time_violations=int(np.sum(fresh_state>e+1e-12)),
                   fresh_episodes_with_state_violation=int(np.any(fresh_state>e+1e-12,axis=1).sum()),fit=fit)
        for method in METHODS:
            for cap in CAPS:
                if np.max(abs(old_d[:cap]))>bound:
                    result=dict(outcome="support_violation",adopted=False,pairs=0)
                else: result=stop(old_d,bound,ELL_ALPHA,method,cap)
                if factor==1 and method=="fixed_mixture" and cap==512:
                    old=next(v for v in json.loads((cal/"certificates.json").read_text()) if v["name"]=="Ridge_d0.05_b81920" and v["samples"]==2048 and v["bound_kind"]=="calibrated_support")
                    assert (result["adopted"],result["pairs"])==(old["adopted"],old["episodes"])
                rows.append(dict(**audit,method=method,cap=cap,alpha=ELL_ALPHA,**result,
                                 independent_gain=float(old_eval.mean()),deployed_gain=float(old_eval.mean()) if result["adopted"] else 0,
                                 used_transitions=81920+4096+80*result["pairs"]))
    return rows


def run(n,seed):
    threadpool_limits(limits=1)
    directory=OUT/f"dim{n}_seed{seed}"
    if (directory/"COMPLETE.json").exists():return json.loads((directory/"COMPLETE.json").read_text())
    directory.mkdir(exist_ok=True)
    if (directory/"STARTED.json").exists():raise RuntimeError(f"Preserve partial execution for audit: {directory}")
    dump(directory/"STARTED.json",dict(utc=datetime.now(timezone.utc).isoformat(),protocol=sha(OUT/"protocol.json")))
    started=time.perf_counter()
    with np.load(OUT/"frozen_policies"/directory.name/"Ridge_d0.05_b81920.npz") as z: coef=z["coef"].copy()
    env=c.StablePlant(n,seed);family=FrozenSphereFamily(coef,n)
    cert=simulate(env,family,4096,960100);np.savez_compressed(directory/"certification.npz",**cert)
    evaluation=simulate(env,family,4096,960200);np.savez_compressed(directory/"evaluation.npz",**evaluation)
    rows=[];metrics=[]
    for j,d in enumerate(DS):
        values=cert["returns"][j+1]-cert["returns"][0]
        ev=evaluation["returns"][j+1]-evaluation["returns"][0]
        metrics.append(dict(d=float(d),gain=float(ev.mean()),gain_over_d=float(ev.mean()/d),
                            gain_mc_se=float(ev.std(ddof=1)/np.sqrt(len(ev))),variance=float(ev.var(ddof=1)),
                            variance_over_d_squared=float(ev.var(ddof=1)/d**2),
                            paired_bound=normalized_support(env,d),ordinary_bound=env.constants["return_range"],
                            max_displacement=float(evaluation["max_action_displacement"][j]),
                            mean_displacement=float(evaluation["mean_action_displacement"][j]),
                            angle_cap_fraction=float(evaluation["angle_cap_fraction"][j])))
        for kind,bound in [("paired",normalized_support(env,d)),("ordinary",env.constants["return_range"])]:
            for method in METHODS:
                for cap in CAPS:
                    result=stop(values,bound,ALPHA,method,cap)
                    rows.append(dict(d=float(d),coupling="CRN",support=kind,method=method,cap=cap,bound=bound,alpha=ALPHA,
                                     independent_gain=float(ev.mean()),deployed_gain=float(ev.mean()) if result["adopted"] else 0,**result))
    ind=cert["returns"][6]-cert["returns"][0]
    crn=cert["returns"][3]-cert["returns"][0]
    for method in METHODS:
        for cap in CAPS:
            result=stop(ind,env.constants["return_range"],ALPHA,method,cap)
            ev=evaluation["returns"][3]-evaluation["returns"][0] # same frozen policy, efficient independent evaluation
            rows.append(dict(d=D0,coupling="independent",support="ordinary",method=method,cap=cap,
                             bound=env.constants["return_range"],alpha=ALPHA,independent_gain=float(ev.mean()),
                             deployed_gain=float(ev.mean()) if result["adopted"] else 0,**result))
    ablation=dict(crn_variance=float(crn.var(ddof=1)),independent_variance=float(ind.var(ddof=1)),
                  variance_ratio=float(ind.var(ddof=1)/crn.var(ddof=1)),
                  crn_cert_mean=float(crn.mean()),independent_cert_mean=float(ind.mean()),
                  independent_initial_and_disturbance=True)
    ell=analyze_ell(env,cert,evaluation)
    dump(directory/"stopping.json",rows);dump(directory/"metrics.json",metrics)
    dump(directory/"ablation.json",ablation);dump(directory/"ell_replay.json",ell)
    result=dict(n=n,seed=seed,seconds=time.perf_counter()-started,fresh_pairs_per_bank=4096,
                actual_simulator_transitions=int(cert["actual_simulator_transitions"]+evaluation["actual_simulator_transitions"]),
                logical_comparison_transitions=int(cert["logical_pair_comparison_transitions"]+evaluation["logical_pair_comparison_transitions"]),
                source_manifest=sha(OUT/"source_hashes.json"),outputs={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
    dump(directory/"COMPLETE.json",result)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--freeze-only",action="store_true");ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args();freeze()
    if not args.freeze_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run,n,s):(n,s) for n in c.DIMENSIONS for s in range(200,208)}
            for i,future in enumerate(as_completed(futures),1):
                result=future.result()
                print(f"{i}/24 dim{result['n']} seed{result['seed']}: {result['seconds']:.1f}s; {result['actual_simulator_transitions']} actual transitions",flush=True)
