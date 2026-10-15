"""Diagnostic review only: no final seeds, training, or adoption decisions.

Checks the supplied theory constants, exact native Gymnasium step semantics,
and feasibility of the proposed fixed Pendulum region. Preserve all drafts.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from threadpoolctl import threadpool_limits
import gymnasium
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from coupled_support_core import StablePlant

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/theory_pendulum_review'
DT=.05
H=100


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def transition(x,command,noise=None,gain=1.,damping=0.):
    u=np.clip(np.asarray(command),-2,2)
    theta,vel=x.T
    reward=-(((theta+np.pi)%(2*np.pi)-np.pi)**2+.1*vel**2+.001*u**2)
    w=0 if noise is None else noise
    # Preserve native order: clip speed BEFORE integrating the angle.
    v=np.clip(vel+(15*np.sin(theta)+3*gain*u-damping*vel+3*w)*DT,-8,8)
    return np.column_stack([theta+DT*v,v]),reward


def jacobian(theta,gain,damping,K,actuator_derivative=1.):
    row=np.array([15*DT*np.cos(theta),1-DT*damping])-3*DT*gain*actuator_derivative*K
    return np.vstack([np.array([1.,0.])+DT*row,row])


def main():
    threadpool_limits(limits=1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise RuntimeError('Preserve completed review; do not overwrite')
    paths=[ROOT/'manuscript/sections/drafts/endtoend_complexity_draft.tex',ROOT/'manuscript/sections/drafts/regional_envelope_draft.tex',ROOT/'EXTERNAL_PENDULUM_PLAN.md',Path(__file__)]
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),purpose='Analytic feasibility review and nonconfirmatory baseline-only pilot; no final seeds or repair learning',
                  input_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
                  pilot_shifts=[[.75,.05],[.75,.15],[.95,.05],[.95,.15]],episodes_per_shift=8192,horizon=H,
                  rng_seed=2026092803,initial_box=[.15,.3],region_box=[.35,1.2],noise_sd=.05)
    dump(OUT/'protocol.json',protocol)
    rng=np.random.default_rng(2026092802)
    x=rng.uniform([-4*np.pi,-8],[4*np.pi,8],size=(10000,2));u=rng.uniform(-4,4,10000)
    states,rewards=transition(x,u)
    env=PendulumEnv()
    discrepancies=[]
    for j in range(len(x)):
        env.state=x[j].copy();_,reward,_,_,_=env.step(np.array([u[j]]))
        discrepancies.append(max(float(np.max(abs(env.state-states[j]))),float(abs(reward-rewards[j]))))
    assert max(discrepancies)<1e-12
    A=jacobian(0,1,0,np.zeros(2))
    G=np.array([[3*DT*DT],[3*DT]])
    S=solve_discrete_are(A,G,np.diag([1.,.1]),np.array([[.001]]))
    K=np.linalg.solve(np.array([[.001]])+G.T@S@G,G.T@S@A).ravel()
    F=jacobian(0,1,0,K)
    P=solve_discrete_lyapunov(F.T,np.eye(2))
    eig,V=np.linalg.eigh(P);Ph=(V*np.sqrt(eig))@V.T;Phi=(V/np.sqrt(eig))@V.T
    point=np.array([.3,1.])
    local=[]
    for gain,damping in protocol['pilot_shifts']:
        saturated=jacobian(point[0],gain,damping,K,0.)
        origin=jacobian(0,gain,damping,K)
        local.append(dict(gain=gain,damping=damping,saturated_interior_point=point.tolist(),
                          unclipped_baseline_torque=float(-K@point),
                          saturated_spectral_radius=float(max(abs(np.linalg.eigvals(saturated)))),
                          saturated_operator_norm=float(np.linalg.norm(saturated,2)),
                          origin_euclidean_norm=float(np.linalg.norm(origin,2)),
                          origin_nominal_P_norm=float(np.linalg.norm(Ph@origin@Phi,2))))
    pilots=[]
    for j,(gain,damping) in enumerate(protocol['pilot_shifts']):
        random=np.random.default_rng([protocol['rng_seed'],j])
        x=random.uniform([-.15,-.3],[.15,.3],size=(8192,2));escaped=np.zeros(len(x),dtype=bool)
        ret=np.zeros(len(x));maxabs=np.zeros(2);saturated=0
        for t in range(H):
            escaped |= np.any(abs(x)>np.array([.35,1.2]),axis=1)
            maxabs=np.maximum(maxabs,abs(x).max(axis=0))
            command=-x@K;saturated+=int(np.sum(abs(command)>2))
            x,reward=transition(x,command,random.normal(0,.05,len(x)),gain,damping)
            ret+=reward
        pilots.append(dict(gain=gain,damping=damping,escapes=int(escaped.sum()),episodes=len(x),
                           mean_return=float(ret.mean()),min_return=float(ret.min()),
                           max_abs_state=maxabs.tolist(),torque_saturation_fraction=saturated/(len(x)*H)))
    margins=[]
    for n in [4,8,16]:
        for seed in range(200,208):
            plant=StablePlant(n,seed);norm=float(np.linalg.norm(plant.A+plant.B@plant.K,2))
            g=1-norm-.005*(1+4*np.sqrt(n))
            term=93*.035**2*(n+np.sqrt(2*np.log(8/.05)))**2/(.5**2*g**2) if g>0 else None
            margins.append(dict(n=n,seed=seed,matrix_norm=norm,g=float(g),noise_term_reset_pairs_at_delta_05=None if term is None else int(np.ceil(term))))
    global_bound=H*(np.pi**2+.1*8**2+.001*2**2)
    p0=-np.expm1(np.log(.005)/200000)
    bet_drift=lambda mu:float((1+mu)/2*np.log(1.5)+(1-mu)/2*np.log(.5))
    result=dict(gymnasium_version=gymnasium.__version__,gymnasium_source_sha256=digest(Path(inspect.getsourcefile(PendulumEnv))),
                native_step_cases=10000,native_max_absolute_error=max(discrepancies),lqr_gain=K.tolist(),nominal_P=P.tolist(),
                maximum_requested_torque_over_proposed_region=float(abs(K)@np.array([.35,1.2])),
                maximum_requested_torque_over_initial_box=float(abs(K)@np.array([.15,.3])),
                regional_counterexamples=local,baseline_pilots=pilots,
                pilot_transitions=4*8192*H,native_parity_transitions=10000,
                global_native_reward_return_bound=float(global_bound),zero_escape_upper_at_200000=float(p0),
                draft_zero_escape_shift=float(2*global_bound*p0),
                tight_shift_limit_when_paired_bound_small=float(global_bound*p0),
                baseline_escape_cost_per_plant=200000*H,
                fixed_lambda_half_counterexample=dict(normalized_mean=.01,expected_log_factor=bet_drift(.01)),
                smallest_grid_lambda_counterexample=dict(normalized_mean=.001,expected_log_factor=float(.5005*np.log1p(1/128)+.4995*np.log1p(-1/128))),
                old_study_margin_audit=margins,
                conclusion='Original saturated LQR rectangle admits a local Jacobian with spectral radius >1, so no induced norm gives uniform one-step contraction there. Pilot escapes are separate empirical diagnostics, not a proof.')
    dump(OUT/'review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
