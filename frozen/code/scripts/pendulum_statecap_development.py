"""Second, recorded exploratory design: state-dependent action cap.

The new cap is zero at equilibrium and enforces the original global d.
Constants are derived before any rollout; oracle pilot is not learner evidence.
"""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.linalg import solve_discrete_are
from pendulum_regional_development import K,G,constants,transition

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_statecap_development'

def design():
    T=np.array([[2.86,.686],[0.,.35]])
    k,noise,kn=constants(T);Ti=np.linalg.inv(T)
    # Round all constants outward. Uniform over gain [.75,.95], damping [.05,.15].
    assert k<.864 and noise<.1351 and kn<7.06
    cfg=dict(T=T.tolist(),K=K.tolist(),kappa=.864,noise_gain=.1351,
             action_gain=.12835,Knorm=7.06,R=.254,rho=.24,
             initial_radius=.19,d=.0125,width=.03,H=100)
    emax=cfg['action_gain']*cfg['d']/(1-cfg['kappa'])
    assert cfg['rho']+emax<cfg['R']
    assert cfg['kappa']*cfg['rho']+cfg['noise_gain']*cfg['width']<cfg['rho']
    assert cfg['Knorm']*cfg['R']+cfg['d']<2
    coords=cfg['R']*np.linalg.norm(Ti,axis=1)
    assert coords[0]<.2
    assert coords[1]+.75*np.sin(coords[0])+.15*.95*2+.15*cfg['width']<8
    Q=np.diag([1.,.1])+.001*np.outer(K,K);Qn=np.linalg.norm(Ti.T@Q@Ti,2)
    assert Qn<1.334
    cfg['Qnorm']=1.334
    cfg['coordinate_radii']=coords.tolist()
    for statecap in [False,True]:
        e=0.;s=cfg['initial_radius'];b=0.;m=0.
        for t in range(cfg['H']):
            a=cfg['d']*min(1,(s+e)/cfg['initial_radius']) if statecap else cfg['d']
            b+=cfg['Qnorm']*(2*s+e)*e+.001*(2*cfg['Knorm']*(s+e)*a+a*a)
            m+=cfg['Qnorm']*(s+e)**2+.001*(2*cfg['Knorm']*(s+e)*a+a*a)
            e=cfg['kappa']*e+cfg['action_gain']*a
            s=cfg['kappa']*s+cfg['noise_gain']*cfg['width']
        cfg['statecap' if statecap else 'constantcap']=dict(paired=b,marginal=m)
    return cfg

def bank(cfg,n,seed):
    rng=np.random.default_rng(seed);ang=rng.uniform(0,2*np.pi,n)
    rad=np.sqrt(rng.uniform(size=n))*cfg['initial_radius']
    x=np.column_stack([np.cos(ang),np.sin(ang)])*rad[:,None]@np.linalg.inv(np.array(cfg['T'])).T
    return x,rng.uniform(-cfg['width'],cfg['width'],(cfg['H'],n))

def rollout(g,c,coeff,bank,cfg,statecap=True):
    x0,w=bank;x=x0.copy();ret=np.zeros(len(x));K=np.array(cfg['K']);T=np.array(cfg['T'])
    for noise in w:
        a=-x@K
        if coeff is not None:
            cap=cfg['d']*np.minimum(1,np.linalg.norm(x@T.T,axis=1)/cfg['initial_radius']) if statecap else cfg['d']
            a+=np.clip(x@coeff,-cap,cap)
        x,r=transition(x,a,noise,g,c);ret+=r
    return ret

def main():
    threadpool_limits(1);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Preserve development output')
    cfg=design();rows=[]
    for g,c in [[.75,.05],[.95,.15]]:
        A=np.array([[1.0375,.05*(1-.05*c)],[.75,1-.05*c]])
        Gg=g*G[:,None];S=solve_discrete_are(A,Gg,np.diag([1.,.1]),np.array([[.001]]))
        Kg=np.linalg.solve(np.array([[.001]])+Gg.T@S@Gg,Gg.T@S@A).ravel()
        bk=bank(cfg,4096,[42835,round(g*100)]);rb=rollout(g,c,None,bk,cfg)
        for statecap in [False,True]:
            diff=rollout(g,c,K-Kg,bk,cfg,statecap)-rb
            b=cfg['statecap' if statecap else 'constantcap']['paired']
            rows.append(dict(g=g,c=c,statecap=statecap,oracle_direction=(K-Kg).tolist(),
                             baseline_return=float(rb.mean()),gain=float(diff.mean()),
                             gain_over_support=float(diff.mean()/b),support=b))
    result=dict(design=cfg,pilots=rows,development_only=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
