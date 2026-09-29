"""Deterministic region design, followed by explicitly exploratory pilots.

No final seeds or claims. All attempted radii and disturbance widths retained.
The state metric is designed using the public uncertainty set only.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import differential_evolution
from threadpoolctl import threadpool_limits
from review_theory_pendulum import jacobian, transition

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_regional_development'
G=np.array([.0075,.15])
A=jacobian(0,1,0,np.zeros(2))
S=solve_discrete_are(A,G[:,None],np.diag([1.,.1]),np.array([[.001]]))
K=np.linalg.solve(np.array([[.001]])+G[None,:]@S@G[:,None],G[None,:]@S@A).ravel()

def metric(z):
    # T defines ||x||_P=||Tx||_2. Fix det(T)=1 to remove scale ambiguity.
    return np.array([[np.exp(z[0]),z[1]],[0.,np.exp(-z[0])]])

def constants(T,angle=.2):
    Ti=np.linalg.inv(T)
    # Jacobian affine in cos(theta), gain, damping: operator norm is convex.
    kap=max(np.linalg.norm(T@jacobian(th,g,c,K)@Ti,2)
            for th in [0,angle] for g in [.75,.95] for c in [.05,.15])
    return float(kap+1e-10),float(np.linalg.norm(T@G)),float(np.linalg.norm(K@Ti))

def design():
    opt=differential_evolution(lambda z:constants(metric(z))[0],
                               [(-1,3),(-3,3)],seed=42832,tol=1e-10)
    T=metric(opt.x);Ti=np.linalg.inv(T);kap,noise,Knorm=constants(T)
    # Full region is P ball radius R. All states there are unsaturated.
    R=min(.2/np.linalg.norm(Ti[0]),1.8/Knorm)
    return dict(T=T.tolist(),K=K.tolist(),kappa=kap,noise_gain=noise,
                action_gain=.95*noise,Knorm=Knorm,R=float(R),
                coordinate_radii=(R*np.linalg.norm(Ti,axis=1)).tolist())

def envelope(cfg,d,width,initial_fraction=.8,H=100):
    T=np.array(cfg['T']);Ti=np.linalg.inv(T);kap=cfg['kappa'];R=cfg['R']
    noise=cfg['noise_gain'];La=cfg['action_gain'];Kn=cfg['Knorm']
    rho=(noise*width/(1-kap))*1.001
    # Use all the remaining region as the invariant baseline ball.
    emax=La*d/(1-kap);rho=max(rho,R-emax)
    valid=rho+emax <= R+1e-12 and rho>=noise*width/(1-kap) and Kn*R+d<2
    x0=initial_fraction*rho
    Q=np.diag([1.,.1])+.001*np.outer(K,K)
    Qp=Ti.T@Q@Ti;Qnorm=float(np.linalg.norm(Qp,2))
    e=0.;s=x0;paired=0.;marginal=0.
    for t in range(H):
        # Both trajectories satisfy norm bound s_t; difference bound e_t.
        # Baseline quadratic reward plus additive-action effort correction.
        paired+=Qnorm*(2*s+e)*e+.001*(2*Kn*(s+e)*d+d*d)
        marginal+=Qnorm*(s+e)**2+.001*(2*Kn*(s+e)*d+d*d)
        s=kap*s+noise*width;e=kap*e+La*d
    # Check speed clipping is inactive over entire full region and noise.
    coords=R*np.linalg.norm(Ti,axis=1)
    vmax=coords[1]+.75*np.sin(coords[0])+.15*.95*(Kn*R+d)+.15*width
    valid=bool(valid and vmax<8 and coords[0]<np.pi)
    return dict(d=d,width=width,rho=rho,initial_radius=x0,emax=emax,
                valid=valid,paired=paired,regional_marginal=marginal,
                speed_raw_bound=float(vmax),Qnorm=Qnorm)

def rollout(g,c,corr,bank,cfg):
    x0,w=bank;x=x0.copy();ret=np.zeros(len(x));K=np.array(cfg['K'])
    for noise in w:
        a=-x@K
        if corr is not None:
            coeff,d=corr;a+=np.clip(x@coeff,-d,d)
        x,r=transition(x,a,noise,g,c);ret+=r
    return ret

def main():
    threadpool_limits(1);OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Preserve completed development')
    cfg=design();attempts=[];rng=np.random.default_rng(42833)
    for width in [.01,.03,.05]:
        for d in [.0125,.025,.05,.1]:
            env=envelope(cfg,d,width);rec=dict(**env)
            if env['valid']:
                Ti=np.linalg.inv(np.array(cfg['T']))
                ang=rng.uniform(0,2*np.pi,2048);rad=np.sqrt(rng.uniform(size=2048))*env['initial_radius']
                x0=np.column_stack([np.cos(ang),np.sin(ang)])*rad[:,None]@Ti.T
                bank=(x0,rng.uniform(-width,width,(100,2048)))
                rec['pilots']=[]
                for g,c in [[.75,.05],[.95,.15]]:
                    b=rollout(g,c,None,bank,cfg)
                    # Oracle linear LQR direction is only a feasibility diagnostic.
                    Ag=jacobian(0,g,c,np.zeros(2));Gg=g*G[:,None]
                    Sg=solve_discrete_are(Ag,Gg,np.diag([1.,.1]),np.array([[.001]]))
                    Kg=np.linalg.solve(np.array([[.001]])+Gg.T@Sg@Gg,Gg.T@Sg@Ag).ravel()
                    r=rollout(g,c,(K-Kg,d),bank,cfg)
                    rec['pilots'].append(dict(gain=g,damping=c,mean_baseline=float(b.mean()),
                         mean_gain=float((r-b).mean()),gain_over_support=float((r-b).mean()/env['paired']),
                         empirical_max_abs=float(abs(r-b).max())))
            attempts.append(rec)
    result=dict(purpose='Development; oracle direction used only to assess headroom',design=cfg,attempts=attempts)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
