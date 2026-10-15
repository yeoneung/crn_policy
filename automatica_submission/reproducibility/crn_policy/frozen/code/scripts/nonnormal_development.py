"""Exploratory nonnormal, correlated-noise feasibility study."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import coupled_support_core as c
from finite_horizon_envelope import calibrate_matrix,return_support
from return_certificate import TwoSidedAdoption

ROOT=Path(__file__).resolve().parents[1]


class NonnormalPlant(c.StablePlant):
    def __init__(self,n,seed):
        super().__init__(n,seed);random=c.gen(856001,n,seed)
        rotation=np.linalg.qr(random.normal(size=(n,n)))[0]
        closed=.9*np.eye(n)+.13*np.diag(np.ones(n-1),1)
        self.linear_baseline=rotation@closed@rotation.T
        self.A=self.linear_baseline-self.B@self.K
        q=np.linalg.qr(random.normal(size=(n,n)))[0]
        self.noise_root=q@np.diag(np.sqrt(np.geomspace(.5,2.,n)))
        self.constants={k:v for k,v in self.constants.items() if k not in ['kappa','actuator_uncertainty_radius']}
    def step(self,x,a,noise):
        return super().step(x,a,noise@self.noise_root.T)
    def support(self,d):
        # Hidden-matrix use is only an evaluation audit, never a learner input.
        kappa=np.linalg.norm(self.linear_baseline,2)+.005
        t=np.arange(c.H);distance=.1*d*(kappa**t-1)/(kappa-1)
        return self.constants['return_range'],distance


def train(env,n,seed,bank,out):
    current=None;xx=[];yy=[];spent=0
    for stage,budget in enumerate([5120,20480,81920]):
        data,cost=c.collect_pool(env,current,16,c.gen(bank,n,seed,stage));spent+=cost
        actions=c.candidates(env.baseline(data['x']),.05)
        pairs,ix=c.select(data,actions,budget-spent,c.gen(bank+100,n,seed,stage))
        values,cost=c.query(env,data,actions,pairs,ix,c.gen(bank+200,n,seed,stage));spent+=cost;assert spent==budget
        si,pi=ix.T;chosen=np.stack([actions[si,0],actions[si,pairs[pi,0]],actions[si,pairs[pi,1]]],1)
        ff=c.features(data['x'][si],data['remaining'][si],chosen,.05);x=ff[:,2]-ff[:,1]
        y=np.diff(values,axis=1).ravel()/.05;xx.append(x);yy.append(y);xa=np.concatenate(xx);ya=np.concatenate(yy)
        coef=np.linalg.solve(xa.T@xa+.1*np.eye(xa.shape[1]),xa.T@ya);current=c.RidgePatch(coef,.05,n)
        np.savez_compressed(out/f'train_b{budget}.npz',**data,actions=actions,pairs=pairs,selected=ix,
            returns=values,fit_x=x,fit_target=y,coef=coef)
    return current


if __name__=='__main__':
    threadpool_limits(limits=1);out=ROOT/'results/nonnormal_development';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Preserve earlier attempt')
    spec=dict(exploratory=True,dimensions=[4,8,16],seeds=[100,101],training=81920,
        ridge=.1,d=.05,calibration=8192,calibration_alpha=.025/6,test_alpha=.025/(6*2),
        final_episodes=1024,certification_pairs=2048,nonlinear_lip=.005,noise_covariance_unknown=True,
        selection='No hyperparameter grid; feasibility only. Do not use final seeds for redesign.')
    (out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    for f in [Path(__file__),Path(__file__).with_name('finite_horizon_envelope.py')]:
        (out/f.name).write_bytes(f.read_bytes())
    rows=[]
    for n in [4,8,16]:
        for seed in [100,101]:
            folder=out/f'dim{n}_seed{seed}';folder.mkdir();env=NonnormalPlant(n,seed);start=time.time()
            policy=train(env,n,seed,856100,folder)
            random=c.gen(856400,n,seed);x=env.initial(8192,random);a=env.baseline(x)
            y1,_=env.step(x,a,random.normal(size=x.shape));y2,_=env.step(x,a,random.normal(size=x.shape))
            np.savez_compressed(folder/'calibration.npz',x=x,y1=y1,y2=y2)
            fit=calibrate_matrix(x,y1,y2,.005,spec['calibration_alpha']);bound,distance,powers=return_support(fit,env.constants,.05)
            ev=c.evaluate(env,[None,policy],1024,c.gen(856500,n,seed),[0,.05]);np.savez_compressed(folder/'evaluation.npz',**ev)
            cert=c.evaluate(env,[None,policy],2048,c.gen(856600,n,seed),[0,.05]);np.savez_compressed(folder/'certification.npz',**cert)
            dif=cert['returns'][1]-cert['returns'][0];gates=[]
            for kind,b in [('matrix_power',bound),('ordinary',env.constants['return_range'])]:
                test=TwoSidedAdoption(b,spec['test_alpha'])
                for i in range(0,2048,16):
                    if test.update(dif[i:i+16]):break
                gates.append(dict(kind=kind,**test.result()))
            row=dict(n=n,seed=seed,fit=fit,bound=bound,gain=float((ev['returns'][1]-ev['returns'][0]).mean()),
                true_linear_norm=float(np.linalg.norm(env.linear_baseline,2)),
                true_matrix_error=float(np.linalg.norm(np.array(fit['matrix'])-env.linear_baseline,2)),
                certificates=gates,seconds=time.time()-start)
            (folder/'COMPLETE').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(n,seed,'gain',row['gain'],'bound',bound,'scalar upper',fit['scalar_lipschitz_upper'],
                'pairs',[(g['kind'],g['episodes'],g['adopted']) for g in gates],flush=True)
    (out/'summary.json').write_text(json.dumps(dict(exploratory=True,rows=rows),indent=2)+'\n')
