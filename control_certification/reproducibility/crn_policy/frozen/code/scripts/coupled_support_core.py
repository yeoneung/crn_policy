"""Black-box bounded patches of a baseline with a known contraction envelope.

The learner sees state/action/scalar returns, baseline actions and the public
displacement/envelope constants. The simulator alone uses the hidden target
matrix and reward geometry. All numerics are float64.
"""
import numpy as np
from scipy.linalg import expm

H=40
GAMMA=float(np.exp(-.01))
DT=.05
DIMENSIONS=[4,8,16]


def gen(bank,n,seed,*extra):return np.random.default_rng([bank,n,seed,*extra])


def unit(x):
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    out=np.zeros_like(x);out[...,0]=1
    return np.divide(x,norm,out=out,where=norm>1e-12)


def pair_bound(kappa,gain,baseline_lip,reward_x,reward_a,displacement):
    if not 0<=kappa<1:raise ValueError('Contraction must be justified')
    t=np.arange(H)
    distance=gain*displacement*(1-kappa**t)/(1-kappa)
    reward=(reward_x+reward_a*baseline_lip)*distance+reward_a*displacement
    return float((GAMMA**t)@reward),distance


class StablePlant:
    def __init__(self,n,seed):
        self.n=self.m=n;self.seed=seed
        random=gen(830010,n,seed)
        orth=np.linalg.qr(random.normal(size=(n,n)))[0]
        skew=random.normal(size=(n,n));skew-=skew.T;skew/=max(np.linalg.norm(skew,2),1e-12)
        self.A=.95*np.eye(n)+.005*skew
        self.B0=.1*orth;self.K=-.5*orth.T
        cross=random.normal(size=(n,n));cross-=cross.T;cross*=.35/max(np.linalg.norm(cross,2),1e-12)
        efficiency=random.uniform(.8,1.,n)
        self.B=self.B0@expm(cross)@np.diag(efficiency)
        self.C=np.linalg.qr(random.normal(size=(n,n)))[0]
        qr=np.linalg.qr(random.normal(size=(n,n)))[0]
        self.Q=qr@np.diag(np.geomspace(.25,2.,n))@qr.T
        self.noise=.035
        # Public uncertainty set: orthogonal actuator rotation spectral angle
        # <=.35 and each efficiency in [.8,1]. No target matrix is used here.
        delta=.1*(2*np.sin(.35/2)+.2)
        kappa=float(np.linalg.norm(self.A+self.B0@self.K,2)+delta*.5+.005)
        self.constants=dict(kappa=kappa,action_gain=.1,baseline_lipschitz=.5,
            reward_state_lipschitz=DT*np.sqrt(4/np.e),
            reward_action_lipschitz=DT*.1*np.sqrt(2/np.e),
            return_range=DT*1.1*sum(GAMMA**t for t in range(H)),
            actuator_uncertainty_radius=delta)
        assert kappa<1
        assert np.linalg.norm(self.B-self.B0,2)<=delta+1e-12
        assert np.linalg.norm(self.A+self.B@self.K,2)+.005<=kappa+1e-12
        assert np.linalg.norm(self.B,2)<=.1+1e-12
    def baseline(self,x):return x@self.K.T
    def initial(self,n,random):return random.normal(size=(n,self.n))*.5
    def step(self,x,a,noise):
        state=np.einsum('bi,ij,bj->b',x,self.Q,x)
        reward=-DT*(-np.expm1(-state)+.1*(-np.expm1(-np.sum(a*a,axis=1))))
        y=x@self.A.T+a@self.B.T+.005*np.tanh(x@self.C.T)+self.noise*noise
        return y,reward
    def support(self,d):
        c=self.constants
        bound,dist=pair_bound(c['kappa'],c['action_gain'],c['baseline_lipschitz'],
            c['reward_state_lipschitz'],c['reward_action_lipschitz'],d)
        return min(bound,c['return_range']),dist


def max_angle(base,d):
    eta=np.linalg.norm(base,axis=1)
    angle=np.minimum(.8,2*np.arcsin(np.minimum(1.,d/(2*np.maximum(eta,1e-12)))))
    return eta,angle


def candidates(base,d):
    eta,theta=max_angle(base,d);u=unit(base);out=[base]
    for fraction in [.25,.5,1.]:
        angle=fraction*theta
        for j in range(base.shape[1]):
            e=np.zeros_like(base);e[:,j]=1;e-=u*u[:,j,None]
            norm=np.linalg.norm(e,axis=1);e=unit(e)
            for sign in [1,-1]:
                a=eta[:,None]*(np.cos(angle)[:,None]*u+sign*np.sin(angle)[:,None]*e)
                a=np.where((norm>1e-10)[:,None],a,base);out.append(a)
    result=np.stack(out,axis=1)
    assert np.max(np.linalg.norm(result-base[:,None],axis=-1))<=d+1e-10
    return result


def context(x,remaining):
    r=np.asarray(remaining)[:,None]/H
    return np.concatenate([np.ones_like(r),x,r*x,r,r*r],axis=-1)


def features(x,remaining,actions,d):
    displacement=actions-actions[:,:1]
    f=(context(x,remaining)[:,None,:,None]*(displacement/d)[:,:,None,:]).reshape(*actions.shape[:2],-1)
    squared=np.sum(displacement**2,axis=-1,keepdims=True)/d
    return np.concatenate([f,squared,squared*(np.asarray(remaining)/H)[:,None,None]],axis=-1)


class RidgePatch:
    def __init__(self,coef,d,m):self.coef=coef;self.d=d;self.m=m
    def action(self,x,remaining,baseline):
        base=baseline(x);acts=candidates(base,self.d);delta=acts-base[:,None]
        field=context(x,remaining)@self.coef[:-2].reshape(-1,self.m)
        linear=np.einsum('bka,ba->bk',delta/self.d,field)
        curve=(self.coef[-2]+self.coef[-1]*np.asarray(remaining)/H)
        score=linear+curve[:,None]*np.sum(delta*delta,axis=-1)/self.d
        ix=score.argmax(1)
        ix=np.where(score[np.arange(len(x)),ix]>0,ix,0)
        return acts[np.arange(len(x)),ix]


def bounded_direction(base,desired,d):
    eta,limit=max_angle(base,d);u=unit(base);v=unit(desired)
    cosine=np.clip(np.sum(u*v,axis=1),-1.,1.)
    angle=np.minimum(np.arccos(cosine),limit)
    tangent=v-cosine[:,None]*u;norm=np.linalg.norm(tangent,axis=1)
    ix=np.argmin(abs(u),axis=1);fallback=np.zeros_like(u);fallback[np.arange(len(u)),ix]=1
    fallback-=u*np.sum(fallback*u,axis=1,keepdims=True)
    tangent=np.where((norm>1e-10)[:,None],unit(tangent),unit(fallback))
    result=eta[:,None]*(np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*tangent)
    return np.where((angle>1e-10)[:,None],result,base)


class ARSPatch:
    def __init__(self,weight,d):self.weight=weight;self.d=d
    def action(self,x,remaining,baseline):
        base=baseline(x);f=np.concatenate([np.tanh(x),np.ones((len(x),1))],axis=1)
        return bounded_direction(base,base+f@self.weight.T,self.d)


def evaluate(env,policies,n,random,displacements=None,batch=256):
    records=[];max_state=np.zeros((len(policies),H));max_action=np.zeros(len(policies))
    max_difference=np.zeros(len(policies));max_radius_error=np.zeros(len(policies))
    for start in range(0,n,batch):
        size=min(batch,n-start);initial=env.initial(size,random)
        x=np.tile(initial,(len(policies),1));ret=np.zeros((len(policies),size))
        for t in range(H):
            acts=[]
            for j,pol in enumerate(policies):
                xx=x[j*size:(j+1)*size]
                a=env.baseline(xx) if pol is None else pol.action(xx,np.full(size,H-t),env.baseline)
                max_action[j]=max(max_action[j],float(np.linalg.norm(a-env.baseline(xx),axis=1).max()))
                max_radius_error[j]=max(max_radius_error[j],float(abs(np.linalg.norm(a,axis=1)-np.linalg.norm(env.baseline(xx),axis=1)).max()))
                acts.append(a)
            a=np.concatenate(acts)
            states=x.reshape(len(policies),size,env.n)
            max_state[:,t]=np.maximum(max_state[:,t],np.linalg.norm(states-states[:1],axis=-1).max(1))
            x,reward=env.step(x,a,np.tile(random.normal(size=(size,env.n)),(len(policies),1)))
            ret+=GAMMA**t*reward.reshape(len(policies),size)
        records.append(ret);max_difference=np.maximum(max_difference,abs(ret-ret[:1]).max(1))
    returns=np.concatenate(records,axis=1)
    if displacements is not None:
        assert policies[0] is None
        for j,d in enumerate(displacements):
            bound,envelope=env.support(d)
            assert max_action[j]<=d+1e-10
            assert max_radius_error[j]<=1e-10
            assert np.max(max_state[j]-envelope)<=1e-10
            assert max_difference[j]<=bound+1e-10
    return dict(returns=returns,max_state_distance=max_state,max_action_displacement=max_action,
        max_return_difference=max_difference,max_same_state_radius_error=max_radius_error,
        transitions=np.array(n*len(policies)*H))


def collect_pool(env,policy,n,random):
    x=env.initial(n,random);states=[];remaining=[]
    for t in range(H):
        if t%8==0:states.append(x.copy());remaining.append(np.full(n,H-t))
        a=env.baseline(x) if policy is None else policy.action(x,np.full(n,H-t),env.baseline)
        x,_=env.step(x,a,random.normal(size=x.shape))
    return dict(x=np.concatenate(states),remaining=np.concatenate(remaining)),n*H


def select(data,actions,budget,random):
    k=actions.shape[1];pairs=np.array([(0,j) for j in range(1,k)]+[(j,j+1) for j in range(1,k,2)])
    cost=2*data['remaining'];ix=[];spent=0
    while True:
        available=np.flatnonzero(cost<=budget-spent)
        if not len(available):break
        s=int(random.choice(available));p=int(random.integers(len(pairs)))
        ix.append([s,p]);spent+=int(cost[s])
    assert spent==budget
    return pairs,np.array(ix)


def query(env,data,actions,pairs,selected,random):
    si,pi=selected.T;count=len(si);x=np.repeat(data['x'][si],2,axis=0)
    rem=np.repeat(data['remaining'][si],2)
    a=np.stack([actions[si,pairs[pi,j]] for j in range(2)],axis=1).reshape(-1,env.m)
    ret=np.zeros(2*count);spent=0
    for t in range(int(rem.max())):
        ix=np.flatnonzero(rem>t)
        noise=np.repeat(random.normal(size=(count,env.n)),2,axis=0)
        y,r=env.step(x[ix],a[ix],noise[ix]);ret[ix]+=GAMMA**t*r;spent+=len(ix)
        x[ix]=y;a[ix]=env.baseline(y)
    return ret.reshape(count,2),spent


def ars_update(env,weight,d,random,step=.1,noise=.1,directions=4,episodes=4):
    delta=random.normal(size=(directions,*weight.shape))/np.sqrt(weight.shape[-1])
    pol=[ARSPatch(weight+sign*noise*z,d) for z in delta for sign in [1,-1]]
    ev=evaluate(env,pol,episodes,random)
    means=ev['returns'].mean(1).reshape(directions,2);best=np.argsort(means.max(1))[-max(1,directions//2):]
    change=((means[best,0]-means[best,1])[:,None,None]*delta[best]).mean(0)
    scale=max(float(means[best].std()),1e-8);updated=weight+step*change/scale
    return updated,dict(delta=delta,means=means,selected=best,weight_before=weight,weight_after=updated,
                       scale=scale,**ev)
