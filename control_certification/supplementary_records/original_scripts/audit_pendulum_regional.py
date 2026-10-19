"""Exact rational inequality checks and an independent post-run audit.

This is analysis code; it is not a dependency of the frozen experiment.
"""
from pathlib import Path
from fractions import Fraction as F
import itertools,json,hashlib
import numpy as np
from scipy.special import logsumexp
from pendulum_regional_study import setup

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'code/results/pendulum_regional_study'
def m(a):return [[F(str(v)) for v in row] for row in a]
def tr(a):return [list(x) for x in zip(*a)]
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def scale(a,s):return [[x*s for x in row] for row in a]
def plus(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def det(a):return a[0][0]*a[1][1]-a[1][0]*a[0][1]
def inv(a):return scale([[a[1][1],-a[0][1]],[-a[1][0],a[0][0]]],1/det(a))
def pd(a):return a[0][0]>0 and det(a)>0
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def exact_constants():
    cfg=setup();T=m(cfg['T']);P=mm(tr(T),T);Pi=inv(P);K=[[F.from_float(k) for k in cfg['K']]]
    checks=[];min_det=None
    for gain,damp,cosine in itertools.product([F(3,4),F(19,20)],[F(1,20),F(3,20)],[F(49,50),F(1)]):
        row=[F(3,4)*cosine-F(3,20)*gain*K[0][0],1-F(1,20)*damp-F(3,20)*gain*K[0][1]]
        A=[[1+F(1,20)*row[0],F(1,20)*row[1]],row]
        gap=plus(scale(P,F(864,1000)**2),scale(mm(mm(tr(A),P),A),-1))
        assert pd(gap);checks.append(float(det(gap)))
    G=m([[.0075],[.15]])
    assert mm(mm(tr(G),P),G)[0][0]<F(1351,10000)**2
    assert mm(mm(K,Pi),tr(K))[0][0]<F(706,100)**2
    Q=plus(m([[1,0],[0,.1]]),scale(mm(tr(K),K),F(1,1000)))
    assert pd(plus(scale(P,F(1334,1000)),scale(Q,-1)))
    assert F(254,1000)**2*Pi[0][0]<F(1,5)**2
    assert F(254,1000)**2*Pi[1][1]<F(73,100)**2
    # Exact invariant/tube/actuator inequalities.
    k=F(864,1000);la=F(12835,100000);d=F(125,10000);rho=F(24,100);radius=F(254,1000)
    assert k*rho+F(1351,10000)*F(3,100)<rho
    assert rho+la*d/(1-k)<radius
    assert F(706,100)*radius+d<2
    # |v_raw| <= .73 + .75*.2 + .15*.95*2 + .15*.03 < 8.
    assert F(73,100)+F(3,4)*F(1,5)+F(3,20)*F(19,20)*2+F(3,20)*F(3,100)<8
    return dict(exact_rational_psd_checks=8,min_psd_determinant=float(min(checks)),
                cosine_lower_bound='.98 <= cos(theta) for |theta|<=.2 by 1-theta^2/2',
                all_geometric_bounds_passed=True)

def independent_log(d,b,alpha,method):
    if method=='fixed_mixture':
        fr=2.**(-np.arange(8));w=np.array([.5]+[.5/7]*7)
        return logsumexp(np.log(w)[:,None]+np.log1p(fr[:,None]*d/b).cumsum(axis=1),axis=0)
    out=[];sx=0.;sq=0.;previous_mean=.5;previous_var=.25;wealth=0.
    for t,y in enumerate(d,1):
        x=(y/b+1)/2
        lam=min(.75,np.sqrt(2*np.log(2/alpha)/(previous_var*t*np.log1p(t))))
        wealth+=lam*(x-.5)-(x-previous_mean)**2*(-np.log1p(-lam)-lam)
        out.append(wealth);sx+=x;mean=(.5+sx)/(t+1);sq+=(x-mean)**2
        previous_var=(.25+sq)/(t+1);previous_mean=mean
    return np.array(out)

def main():
    result=exact_constants();checks=0;hashes={}
    for mode in ['development','final']:
        folder=OUT/mode
        if not (folder/'summary.json').exists():continue
        protocol=json.loads((folder/'protocol.json').read_text())
        for k,v in protocol['source_hashes'].items():assert sha(ROOT/k.replace('\\','/'))==v;hashes[k]=v
        for p in sorted(folder.glob('seed*/metrics.json')):
            row=json.loads(p.read_text());directory=p.parent
            for k,v in json.loads((directory/'COMPLETE.json').read_text())['outputs'].items():assert sha(directory/k)==v
            with np.load(directory/'certification.npz') as bank:d=bank['returns'][1]-bank['returns'][0]
            for r in row['results']:
                pos=independent_log(d,r['bound'],r['alpha'],r['method'])
                neg=independent_log(-d,r['bound'],r['alpha'],r['method'])
                idx=np.arange(15,len(d),16);threshold=np.log(1/r['alpha']);outcome='inconclusive';n=len(d)
                crossings=idx[(pos[idx]>=threshold)|(neg[idx]>=threshold)]
                if len(crossings):
                    j=crossings[0];n=int(j+1);outcome='adopt' if pos[j]>=threshold else 'reject'
                assert n==r['pairs'] and outcome==r['outcome']
                assert abs(pos[n-1]-r['positive_log_wealth'])<1e-8
                assert abs(d).max()<r['bound'];checks+=1
            assert row['certification_audit']['state_envelope_violations']==0
            assert row['evaluation_audit']['state_envelope_violations']==0
            with np.load(directory/'evaluation.npz') as bank:
                assert abs(np.mean(bank['returns'][1]-bank['returns'][0])-row['independent_gain'])<1e-14
            if mode=='final':
                # Reconstruct held-out returns through a separate scalar dynamics implementation.
                cfg=protocol['config'];T=np.array(cfg['T']);K=np.array(cfg['K'])
                coeff=np.array(json.loads((directory/'frozen_policy.json').read_text())['coef'])
                rng=np.random.default_rng([row['seed'],4]);angles=rng.uniform(0,2*np.pi,4096)
                radii=np.sqrt(rng.uniform(size=4096))*cfg['initial_radius']
                starts=np.column_stack([np.cos(angles),np.sin(angles)])*radii[:,None]@np.linalg.inv(T).T
                noises=rng.uniform(-cfg['width'],cfg['width'],(cfg['H'],4096))
                with np.load(directory/'evaluation.npz') as bank: expected=bank['returns'][:,:128]
                recovered=np.zeros((2,128))
                for branch in range(2):
                    for episode in range(128):
                        theta,vel=starts[episode]
                        for t in range(cfg['H']):
                            features=T@np.array([theta,vel]);u=-K@np.array([theta,vel])
                            if branch:u+=cfg['d']*min(1,np.linalg.norm(features)/cfg['initial_radius'])*np.sign(features@coeff)
                            u=min(2,max(-2,u));angle=(theta+np.pi)%(2*np.pi)-np.pi
                            recovered[branch,episode]-=angle*angle+.1*vel*vel+.001*u*u
                            vel=min(8,max(-8,vel+.75*np.sin(theta)+.15*row['gain']*u-.05*row['damping']*vel+.15*noises[t,episode]))
                            theta+=.05*vel
                assert np.max(abs(recovered-expected))<1e-12
    result.update(stopping_checks=checks,independent_scalar_transition_checks=81920 if (OUT/'final/summary.json').exists() else 0,
                  source_hashes=hashes,audit_source_sha256=sha(Path(__file__)),all_passed=True)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
