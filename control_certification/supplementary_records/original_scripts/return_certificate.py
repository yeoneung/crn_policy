"""Established bounded-mean betting for independent whole-policy adoption.

This module is intentionally independent of neural calibration. D is a full
episode return difference with a justified support [-bound,bound]. Under a
non-improving fixed proposal each factor has conditional expectation <=1.
Mixture wealth is a nonnegative supermartingale; Ville's inequality controls
false adoption, including stopping at data-dependent times.
"""
import numpy as np
from scipy.special import logsumexp

FRACTIONS=2.**(-np.arange(8))
WEIGHTS=np.array([.5]+[.5/7]*7)


def log_wealth(d,bound,margin=0.):
    d=np.asarray(d,dtype=float)
    if bound<=0 or d.ndim!=1 or not np.isfinite(d).all() or np.max(abs(d),initial=0)>bound:
        raise ValueError('Known support violated or invalid inputs')
    if not 0<=margin<=bound:raise ValueError('Only nonnegative improvement margins supported')
    x=(d-margin)/(bound+margin)
    factors=1+FRACTIONS[:,None]*x[None,:]
    # Exact support endpoints may produce a zero component wealth, which is
    # valid. Do not replace it by a positive numerical floor.
    with np.errstate(divide='ignore'):
        log_components=np.log(factors).sum(1)
    return float(logsumexp(np.log(WEIGHTS)+log_components))


def deployed_gain_lower(d,bound,alpha):
    """Lower bound for the adopt-or-fallback output; not for a rejected proposal."""
    if not 0<alpha<1:raise ValueError('alpha must lie in (0,1)')
    if log_wealth(d,bound)<np.log(1/alpha):return 0.
    lo,hi=0.,bound
    for _ in range(55):
        mid=(lo+hi)/2
        if log_wealth(d,bound,mid)>=np.log(1/alpha):lo=mid
        else:hi=mid
    return lo


class ReturnCertificate:
    def __init__(self,bound,alpha=.05):
        if bound<=0 or not 0<alpha<1:raise ValueError('Invalid bound or level')
        self.bound=float(bound);self.alpha=float(alpha);self.parts=[];self.adopted=False
        self.episodes=0;self.trace=[];self.lower=0.
    def update(self,d):
        if self.adopted:raise RuntimeError('A completed deployment test must not be updated')
        d=np.asarray(d,dtype=float)
        log_wealth(d,self.bound) # Validate before mutating state.
        self.parts.append(d.copy());self.episodes+=len(d)
        values=np.concatenate(self.parts);wealth=log_wealth(values,self.bound)
        self.trace.append([self.episodes,wealth])
        if wealth>=np.log(1/self.alpha):
            self.adopted=True;self.lower=deployed_gain_lower(values,self.bound,self.alpha)
        return self.adopted
    def result(self):
        return dict(adopted=self.adopted,episodes=self.episodes,
            deployed_gain_lower=self.lower,proposal_gain_lower=self.lower if self.adopted else None,
            alpha=self.alpha,bound=self.bound,trace=self.trace,
            test='Fixed nonnegative mixture of bounded-mean bets; standard Ville guarantee')


class TwoSidedAdoption:
    """Accept improvement, reject on negative evidence, or exhaust the budget.

    False adoption is controlled by alpha. The negative-mean test is a
    futility rule with its own same-sized level, and cannot inflate false
    adoption. Stating a joint error guarantee for both labels requires
    counting both tails, rather than silently reusing a one-sided alpha.
    """
    def __init__(self,bound,alpha=.05):
        self.positive=ReturnCertificate(bound,alpha);self.negative=ReturnCertificate(bound,alpha)
        self.outcome='inconclusive'
    def update(self,d):
        if self.outcome!='inconclusive':raise RuntimeError('Completed test')
        yes=self.positive.update(d);no=self.negative.update(-np.asarray(d))
        assert not (yes and no)
        if yes:self.outcome='adopt'
        elif no:self.outcome='reject'
        return self.outcome!='inconclusive'
    def result(self):
        result=self.positive.result();result['outcome']=self.outcome
        result['futility_trace']=self.negative.trace
        result['joint_two_tail_error_bound']=2*self.positive.alpha
        return result
