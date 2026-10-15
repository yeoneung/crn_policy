"""Independent baseline-system calibration with unknown Gaussian noise scale.

Observe two independent next states at each reset state, under the baseline.
Assume F(x)=M x+h(x), h(0)=0 and a known global Lipschitz bound ell for h.
The learner gets neither M nor h nor the noise variance. A public action-gain
bound and reward constants remain necessary for paired-return certification.
"""
import numpy as np
from scipy.stats import f as f_distribution


def estimate_contraction(x,y1,y2,nonlinear_lip,alpha):
    x=np.asarray(x,dtype=float);y1=np.asarray(y1,dtype=float);y2=np.asarray(y2,dtype=float)
    if x.ndim!=2 or y1.shape!=x.shape or y2.shape!=x.shape:raise ValueError('Expected N by n observations')
    n_samples,n=x.shape
    if nonlinear_lip<0 or not 0<alpha<1:raise ValueError('Invalid assumptions')
    if n_samples<n or np.linalg.matrix_rank(x)<n:return dict(valid=False,reason='rank deficient')
    gram=x.T@x;smallest=float(np.linalg.eigvalsh(gram)[0])
    estimate=np.linalg.solve(gram,x.T@((y1+y2)/2)).T
    variance=float(np.sum((y1-y2)**2)/(2*n*n_samples))
    degrees=n*n_samples;parameters=n*n
    radius=float(np.sqrt(parameters*f_distribution.ppf(1-alpha,parameters,degrees)))
    stochastic=float(np.sqrt(variance/2)*radius/np.sqrt(smallest))
    nonlinear_bias=float(nonlinear_lip*np.sqrt(np.trace(gram)/smallest))
    kappa=float(np.linalg.norm(estimate,2)+stochastic+nonlinear_bias+nonlinear_lip)
    return dict(valid=kappa<1,kappa=kappa,estimate=estimate.tolist(),noise_variance=variance,
        stochastic_error=stochastic,nonlinear_bias=nonlinear_bias,nonlinear_lipschitz=nonlinear_lip,
        alpha=alpha,samples=n_samples,transitions=2*n_samples,gram_min_eigenvalue=smallest,
        radius=radius,reason='certified contraction' if kappa<1 else 'contraction not certified')


def paired_support(constants,d,kappa,horizon=40,gamma=float(np.exp(-.01))):
    if not 0<=kappa<1:raise ValueError('No certified contraction')
    t=np.arange(horizon)
    distance=constants['action_gain']*d*(1-kappa**t)/(1-kappa)
    lipschitz=constants['reward_state_lipschitz']+constants['reward_action_lipschitz']*constants['baseline_lipschitz']
    return min(constants['return_range'],float((gamma**t)@(lipschitz*distance)))
