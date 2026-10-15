"""Matrix-power return envelope; no one-step contraction assumption.

The rowwise Gaussian calibration permits an unknown full disturbance
covariance. Correlations across coordinates need not be estimated: a union
bound combines valid marginal regression pivots.
"""
import numpy as np
from scipy.stats import f as f_distribution


def calibrate_matrix(x,y1,y2,nonlinear_lip,alpha):
    x=np.asarray(x,dtype=float);y1=np.asarray(y1,dtype=float);y2=np.asarray(y2,dtype=float)
    count,n=x.shape
    if y1.shape!=x.shape or y2.shape!=x.shape or np.linalg.matrix_rank(x)<n:raise ValueError('Full-rank replicated observations required')
    gram=x.T@x;smallest=float(np.linalg.eigvalsh(gram)[0])
    matrix=np.linalg.solve(gram,x.T@((y1+y2)/2)).T
    variances=np.sum((y1-y2)**2,axis=0)/(2*count)
    radius=float(np.sqrt(n*f_distribution.ppf(1-alpha/n,n,count)))
    noise=float(radius*np.sqrt(variances.sum()/2)/np.sqrt(smallest))
    bias=float(nonlinear_lip*np.sqrt(np.trace(gram)/smallest))
    return dict(matrix=matrix.tolist(),noise_variances=variances.tolist(),noise_error=noise,
        nonlinear_bias=bias,matrix_error=noise+bias,nonlinear_lip=nonlinear_lip,
        alpha=alpha,samples=count,transitions=2*count,gram_min_eigenvalue=smallest,
        scalar_lipschitz_upper=float(np.linalg.norm(matrix,2)+noise+bias+nonlinear_lip))


def power_envelope(matrix,residual_lip,action_gain,d,horizon):
    matrix=np.asarray(matrix,dtype=float)
    if residual_lip<0 or action_gain<0 or d<0 or horizon<1:raise ValueError('Invalid constants')
    power=np.eye(len(matrix));norms=[]
    for _ in range(horizon):
        norms.append(float(np.linalg.norm(power,2)));power=power@matrix
    norms=np.asarray(norms);distance=np.zeros(horizon)
    for t in range(1,horizon):
        distance[t]=action_gain*d*norms[:t].sum()+residual_lip*np.dot(norms[:t][::-1],distance[:t])
    return distance,norms


def return_support(fit,constants,d,horizon=40,gamma=float(np.exp(-.01))):
    distance,powers=power_envelope(fit['matrix'],fit['matrix_error']+fit['nonlinear_lip'],constants['action_gain'],d,horizon)
    reward_lip=constants['reward_state_lipschitz']+constants['reward_action_lipschitz']*constants['baseline_lipschitz']
    bound=min(constants['return_range'],float((gamma**np.arange(horizon))@(reward_lip*distance)))
    return bound,distance,powers
