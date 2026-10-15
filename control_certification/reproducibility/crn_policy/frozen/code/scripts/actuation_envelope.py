"""Finite-horizon forcing envelope with a known actuator image.

Assume f(x,a)-f(x,B(x))=G0 v(x,a), with ||v(x,a)||<=||a-B(x)||.
For linear uncertain input B_target=G0 T with ||T||<=1 this holds directly.
The same calibrated reference matrix error/nonlinear allowance is used.
"""
import numpy as np


def actuation_envelope(matrix, residual_lip, action_factor, displacement, horizon):
    matrix=np.asarray(matrix,dtype=float);factor=np.asarray(action_factor,dtype=float)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.shape[0]<1:
        raise ValueError('Square reference matrix required')
    if factor.ndim!=2 or factor.shape[0]!=matrix.shape[0] or factor.shape[1]<1:
        raise ValueError('Action factor must map into the state space')
    if not np.isfinite(matrix).all() or not np.isfinite(factor).all():
        raise ValueError('Finite matrices required')
    if not np.isfinite(residual_lip) or residual_lip<0 or not np.isfinite(displacement) or displacement<0:
        raise ValueError('Nonnegative finite allowances required')
    if not isinstance(horizon,(int,np.integer)) or horizon<1:raise ValueError('Positive integer horizon required')
    power=np.eye(len(matrix));powers=[];forcing=[]
    for _ in range(horizon):
        powers.append(float(np.linalg.norm(power,2)))
        forcing.append(float(np.linalg.norm(power@factor,2)))
        power=power@matrix
    powers=np.asarray(powers);forcing=np.asarray(forcing);distance=np.zeros(horizon)
    for t in range(1,horizon):
        distance[t]=displacement*forcing[:t].sum()+residual_lip*np.dot(powers[:t][::-1],distance[:t])
    return distance,powers,forcing


def actuation_return_support(fit,constants,action_factor,displacement,horizon=40,gamma=float(np.exp(-.01))):
    distance,powers,forcing=actuation_envelope(fit['matrix'],fit['matrix_error']+fit['nonlinear_lip'],
                                             action_factor,displacement,horizon)
    reward_lip=constants['reward_state_lipschitz']+constants['reward_action_lipschitz']*constants['baseline_lipschitz']
    bound=min(constants['return_range'],float((gamma**np.arange(horizon))@(reward_lip*distance)))
    return bound,distance,powers,forcing
