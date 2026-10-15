"""Numerically amended mechanical study; physical plant and objective unchanged."""
import numpy as np
import coupled_support_core as c
from mechanical_repair_core import MechanicalPlant
from stable_sphere_candidates import candidates, RidgePatch


def train(env, seed, d, bank, out):
    current=None;xs=[];ys=[];spent=0
    for stage,budget in enumerate([5120,20480,81920]):
        data,cost=c.collect_pool(env,current,16,c.gen(bank,env.n,seed,stage));spent+=cost
        actions=candidates(env.baseline(data['x']),d)
        pairs,selected=c.select(data,actions,budget-spent,c.gen(bank+100,env.n,seed,stage))
        values,cost=c.query(env,data,actions,pairs,selected,c.gen(bank+200,env.n,seed,stage))
        spent+=cost;assert spent==budget
        si,pi=selected.T
        chosen=np.stack([actions[si,0],actions[si,pairs[pi,0]],actions[si,pairs[pi,1]]],1)
        feature=c.features(data['x'][si],data['remaining'][si],chosen,d)
        x=feature[:,2]-feature[:,1];y=np.diff(values,axis=1).ravel()/d
        xs.append(x);ys.append(y);xa=np.concatenate(xs);ya=np.concatenate(ys)
        coef=np.linalg.solve(xa.T@xa+.1*np.eye(xa.shape[1]),xa.T@ya)
        current=RidgePatch(coef,d,env.m)
        np.savez_compressed(out/f'train_b{budget}.npz',**data,actions=actions,pairs=pairs,
                            selected=selected,returns=values,fit_x=x,fit_target=y,coef=coef)
    return current
