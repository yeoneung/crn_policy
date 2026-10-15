"""A synthetic underactuated spring-mass chain with a frozen nominal LQR.

The chain has p positions and p velocities, with p/2 force actuators. It is
an interpretable structured simulator, not a standard robotics benchmark.
All queries retain the scalar-return interface of the controlled study.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_discrete_are
import coupled_support_core as c
from finite_horizon_envelope import power_envelope


class MechanicalPlant:
    def __init__(self, n, seed):
        assert n in [8, 16]
        self.n = n
        p = n // 2
        self.m = p // 2
        self.seed = seed
        random = c.gen(868001, n, seed)
        incidence = np.eye(p) - np.eye(p, k=1)
        springs = np.diag(random.uniform(.8, 1.2, p)) + .25 * incidence.T @ incidence
        damping = np.diag(random.uniform(.35, .45, p))
        h = c.DT
        source_a = np.block([[np.eye(p), h * np.eye(p)], [-h * springs, np.eye(p) - h * damping]])
        self.A = np.block([[np.eye(p), h * np.eye(p)], [-h * springs, np.eye(p) - .85*h*damping]])
        allocation = np.eye(p)[:, ::2]
        self.B0 = np.vstack([np.zeros((p, self.m)), h * allocation])
        q_lqr = np.diag([2.] * p + [.3] * p)
        r_lqr = .5 * np.eye(self.m)
        riccati = solve_discrete_are(source_a, self.B0, q_lqr, r_lqr)
        self.K = -np.linalg.solve(r_lqr + self.B0.T @ riccati @ self.B0,
                                  self.B0.T @ riccati @ source_a)
        skew = random.normal(size=(self.m, self.m));skew -= skew.T
        skew *= .35 / max(np.linalg.norm(skew, 2), 1e-12)
        self.B = self.B0 @ expm(skew) @ np.diag(random.uniform(.8, 1., self.m))
        self.linear_baseline = self.A + self.B @ self.K
        self.Q = q_lqr
        noise_rotation = np.linalg.qr(random.normal(size=(n, n)))[0]
        self.noise_root = .015 * noise_rotation @ np.diag(np.sqrt(np.geomspace(.5, 2., n)))
        self.nonlinear_lip = .02 * h
        self.constants = dict(action_gain=h, baseline_lipschitz=float(np.linalg.norm(self.K, 2)),
                              reward_state_lipschitz=h*np.sqrt(2*np.linalg.norm(q_lqr, 2)/np.e),
                              reward_action_lipschitz=h*.02*np.sqrt(2/np.e),
                              return_range=h*1.02*sum(c.GAMMA**t for t in range(c.H)))
    def baseline(self, x):
        return x @ self.K.T
    def initial(self, count, random):
        return random.normal(size=(count, self.n)) * .3
    def step(self, x, a, noise):
        state = np.einsum('bi,ij,bj->b', x, self.Q, x)
        reward = -c.DT * (-np.expm1(-state) + .02*(-np.expm1(-np.sum(a*a, axis=1))))
        nonlinear = np.zeros_like(x)
        nonlinear[:, self.n//2:] = -self.nonlinear_lip*np.tanh(x[:, :self.n//2])
        y = x @ self.A.T + a @ self.B.T + nonlinear + noise @ self.noise_root.T
        return y, reward
    def support(self, d):
        # Hidden matrix enters evaluation diagnostics only, never learned fits.
        distance, _ = power_envelope(self.linear_baseline, self.nonlinear_lip,
                                     self.constants['action_gain'], d, c.H)
        return self.constants['return_range'], distance


def train(env, seed, d, bank, out):
    current = None;xs = [];ys = [];spent = 0
    for stage, budget in enumerate([5120, 20480, 81920]):
        data, cost = c.collect_pool(env, current, 16, c.gen(bank, env.n, seed, stage));spent += cost
        actions = c.candidates(env.baseline(data['x']), d)
        pairs, selected = c.select(data, actions, budget-spent, c.gen(bank+100, env.n, seed, stage))
        values, cost = c.query(env, data, actions, pairs, selected, c.gen(bank+200, env.n, seed, stage))
        spent += cost;assert spent == budget
        si, pi = selected.T
        chosen = np.stack([actions[si, 0], actions[si, pairs[pi, 0]], actions[si, pairs[pi, 1]]], 1)
        feature = c.features(data['x'][si], data['remaining'][si], chosen, d)
        x = feature[:, 2]-feature[:, 1];y = np.diff(values, axis=1).ravel()/d
        xs.append(x);ys.append(y);xa = np.concatenate(xs);ya = np.concatenate(ys)
        coef = np.linalg.solve(xa.T@xa+.1*np.eye(xa.shape[1]), xa.T@ya)
        current = c.RidgePatch(coef, d, env.m)
        np.savez_compressed(out/f'train_b{budget}.npz', **data, actions=actions, pairs=pairs,
                            selected=selected, returns=values, fit_x=x, fit_target=y, coef=coef)
    return current
