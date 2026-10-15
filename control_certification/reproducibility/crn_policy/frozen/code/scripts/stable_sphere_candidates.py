"""Coordinate tangent bank without subtraction of nearly parallel vectors."""
import numpy as np
import coupled_support_core as c


def candidates(base, d):
    base = np.asarray(base, dtype=float)
    eta, theta = c.max_angle(base, d)
    u = c.unit(base)
    out = [base]
    for fraction in [.25, .5, 1.]:
        angle = fraction*theta
        for j in range(base.shape[1]):
            remainder = u.copy();remainder[:, j] = 0
            length = np.linalg.norm(remainder, axis=1)
            e = np.divide(-u[:, j, None]*remainder, length[:, None],
                          out=np.zeros_like(u), where=length[:, None]>1e-10)
            e[:, j] = length
            for sign in [1, -1]:
                a = eta[:, None]*(np.cos(angle)[:, None]*u + sign*np.sin(angle)[:, None]*e)
                out.append(np.where((length>1e-10)[:, None], a, base))
    result = np.stack(out, axis=1)
    assert np.max(np.linalg.norm(result-base[:, None], axis=-1))<=d+1e-10
    return result


class RidgePatch(c.RidgePatch):
    def action(self, x, remaining, baseline):
        base = baseline(x);acts = candidates(base, self.d);delta = acts-base[:, None]
        field = c.context(x, remaining)@self.coef[:-2].reshape(-1, self.m)
        linear = np.einsum('bka,ba->bk', delta/self.d, field)
        curve = self.coef[-2]+self.coef[-1]*np.asarray(remaining)/c.H
        score = linear+curve[:, None]*np.sum(delta*delta, axis=-1)/self.d
        ix = score.argmax(1)
        ix = np.where(score[np.arange(len(x)), ix]>0, ix, 0)
        return acts[np.arange(len(x)), ix]
