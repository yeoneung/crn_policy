"""Independent arithmetic/legacy-evaluator audit; never modifies frozen runs."""
import json
import math
from datetime import datetime, timezone
import numpy as np
from threadpoolctl import threadpool_limits
from coupling_sensitivity import ROOT, OUT, DS, D0, FrozenSphereFamily, sha
import coupled_support_core as c
from return_certificate import log_wealth


def independent_stop(values, row):
    """Legacy mixture sums or a separately written scalar PM-EB recursion."""
    bound, alpha, cap = row['bound'], row['alpha'], row['cap']
    assert np.max(abs(values[:cap])) <= bound
    threshold = math.log(1/alpha)
    sums = [.5, .5]; squares = [.25, .25]; predictors = [.5, .5]
    variances = [.25, .25]; wealth = [0., 0.]
    for t in range(1, cap+1):
        if row['method'] == 'predictable_empirical_bernstein':
            for j, sign in enumerate([1, -1]):
                x = (sign*float(values[t-1])/bound+1)/2
                lam = min(.75, math.sqrt(2*math.log(2/alpha)/(variances[j]*t*math.log1p(t))))
                wealth[j] += lam*(x-.5) - (x-predictors[j])**2*(-math.log1p(-lam)-lam)
                sums[j] += x
                updated_mean = sums[j]/(t+1)
                squares[j] += (x-updated_mean)**2
                variances[j] = squares[j]/(t+1)
                predictors[j] = updated_mean
        if t % 16 and t != cap:
            continue
        if row['method'] == 'fixed_mixture':
            wealth = [log_wealth(sign*values[:t], bound) for sign in [1,-1]]
        if wealth[0] >= threshold: return 'adopt', t
        if wealth[1] >= threshold: return 'reject', t
    return 'inconclusive', cap


class OneRadius:
    def __init__(self, family, radius): self.family, self.radius = family, radius
    def action(self, x, remaining, baseline):
        return self.family.action(x, remaining, baseline, np.full(len(x), self.radius))[0]


def audit():
    threadpool_limits(limits=1)
    hashes = 0
    for name in ['source_hashes.json', 'input_hashes.json']:
        for relative, expected in json.loads((OUT/name).read_text()).items():
            assert sha(ROOT/relative) == expected, relative
            hashes += 1
    cells = 0; rollout_checks = []
    for p in sorted(OUT.glob('dim*_seed*')):
        completed = json.loads((p/'COMPLETE.json').read_text())
        for name, expected in completed['outputs'].items():
            assert sha(p/name) == expected, (p,name)
            hashes += 1
        with np.load(p/'certification.npz') as z: returns = z['returns'].copy()
        for row in json.loads((p/'stopping.json').read_text()):
            index = 6 if row['coupling']=='independent' else int(np.flatnonzero(DS==row['d'])[0])+1
            assert independent_stop(returns[index]-returns[0],row) == (row['outcome'],row['pairs']), (p,row)
            cells += 1
        old = ROOT/'code/results/coupled_support_confirmatory'/p.name
        names = json.loads((old/'pre_evaluation.json').read_text())['names']
        with np.load(old/'certification.npz') as z:
            values = z['returns'][names.index('Ridge_d0.05_b81920')]-z['returns'][0]
        for row in json.loads((p/'ell_replay.json').read_text()):
            assert independent_stop(values,row) == (row['outcome'],row['pairs']), (p,row)
            cells += 1
        if completed['seed'] == 200:
            env = c.StablePlant(completed['n'],completed['seed'])
            with np.load(OUT/'frozen_policies'/p.name/'Ridge_d0.05_b81920.npz') as z:
                family = FrozenSphereFamily(z['coef'],env.n)
            legacy = c.evaluate(env,[None]+[OneRadius(family,d) for d in DS],128,c.gen(960100,env.n,env.seed),batch=128)
            np.testing.assert_allclose(legacy['returns'],returns[:6,:128],rtol=1e-12,atol=1e-12)
            rollout_checks.append(dict(plant=p.name,policies=6,episodes=128,max_absolute_difference=float(abs(legacy['returns']-returns[:6,:128]).max())))
        print(p.name,'checked',flush=True)
    assert cells == 1536
    result = dict(utc=datetime.now(timezone.utc).isoformat(),all_passed=True,
                  hashes_verified=hashes,independent_stopping_cell_checks=cells,
                  checks='All fixed-mixture cells checked with legacy direct-prefix log wealth; all PM-EB cells checked with independent scalar recursion',
                  legacy_rollout_checks=rollout_checks,
                  additional_audit_simulator_transitions=3*6*128*c.H,
                  audit_source_sha256=sha(__import__('pathlib').Path(__file__)))
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': audit()
