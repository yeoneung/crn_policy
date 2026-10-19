# Secondary physical-cost evaluation of frozen mechanical repairs

Recorded 2026-09-27 19:59 UTC, after the primary bounded-return results but
before these new episode banks or metric outcomes are generated.

This is a descriptive secondary evaluation of the 16 existing input-image
confirmation plants (8/16 states, seeds 700--707). It is not 16 new plants,
does not change their policies or primary adoption decisions, and does not
turn the bounded-return certificate into a guarantee for unbounded costs.

For each plant, load the already frozen 81,920-transition ridge repair with
displacement .15. Compare that exact repair with its nominal LQR baseline on
4,096 new paired episodes, sharing initial states and all Gaussian noise.
Use bank [872700,n,seed], batch 256, the unchanged H=40, h=.05 and discount
exp(-.01). No metric is used for hyperparameters, action selection, or testing.

Retain every episode value for all six metrics:

1. The original bounded objective, as negative return (cost).
2. h*sum gamma^t q_t' q_t / p: mean position energy per coordinate.
3. h*sum gamma^t v_t' v_t / p: mean velocity energy per coordinate.
4. h*sum gamma^t x_t' Q x_t: uncapped quadratic state cost.
5. h*sum gamma^t .5*||a_t||^2: uncapped effort cost.
6. The sum of 4 and 5: finite-horizon nominal LQR quadratic cost.

Here p=n/2, Q=diag(2I,.3I), and actions are the actually applied actions in
this synthetic plant. Improvement is baseline cost minus repair cost. Report
all metrics, their plant-level means/SE and relative cost changes, including
negative cases. These across-plant summaries are descriptive; no new primary
hypothesis test or finite-sample safety claim is assigned to these unbounded
Gaussian-state metrics. Counts/uncertainty use 16 plant units, not episodes.

Save hashes of execution sources and all loaded original policy arrays before
simulating, retain raw per-episode values and complete outcomes, charge all
16*4096*2*40=5,242,880 transitions as additional research evaluation, and
preserve any incomplete attempt rather than replacing it silently.
