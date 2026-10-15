# Independent calibration extension of the frozen support study

This extension is fixed after development on seeds 100--101, and before
reading any final controlled-support return outcomes. It reuses the same
24 frozen final policies and return banks; it is **not an independent
replication** of those return outcomes. New baseline-system measurements
are independent of policy training and return testing.

Replace the supplied contraction constant by a finite-sample upper bound.
The baseline closed loop is assumed to have form Mx+h(x), with h(0)=0 and
known global Lipschitz constant at most ell=.005. M, h, and the Gaussian
noise variance are not supplied to the estimator. The baseline is given;
the action-gain upper bound .1 and reward Lipschitz bounds remain public
engineering specifications. Additive next-state noise is isotropic,
Gaussian, independent across resets/replicates and independent of the reset
state. This is a declared, restricted calibration model.

At each reset state x, execute the baseline twice with independent noise,
observing y1,y2. Use N=512, 2,048 and 8,192 reset pairs (2N charged target
transitions). Regress the average next state on x. The replicate differences
estimate unknown noise variance without bias from the nonlinear drift.
An exact Gaussian F pivot bounds the random matrix error; the known
Lipschitz residual bounds deterministic regression bias. Add ell to the
linear norm upper bound to bound the entire baseline Lipschitz constant.
If the resulting kappa upper bound is at least one, this study falls back
without claiming certified contraction.

Allocate delta_cal=.025/(24*3) per calibration prefix and
delta_test=.025/(24*3*2*3*2*2) per positive return-test tail. There are 72
calibration events and 1,728 return-test cells (24 plants, 3 calibration
budgets, 2 radii, 3 learning budgets, 2 learners, 2 ranges). The combined
familywise false-adoption bound is .05 by a union bound. The negative
futility label uses its own tail; its error budget is not silently included
in the positive .05 claim. One calibration event serves all policies of
that plant/prefix.

Primary extension: Ridge, displacement .05, training 81,920, calibration
2,048 pairs (4,096 transitions), and at most 512 paired return episodes
(40,960 transitions). Total allowed cost is **126,976 transitions**. The
ordinary-range comparator receives the same total budget: no dynamics
calibration and up to floor((40,960+4,096)/80)=563 return pairs. Report its
unused 16-transition remainder. Both use the identical stored return
stream and the same delta_test. The final independent bank evaluates the
adopt-or-baseline output. Report plant-level paired differences and an
approximate stratified 20,000-resample bootstrap interval; interpret this
as a planned robustness extension of the original primary result, not a
second independent replication. All other cells are secondary.

Use new calibration bank 853100 indexed by dimension and seed. Generate
8,192 reset pairs per plant and process the three fixed prefixes. The full
research measurement cost is recorded separately from the deployment
prefix cost. No return tail or final evaluation chooses a calibration
budget. Store reset states and both observed next states. Retain estimated
noise, Gram conditioning, each error term and the contraction upper bound.
Hidden true matrices are used only in a post-fit coverage audit. The
calibration function accepts only observed x,y1,y2, ell and alpha.
