# Frozen final study: certification cost of a bounded policy repair

Development is complete on dimensions 4, 8, 16 and seeds 100--101. The
54-cell grid in `code/results/coupled_support_tuning/grid.json` selects
ridge 0.1 and ARS step 0.01 with four directions by the declared mean-gain
rule. The final target seeds 200--207 have not been evaluated at this freeze.

## Question and population

On the **same frozen policy and same paired return observations**, does a
justified contraction-based support make reliable deployment affordable
when the ordinary return range does not? This tests a verification benefit,
not superiority of a new learning algorithm. Plants have state and action
dimensions 4, 8, 16 (eight independent seeds each). The target dynamics are
hidden from learners. A public actuator uncertainty set and known baseline
yield analytic contraction and reward Lipschitz constants. They are not
estimated from observed extrema. The source baseline is given, not learned;
its engineering/development cost is outside the target-interaction budget.

## Frozen learners and costs

Uniform scalar-return ridge calibration and a tuned ARS control use the
same target training budgets 5,120, 20,480 and 81,920 transitions. They obey
pointwise baseline displacement limits d=0.05 and d=0.15 and preserve the
baseline action norm. Ridge uses regularization 0.1 and its own current
policy's occupation pool; ARS uses step 0.01, perturbation 0.1, four
directions and four trajectories per sign. Every queried transition,
including collecting anchor states and both branches, is charged.

After all policies are frozen, 4,096 independent paired trajectories are
simulated, using identical exogenous noise for policy and baseline. Two
standard bounded-mean betting tests process the identical ordered stream:

1. Ordinary global return range.
2. Normalization-aware paired support: min(global range,
   sum gamma^t (L_rx + L_ra L_B) D_t), with
   D_t=L_a d (1-kappa^t)/(1-kappa).

The second bound uses equality of action norms to cancel the same-state
action-effort difference. It requires the stated dynamics/reward structure;
it is not a learned critic confidence bound. A more general paired bound
without this cancellation is retained as a deterministic analytic ablation,
without a third adaptive decision rule.

The one-sided false-adoption level is .05/(24*2*3*2*2) across all 576 planned
policy/radius/budget/range cells. A separate negative-mean test at the same
level permits early futility. False adoption has familywise bound .05;
joint incorrect positive/negative labeling has bound .10. Policies are
frozen before testing. Reject/inconclusive means deploy the baseline.
Tests check every 16 pairs and have no fitted betting parameter. All
prefixes of a given test use its one anytime-valid event, not new tests.

## Primary endpoint and secondary analyses

Primary: ridge, d=.05, training budget 81,920, certification cap **512
pairs**. Total allowed deployment cost is 81,920+512*80=122,880 target
transitions. Compare the independent return of the deployed output under
paired support against the output under the ordinary range across 24 plant
units. Report mean paired difference, plant-level standard error, two-sided
paired t-test, and a stratified 20,000-resample percentile bootstrap interval
(eight seeds resampled within each fixed dimension). This is approximate
cross-plant inference, distinct from the finite-sample adoption guarantee.
Also report adoption counts and dimension-specific means; there is only one
primary superiority comparison. No post hoc change of primary radius,
budget or learner is permitted.

Secondary: all learners, dimensions, radii and budgets; certification caps
128, 512, 2,048 and 4,096 pairs; stopping costs and censoring; positive,
negative and inconclusive outcomes; raw proposal return; baseline-relative
gain. No single conditional mean stopping-time ratio should discard cases
that fail to certify. Plot the censored adoption-versus-cost curves.

The complete fixed research bank is generated to compare prefixes. Its full
simulation cost is reported separately from the cost of the predeclared
deployment procedure, which stops at the first positive/negative crossing
or its cap. Shared baseline simulations across research arms never reduce
the charged 80-transition cost per deployed pair. Tail data never affect a
prefix decision. Final return evaluation uses 4,096 fresh episodes for all
13 policies and is unavailable during fitting and certification. These
evaluation trajectories are research cost, not deployment cost.

## Randomness and audit

Training pools, query selection, query noises and ARS use banks 851100,
851200, 851300 and 851500, indexed by dimension/seed/radius/stage as
appropriate. Certification uses 851600, final evaluation 851700, bootstrap
851900. Store every scalar query and ARS update, fitted coefficient,
pre-evaluation hash, all returns, observed path-envelope checks and analytic
constants. Source hashes and a source snapshot are written before any final
run. Partial attempts are retained and require a recorded audit to resume.
Development and final results remain separate, including failures.
