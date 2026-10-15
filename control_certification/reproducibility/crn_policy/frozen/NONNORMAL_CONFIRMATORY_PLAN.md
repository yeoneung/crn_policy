# Frozen finite-horizon extension beyond one-step contraction

Development used only dimensions 4,8,16 and seeds 100--101 with a single
declared configuration. All six development plants had a baseline linear
operator norm above one; the scalar contraction certificate was unavailable,
but a matrix-power trajectory bound remained useful. The final study now
uses fresh seeds 400--407 in each dimension (24 plants). No settings are
selected using those outcomes.

The baseline linear part is an orthogonal rotation of .9I+.13J, with J the
upper shift matrix. The plant A is chosen so that A+BK equals this matrix;
the same bounded nonlinear drift .005 tanh(Cx) remains. Additive Gaussian
noise has unknown full covariance with eigenvalues proportional to the
geometric sequence .5--2 and random eigenvectors. The learner receives
neither the matrix, nonlinear drift nor noise covariance. It has the given
linear baseline, known residual Lipschitz bound .005, public action gain .1
and reward sensitivity bounds. No one-step contraction is assumed or
inferred from an empirical maximum.

Train the same uniform scalar-return ridge .1 repair with d=.05 and
cumulative budgets 5,120,20,480,81,920; only the final policy is tested.
Independent dynamics calibration uses 8,192 reset states and two baseline
next-state replicates each, costing 16,384 transitions. A rowwise Gaussian
F pivot and a union bound give a matrix spectral-norm error allowance with
unknown correlated noise. The nonlinear regression-bias allowance remains.
The fixed nominal-matrix powers and that residual allowance define a
Volterra trajectory envelope; it is valid at finite horizon whether or not
the one-step norm is below one. Intersect its normalized-reward return
support with the ordinary range.

The primary procedure uses at most 512 paired validation episodes, plus
training and calibration: 81,920+16,384+40,960=139,264 allowed target
transitions. The ordinary-range comparator receives that identical total
budget, allocating no dynamics calibration and at most 716 pairs (64
transitions remain unused). Both apply the same betting mixture to the
same ordered independent return bank. The calibrated proposal is adopted
only upon a positive crossing, with baseline fallback otherwise.

Allocate .025/24 to each plant's calibration event and .025/(24*2) to each
positive return test. This bounds false adoption anywhere in the planned
family by .05. The separate negative-tail futility test does not increase
that bound; no joint two-label .05 claim is made. Checks occur every 16
pairs with a shorter final block when needed. The complete independent
research bank contains 4,096 pairs so that stopping/censoring can also be
reported; this tail cannot change a primary-prefix decision. Each pair
costs 80 transitions. A further independent 4,096-pair bank evaluates final
return. Record full research simulation cost and prefix deployment cost.

Primary endpoint: plant-level paired difference of deployed return under
matrix-power support versus equal-total ordinary range. Report mean, SE,
two-sided paired t-test and a 20,000-resample bootstrap interval stratified
by dimension. This is a new robustness study, separate from the original
primary hypothesis. Do not interpret it as general optimizer superiority.
Secondary endpoints include used transition costs, 4,096-pair stopping
curves/censoring, raw gains, matrix-coverage audits, all scalar Lipschitz
upper bounds and the magnitude of each residual term. Return-bound
violations, if encountered, cause recorded fallback rather than clipping.

Training banks 857100/200/300, dynamics calibration 857400, certification
857600, final evaluation 857700, bootstrap 857900, each indexed by dimension
and seed. Freeze all source hashes and this plan before final target data.
Freeze learned coefficients and calibration before reading return banks.
Retain every raw measurement, policy, audit and negative outcome.
