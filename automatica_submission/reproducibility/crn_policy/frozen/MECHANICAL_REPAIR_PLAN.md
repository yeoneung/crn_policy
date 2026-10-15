# Structured mechanical extension

2026-09-27 UTC. This extension follows the completed random-matrix studies.
It asks whether the same certificate connection survives in a synthetic,
mechanically interpretable underactuated spring-mass chain. It is not an
unmodified external benchmark or a robot experiment. No mechanical final
plant has been run when this plan is written.

There are p=4 or 8 positions and as many velocities (8/16 states), with
p/2 force actuators on alternating masses (2/4 actions). A nominal discrete
LQR is computed from source dynamics and quadratic weights diag(2,.3), R=.5I.
The target reduces damping by 15%, rotates actuator coordinates by spectral
angle .35 and applies efficiencies in [.8,1]. Euler interval .05, horizon40,
discount exp(-.01). Additional restoring force is -.02*tanh(q); its discrete
global Lipschitz allowance .001 is known. Additive Gaussian covariance is
unknown to calibration. Source LQR preparation uses the known nominal model;
target learning observes scalar returns, states and baseline actions only.

True reward is -.05*[1-exp(-x'Qx)+.02*(1-exp(-||a||^2))], Q=diag(2,.3).
Initial state is N(0,.09I). Actions preserve the baseline norm at each state
and obey a pointwise displacement limit. Public action and reward bounds
are analytic; target matrices and noise covariance are hidden from learning
and calibration. The target simulator does not represent arbitrary learned
neural dynamics. Common-noise rollout access and independent reset calibration
are assumptions.

Development dimensions 8/16 and seeds100/101 compare only displacements .05
and .15. Ridge .1, budgets5120/20480/81920 and the existing candidate/query
design stay fixed. Selection is the highest mean relative raw gain across
the four development instances, at 81920; no favorable-family selection.
All development outcomes remain, including failure. No final outcomes select
the displacement. Use the selected setting for all final plants regardless
of development sign.

Final dimensions8/16 and seeds600--607 give16 new plants. Train once at81920.
Collect8192 pairs of independent baseline transitions at fixed independent
N(0,.25I) reset states; charge16384 transitions. Use the original rowwise-F
matrix confidence and finite-horizon matrix-power envelope, not the later
exploratory spectral alternative. Calibration alpha=.025/16. Positive test
alpha=.025/(16*2); futility is a separate tail. Both certificate arms receive
the same frozen learned policy and ordered paired-return bank.

Primary total allowance139264: the calibrated arm gets512 return pairs;
ordinary range gets716 pairs, with64 transitions unused. Each pair costs80.
Checks every16 pairs, with a final partial block. Research extension4096
pairs is descriptive. Independent final evaluation4096 paired episodes.
Primary comparison is calibrated-minus-ordinary deployed gain across16
plant units, with two-sided t interval/test and bootstrap stratified by
dimension. No comparison against a new learner or state-safety claim is made.

Training bank869100, calibration869400, certification869600, final869700,
always combined with dimension and plant seed. Save checkpoints, calibration
and source hashes before return testing. Never replace failed seeds or
change the reward/budget after final outcomes. Report all16 plants and both
dimensions even if calibration or certification is unsuccessful.
