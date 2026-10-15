# Fresh mechanical confirmation of known input structure

2026-09-27 UTC. This follow-up follows the amended 16-plant mechanical study
and an explicitly exploratory replay on those plants. The exploratory replay
found that retaining the known actuator image roughly halved the return
support. It is not counted as confirmation. The following 16 new plants have
not been run or inspected when this plan is frozen.

Use exactly the mechanical model, nominal LQR source, reward, horizon,
candidate class, ridge .1 and displacement .15 of MECHANICAL_REPAIR_PLAN.md,
with the numerically stable tangent formula in the documented amendment.
Dimensions8/16, source/plant seeds700--707, all retained regardless of sign.
No further development grid or choice of budget/displacement is allowed.

Additional public structure: the target input matrix is G0*T, where G0=B0
is the known nominal force-allocation matrix and ||T||_2<=1. The target
actuator rotation and efficiencies remain unknown. This is justified by
the declared physical input uncertainty set, not fitted to return outcomes.
The source baseline already uses the nominal model. No extra target
observations are charged or supplied beyond the same matrix calibration.

For reference-matrix estimate Mhat with error epsilon and nonlinear limit
ell, let p_j=||Mhat^j|| and q_j=||Mhat^j G0||. The input-aware envelope is

    E_0=0,
    E_t=d sum_{j<t} q_j + (epsilon+ell) sum_{s<t} p_{t-1-s} E_s.

It follows from variation of constants and ||T delta_a||<=d. The original
matrix envelope instead uses d*||G0||*sum p_j. Both use the identical
rowwise-F matrix fit on8192 independent reset pairs, known ell=.001 and
calibration alpha=.025/16. The ordinary comparator receives no calibration.

Train once per plant at81920 transitions. Primary total allowance139264:
input-aware and original-matrix arms receive512 paired return episodes after
16384 calibration transitions; ordinary range receives716 pairs, leaving64
unused transitions. All three use the same frozen policy and ordered return
bank, positive alpha=.025/(16*3), separate same-sized futility, checks every16
pairs with a final partial block. Research cap4096 is separately descriptive.
All three receive the same independent4096-pair final evaluation bank.

Primary comparisons: input-aware minus original-matrix deployed gain and
input-aware minus ordinary-range deployed gain. Use16 plant units, two-sided
t intervals/tests with Holm correction across these two comparisons; also
report dimension-stratified bootstrap intervals. A joint success requires
positive means and both Holm p<.05. Report raw gains, all adoption counts,
actual stopping costs, support ratios and every plant. No general learner
superiority or real-robot safety claim is made.

Training bank870100, reset calibration870400, certification870600, final870700,
each combined with dimension and seed. Freeze source/plan hashes first and
all learned/calibration arrays before certification. Preserve all runs and
any failure. Do not alter the earlier mechanical primary result (2/16 vs0/16).
