# Control certification code and data

Tools for paired-return verification, finite-power sensitivity bounds,
replicated-transition calibration and reproducible simulation comparisons.

## Install and run

Extract `control_certification.zip`, then enter its `control_certification`
directory. Use Python 3.11 and a CPU:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py quick
python reproduce.py current
```

Installation needs access to the Python package index (or local wheels).
**Once dependencies are installed, all these commands run without network
access.** The approximately 291 MB recorded-data archive is already included;
do not run the legacy download command. No GPU, PyTorch or MuJoCo is needed.
Gymnasium is needed by the original Pendulum audit imports.

Each command creates a new directory under `runs/`, with logs and a
`run_report.json`. `--work-dir /path/to/new-folder` selects a different new
directory. Existing output directories are never overwritten. The root runner
writes regenerated evidence and figures into that run directory and preserves
the packaged input files and their checksums.

| Command | Checks / outputs |
|---|---|
| `verify` | Release file hashes, 40 frozen inputs and 984 recorded data members |
| `quick` | 16 new tests, 3 legacy runner tests, 12 original verifier/coupling tests, stress checks, fresh example |
| `current` | All active comparisons: prefix calibration, noise/displacement, support/verifier comparisons, secondary costs, regional regulation and finite-power study |
| `all` | Complete original replay, including older diagnostic/reference groups |

`current` replays retained data and runs targeted simulator checks. It does not
retrain all original policies or regenerate all simulation banks. To run fresh
original studies, follow
`reproducibility/crn_policy/docs/REPRODUCIBILITY.md` and use its isolated
`prepare --without-data` workflow. Do not run original simulation scripts
directly inside `frozen/`.

## Experiment source map

[EXPERIMENTS.md](EXPERIMENTS.md) maps each active comparison to its simulator, learner, raw records and audit. It includes isolated-workspace commands for fresh simulation and identifies reused policies and aggregate-only historical records. `current` selects these comparisons; `all` also runs older diagnostic/reference analyses.

## New implementation

`certification/finite_power.py` exposes:

| Function | Meaning |
|---|---|
| `calibrate_matrix` | Replicated independent baseline transitions; unknown isotropic noise scale, or unknown full covariance with `noise_model='full'` |
| `block_certificate` | Search all eligible lengths up to a fixed cap; uniform state bound including matrix uncertainty and nonlinear residual |
| `return_support` | Convert the state bound to a discounted paired-return support; general reward or pointwise reward identity |
| `fixed_pair_test` | Fixed-sample Hoeffding adoption rule on fresh paired returns; rejects observations outside the asserted support |
| `perturbation_majorant` | Deterministic perturbation analysis constants for a specified true matrix |
| `calibration_budget` | Explicit sufficient replicated-reset count in the explicit Gaussian-reset allocation |
| `validation_budget` | Fixed pair count given externally justified support/gain coefficients |

Example integration, with application-supplied arrays and constants:

```python
from certification.finite_power import (
    calibrate_matrix, block_certificate, return_support, fixed_pair_test,
)

# Freeze the policy and allocate N_v before seeing validation observations.
fit = calibrate_matrix(X, Y1, Y2, residual_lip=ell,
                       failure_probability=delta / 4)
certificate = block_certificate(fit.matrix, fit.error_radius, ell,
                                cap=K, action_gain=L_a)
if certificate is None:
    adopted = False                 # Retain the baseline.
else:
    b = return_support(certificate, d, H, gamma,
                       reward_state_lip=L_x, reward_action_lip=L_u,
                       baseline_lip=L_B)
    decision = fixed_pair_test(fresh_paired_differences, b, alpha,
                               expected_pairs=N_v)
    adopted = decision['adopted']
```

`X`, `Y1`, `Y2` have shape `(N_c,n)`. Calibration replicates have independent
noise; the within-trajectory-pair noises used for return validation are shared.
Validation pairs are independent of one another, calibration and policy
selection. No API can establish these sampling conditions from supplied arrays.

For known forcing image `G_0`, use `actuator=G_0` instead of `action_gain` in
`block_certificate`, where the actual forcing is `G_0 v`, `||v||<=d`. Its scale
must already be included in `G_0`. The implementation minimizes the actuator
numerator divided by the robust margin over **all** feasible block lengths.
There is only one matrix-confidence event, with no extra error allocation
across lengths. Multiple separate adoption decisions still need their own
testing-level allocation.

For a pointwise reward identity, pass
`equal_state_reward_lip=L_rB` to `return_support`. This asserts a pointwise
identity; marginal distribution equality is insufficient. For infinite
discounted support use `horizon=None` and `gamma<1`. A finite simulation still
has its actual horizon and transition cost.

## Fresh analytic example

```sh
python -m examples.nonnormal_certificate
```

The example uses
`M=[[0.5,1],[0,0.5]]`, `h(x)=0.001*tanh(x)`, additive Gaussian noise with
standard deviation `0.005`, baseline action zero, constant candidate action
`d`, initial state zero and reward `tanh(x_1)`. The action affects the second
coordinate. Calibration and validation have separate random streams. The
estimator is not supplied the true matrix or noise scale.

At the default seed `20261015`, `d=0.05`, horizon `3`, discount `0.95`:

- The matrix has norm `1.2071`; the fitted one-step bound exceeds one.
- The sufficient allocation is `173` replicated resets (`346` transitions).
- A block of length `4` passes; the calibrated return support is about `1.2085`.
- The conservative sufficient fixed-sample allocation is `317182` validation pairs;
  the mean paired gain is about `0.04508`, and the update is adopted.
- The complete count is `346 + 2*3*317182 = 1903438` transitions.

This intentionally uses the explicit sufficient allocation, rather than
tuning a count to observed success. It shows an executable certificate and
the cost accounting; it does not show that the sufficient constants are
practically optimal. The analysis budget uses justified simulator parameters;
the estimator itself remains unaware of the true matrix and noise scale.
Alternative `--displacement`, `--horizon`, `--seed`, `--output` arguments are
available. The analytic gain guarantee requires `0<d<=0.1` and `H>=3`.

The separate stress audit checks 4,862 noncommuting perturbation inequalities,
80,000 residual-forced trajectory steps, 1,000 Gaussian calibration replications
and the analytic time-two gain identity. These numerical checks do not replace
the analytic arguments. The implementation uses double precision and rejects
near-zero margins; it is not a formally verified interval-arithmetic tool.

## Additional finite-power study and model-based reference

```sh
# Replay the included final banks into a NEW directory:
python -m examples.finite_power_study --replay-from evidence/finite_power_study --output-dir runs/block-replay
python audit_finite_power_study.py
# Regenerate from the declared final seed into another NEW directory:
python -m examples.finite_power_study --output-dir runs/block-fresh
```

The final protocol has master seed `20261017`, 24 independent reset/validation
replications, actions `0.05, 0.025, 0.0125, 0, -0.05`, horizon 3 and discount
0.95. These are replications of one analytic nonnormal plant, not 24 different
plants. Calibration uses 173 replicated resets (346 transitions). Matrix
confidence failure is 0.0125, calibrated return-test level 0.0375; ordinary
testing uses 0.05. The example-specific model reference allocates 0.0375 to
the noise-scale confidence bound. Every procedure thus has a per-decision
false-adoption bound of 0.05. No family-wise claim is made across decisions.

Methods share the same ordered fresh return bank within each replication.
Actions share validation noises, so the displacement comparisons are dependent.
Calibration and validation streams are independent, as are replications.
The ordinary marginal half-width is `2*(gamma+gamma^2)` because the known
initial reward is zero. Supported pair caps are 2048; ordinary testing may
use 2105 pairs after converting its calibration saving. Four transitions of
rounding remain unused by that ordinary arm. All total-cost records charge
the full calibration cost to each informed method separately. Costs count
the return prefix consumed until each decision. Full banks are generated
for reproducible paired comparisons; their generation cost is not added
to each method's stopping cost, and wall-clock savings are not measured.

All 24 one-step uniform gates fail and all 24 block certificates pass at
length four. For the three positive actions, all methods adopt in every
replication. Mean total transitions are:

| d | Ordinary | Finite horizon | Finite power | Robust model |
|---|---:|---:|---:|---:|
| 0.05 | 1824 | 442 | 1018 | 346 |
| 0.025 | 3648 | 442 | 1018 | 346 |
| 0.0125 | 7200 | 442 | 1018 | 346 |

Finite-power testing uses 112 pairs, finite-horizon testing 16. This is the
fixed eight-stake sequential mixture, **not** the conservative fixed-sample
allocation of 317182 pairs in the separate default demonstration.
The finite-horizon envelope is tighter at this short horizon. Uniformity in
time does not make the block bound numerically smallest at every horizon.
Zero and negative action controls have no adoptions; these diagnostics are
not empirical proof of a general coverage guarantee.

`certification/model_reference.py` derives an expected-gain interval using
the same matrix-confidence ball and an exact upper confidence bound for
unknown Gaussian noise scale. It needs no return validation here, and is
cheapest. The implementation bounds both matrix error and noise-scale uncertainty. The known input location, constant
candidate and horizon-three reward structure make this reference tractable;
it is not a general replacement for arbitrary policies and horizons.
The comparison makes explicit that the ordinary-range lower bound does not
apply to procedures allowed to use a model-derived gain interval.

`evidence/finite_power_study/` contains the final protocol, raw reset arrays,
return banks, all 480 decision records and summaries. Development records
from seeds 20261015 and 20261016 are retained separately in
`evidence/finite_power_development/`. Development tightened the ordinary
range to account for the known zero initial reward and corrected a metadata
field name. The final seed is disjoint and was fixed before drawing its
banks; all 24 final replications are retained. The audit independently
checks 252600 return values against the analytic time-two identity, the
decision cost accounting, and every numerical cell in the recorded cost summary.

The original ridge learner details remain reproducible in the frozen code:
ridge penalty 0.1; features `phi=(1,x,v*x,v,v^2)` and
`z=[phi tensor (u/d), ||u||^2/d, v*||u||^2/d]`; rotation fractions
0.25, 0.5, 1; maximum angle `min(0.8,2*asin(min(1,d/(2*||B(x)||))))`.
Both signs of coordinate tangent directions are queried and the unchanged
action is retained when every score is nonpositive.

## Assumptions and interpretation

The global residual/action/reward constants and globally enforced displacement
are required inputs. Reset data do not establish global Lipschitz regularity.
The assumed model is `F_B(x)=M x+h(x)`, `h(0)=0`, with a known bound on `Lip(h)`.
Unknown full covariance is allowed for calibration; the explicit sufficient
reset allocation is for isotropic Gaussian resets and noise.

The fixed-budget constant-in-`d` conclusion requires a fixed positive robust margin
and gain lower bound `mu_d>=m*d`, `m>0`. `validation_budget` does not estimate
that gain bound. Do not choose a count or support from the same validation
returns, stop early with the fixed-sample test, silently clip observed returns,
or interpret a calibration failure as permission to adopt. False adoption adds
the calibration failure allowance to the conditional testing level. The frozen
experiments' sequential betting verifiers remain in the original code.

## Contents and rebuilding

- `certification/`, `examples/`, `tests/`: implementation, demonstrations and tests.
- `reproducibility/crn_policy/`: byte-preserved original experiment sources,
  summaries, data archive, checksums, protocols and dependency notices.
- `evidence/`: reset data, return banks, numerical summaries and diagnostics.
- `CODE_MANIFEST.json`: byte sizes and SHA-256 hashes of package inputs.
- `package_code.py`: builds `control_certification.zip` and its checksum.

For a Git checkout, run `python download_recorded_data.py` once to retrieve and
verify the original recorded-data archive before `python reproduce.py verify`.
The self-contained ZIP already includes the data. After changing package inputs,
run `python package_code.py` to refresh the manifest and ZIP.

Licensing status is recorded in `LICENSE_NOTICE.md`.

## Additional recorded experiments

`supplementary_records/` includes 16 raw secondary mechanical metric banks,
their six-cost summaries, eight raw bounded-noise regulation studies with
an independent support-and-stopping audit, original source snapshots,
verifier and residual comparisons, and historical aggregate/protocol records. No article
sources or publication metadata are included. Historical aggregate-only
studies do not include all original raw banks.

```sh
python supplementary_records/audit.py
python supplementary_records/audit.py --recorded-data reproducibility/crn_policy/data/crn_policy_experiment_data.zip
python supplementary_records/pendulum_audit.py
```

The audits recompute mechanical metrics, check invariant-region inequalities
and replay both verifiers on retained observations. No new training runs or
experimental observations are generated by these audits. See the additional
records' README for scope and limitations.
