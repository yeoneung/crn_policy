# Active simulation comparisons

This catalog identifies the simulator, learner, recorded observations and
replay checks for each active comparison. All commands start in this directory.

## One command for the recorded comparisons

```sh
# Git checkouts only; the self-contained ZIP already contains this asset:
python download_recorded_data.py
python reproduce.py current
```

The runner checks input hashes, runs the existing 31 implementation/runner
tests, checks the finite-power certificate, replays all comparisons below,
and regenerates the combined diagnostic figure. Each run creates a new
`runs/current-<timestamp>/` with logs, JSON summaries and a run report.
No full benchmark retraining is performed by this command.

Let `S` below denote `reproducibility/crn_policy/frozen/code/scripts/`, and
`D` denote the original `crn_policy_experiment_data.zip` release asset under
`reproducibility/crn_policy/data/`. Source files under `S` retain their
original bytes and hashes. Records inside `D` use the prefix `code/results/`.

| Comparison | Population / reuse | Simulation and learning sources under S | Raw records / checks |
|---|---|---|---|
| Stable support | 24 plants, dimensions 4/8/16, seeds 200–207 | `coupled_support_core.py`, `stable_sphere_candidates.py`, `coupled_support_confirmatory.py` | D: `coupled_support_confirmatory/`; `audit_evidence.py` |
| Noise and displacement | The same 24 frozen directions, new ordered noise banks, five displacement limits | `coupling_sensitivity.py` | D: `v2_coupling_sensitivity/`; original coupling replay and independent audit |
| Calibration prefixes | Same 24 stable policies and returns; nested 512/2048/8192 reset prefixes | `stability_calibration.py`, `stability_calibration_confirmatory.py` | D: `stability_calibration_confirmatory/`; `audit_included_experiments.py` recomputes 72 fits and 144 decisions |
| Residual-bound sensitivity | Same 24 calibration and return banks; factors .5/1/2/5/10 | `coupling_sensitivity.py` | D: `v2_coupling_sensitivity/*/ell_replay.json`; included-comparison audit |
| Mechanical input geometry | 16 plants, state dimensions 8/16, seeds 700–707 | `mechanical_repair_core_v2.py`, `actuation_envelope.py`, `actuation_structure_confirmatory.py` | D: `actuation_structure_confirmatory/`; primary/extended checks in `audit_evidence.py` |
| Secondary mechanical costs | Same 16 policies, separate 4096-pair banks, six cost measures | `supplementary_records/original_scripts/mechanical_physical_metrics.py` | `supplementary_records/data/mechanical_physical_metrics/`; NumPy metric audit |
| Nonnormal dynamics | 24 distinct plants, dimensions 4/8/16, seeds 400–407 | `nonnormal_development.py`, `nonnormal_confirmatory.py`, `finite_horizon_envelope.py` | D: `nonnormal_confirmatory/`; included-comparison and original source checks |
| Verifier comparison | Replays of the 24 stable, 24 nonnormal and 16 mechanical plants | `variance_replay.py`, `return_certificate.py` | Same return banks; included-comparison audit replays fixed mixture and PM-EB |
| Finite-power acquisition | One analytic plant, 24 independent calibration/validation repetitions, five signed actions | `examples/finite_power_study.py`, `certification/finite_power.py`, `certification/model_reference.py` | `evidence/finite_power_study/`; full bank, cost and zero/negative-control audit |
| Regional regulation | Eight upright Pendulum variants, seeds 1000–1007, four support bounds and two verifiers | `pendulum_regional_study.py`, `pendulum_statecap_development.py`, `pendulum_regional_development.py`, `review_theory_pendulum.py` | `supplementary_records/data/pendulum_regional_study/`; 64 decisions, exact-rational inequalities and independent scalar check |

The finite-power repetitions are not 24 different plants. Replaying a
second verifier, varying a bound or changing the displacement of a frozen
direction does not create an independent plant population. For calibration
prefixes, supported adoptions are 16/24, 22/24 and 24/24; acquisition costs
are 1024, 4096 and 16384 transitions. Their total allowances differ.
The small regional gains are evaluated under bounded torque noise and
upright initialisation, distinct from standard Pendulum swing-up.

## Run individual checks

```sh
python audit_evidence.py --output runs/primary.json
python audit_included_experiments.py --output runs/included.json
python supplementary_records/audit.py
python supplementary_records/pendulum_audit.py --output runs/regional.json
python audit_finite_power_study.py --output runs/finite-power.json
python -m examples.finite_power_study --replay-from evidence/finite_power_study --output-dir runs/new-block-replay
```

These commands preserve source inputs. The primary audit only aggregates
recorded certificates; the included-comparison audit additionally refits
calibration prefixes and replays verifiers on raw observations. The regional
audit independently reconstructs a subset of the held-out returns.

## Fresh simulation sources

Prepare an isolated source workspace. Original entry points use fixed relative
paths and must be executed from the prepared copy, never inside `frozen/`.
If a workspace path includes a filesystem alias, use the canonical prepared
path printed by the helper when invoking an original script.

```sh
python prepare_sources.py --mode fresh --output runs/fresh
# One stable plant; omit both options to generate the full stable population:
python runs/fresh/code/scripts/coupled_support_confirmatory.py --dimension 4 --seed 200
# This extension needs the completed stable population:
python runs/fresh/code/scripts/stability_calibration_confirmatory.py
# Independent nonnormal and mechanical populations:
python runs/fresh/code/scripts/nonnormal_confirmatory.py
python runs/fresh/code/scripts/actuation_structure_confirmatory.py
# Regional development followed by the frozen final protocol:
python runs/fresh/code/scripts/pendulum_regional_study.py --development
python runs/fresh/code/scripts/pendulum_regional_study.py --final
```

Each complete population can be substantially more expensive than replay.
The source workspace includes the fixed mechanical development choice and
the secondary-cost script and its protocol. After generating all 16 mechanical
plants, the separate secondary evaluation can be run there:

```sh
python runs/fresh/code/scripts/mechanical_physical_metrics.py
```

To regenerate noise/displacement banks from the retained frozen checkpoints,
prepare their prerequisite records without any completed coupling output cells:

```sh
python prepare_sources.py --mode coupling --output runs/new-coupling
python runs/new-coupling/code/scripts/coupling_sensitivity.py --workers 1
```

This bypasses the original one-time checkpoint-import branch, which referred
to an old external workspace. The preserved protocol and checkpoint hashes
are used directly; the source is not rewritten. To inspect all original
observations in their expected directory layout, use `--mode records` instead.

The finite-power study has a direct new-run interface:

```sh
python -m examples.finite_power_study --output-dir runs/new-block-simulation
```

Never reuse an existing output path for a new attempt. Completed attempts,
development choices and failed/inconclusive outcomes remain identifiable.
The Gaussian calibration source requires independent reset replicates;
within each return pair the initial state and disturbances are shared.

## Additional retained records

The original data asset and source snapshots also contain older diagnostic,
reference and development studies. Historical aggregate-only records under
`supplementary_records/historical/` are retained for provenance; their raw
banks are not all distributed. They are not part of the active comparison
set or the default `current` command. The older `all` command remains available
for the complete original replay, including extra reference studies.
