# Running the experiments

## Recorded-data replay

```sh
python download_data.py
python reproduce.py all
```

The runner checks source and data hashes, creates a new workspace, and runs tests, aggregation and independent audits. Plots and tables are under `runs/<id>/artifacts/`; summaries and audit results are under `runs/<id>/code/results/`. Logs include failures as well as successes.

The original scripts use fixed relative paths. They are copied unchanged into each workspace so their recorded source hashes remain meaningful. Temporary output paths are collected into `artifacts/` at the end. Do not execute legacy scripts inside `frozen/`, where their relative paths could create outputs beside the preserved inputs.

To inspect a data-populated workspace without analysis:

```sh
python reproduce.py prepare --work-dir runs/inspection
```

The specified directory must not exist. Use a new name for each experiment.

## Fresh simulation

Create an empty simulation workspace:

```sh
python reproduce.py prepare --without-data --work-dir runs/new-study
python runs/new-study/code/scripts/coupled_support_confirmatory.py --dimension 4 --seed 200
```

The second command trains and evaluates one stable plant and is considerably more expensive than a smoke test. Omitting the dimension and seed runs the full stable protocol. The calibration extension requires the corresponding completed stable runs:

```sh
python runs/new-study/code/scripts/stability_calibration_confirmatory.py
```

Other fresh-study entry points are:

```sh
python runs/new-study/code/scripts/nonnormal_confirmatory.py
python runs/new-study/code/scripts/actuation_structure_confirmatory.py
python runs/new-study/code/scripts/pendulum_regional_study.py --development
python runs/new-study/code/scripts/pendulum_regional_study.py --final
python runs/new-study/code/scripts/pendulum_nominal_reference.py
python runs/new-study/code/scripts/pendulum_oracle_reference.py
```

The last two require final Pendulum banks; the oracle reference also requires the nominal reference. Original protocols define seeds and ordering. Completed-output checks prevent accidental regeneration over retained observations.

For the displacement sweep, start from a data-populated workspace with the retained `v2_coupling_sensitivity/frozen_policies/` and source/input manifests. The old one-time checkpoint-import branch is not a standalone entry point; supported replay uses the supplied checkpoints directly. The historical diagnostic `review_theory_pendulum.py` supplies transition/Jacobian helpers to the simulator; its old standalone review entry point is not part of this experiment interface.

## Interpretation and numerical reproducibility

- Use Python 3.11 and the pinned dependencies. No GPU is needed for the replay commands.
- Archived inputs must be byte-identical. Numerical results and stopping decisions are compared with their recorded references; new timestamps and generated image metadata need not match old output bytes.
- Audits include small independent simulator checks. The `all` command does not constitute a new full training study.
- Failures, censoring and dependent reuse of data remain visible. Do not count multiple analyses of the same plant/bank as new independent replications.
- Full simulation sources are included, but release validation reruns replay and targeted checks rather than every expensive training run.
