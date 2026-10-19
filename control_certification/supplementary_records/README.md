# Additional recorded experiments

This archive contains retained synthetic observations and aggregate records.
It contains no article sources or publication metadata.

- `data/mechanical_physical_metrics/`: 16 independent secondary evaluation banks for existing frozen mechanical repairs, 4,096 coupled pairs per bank and six costs per trajectory. Positive cost reductions mean lower cost under the repair.
- `summaries/mechanical_physical_metrics.json`: per-plant values and aggregates for bounded cost, position energy, velocity energy, quadratic state cost, effort and total LQR cost.
- `summaries/verifier_comparison.json`: fixed betting and predictable empirical-Bernstein comparisons on the same 64 existing plants.
- `summaries/coupling_sensitivity.json`: retained displacement, noise-coupling and residual-bound replays on 24 stable plants.
- `summaries/nonnormal.json`: finite-horizon matrix-envelope comparisons on 24 distinct nonnormal plants.
- `data/pendulum_regional_study/`: eight frozen regulation policies, training records, 32,768-pair certification banks and 4,096-pair evaluation banks, plus original completion hashes, protocols, development summaries and the retained independent audit. The final study uses bounded torque noise and a restricted upright initial disk.
- `summaries/pendulum_record_manifest.json`: provenance and hashes of 53 unmodified regulation records extracted from the original recorded-data archive.
- `historical/`: aggregate/protocol records of earlier physical-interval, learner, Reacher and budget-selection studies, including unsuccessful outcomes. Their historical raw banks are not included; these records do not establish full raw-data reproducibility for those studies.
- `original_scripts/`: unmodified source snapshots and the mechanical metric protocol. These scripts use the original experiment directory layout; use `audit.py` for the portable record check.

## Check the retained observations

Python 3.11 with NumPy is sufficient for the metric-bank audit:

```sh
python audit.py
```

For verifier replay, also install SciPy and provide the original approximately 291 MB recorded-data ZIP:

```sh
python audit.py --recorded-data /path/to/crn_policy_experiment_data.zip
```

In the complete code distribution that file is under `reproducibility/crn_policy/data/`.
The audit recomputes all six mechanical aggregates from the raw arrays, checks bank hashes against original completion records, replays the two verifiers at their original levels, and reaggregates the residual diagnostics. It does not retrain policies or regenerate observations.

The self-contained regulation audit needs NumPy and SciPy:

```sh
python pendulum_audit.py
```

It checks all copied record hashes, repeats eight exact-rational matrix inequalities and the invariant-region, input and reward bounds, recomputes all four supports, and replays all 64 final stopping decisions. It also reconstructs 128 recorded evaluation episodes per policy branch from their fixed seeds through an independent scalar implementation. This is a check of existing observations, not a new experiment. Gymnasium is not needed for this audit. The unmodified original audit snapshot in `original_scripts/` uses the original layout and dependencies.

Regional marginal support exploits the same invariant geometry and action cap as the paired bounds. The four supports use the same policies and observations. Final independent gains are small; this regulation variant is distinct from the standard swing-up task. The earlier Gaussian-noise development is not covered by the final invariant-region certificate.

The secondary mechanical costs are unbounded and have no adoption certificate. Mean velocity and effort costs increase despite a reduction in the bounded objective. Verifier and residual comparisons reuse observations; they are not independent additional experiments. An underestimated residual bound violates the stated premise even when no sampled support violation is observed.
