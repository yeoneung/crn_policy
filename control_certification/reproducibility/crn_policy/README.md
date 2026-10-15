# CRN policy experiments

Simulation and replay tools for comparing small policy repairs with **common random numbers (CRN)**. The experiments separate noise coupling from the support bound supplied to a sequential test.

Included studies cover stable controlled plants, calibration of dynamics bounds, nonnormal and underactuated systems, displacement sweeps, independent-noise ablations, residual-bound sensitivity, and a Pendulum regulation variant. Recorded outcomes include adoption, rejection, censoring and fallback.

## Setup

Use Python 3.11 and a CPU:

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py tests
```

CUDA, PyTorch and MuJoCo are not required for these commands.

## Replay the recorded experiments

Download the approximately 291 MB data asset and verify its checksum:

```sh
python download_data.py
python reproduce.py all
```

Alternatively, obtain [crn_policy_experiment_data.zip](https://github.com/yeoneung/crn_policy/releases/download/experiments-v1/crn_policy_experiment_data.zip) and supply its path:

```sh
python reproduce.py all --data /path/to/crn_policy_experiment_data.zip
```

Every command creates a new directory under `runs/`. Generated plots are in `runs/<run>/artifacts/figures/`, tables in `artifacts/tables/`, and recomputed results in `code/results/`. Logs and `run_report.json` record status, timings and dependency versions. Existing run directories are never overwritten.

| Command | Experiments | Data needed |
|---|---|---|
| `python reproduce.py tests` | Betting/PM-EB recursions, support checks, direction scaling and CRN parity | No |
| `python reproduce.py support` | Stable support, calibration and nonnormal comparisons | Yes |
| `python reproduce.py variance` | Two verifiers on 64 plants and an analytic noise-cancellation example | Yes |
| `python reproduce.py coupling` | Displacement sweep, noise ablation, residual-bound sensitivity and audit | Yes |
| `python reproduce.py pendulum` | Regional-support comparison, dynamics/stopping checks and LQR references | Yes |
| `python reproduce.py all` | All commands above in one workspace | Yes |

Replay uses retained observations and small independent transition checks. It does not retrain all policies or regenerate the full simulation banks. Fresh simulation instructions are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Files

- `reproduce.py`, `download_data.py`: portable execution and data-download tools.
- `frozen/code/scripts/`: original experiment code, preserved byte for byte.
- `frozen/*.md`: experiment protocols and numerical implementation amendments.
- `frozen/code/results/`: one fixed development choice used by mechanical simulations.
- `recorded_summaries/`: small aggregate records for inspection without downloading raw banks.
- `data/manifest.json`: sizes and SHA-256 hashes for 40 frozen inputs and 984 recorded files.
- `docs/EXPERIMENTS.md`: study populations, dependency structure and data coverage.
- `third_party_licenses/`: dependency notices; no package binaries are redistributed.

Run the original code through `reproduce.py`, which creates the expected relative directory layout in an isolated workspace. `.gitattributes` preserves original file bytes across Windows and POSIX Git checkouts.

The Pendulum environment is an upright regulation variant with declared actuator and disturbance changes, not an unmodified swing-up benchmark. PM-EB and residual-bound analyses reuse existing banks; they are not independent replications. All retained outcomes for the included studies remain available.

See [docs/VALIDATION.md](docs/VALIDATION.md) for execution checks and [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the current licensing status.
