# Control certification code and data

Simulation tools for comparing small controller updates using common random
numbers, justified paired-return bounds and independent transition calibration.

The implementation, preserved experiment sources and recorded inputs are in
[`control_certification/`](control_certification/).
The [experiment catalog](control_certification/EXPERIMENTS.md) maps every active
comparison to its simulator, learner, raw banks and replay checks.
See the [implementation guide](control_certification/CODE_README.md) for APIs
and assumptions.

## Install and reproduce

Use Python 3.11 and a CPU. No GPU, PyTorch or MuJoCo is needed.

```sh
cd control_certification
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_recorded_data.py
python reproduce.py verify
python reproduce.py current
```

The download helper retrieves the original approximately 291 MB data asset
and checks its byte count and SHA-256. Additional metric banks and regional
regulation records are tracked directly. Once dependencies and inputs are
available, replay runs without network access.

`current` runs 31 existing implementation/runner tests, certificate checks,
recorded comparison audits and figure regeneration. It creates a new output
folder and preserves all input files. It includes:

- Stable controller updates, noise/displacement comparisons and reset-prefix calibration.
- Residual-bound sensitivity, nonnormal dynamics and underactuated input geometry.
- Fixed betting and predictable empirical-Bernstein replay on the same 64 plants.
- Six secondary mechanical costs, including adverse velocity and effort outcomes.
- Finite-power acquisition with a model-based reference and signed-action controls.
- Regional upright Pendulum regulation with four analytically justified bounds.

Replays and bound variations reuse policies and observations; they are not
additional independent plant populations. Full benchmark retraining is not
performed by `current`. Fresh simulation sources and isolated-workspace
commands are documented in the experiment catalog.

## Self-contained download

The [v2 code and data release](https://github.com/yeoneung/crn_policy/releases/tag/control-certification-v2)
contains `control_certification.zip` and `control_certification.zip.sha256`.
The approximately 307 MB ZIP includes all required recorded inputs and needs
no separate data download. Extract it and follow `CODE_README.md`.

The v1 and original-data releases remain available as earlier snapshots.
The original source artifact is preserved once under
`control_certification/reproducibility/crn_policy/`, rather than duplicated
at the repository root. Older aggregate-only studies remain identified as
historical records; they are outside the default active-comparison command.

## Checks and rebuilding

```sh
cd control_certification
python reproduce.py quick
python prepare_sources.py --mode fresh --output runs/new-source-workspace
python package_code.py
```

Packaging refreshes the file manifest and produces a ZIP plus checksum.
Keep newly generated runs outside preserved source and evidence directories.
Licensing and dependency notices are in [LICENSE_NOTICE.md](LICENSE_NOTICE.md)
and the preserved artifact's `third_party_licenses/` directory.
