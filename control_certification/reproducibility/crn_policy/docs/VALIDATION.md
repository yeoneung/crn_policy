# Release validation

The experiment-only export passed the checks below with Python 3.11.16 on Windows and the pinned numerical dependencies. The machine-readable record is in [validation.json](validation.json).

Completed checks:

- SHA-256 verification of all 40 frozen inputs and 984 data members.
- The 12 existing verifier/coupling tests and 3 runner tests.
- All supported replay commands in a new workspace.
- Eight recomputed support, calibration, verifier, coupling and Pendulum result files matched their recorded values: 27,688 numeric fields at `rtol=1e-10`, `atol=1e-12`.
- Independent stopping audits covering 1,536 coupling cells and 96 Pendulum cells, along with geometry and transition/reference checks.
- Byte preservation through Git staging, including Windows line-ending settings.

Full policy training and all large simulation banks are not repeated as part of release validation. The CI workflow runs source integrity and smoke checks without downloading the large data asset.
