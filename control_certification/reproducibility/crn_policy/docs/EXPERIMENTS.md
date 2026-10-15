# Experiment map

Paths below refer to `code/results/` inside a run or the recorded data ZIP.

| Study | Plants | Data directory | Replay command |
|---|---:|---|---|
| Stable paired-support comparison | 24 | `coupled_support_confirmatory/` | `support` |
| Independent dynamics calibration | Same 24 | `stability_calibration_confirmatory/` | `support` |
| Nonnormal dynamics | 24 | `nonnormal_confirmatory/` | `support`, `variance` |
| Underactuated mechanical systems | 16 | `actuation_structure_confirmatory/` | `variance` |
| Displacement/noise/residual sensitivity | Same 24 stable plants | `v2_coupling_sensitivity/` | `coupling` |
| PM-EB versus fixed betting | 64 pooled plants | `v2_variance_replay/` | `variance` |
| Regional Pendulum | 8 final, 4 development | `pendulum_regional_study/` | `pendulum` |
| Nominal and target-LQR references | Same 8 final plants | `pendulum_nominal_reference/`, `pendulum_oracle_reference/` | `pendulum` |

The 64 verifier-replay plants are the 24 stable, 24 nonnormal and 16 mechanical plants, not additional independent experiments. Calibration prefixes, residual-bound sensitivity and verifier comparisons share earlier data as declared in their protocols. Displacement and independent-noise banks use new streams on fixed learned policies.

The nominal and target-LQR comparisons are post-hoc descriptive references; they do not change the original adoption decisions. Pendulum development diagnostics are retained, including unsuccessful earlier region choices. Its dynamics helpers include the original native-step and Jacobian routines used by the regional simulator.

Only the studies listed above and their supporting development records are distributed here. The exact member list is in `data/manifest.json`. Frozen protocols retain the original seed ranges, levels, displacement caps, research and deployment cost definitions, and rules for keeping failures.
