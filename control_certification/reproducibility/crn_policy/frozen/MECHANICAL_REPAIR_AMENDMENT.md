# Numerical implementation amendment, before aggregate inspection

2026-09-27 UTC. The original mechanical runner completed 8-state seeds600,
601,602 and then failed an action-radius assertion during the independent
evaluation of seed603. No aggregate comparison or outcome-directed choice
was made. Every original completed/partial file remains under
`mechanical_repair_confirmatory`; that attempt is marked incomplete and is
not pooled with the amended study.

Diagnosis: subtracting u*u_j from coordinate vector e_j can lose relative
precision when u is almost parallel to that coordinate. Normalizing the tiny
remainder can then lose tangency. The mathematical candidate has not changed.
For r=||u_{-j}||, its unit tangent is computed directly as t_j=r and
t_{-j}=-u_j*u_{-j}/r, with the same r>1e-10 degeneracy rule. This avoids the
subtraction. Dedicated tests cover almost axial 2D actions down to1e-14 and
agreement with the original formula away from degeneracy.

Use new source/result names ending `_v2`, preserve the original implementation
and all its results, rerun every development cell by the original selection
rule, and rerun all16 final instances with the same seeds and random banks.
No seed, reward, dynamics, statistical unit, budget, stopping threshold or
hyperparameter-selection criterion changes. The amended study is transparently
identified as an implementation restart, not an independent replication of
the initial completed cells. Its source snapshot includes this amendment.
