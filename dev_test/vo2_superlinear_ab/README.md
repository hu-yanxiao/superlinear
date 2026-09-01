# VO2 release3 versus superlinear A/B

This server experiment freezes one model checkpoint from
`/work/phy-weigw/hyx/vo2/9.1` and uses the same 2,062-configuration training
set, 200 MPI ranks, five 40-rank nodes, BFGS/Wolfe settings, weights, and 100
optimizer iterations for both branches.

- A loads the frozen release3 checkpoint with the formal release3 binary.
- B loads the same checkpoint after the dedicated zero-perturbation radial
  mixing upgrade and uses the superlinear binary.

Before submission, run `calc-efs` on a frozen smoke subset for the original
checkpoint and the zero-upgraded checkpoint.  Numeric parity must pass at
`1e-12`.  Do not modify or suspend the live VO2 training job.
