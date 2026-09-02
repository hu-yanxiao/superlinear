# VO2 fresh-start release3/superlinear A/B audit

This protocol compares the current release3 model against the superlinear
radial mixer without loading any trained checkpoint.  All raw starting models
are generated with `--max-iter=0`, `--init-params=random`, and the same explicit
64-bit seed.  The raw artifact is saved before shift correction, linear
warmup, readout fitting, or BFGS.

Variants:

- `release3`: formal release3 binary, loading the raw mixer-off A model.
- `superlinear_off`: superlinear binary, loading the same raw A model.
- `superlinear_zero`: superlinear binary, common coefficients identical to A,
  radial-mixing metadata enabled, all mixer coefficients exactly zero.
- `superlinear_mix`: superlinear binary, common coefficients identical to A,
  mixer coefficients sampled uniformly in `[-1e-4, 1e-4]` after the common
  random stream.

The topology matches `/work/phy-weigw/hyx/vo2/9.1`: two species, L3/K8,
body order 6, body-specific L cutoffs `3,3,3,2,1`, scaling map K, 6.5 A
cutoff, 10 Chebyshev radial functions, full first gate truncated at body order
4, gate K-rank 4, and outer K-rank 2.

Run `prepare_fresh_models.sh` once in a new campaign directory outside
`/work/phy-weigw/hyx/vo2/9.1`.  It refuses both the live directory and any of
its descendants, refuses to reuse existing seed artifacts, verifies every
common trainable block including the multiline base-radial block, and performs
three one-configuration checks:

1. formal release3 A versus superlinear A numerical parity;
2. superlinear A versus zero-mixer B0 numerical parity;
3. active mixer B versus B0 finite, nonzero effect.

Only after those checks pass should `bsub_fresh_profile.lsf` be submitted.  The
job runs the four variants twice in one 200-rank allocation.  Round 1 uses
`release3, off, zero, mix`; round 2 reverses the order as
`mix, zero, off, release3`.  This controls both queue/node variation and the
first-to-last cache/thermal ordering bias.  Each run permits three BFGS outer
iterations by default and records accepted-step traces, wall time,
binary/model hashes, and superlinear objective/line-search counters.  Compare
time per objective/gradient call and per accepted step across both rounds;
do not infer mixer overhead from aggregate wall time alone.

The short profile is intentionally non-resumable.  Every output directory must
be absent before submission, and any failed campaign must be diagnosed and
rerun in a newly named campaign directory.  This prevents a partially trained
checkpoint from silently entering what is required to remain a fresh-start
A/B comparison.

The live VO2 job in `/work/phy-weigw/hyx/vo2/9.1` is an external reference only.
This protocol must not suspend, overwrite, or continue from that job.
