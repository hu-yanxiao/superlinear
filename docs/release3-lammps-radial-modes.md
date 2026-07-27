# Release3 LAMMPS radial evaluation modes

Release3 LAMMPS exposes the radial evaluation path as a runtime option:

```lammps
pair_style sus2mtp/kk p.mtp chunksize 104544 radial_mode direct
pair_style sus2mtp/kk p.mtp chunksize 104544 radial_mode table tabstep 1.0e-4
```

`radial_mode` accepts `auto`, `table`, or `direct`. The default `auto`
preserves the existing model-file convention: an `_lmp` radial-basis marker
selects the table and the corresponding marker without `_lmp` selects direct
evaluation. Explicit `direct` permits an `_lmp` model to bypass the table
without changing the model file. Explicit `table` requires an `_lmp` marker.

The table spacing has not been enlarged. Its default and the release
performance validation both use `1.0e-4 A`.

## Mathematical path

For Chebyshev radial functions, the direct Kokkos path evaluates

```text
x(r)       = tanh(a (r - b) / 2)
phi_n(r)   = s (r - r_cut)^2 T_n(x(r))
```

and obtains `d phi_n / d r` by differentiating the same Chebyshev recurrence.
For the logarithmic Laguerre form, it evaluates

```text
u(r)       = a log(1 + r / rho)
phi_n(r)   = s (r - r_cut)^2 exp(-u(r) / 2) L_n(u(r))
```

and differentiates the Laguerre recurrence analytically. The positive
Laguerre variant first applies its serialized positive-parameter transform to
`a` and `rho`.

The shared gate radial channel uses the same coefficients as the standalone
model:

```text
R_ij,mu(r) =
  c_type(Z_i, mu) c_type(Z_j, mu)
  sum_n c_radial(mu, n) phi_n(r).
```

Thus `direct` changes only how the already-defined radial functions are
evaluated. It does not add an approximation or alter the model topology.
`table` evaluates the same functions on the fixed grid and linearly
interpolates the stored value and derivative.

Kokkos direct evaluation supports:

- `RBChebyshev_sss[_lmp]`
- `RBLaguerre_log1p[_lmp]`
- `RBLaguerre_log1p_pos[_lmp]`

Unsupported direct bases fail during setup. This includes `auto` resolving a
non-`_lmp` unsupported basis to direct, preventing a different analytic
formula from being selected silently.

## Table layout optimization

The three-dimensional radial tables use `Kokkos::LayoutRight` for the logical
layout `[species_pair][grid_point][mu_group]`. The innermost `mu_group`
dimension is therefore contiguous, matching the access pattern of the
release3 Kokkos kernels. This changes storage order only; table values,
derivatives, interpolation, and model mathematics are unchanged.

## Validation record

All server tests used the release3 VO2 model and an A100 unless noted.

- Final GPU candidate build: job `4012759`, binary SHA-256
  `0c07879c7af1cecbfc0cbd6e2ac2d477d80edc991fc9419bf24858a9569179ba`.
- Final static validation: job `4012877`,
  `RADIAL_MODES_STATIC_VALIDATION_PASS`. CPU/GPU table and direct energies and
  forces agree at the configured numerical limits. Explicit and automatic
  path selection agree exactly. Direct mode creates no radial table.
  Unsupported rational-direct and non-`_lmp` table requests are rejected.
- CPU candidate validation: job `4012824`,
  `RADIAL_MODES_CPU_VALIDATION_PASS`.
- Full topology validation: job `4014137`,
  `DIRECT_TOPOLOGY_VALIDATION_PASS`, with `chunksize=3`. It covers historical
  full-K, outer rank 1/2/K, gate and outer rank combinations, the default gate
  mode, zero and nonzero single-layer readout, and the expected unsupported
  modes. It passes for Chebyshev table, Chebyshev direct, Laguerre direct, and
  positive Laguerre direct.

The 104,544-atom, 300 K NVT performance tests used 100 warm-up steps followed
by 1,000 timed steps, full chunking, and eight-run ABBA orderings:

| Comparison | Mean time | Throughput | Peak GPU memory |
| --- | ---: | ---: | ---: |
| Previous release3 table | 420.7655 s | 248,461 atom-step/s | 14,485 MiB |
| LayoutRight table | 375.6653 s | 278,290 atom-step/s | 14,485 MiB |
| Runtime table | 375.8000 s | 278,191 atom-step/s | 14,485 MiB |
| Runtime direct | 347.5135 s | 300,834 atom-step/s | 14,137 MiB |

The table-layout change is 12.0054% faster than the previous release3 table.
Direct evaluation is another 8.1397% faster than the optimized table and uses
348 MiB less peak GPU memory in this benchmark.

Performance jobs `4008373` and `4008374` used candidate SHA
`1da79405c8be102598dfb778798b88fce177449633074c13f4c28bd0d9e6130e`.
The final candidate differs only by the fail-fast guard for an unsupported
non-`_lmp` direct basis; the supported Chebyshev table/direct kernels measured
by the benchmark are unchanged.
