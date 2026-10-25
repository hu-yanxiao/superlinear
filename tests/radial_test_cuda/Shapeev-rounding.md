# Shapeev CUDA r1 failure diagnosis and r2 repair

Authoritative r1 GPU job 4349879 compiled successfully and ran on A100-SXM4-80GB,
then exited 1 at RBShapeev, B=12, r=2.5, n=10. Its evidence is retained, not
overwritten. The failure was an actual numerical parity failure, not a fallback
to Serial and not a successful qualification.

CPU `sus2_sl_radial_math.cpp` and the device implementation both accumulate
monomials in increasing exponent order i=-2..n. Neither uses Horner. Both sets
of coefficient formulas were compared bit-for-bit locally: zero mismatches.
With the exact same coefficients, the independent diagnostic reproduces:

| Operation sequence | value |
|---|---:|
| CPU / rounded multiply then add | -0.00070551076337466156 |
| explicit FMA / actual r1 CUDA | -0.00070551075664588843 |
| difference | 6.7287731376347581502e-12 |

Both reproduced outputs match the reported remote CPU/GPU values exactly.
At this r, repeated integer powers and libm powers give the same final FMA
result; no power-algorithm or coefficient difference is needed to explain the
observed failure.

Decimal precision 90, using exact binary64 coefficients and scaling, gives
the mathematical sum -0.00070551075383141968834859436584975804399604229443607.
Sum of absolute monomial terms is 140028.11212468933597, yielding cancellation
condition sum(abs(terms))/abs(sum) = 198477643.84063345. CPU accumulation differs
from that exact stored-coefficient sum by -9.54324187639106e-12; FMA differs by
-2.81446873875630e-12. Thus FMA is not intrinsically less accurate here: it is
different from the frozen native CPU operation sequence in an ill-conditioned
representation. Stable Horner/orthogonal recurrences would also change native
rounding and are not silently substituted.

r2 repairs only the CUDA device Shapeev branch: value uses
`__dmul_rn(scaling,c)`, `__dmul_rn(a,power)`, `__dadd_rn(sum,term)`; derivative
also explicitly rounds multiplication by i before multiplying by power(r,i-1).
This prevents contraction while preserving the CPU multiplication order.
Non-CUDA code, coefficients, all other families, model physics, table math and
the global `3e-12*(1+abs(reference))` tolerance are unchanged.

Reproduction diagnostics are `tests/radial_test_shapeev_rounding.cpp` and
`tests/radial_test_shapeev_precision.py`. Compile the C++ diagnostic with the
same sources/libraries as the standalone test and `-ffp-contract=off`, then
run `python radial_test_shapeev_precision.py <diagnostic executable>`.
The diagnostic also checks all 12 rows' native value/derivative against
separate-product accumulation. No diagnostic alters production source.

Full 4.7.1 Serial qualification and host syntax checks must pass after repair.
CUDA libm differences at other coordinates remain something the unchanged
full test must actually measure; r2 is not GPU-qualified until rerun. No GPU
resubmission is performed by this change.
