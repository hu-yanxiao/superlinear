#!/usr/bin/env python3
"""Source and math contract checks for the release3 LAMMPS cutoff port."""

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def factored(x, p):
    polynomial = 0.0
    for k in range(p - 1, -1, -1):
        polynomial = polynomial * x + 0.5 * (k + 1) * (k + 2)
    u = 1.0 - x
    value = u * u * u * polynomial
    derivative_x = -0.5 * p * (p + 1) * (p + 2) * x ** (p - 1) * u * u
    return value, derivative_x


def expanded(x, p):
    return (
        1.0
        - 0.5 * (p + 1) * (p + 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - 0.5 * p * (p + 1) * x ** (p + 2)
    )


for p in (1, 2, 3, 4, 5, 6, 16, 64):
    for x in (0.0, 0.05, 0.25, 0.5, 0.8, 0.95):
        value, derivative_x = factored(x, p)
        assert math.isclose(value, expanded(x, p), rel_tol=2e-12, abs_tol=2e-13)
        h = 1.0e-6
        finite_difference = (factored(x + h, p)[0] - factored(x - h, p)[0]) / (2 * h)
        assert math.isclose(derivative_x, finite_difference, rel_tol=2e-5, abs_tol=2e-7)

pair = (ROOT / "interfaces/lammps/ML-SUS2/pair_sus2_mtp.cpp").read_text()
cpu = (ROOT / "interfaces/lammps/ML-SUS2/sus2_mtp_rb_chebyshev_ss_basis.cpp").read_text()
kk = (ROOT / "interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.cpp").read_text()

assert 'radial_cutoff_type_index = SUS2RadialMTPBasis::CUTOFF_LEGACY' in pair
assert 'radial_cutoff_p = 5' in pair
assert 'radial_cutoff_p requires preceding radial_cutoff_type' in pair
assert 'radial_cutoff_p should be in [1, 64]' in pair
assert pair.count('MPI_Bcast(&radial_cutoff_') == 2

for function in (
    "RBChebyshev_sss::calc_radial_basis",
    "RBChebyshev_sss_lmp::calc_radial_basis",
    "RBChebyshev_sss_rational::calc_radial_basis",
    "RBChebyshev_sss_rational_lmp::calc_radial_basis",
    "RBLaguerre_log1p::calc_radial_basis",
    "RBLaguerre_log1p_lmp::calc_radial_basis",
):
    start = cpu.index(function)
    end = cpu.find("\n}\n", start)
    assert "CalcRadialCutoff" in cpu[start:end], function

assert "basis.CalcRadialCutoff" in cpu
assert "eval_radial_cutoff" in kk
assert kk.count("eval_radial_cutoff(dist, cutoff, cutoff_der)") == 4

print("LAMMPS release3 radial cutoff source/math checks passed")
