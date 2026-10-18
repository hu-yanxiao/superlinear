#include <cmath>
#include <cstdlib>
#include <iostream>

#include "interfaces/lammps/ML-SUS2/sus2_mtp_zbl.h"

namespace {

void CheckClose(double actual, double expected, double tol, const char *name)
{
  if (std::abs(actual - expected) > tol) {
    std::cerr << name << " mismatch: actual=" << actual
              << " expected=" << expected << " tol=" << tol << "\n";
    std::exit(1);
  }
}

SUS2MTPZBLPairValue NepReferenceZBLPair(int zi, int zj, double r, double inner, double outer)
{
  const double coefficients[4] = {0.18175, 0.50986, 0.28022, 0.02817};
  const double exponents[4] = {3.1998, 0.94229, 0.4029, 0.20162};
  const double ev_angstrom_per_e2 = 14.3996454784255;
  const double a_inv =
      2.134563 * (std::pow(static_cast<double>(zi), 0.23) +
                  std::pow(static_cast<double>(zj), 0.23));
  const double x = a_inv * r;
  double phi = 0.0;
  double dphi_dr = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double exp_value = std::exp(-exponents[i] * x);
    phi += coefficients[i] * exp_value;
    dphi_dr -= coefficients[i] * exponents[i] * a_inv * exp_value;
  }
  const double prefactor = ev_angstrom_per_e2 * zi * zj;
  const double base_energy = prefactor * phi / r;
  const double base_dEdr = prefactor * (dphi_dr / r - phi / (r * r));

  double fc = 1.0;
  double fcp = 0.0;
  if (r >= outer) {
    fc = 0.0;
  } else if (r >= inner) {
    const double pi_factor = std::acos(-1.0) / (outer - inner);
    fc = 0.5 * std::cos(pi_factor * (r - inner)) + 0.5;
    fcp = -0.5 * pi_factor * std::sin(pi_factor * (r - inner));
  }
  return SUS2MTPZBLPairValue{base_energy * fc, base_dEdr * fc + base_energy * fcp};
}

} // namespace

int main()
{
  CheckClose(sus2_mtp_zbl::DefaultTypewiseCutoffFactor(), 0.7, 1.0e-15,
             "LAMMPS-SH default typewise factor");
  CheckClose(sus2_mtp_zbl::CovalentRadius(1), 0.426667, 1.0e-15,
             "LAMMPS-SH H covalent radius");
  CheckClose(sus2_mtp_zbl::CovalentRadius(6), 1.0, 1.0e-15,
             "LAMMPS-SH C covalent radius");

  const double hc_typewise_outer = sus2_mtp_zbl::PairOuterCutoff(1, 6, 1.4, 0.7);
  CheckClose(hc_typewise_outer, 0.7 * (0.426667 + 1.0), 1.0e-12,
             "LAMMPS-SH typewise outer cutoff");
  const int atomic_numbers[2] = {1, 6};
  double pair_inner[4] = {-1.0, -1.0, -1.0, -1.0};
  double pair_outer[4] = {-1.0, -1.0, -1.0, -1.0};
  double pair_outer_sq[4] = {-1.0, -1.0, -1.0, -1.0};
  sus2_mtp_zbl::FillPairCutoffTables(2, atomic_numbers, 0.7, 1.4, true, 0.7,
                                     pair_inner, pair_outer, pair_outer_sq);
  CheckClose(pair_inner[0], 0.0, 1.0e-15, "cached H-H typewise inner");
  CheckClose(pair_inner[1], 0.0, 1.0e-15, "cached H-C typewise inner");
  CheckClose(pair_outer[0], 0.7 * (0.426667 + 0.426667), 1.0e-12,
             "cached H-H typewise outer");
  CheckClose(pair_outer[1], hc_typewise_outer, 1.0e-12,
             "cached H-C typewise outer");
  CheckClose(pair_outer[2], hc_typewise_outer, 1.0e-12,
             "cached C-H typewise outer");
  CheckClose(pair_outer[3], 1.4, 1.0e-12, "cached C-C typewise outer");
  CheckClose(pair_outer_sq[1], hc_typewise_outer * hc_typewise_outer, 1.0e-12,
             "cached H-C typewise outer squared");

  const SUS2MTPZBLPairValue typewise_reference =
      NepReferenceZBLPair(1, 6, 0.9, 0.0, hc_typewise_outer);
  const SUS2MTPZBLPairValue typewise_implementation =
      sus2_mtp_zbl::ComputePairHost(1, 6, 0.9, 0.7, 1.4, 0.7);
  const SUS2MTPZBLPairValue cached_typewise_implementation =
      sus2_mtp_zbl::ComputePairHost(1, 6, 0.9, pair_inner[1], pair_outer[1]);
  CheckClose(typewise_implementation.energy, typewise_reference.energy, 1.0e-12,
             "LAMMPS-SH typewise ZBL energy");
  CheckClose(typewise_implementation.dEdr, typewise_reference.dEdr, 1.0e-12,
             "LAMMPS-SH typewise ZBL dEdr");
  CheckClose(cached_typewise_implementation.energy, typewise_reference.energy, 1.0e-12,
             "LAMMPS-SH cached typewise ZBL energy");
  CheckClose(cached_typewise_implementation.dEdr, typewise_reference.dEdr, 1.0e-12,
             "LAMMPS-SH cached typewise ZBL dEdr");
  const SUS2MTPZBLPairValue outside_typewise =
      sus2_mtp_zbl::ComputePairHost(1, 6, 1.1, 0.7, 1.4, 0.7);
  CheckClose(outside_typewise.energy, 0.0, 1.0e-15,
             "LAMMPS-SH outside typewise energy");
  CheckClose(outside_typewise.dEdr, 0.0, 1.0e-15,
             "LAMMPS-SH outside typewise dEdr");
  const SUS2MTPZBLPairValue cached_outside_typewise =
      sus2_mtp_zbl::ComputePairHost(1, 6, 1.1, pair_inner[1], pair_outer[1]);
  CheckClose(cached_outside_typewise.energy, 0.0, 1.0e-15,
             "LAMMPS-SH cached outside typewise energy");
  CheckClose(cached_outside_typewise.dEdr, 0.0, 1.0e-15,
             "LAMMPS-SH cached outside typewise dEdr");

  sus2_mtp_zbl::FillPairCutoffTables(2, atomic_numbers, 0.7, 1.4, false, 0.0,
                                     pair_inner, pair_outer, pair_outer_sq);
  CheckClose(pair_inner[1], 0.7, 1.0e-15, "cached fixed inner");
  CheckClose(pair_outer[1], 1.4, 1.0e-15, "cached fixed outer");
  CheckClose(pair_outer_sq[1], 1.96, 1.0e-15, "cached fixed outer squared");

  const SUS2MTPZBLPairValue reference = NepReferenceZBLPair(6, 8, 0.9, 0.7, 1.4);
  const SUS2MTPZBLPairValue implementation =
      sus2_mtp_zbl::ComputePairHost(6, 8, 0.9, 0.7, 1.4);
  CheckClose(implementation.energy, reference.energy, 1.0e-12, "LAMMPS-SH ZBL energy");
  CheckClose(implementation.dEdr, reference.dEdr, 1.0e-12, "LAMMPS-SH ZBL dEdr");

  const SUS2MTPZBLPairValue outside =
      sus2_mtp_zbl::ComputePairHost(1, 6, 1.401, 0.7, 1.4);
  CheckClose(outside.energy, 0.0, 1.0e-15, "LAMMPS-SH outside energy");
  CheckClose(outside.dEdr, 0.0, 1.0e-15, "LAMMPS-SH outside dEdr");

  std::cout << "lammps_sh_zbl_pair_check passed\n";
  return 0;
}
