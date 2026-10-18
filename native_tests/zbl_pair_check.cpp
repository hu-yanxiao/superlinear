#include <cmath>
#include <iostream>

#include "src/zbl.h"

namespace {

void CheckClose(double actual, double expected, double tol, const char* name)
{
	if (std::abs(actual - expected) > tol) {
		std::cerr << name << " mismatch: actual=" << actual
		          << " expected=" << expected << " tol=" << tol << "\n";
		std::exit(1);
	}
}

ZBLPairValue NepReferenceZBLPair(int zi, int zj, double r, double inner, double outer)
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

	return ZBLPairValue{base_energy * fc, base_dEdr * fc + base_energy * fcp};
}

} // namespace

int main()
{
	CheckClose(DefaultZBLInnerCutoff(), 0.7, 1.0e-15, "default inner cutoff");
	CheckClose(DefaultZBLOuterCutoff(), 1.4, 1.0e-15, "default outer cutoff");
	CheckClose(DefaultZBLTypewiseCutoffFactor(), 0.7, 1.0e-15,
	           "default typewise cutoff factor");
	CheckClose(ZBLCovalentRadius(1), 0.426667, 1.0e-15, "H covalent radius");
	CheckClose(ZBLCovalentRadius(6), 1.0, 1.0e-15, "C covalent radius");

	const double hc_typewise_outer = ZBLTypewiseOuterCutoff(1, 6, 1.4, 0.7);
	CheckClose(hc_typewise_outer, 0.7 * (0.426667 + 1.0), 1.0e-12,
	           "H-C typewise outer cutoff");
	const ZBLPairValue outside_typewise =
		ComputeZBLPair(1, 6, 1.1, 0.7, 1.4, 0.7);
	CheckClose(outside_typewise.energy, 0.0, 1.0e-15,
	           "outside typewise energy");
	CheckClose(outside_typewise.dEdr, 0.0, 1.0e-15,
	           "outside typewise dEdr");

	const ZBLPairValue typewise_reference =
		NepReferenceZBLPair(1, 6, 0.9, 0.0, hc_typewise_outer);
	const ZBLPairValue typewise_implementation =
		ComputeZBLPair(1, 6, 0.9, 0.7, 1.4, 0.7);
	CheckClose(typewise_implementation.energy, typewise_reference.energy, 1.0e-12,
	           "typewise NEP energy");
	CheckClose(typewise_implementation.dEdr, typewise_reference.dEdr, 1.0e-12,
	           "typewise NEP dEdr");

	std::vector<double> core_pair_inner;
	std::vector<double> core_pair_outer;
	std::vector<double> core_pair_outer_sq;
	FillZBLPairCutoffTables(std::vector<int>{1, 6}, 0.7, 1.4, true, 0.7,
	                        core_pair_inner, core_pair_outer, core_pair_outer_sq);
	CheckClose(core_pair_inner[1], 0.0, 1.0e-15,
	           "cached core H-C typewise inner");
	CheckClose(core_pair_outer[1], hc_typewise_outer, 1.0e-12,
	           "cached core H-C typewise outer");
	CheckClose(core_pair_outer[2], hc_typewise_outer, 1.0e-12,
	           "cached core C-H typewise outer");
	CheckClose(core_pair_outer[3], 1.4, 1.0e-12,
	           "cached core C-C typewise outer");

	FillZBLPairCutoffTables(std::vector<int>{1, 6}, 0.7, 1.4, false, 0.0,
	                        core_pair_inner, core_pair_outer, core_pair_outer_sq);
	CheckClose(core_pair_inner[1], 0.7, 1.0e-15,
	           "cached core fixed inner");
	CheckClose(core_pair_outer[1], 1.4, 1.0e-15,
	           "cached core fixed outer");

	const ZBLPairValue short_pair = ComputeZBLPair(1, 6, 0.5, 0.7, 1.4);
	if (!(short_pair.energy > 0.0) || !(short_pair.dEdr < 0.0)) {
		std::cerr << "short-range H-C ZBL pair should be repulsive\n";
		return 1;
	}

	const ZBLPairValue outside = ComputeZBLPair(1, 6, 1.401, 0.7, 1.4);
	CheckClose(outside.energy, 0.0, 1.0e-15, "outside energy");
	CheckClose(outside.dEdr, 0.0, 1.0e-15, "outside dEdr");

	const double r = 1.0;
	const double h = 1.0e-6;
	const ZBLPairValue mid = ComputeZBLPair(1, 6, r, 0.7, 1.4);
	const double eplus = ComputeZBLPair(1, 6, r + h, 0.7, 1.4).energy;
	const double eminus = ComputeZBLPair(1, 6, r - h, 0.7, 1.4).energy;
	const double fd = (eplus - eminus) / (2.0 * h);
	CheckClose(mid.dEdr, fd, 1.0e-6, "dEdr finite difference");

	const ZBLPairValue nep_reference = NepReferenceZBLPair(6, 8, 0.9, 0.7, 1.4);
	const ZBLPairValue implementation = ComputeZBLPair(6, 8, 0.9, 0.7, 1.4);
	CheckClose(implementation.energy, nep_reference.energy, 1.0e-12, "NEP cosine-taper energy");
	CheckClose(implementation.dEdr, nep_reference.dEdr, 1.0e-12, "NEP cosine-taper dEdr");

	std::cout << "zbl_pair_check passed\n";
	return 0;
}
