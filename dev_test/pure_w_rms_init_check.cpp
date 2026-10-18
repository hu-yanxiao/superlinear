#include "../dev_src/mtpr.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef MLIP_MPI
#include <mpi.h>
#endif

namespace {
void Require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}

std::vector<double> Coefficients(MLMTPR& model)
{
	return std::vector<double>(model.Coeff(), model.Coeff() + model.CoeffCount());
}

void SameOutsideW(MLMTPR& model, const std::vector<double>& before)
{
	const int first = model.TwoLayerGateRadialMixingCoeffOffset();
	const int last = first + model.TwoLayerGateRadialMixingCoeffCount();
	for (int i = 0; i < model.CoeffCount(); ++i)
		if (i < first || i >= last)
			Require(model.Coeff()[i] == before[i], "initialization changed a non-W coefficient");
}

void NonzeroGate(MLMTPR& model)
{
	for (int i = 0; i < model.TwoLayerGateWeightCount(); ++i)
		model.Coeff()[model.TwoLayerGateWeightOffset() + i] = 0.04 * std::cos(0.71 * (i + 1));
	for (int i = 0; i < model.TwoLayerGateRadialCoeffCount(); ++i)
		model.Coeff()[model.TwoLayerGateRadialCoeffOffset() + i] = 0.3 * std::cos(0.39 * (i + 1)) + 0.05;
	for (int i = 0; i < model.TwoLayerGateAdditiveCoeffCount(); ++i)
		model.Coeff()[model.TwoLayerGateAdditiveCoeffOffset() + i] = 0.7 + 0.1 * std::sin(i + 1);
	model.DistributeCoeffs();
}

std::vector<double> Statistics(MLMTPR& model, const Configuration& cfg)
{
	std::vector<double> sums;
	std::uint64_t count = 0;
	model.AccumulatePureWOuterBlockSecondMoments(cfg, sums, count);
	Require(count == static_cast<std::uint64_t>(cfg.size()), "incorrect statistics site count");
	Require(sums.size() == static_cast<size_t>(model.L * model.OuterKRank()), "incorrect block order/count");
	return sums;
}

void Compare(const std::vector<double>& a, const std::vector<double>& b, double tolerance,
             const char* message)
{
	Require(a.size() == b.size(), "comparison dimension mismatch");
	for (size_t i = 0; i < a.size(); ++i)
		Require(std::abs(a[i] - b[i]) <= tolerance * std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i]))), message);
}

void CheckPrefix(MLMTPR& narrow, MLMTPR& wide)
{
	Require(narrow.L == wide.L && narrow.radial_func_count == wide.radial_func_count,
	        "prefix fixture dimensions differ");
	const int K = narrow.radial_func_count / narrow.L;
	for (int l = 0; l < narrow.L; ++l)
		for (int o = 0; o < narrow.OuterKRank(); ++o)
			for (int k = 0; k < K; ++k)
				for (int b = 0; b < narrow.p_RadialBasis->rb_size; ++b)
					Require(narrow.Coeff()[narrow.TwoLayerGateRadialMixingCoeffIndex(l, o, k, b)]
					        == wide.Coeff()[wide.TwoLayerGateRadialMixingCoeffIndex(l, o, k, b)],
					        "indexed W changed its common output prefix across O");
}

template<class Function>
void MustRejectUnchanged(MLMTPR& model, Function function)
{
	const std::vector<double> before = Coefficients(model);
	bool rejected = false;
	try { function(); } catch (...) { rejected = true; }
	Require(rejected, "invalid initialization operation was accepted");
	Require(before == Coefficients(model), "rejected initialization partially changed coefficients");
}

void Run(int argc, char** argv)
{
	Require(argc == 7, "usage: rms-init narrow.sus2 wide.sus2 k1.sus2 legacy.sus2 input.cfg roundtrip.sus2");
	MLMTPR narrow(argv[1]), wide(argv[2]), k1(argv[3]), legacy(argv[4]);
	Configuration cfg;
	std::ifstream stream(argv[5]);
	Require(cfg.Load(stream), "could not read initialization test configuration");
	const Configuration original_cfg(cfg);
	NonzeroGate(narrow); NonzeroGate(wide); NonzeroGate(k1);
	const std::vector<double> before = Coefficients(wide);
	const std::uint64_t seed = UINT64_C(0xfedcba9876543210);
	narrow.InitializePureWDirections(seed);
	wide.InitializePureWDirections(seed);
	k1.InitializePureWDirections(seed);
	SameOutsideW(wide, before);
	CheckPrefix(narrow, wide);
	const std::vector<double> seeded = Coefficients(wide);
	wide.InitializePureWDirections(seed);
	Require(seeded == Coefficients(wide), "same indexed seed is not reproducible");
	wide.InitializePureWDirections(seed + 1);
	Require(seeded != Coefficients(wide), "different seed produced identical directions");
	wide.InitializePureWDirections(seed);
	const int K = wide.radial_func_count / wide.L, B = wide.p_RadialBasis->rb_size;
	bool distinct_outputs = false;
	for (int l = 0; l < wide.L; ++l)
		for (int k = 0; k < K; ++k)
			for (int b = 0; b < B; ++b) {
				const double first = wide.Coeff()[wide.TwoLayerGateRadialMixingCoeffIndex(l, 0, k, b)];
				const double second = wide.Coeff()[wide.TwoLayerGateRadialMixingCoeffIndex(l, 1, k, b)];
				distinct_outputs = distinct_outputs || first != second;
				Require(std::abs(first) <= std::sqrt(3.0 / (K * B)), "indexed fan-in amplitude is invalid");
			}
	Require(distinct_outputs, "output directions are permutation locked");

	const std::vector<double> stats = Statistics(wide, cfg);
	Require(Coefficients(wide) == seeded, "RMS statistics changed physical coefficients");
	Require(cfg.energy == original_cfg.energy, "statistics modified input energy");
	for (int i = 0; i < cfg.size(); ++i)
		for (int axis = 0; axis < 3; ++axis)
			Require(cfg.pos(i)[axis] == original_cfg.pos(i)[axis]
			        && cfg.force(i)[axis] == original_cfg.force(i)[axis], "statistics modified input positions/forces");
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			Require(cfg.stresses[i][j] == original_cfg.stresses[i][j], "statistics modified input stress");
	Require(*std::max_element(stats.begin(), stats.end()) > 1e-8, "vacuous zero primitive fixture");
	const std::vector<double> narrow_stats = Statistics(narrow, cfg);
	for (int l = 0; l < narrow.L; ++l)
		Require(std::abs(narrow_stats[l] - stats[l * wide.OuterKRank()])
		        <= 5e-12 * std::max(1.0, narrow_stats[l]), "common raw output RMS changed across O");
	for (int i = 0; i < wide.TwoLayerGateWeightCount(); ++i)
		wide.Coeff()[wide.TwoLayerGateWeightOffset() + i] = 0.0;
	const std::vector<double> no_gate_stats = Statistics(wide, cfg);
	double gate_change = 0.0;
	for (size_t i = 0; i < stats.size(); ++i) gate_change = std::max(gate_change, std::abs(stats[i] - no_gate_stats[i]));
	Require(gate_change > 1e-8, "RMS statistics omitted the nonzero physical gate signal");
	std::copy(seeded.begin(), seeded.end(), wide.Coeff());
	std::vector<double> additive = stats;
	std::uint64_t count = cfg.size();
	wide.AccumulatePureWOuterBlockSecondMoments(cfg, additive, count);
	Require(count == static_cast<std::uint64_t>(2 * cfg.size()), "statistics did not accumulate site counts");
	for (size_t i = 0; i < stats.size(); ++i)
		Require(std::abs(additive[i] - 2.0 * stats[i]) <= 2e-13 * std::max(1.0, stats[i]), "second moments did not accumulate");

	// General 3-D rotation (Rodrigues, axis(1,2,3)), not just a z rotation.
	const double axis[3] = {1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0)};
	const double c = std::cos(0.731), s = std::sin(0.731);
	Matrix3 rotation;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			rotation[i][j] = (i == j ? c : 0.0) + (1.0 - c) * axis[i] * axis[j];
	rotation[0][1] -= s * axis[2]; rotation[1][0] += s * axis[2];
	rotation[0][2] += s * axis[1]; rotation[2][0] -= s * axis[1];
	rotation[1][2] -= s * axis[0]; rotation[2][1] += s * axis[0];
	Configuration rotated(cfg);
	rotated.Deform(rotation);
	Compare(stats, Statistics(wide, rotated), 5e-11, "outer tensor RMS is not rotation invariant");

	std::vector<double> factors(stats.size());
	for (size_t i = 0; i < factors.size(); ++i) factors[i] = 0.6 + 0.13 * i;
	wide.ApplyPureWOutputScales(factors);
	SameOutsideW(wide, seeded);
	const std::vector<double> scaled_stats = Statistics(wide, cfg);
	for (size_t i = 0; i < stats.size(); ++i)
		Require(std::abs(scaled_stats[i] - stats[i] * factors[i] * factors[i])
		        <= 5e-12 * std::max(1.0, scaled_stats[i]), "RMS scaling is not a primitive factor squared");
	// Head changes cannot affect A; catches accidental scalar/readout statistics.
	for (int i = wide.LinearCoeffOffset(); i < wide.CoeffCount(); ++i)
		wide.Coeff()[i] = 7.0 + 0.1 * i;
	Compare(scaled_stats, Statistics(wide, cfg), 0.0, "RMS statistics depend on fitted heads");
	wide.Save(argv[6]);
	MLMTPR roundtrip(argv[6]);
	Compare(Coefficients(wide), Coefficients(roundtrip), 2e-14, "W metadata did not roundtrip");
	Compare(scaled_stats, Statistics(roundtrip, cfg), 1e-11, "scaled W statistics changed after roundtrip");
	std::vector<double> calibration(scaled_stats.size());
	for (size_t i = 0; i < calibration.size(); ++i) {
		Require(scaled_stats[i] > 1e-12, "vacuous RMS calibration fixture");
		calibration[i] = std::sqrt(static_cast<double>(cfg.size()) / scaled_stats[i]);
	}
	wide.ApplyPureWOutputScales(calibration);
	const std::vector<double> normalized = Statistics(wide, cfg);
	for (double value : normalized)
		Require(std::abs(std::sqrt(value / cfg.size()) - 1.0) < 2e-11, "physical outer RMS target was not attained");
	const std::vector<double> boundary = Statistics(k1, cfg);
	Require(boundary.size() == static_cast<size_t>(k1.L * k1.OuterKRank()), "K1/R>KO boundary failed");

	MustRejectUnchanged(legacy, [&]() { legacy.InitializePureWDirections(seed); });
	MustRejectUnchanged(legacy, [&]() { legacy.ApplyPureWOutputScales(factors); });
	MustRejectUnchanged(legacy, [&]() { Statistics(legacy, cfg); });
	MustRejectUnchanged(wide, [&]() { wide.ApplyPureWOutputScales(std::vector<double>()); });
	std::vector<double> bad = factors;
	bad.back() = std::numeric_limits<double>::quiet_NaN();
	MustRejectUnchanged(wide, [&]() { wide.ApplyPureWOutputScales(bad); });
	bad.back() = 0.0;
	MustRejectUnchanged(wide, [&]() { wide.ApplyPureWOutputScales(bad); });
	bad.back() = -1.0;
	MustRejectUnchanged(wide, [&]() { wide.ApplyPureWOutputScales(bad); });
	const int first = wide.TwoLayerGateRadialMixingCoeffOffset();
	wide.Coeff()[first] = std::numeric_limits<double>::max();
	MustRejectUnchanged(wide, [&]() { wide.ApplyPureWOutputScales(std::vector<double>(factors.size(), 2.0)); });
	wide.InitializePureWDirections(seed);
	Configuration invalid_cfg(cfg);
	invalid_cfg.pos(0)[1] = std::numeric_limits<double>::quiet_NaN();
	MustRejectUnchanged(wide, [&]() { Statistics(wide, invalid_cfg); });
	std::vector<double> overflow = stats;
	std::uint64_t overflow_count = std::numeric_limits<std::uint64_t>::max();
	MustRejectUnchanged(wide, [&]() { wide.AccumulatePureWOuterBlockSecondMoments(cfg, overflow, overflow_count); });
	Require(overflow == stats && overflow_count == std::numeric_limits<std::uint64_t>::max(), "rejected statistics changed its accumulators");
	for (int i = 0; i < wide.TwoLayerGateRadialMixingCoeffCount(); ++i) wide.Coeff()[first + i] = 0.0;
	const std::vector<double> zeros = Statistics(wide, cfg);
	Require(std::all_of(zeros.begin(), zeros.end(), [](double value) { return value == 0.0; }), "zero W used a legacy fallback in RMS statistics");
	std::cout << "pure_w_rms_init_api_ok blocks=" << stats.size() << " seed=" << seed << '\n';
}
} // namespace

int main(int argc, char** argv)
{
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
#endif
	int result = 0;
	try { Run(argc, argv); }
	catch (const std::exception& e) { std::cerr << e.what() << '\n'; result = 1; }
	catch (...) { std::cerr << "RMS initialization test failed with an MLIP exception\n"; result = 1; }
#ifdef MLIP_MPI
	MPI_Finalize();
#endif
	return result;
}
