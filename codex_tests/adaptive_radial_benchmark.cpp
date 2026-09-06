#include "dev_src/mtpr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef MLIP_MPI
#include <mpi.h>
#endif

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

struct Timing {
	double seconds = 0.0;
	double checksum = 0.0;
};

struct Summary {
	double min_value = 0.0;
	double max_value = 0.0;
	double sum_value = 0.0;
};

struct GradientReuseResult {
	double max_scaled_error = 0.0;
	double min_abs_checksum = HUGE_DOUBLE;
};

double Now()
{
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string HostName()
{
	char name[256] = {0};
#if defined(_WIN32)
	if (gethostname(name, sizeof(name) - 1) != 0)
		return "unknown";
#else
	if (::gethostname(name, sizeof(name) - 1) != 0)
		return "unknown";
#endif
	return name;
}

std::vector<Configuration> LoadConfigurations(const std::string& path,
                                              int max_configs)
{
	std::ifstream input(path, std::ios::binary);
	if (!input)
		ERROR("adaptive_radial_benchmark: cannot open configuration file");
	std::vector<Configuration> out;
	Configuration cfg;
	while ((max_configs <= 0 || static_cast<int>(out.size()) < max_configs)
	       && cfg.Load(input)) {
		if (cfg.size() > 0)
			out.push_back(cfg);
	}
	if (out.empty())
		ERROR("adaptive_radial_benchmark: no configurations loaded");
	return out;
}

double AccumulateEFSChecksum(const Configuration& cfg)
{
	double sum = cfg.energy;
	for (int i = 0; i < cfg.size(); ++i) {
		if (i < static_cast<int>(cfg.cal_se.size()))
			sum += 0.031 * cfg.cal_se[i];
		for (int a = 0; a < 3; ++a)
			sum += (0.013 + 0.007 * a) * cfg.force(i, a);
	}
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			sum += (0.002 + 0.001 * (3 * a + b)) * cfg.stresses[a][b];
	return sum;
}

double AccumulateGradientChecksum(const Array1D& gradient)
{
	double sum = 0.0;
	for (int i = 0; i < static_cast<int>(gradient.size()); ++i)
		sum += gradient[i] * std::sin(0.17 * (i + 1));
	return sum;
}

double AccumulateComponentChecksum(MLMTPR& model,
                                   const Configuration& cfg)
{
	const int columns = model.LinearEquationCount();
	double sum = 0.0;
	for (int c = 0; c < columns; ++c) {
		sum += model.energy_cmpnts[c] * std::sin(0.11 * (c + 1));
		for (int i = 0; i < cfg.size(); ++i)
			for (int a = 0; a < 3; ++a)
				sum += model.forces_cmpnts(i, c, a)
				     * std::cos(0.07 * (1 + i + 3 * a + 5 * c));
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				sum += model.stress_cmpnts[c][a][b]
				     * std::sin(0.05 * (1 + 3 * a + b + 7 * c));
	}
	return sum;
}

std::vector<double> EnergyWeights(const Configuration& cfg)
{
	std::vector<double> weights(cfg.size(), 0.0);
	for (int i = 0; i < cfg.size(); ++i)
		weights[i] = 0.37 + 0.03 * std::sin(0.4 * (i + 1));
	return weights;
}

std::vector<Vector3> ForceWeights(const Configuration& cfg)
{
	std::vector<Vector3> weights(cfg.size(), Vector3(0.0, 0.0, 0.0));
	for (int i = 0; i < cfg.size(); ++i)
		for (int a = 0; a < 3; ++a)
			weights[i][a] = 0.09 * std::cos(0.31 * (1 + 3 * i + a));
	return weights;
}

Matrix3 StressWeights()
{
	Matrix3 weights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			weights[a][b] = 0.017 * std::sin(0.23 * (1 + 3 * a + b));
	return weights;
}

Array1D ComputeCombinationGradient(MLMTPR& model, Configuration& cfg)
{
	std::vector<double> energy_weights = EnergyWeights(cfg);
	const std::vector<Vector3> force_weights = ForceWeights(cfg);
	const Matrix3 stress_weights = StressWeights();
	Array1D gradient(model.CoeffCount());
	std::fill(gradient.begin(), gradient.end(), 0.0);
	model.AccumulateEFSCombinationGrad(
		cfg, energy_weights, force_weights, stress_weights, gradient);
	return gradient;
}

Timing TimeCalcEFS(MLMTPR& model,
                   const std::vector<Configuration>& configs,
                   int repeats)
{
	Timing out;
	const double start = Now();
	for (int repeat = 0; repeat < repeats; ++repeat) {
		for (const Configuration& source : configs) {
			Configuration cfg = source;
			model.CalcEFS(cfg);
			out.checksum += AccumulateEFSChecksum(cfg);
		}
	}
	out.seconds = Now() - start;
	return out;
}

Timing TimeFullCombinationGrad(MLMTPR& model,
                               const std::vector<Configuration>& configs,
                               int repeats)
{
	Timing out;
	const Matrix3 stress_weights = StressWeights();
	const double start = Now();
	for (int repeat = 0; repeat < repeats; ++repeat) {
		for (const Configuration& source : configs) {
			Configuration cfg = source;
			std::vector<double> energy_weights = EnergyWeights(cfg);
			const std::vector<Vector3> force_weights = ForceWeights(cfg);
			Array1D gradient(model.CoeffCount());
			std::fill(gradient.begin(), gradient.end(), 0.0);
			model.RequestTwoLayerFullEdgeCacheForNextCalcEFS();
			model.CalcEFS(cfg);
			model.AccumulateEFSCombinationGrad(
				cfg, energy_weights, force_weights, stress_weights, gradient);
			out.checksum += AccumulateGradientChecksum(gradient);
		}
	}
	out.seconds = Now() - start;
	return out;
}

Timing TimeComponents(MLMTPR& model,
                      const std::vector<Configuration>& configs,
                      int repeats)
{
	Timing out;
	const double start = Now();
	for (int repeat = 0; repeat < repeats; ++repeat) {
		for (const Configuration& source : configs) {
			Configuration cfg = source;
			model.CalcEFSComponents(cfg, true, true);
			out.checksum += AccumulateComponentChecksum(model, cfg);
		}
	}
	out.seconds = Now() - start;
	return out;
}

void MutateCoefficients(MLMTPR& model)
{
	if (model.CoeffCount() == 0)
		return;
	const int linear = std::min(model.LinearCoeffOffset(), model.CoeffCount() - 1);
	model.Coeff()[linear] += 1.0e-4;
	const int amplitude_begin = model.AdaptiveRadialAmplitudeOffset();
	const int amplitude_count = model.AdaptiveRadialRowCount();
	if (amplitude_count > 0 && amplitude_begin < model.CoeffCount())
		model.Coeff()[amplitude_begin] += 2.5e-4;
	const int weight_begin = model.AdaptiveRadialWeightOffset();
	const int weight_count =
		model.AdaptiveRadialRowCount() * model.TwoLayerGateScalarCount();
	if (weight_count > 0 && weight_begin < model.CoeffCount())
		model.Coeff()[weight_begin] -= 3.0e-4;
}

void MutateModelState(MLMTPR& model)
{
	MutateCoefficients(model);
	model.scaling *= 1.0 + 7.0e-5;
}

void MutateConfiguration(Configuration& cfg, int species_count)
{
	if (cfg.size() == 0)
		return;
	for (int i = 0; i < cfg.size(); ++i)
		for (int a = 0; a < 3; ++a)
			cfg.pos(i, a) += 2.0e-4 * std::sin(0.41 * (1 + i + 3 * a));
	if (species_count > 1)
		cfg.type(0) = (cfg.type(0) + 1) % species_count;
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			cfg.lattice[a][b] *= 1.0 + (a == b ? 3.0e-5 : -1.0e-5);
}

void MutateGeometry(Configuration& cfg)
{
	for (int i = 0; i < cfg.size(); ++i)
		for (int a = 0; a < 3; ++a)
			cfg.pos(i, a) += 2.0e-4 * std::sin(0.41 * (1 + i + 3 * a));
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			cfg.lattice[a][b] *= 1.0 + (a == b ? 3.0e-5 : -1.0e-5);
}

void MutateSpecies(Configuration& cfg, int species_count)
{
	if (cfg.size() > 0 && species_count > 1)
		cfg.type(0) = (cfg.type(0) + 1) % species_count;
}

void ApplyGradientReuseMutation(int kind, MLMTPR& model, Configuration& cfg)
{
	switch (kind) {
	case 1:
		MutateGeometry(cfg);
		break;
	case 2:
		MutateSpecies(cfg, model.species_count);
		break;
	case 3:
		model.scaling *= 1.0 + 7.0e-5;
		break;
	case 4:
		MutateCoefficients(model);
		break;
	default:
		break;
	}
}

double CompareEFS(const Configuration& lhs, const Configuration& rhs)
{
	double max_error = 0.0;
	auto compare = [&max_error](double a, double b) {
		if (!std::isfinite(a) || !std::isfinite(b)) {
			max_error = std::max(max_error, 1.0);
			return;
		}
		max_error = std::max(max_error,
			std::abs(a - b) / std::max(1.0, std::max(std::abs(a), std::abs(b))));
	};
	compare(lhs.energy, rhs.energy);
	for (int i = 0; i < lhs.size(); ++i) {
		if (i < static_cast<int>(lhs.cal_se.size())
		    && i < static_cast<int>(rhs.cal_se.size()))
			compare(lhs.cal_se[i], rhs.cal_se[i]);
		for (int a = 0; a < 3; ++a)
			compare(lhs.force(i, a), rhs.force(i, a));
	}
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			compare(lhs.stresses[a][b], rhs.stresses[a][b]);
	return max_error;
}

double CompareGradients(const Array1D& lhs, const Array1D& rhs)
{
	if (lhs.size() != rhs.size())
		return 1.0;
	double max_error = 0.0;
	for (int i = 0; i < static_cast<int>(lhs.size()); ++i) {
		if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]))
			return 1.0;
		max_error = std::max(max_error,
			std::abs(lhs[i] - rhs[i])
			/ std::max(1.0, std::max(std::abs(lhs[i]), std::abs(rhs[i]))));
	}
	return max_error;
}

GradientReuseResult FullCacheGradientReuseCheck(const std::string& model_path,
                                                const Configuration& seed)
{
	GradientReuseResult result;
	for (int kind = 0; kind <= 4; ++kind) {
		MLMTPR cached_model(model_path);
		Configuration cached_cfg = seed;
		cached_model.RequestTwoLayerFullEdgeCacheForNextCalcEFS();
		cached_model.CalcEFS(cached_cfg);
		ApplyGradientReuseMutation(kind, cached_model, cached_cfg);
		Array1D cached_gradient =
			ComputeCombinationGradient(cached_model, cached_cfg);

		MLMTPR fresh_model(model_path);
		Configuration fresh_cfg = seed;
		ApplyGradientReuseMutation(kind, fresh_model, fresh_cfg);
		Array1D fresh_gradient =
			ComputeCombinationGradient(fresh_model, fresh_cfg);

		const double checksum = AccumulateGradientChecksum(cached_gradient);
		if (!std::isfinite(checksum))
			result.max_scaled_error = 1.0;
		result.min_abs_checksum =
			std::min(result.min_abs_checksum, std::abs(checksum));
		result.max_scaled_error = std::max(
			result.max_scaled_error,
			CompareGradients(cached_gradient, fresh_gradient));
	}
	return result;
}

double MutableStateError(const std::string& model_path,
                         const Configuration& seed)
{
	MLMTPR model(model_path);
	Configuration efs_before = seed;
	model.CalcEFS(efs_before);

	Configuration full_grad_cfg = seed;
	std::vector<double> energy_weights = EnergyWeights(full_grad_cfg);
	const std::vector<Vector3> force_weights = ForceWeights(full_grad_cfg);
	const Matrix3 stress_weights = StressWeights();
	Array1D gradient(model.CoeffCount());
	std::fill(gradient.begin(), gradient.end(), 0.0);
	model.RequestTwoLayerFullEdgeCacheForNextCalcEFS();
	model.CalcEFS(full_grad_cfg);
	model.AccumulateEFSCombinationGrad(
		full_grad_cfg, energy_weights, force_weights, stress_weights, gradient);

	Configuration energy_only = seed;
	model.CalcEnergyAndSiteEnergies(energy_only);
	MutateModelState(model);
	Configuration after = seed;
	MutateConfiguration(after, model.species_count);
	model.CalcEFS(after);

	MLMTPR reference(model_path);
	MutateModelState(reference);
	Configuration expected = seed;
	MutateConfiguration(expected, reference.species_count);
	reference.CalcEFS(expected);
	return CompareEFS(after, expected);
}

Summary ReduceSummary(double local)
{
	Summary out;
#ifdef MLIP_MPI
	MPI_Allreduce(&local, &out.min_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&local, &out.max_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&local, &out.sum_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
	out.min_value = local;
	out.max_value = local;
	out.sum_value = local;
#endif
	return out;
}

int Rank()
{
#ifdef MLIP_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;
#else
	return 0;
#endif
}

int Size()
{
#ifdef MLIP_MPI
	int size = 1;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;
#else
	return 1;
#endif
}

void Barrier()
{
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

int Run(int argc, char** argv)
{
	if (argc < 3 || argc > 5) {
		if (Rank() == 0)
			std::cerr << "usage: adaptive_radial_benchmark model.mtp data.cfg "
			          << "[repeats=3] [max-configs=3]\n";
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string data_path = argv[2];
	const int repeats = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 3;
	const int max_configs = argc >= 5 ? std::atoi(argv[4]) : 3;
	std::vector<Configuration> configs = LoadConfigurations(data_path, max_configs);

	MLMTPR model(model_path);
	const int rank = Rank();
	const int ranks = Size();
	if (rank == 0) {
		size_t atoms = 0;
		for (const Configuration& cfg : configs)
			atoms += static_cast<size_t>(cfg.size());
		std::cout << std::setprecision(12)
		          << "adaptive_radial_benchmark"
		          << " host=" << HostName()
		          << " ranks=" << ranks
		          << " configs_per_rank=" << configs.size()
		          << " atoms_per_rank=" << atoms
		          << " repeats=" << repeats
		          << " coeff_count=" << model.CoeffCount()
		          << " linear_columns=" << model.LinearEquationCount()
		          << " adaptive_rows=" << model.AdaptiveRadialRowCount()
		          << " gate_scalars=" << model.TwoLayerGateScalarCount()
		          << '\n';
	}

	Barrier();
	Timing calc_efs = TimeCalcEFS(model, configs, 1);
	Timing full_grad = TimeFullCombinationGrad(model, configs, 1);
	Timing components = TimeComponents(model, configs, 1);
	(void)calc_efs;
	(void)full_grad;
	(void)components;

	Barrier();
	calc_efs = TimeCalcEFS(model, configs, repeats);
	Barrier();
	full_grad = TimeFullCombinationGrad(model, configs, repeats);
	Barrier();
	components = TimeComponents(model, configs, repeats);
	Barrier();
	const double state_error = MutableStateError(model_path, configs.front());
	const GradientReuseResult reuse_result =
		FullCacheGradientReuseCheck(model_path, configs.front());

	const Summary efs_s = ReduceSummary(calc_efs.seconds);
	const Summary grad_s = ReduceSummary(full_grad.seconds);
	const Summary comp_s = ReduceSummary(components.seconds);
	const Summary state_s = ReduceSummary(state_error);
	const Summary reuse_error_s = ReduceSummary(reuse_result.max_scaled_error);
	const Summary reuse_checksum_s = ReduceSummary(reuse_result.min_abs_checksum);
	const Summary checksum_s = ReduceSummary(
		calc_efs.checksum + full_grad.checksum + components.checksum);
	if (rank == 0) {
		std::cout << std::setprecision(12)
		          << "calc_efs_seconds_min=" << efs_s.min_value
		          << " calc_efs_seconds_max=" << efs_s.max_value
		          << " calc_efs_seconds_avg=" << efs_s.sum_value / ranks
		          << '\n'
		          << "full_combination_grad_seconds_min=" << grad_s.min_value
		          << " full_combination_grad_seconds_max=" << grad_s.max_value
		          << " full_combination_grad_seconds_avg=" << grad_s.sum_value / ranks
		          << '\n'
		          << "components_seconds_min=" << comp_s.min_value
		          << " components_seconds_max=" << comp_s.max_value
		          << " components_seconds_avg=" << comp_s.sum_value / ranks
		          << '\n'
		          << "mutable_state_scaled_error_max=" << state_s.max_value
		          << " full_cache_gradient_scaled_error_max=" << reuse_error_s.max_value
		          << " full_cache_gradient_min_abs_checksum="
		          << reuse_checksum_s.min_value
		          << " combined_checksum_avg=" << checksum_s.sum_value / ranks
		          << '\n';
	}
	return state_s.max_value < 2.0e-10
	    && reuse_error_s.max_value < 2.0e-10
	    && reuse_checksum_s.min_value > 1.0e-14
		? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
#endif
	int rc = 1;
	try {
		rc = Run(argc, argv);
	} catch (const MlipException& exc) {
		if (Rank() == 0)
			std::cerr << exc.What() << std::endl;
		rc = 1;
	} catch (const std::exception& exc) {
		if (Rank() == 0)
			std::cerr << exc.what() << std::endl;
		rc = 1;
	}
#ifdef MLIP_MPI
	MPI_Finalize();
#endif
	return rc;
}
