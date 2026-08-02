/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include "mtpr_trainer.h"
#include "optimizer_accepted_state.h"
#include "optimizer_checkpoint.h"
#include "diagonal_coordinate_scaling.h"
#include "radial_weight_norm.h"

#ifdef ALGLIB
#	include "alglib/optimization.h"
#endif

#ifdef MLIP_INTEL_MKL
#	include <mkl_lapacke.h>
#	include <mkl_cblas.h>
#	include <mkl_service.h>
#else
#	include <cblas.h>
#endif

#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>

using namespace std;

void Rescale(MTPR_trainer& trainer, MLMTPR& mtpr, const std::vector<Neighborhoods>* training_neighborhoods);

namespace {

std::string CurrentTimestamp()
{
	std::time_t now = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
	return std::string(buf);
}

std::string CurrentFilenameTimestamp()
{
	std::time_t now = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&now));
	return std::string(buf);
}

constexpr std::size_t kNeighborhoodCacheBudgetBytesPerRank = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr int kOptimizerReducePrefix = 1;
constexpr int kAcceptedDiagDoubleCount = 11;
constexpr int kAcceptedDiagCountCount = 3;
constexpr int kTraceFlushInterval = 16;
constexpr std::uint64_t kCoordinateProbeSeed = 0x9e3779b97f4a7c15ULL;

std::uint64_t SplitMix64(std::uint64_t value)
{
	value += 0x9e3779b97f4a7c15ULL;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

double CoordinateProbeSign(int probe,
	                       int rank,
	                       std::size_t configuration_index,
	                       std::uint64_t output_index)
{
	std::uint64_t key = kCoordinateProbeSeed;
	key = SplitMix64(key ^ static_cast<std::uint64_t>(probe + 1));
	key = SplitMix64(key ^ static_cast<std::uint64_t>(rank + 1));
	key = SplitMix64(key ^ static_cast<std::uint64_t>(configuration_index + 1));
	key = SplitMix64(key ^ (output_index + 1));
	return (key & 1ULL) == 0ULL ? -1.0 : 1.0;
}

double MedianPositive(std::vector<double> values)
{
	values.erase(
		std::remove_if(
			values.begin(),
			values.end(),
			[](double value) { return !std::isfinite(value) || value <= 0.0; }),
		values.end());
	if (values.empty())
		return 0.0;
	const std::size_t middle = values.size() / 2;
	std::nth_element(values.begin(), values.begin() + middle, values.end());
	const double upper = values[middle];
	if ((values.size() & 1U) != 0U)
		return upper;
	std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
	return 0.5 * (values[middle - 1] + upper);
}

std::size_t UpperTrianglePackedSize(int n)
{
	return static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) / 2;
}

void PackUpperTriangle(const double* matrix, int n, std::vector<double>& packed)
{
	packed.resize(UpperTrianglePackedSize(n));
	std::size_t pos = 0;
	for (int i = 0; i < n; ++i) {
		const double* row = matrix + static_cast<std::size_t>(i) * n;
		for (int j = i; j < n; ++j)
			packed[pos++] = row[j];
	}
}

void UnpackUpperTriangle(const std::vector<double>& packed, int n, double* matrix)
{
	std::size_t pos = 0;
	for (int i = 0; i < n; ++i) {
		double* row = matrix + static_cast<std::size_t>(i) * n;
		for (int j = i; j < n; ++j)
			row[j] = packed[pos++];
	}
}

double EstimateConfigurationVolume(const Configuration& cfg)
{
	double volume = std::abs(cfg.lattice.det());
	if (volume > 1.0e-12)
		return volume;

	if (cfg.size() == 0)
		return 0.0;

	Vector3 min_pos = cfg.pos(0);
	Vector3 max_pos = cfg.pos(0);
	for (int i = 1; i < cfg.size(); ++i) {
		for (int a = 0; a < 3; ++a) {
			min_pos[a] = std::min(min_pos[a], cfg.pos(i, a));
			max_pos[a] = std::max(max_pos[a], cfg.pos(i, a));
		}
	}

	const double dx = std::max(1.0e-6, max_pos[0] - min_pos[0]);
	const double dy = std::max(1.0e-6, max_pos[1] - min_pos[1]);
	const double dz = std::max(1.0e-6, max_pos[2] - min_pos[2]);
	return dx * dy * dz;
}

std::size_t EstimateNeighborhoodCacheBytes(const std::vector<Configuration>& configs, double cutoff)
{
	const double sphere_volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;
	long double total_bytes = 0.0L;
	for (const Configuration& cfg : configs) {
		if (cfg.size() == 0)
			continue;
		const double volume = EstimateConfigurationVolume(cfg);
		const double number_density = volume > 1.0e-12 ? static_cast<double>(cfg.size()) / volume : 0.0;
		const double expected_neighbors = std::max(1.0, sphere_volume * number_density);
		const long double entries = static_cast<long double>(cfg.size()) * expected_neighbors;
		total_bytes += static_cast<long double>(cfg.size()) * 160.0L;
		total_bytes += entries * 96.0L;
	}
	if (total_bytes < 0.0L)
		return 0;
	if (total_bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
		return std::numeric_limits<std::size_t>::max();
	return static_cast<std::size_t>(total_bytes);
}

bool ShouldCacheNeighborhoods(const std::vector<Configuration>& configs, double cutoff, std::size_t* estimated_bytes = nullptr)
{
	const std::size_t bytes = EstimateNeighborhoodCacheBytes(configs, cutoff);
	if (estimated_bytes != nullptr)
		*estimated_bytes = bytes;
	const char* env = std::getenv("MLIP_CACHE_NEIGHBORHOODS");
	if (env != nullptr) {
		if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0)
			return false;
		if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0)
			return true;
	}
	return bytes <= kNeighborhoodCacheBudgetBytesPerRank;
}

bool TryBuildNeighborhoods(const std::vector<Configuration>& configs,
						   double cutoff,
						   std::vector<Neighborhoods>& neighborhoods_out)
{
	try {
		neighborhoods_out.clear();
		neighborhoods_out.reserve(configs.size());
		for (const Configuration& cfg : configs)
			neighborhoods_out.emplace_back(cfg, cutoff);
		return true;
	}
	catch (const std::bad_alloc&) {
		neighborhoods_out.clear();
		neighborhoods_out.shrink_to_fit();
		return false;
	}
}

int FirstNonFinite(const double* values, int count)
{
	for (int i = 0; i < count; ++i)
		if (!std::isfinite(values[i]))
			return i;
	return -1;
}

std::string NonFiniteMessage(const std::string& name, int index)
{
	std::ostringstream oss;
	oss << name << " contains a non-finite value";
	if (index >= 0)
		oss << " at index " << index;
	return oss.str();
}

bool IsFiniteArray(const double* values, int count)
{
	return FirstNonFinite(values, count) < 0;
}

void RequireFiniteArray(const double* values, int count, const std::string& name)
{
	const int bad_index = FirstNonFinite(values, count);
	if (bad_index >= 0)
		ERROR(NonFiniteMessage(name, bad_index));
}

bool EnvFlagEnabled(const char* name)
{
	const char* value = std::getenv(name);
	return value != nullptr
	    && value[0] != '\0'
	    && std::strcmp(value, "0") != 0
	    && std::strcmp(value, "false") != 0
	    && std::strcmp(value, "FALSE") != 0;
}

int SuggestedLinearSolveThreads(int n)
{
#ifdef MLIP_INTEL_MKL
	const char* env = std::getenv("MLIP_LINEAR_SOLVE_THREADS");
	if (env != nullptr) {
		int parsed = std::atoi(env);
		if (parsed > 0)
			return parsed;
	}
	if (n >= 1000 && n <= 5000)
		return 8;
#endif
	return 1;
}

void SolveSLAEGaussian(int n, double* matrix, double* rhs)
{
	for (int i = 0; i < (n - 1); i++) {
		for (int j = (i + 1); j < n; j++) {
			double ratio = matrix[j*n + i] / matrix[i*n + i];
			for (int count = i; count < n; count++)
				matrix[j*n + count] -= ratio * matrix[i*n + count];
			rhs[j] -= ratio * rhs[i];
		}
	}

	rhs[n - 1] /= matrix[(n - 1)*n + (n - 1)];
	for (int i = (n - 2); i >= 0; i--) {
		double temp = rhs[i];
		for (int j = (i + 1); j < n; j++)
			temp -= matrix[i*n + j] * rhs[j];
		rhs[i] = temp / matrix[i*n + i];
	}
}

}


void MTPR_trainer::shift(bool shift_)
{
p_mlmtpr->shift_ = shift_;

}

#ifdef MLIP_MPI
void MTPR_trainer::ConfigureTrainComm(bool has_local_work, int world_rank, int world_size)
{
	if (train_comm_owned_ && train_comm_ != MPI_COMM_NULL) {
		MPI_Comm_free(&train_comm_);
	}
	train_comm_owned_ = false;
	train_rank_active_ = has_local_work;
	train_comm_is_world_ = true;
	train_rank_ = 0;
	train_size_ = 1;
	train_comm_ = MPI_COMM_WORLD;

	if (world_size <= 1) {
		train_rank_active_ = true;
		return;
	}

	MPI_Comm_split(MPI_COMM_WORLD, has_local_work ? 0 : MPI_UNDEFINED, world_rank, &train_comm_);
	int active_count = 0;
	int active_flag = has_local_work ? 1 : 0;
	MPI_Allreduce(&active_flag, &active_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	train_comm_is_world_ = (active_count == world_size);
	if (!has_local_work) {
		train_comm_ = MPI_COMM_NULL;
		train_rank_ = -1;
		train_size_ = 0;
		return;
	}

	train_comm_owned_ = true;
	MPI_Comm_rank(train_comm_, &train_rank_);
	MPI_Comm_size(train_comm_, &train_size_);
}

void MTPR_trainer::BroadcastCoeffsWorld(int root_world_rank)
{
	MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, root_world_rank, MPI_COMM_WORLD);
}
#endif


void MTPR_trainer::LoadWeights(ifstream& ifs)
{

	string next;


	ifs >> wgt_eqtn_forces;
	//cout << "forces coeffitient = " << wgt_eqtn_forces << "\n";
	ifs >> next;
	ifs >> next;

	ifs >> wgt_eqtn_stress;
	//cout << "stress coeffitient = " << wgt_eqtn_stress << "\n";
	ifs >> next;
	ifs >> next;

	ifs >> wgt_eqtn_constr;
	//cout << "stress coeffitient = " << wgt_eqtn_stress << "\n";
	ifs >> next;
	ifs >> next;

}

bool MTPR_trainer::HasFixedAtomicEnergies() const
{
	return !fixed_atomic_energies.empty();
}

void MTPR_trainer::ValidateFixedAtomicEnergies() const
{
	if (!HasFixedAtomicEnergies())
		return;
	if (static_cast<int>(fixed_atomic_energies.size()) != p_mlmtpr->species_count)
		ERROR("--atomic-energies count should match species_count");
	for (double value : fixed_atomic_energies)
		if (!std::isfinite(value))
			ERROR("--atomic-energies contains a non-finite value");
	if (fixed_atomic_energy_weight < 0.0 || !std::isfinite(fixed_atomic_energy_weight))
		ERROR("--atomic-energy-weight should be a finite non-negative value");
}

double MTPR_trainer::FixedAtomicEnergySum(const Configuration& cfg) const
{
	double energy = 0.0;
	for (int i = 0; i < cfg.size(); ++i) {
		const int type = cfg.type(i);
		if (type < 0 || type >= static_cast<int>(fixed_atomic_energies.size()))
			ERROR("Configuration atom type is out of range for --atomic-energies");
		energy += fixed_atomic_energies[type];
	}
	return energy;
}

void MTPR_trainer::ApplyFixedAtomicEnergyGauge(int n)
{
	if (!HasFixedAtomicEnergies())
		return;
	ValidateFixedAtomicEnergies();
	const int C = p_mlmtpr->species_count;
	for (int type = 0; type < C; ++type) {
		const int active_type = LinearSolveActiveIndex(type);
		if (active_type < 0)
			continue;
		for (int j = 0; j < n; ++j) {
			quad_opt_matr[active_type * n + j] = 0.0;
			quad_opt_matr[j * n + active_type] = 0.0;
		}
		quad_opt_matr[active_type * n + active_type] = 1.0;
		quad_opt_vec[active_type] = 1.0;
	}
}

bool MTPR_trainer::LinearColumnActiveForStage(int column) const
{
	if (p_mlmtpr == nullptr || !p_mlmtpr->TwoLayerResidualEnabled())
		return true;

	const int linear_count = p_mlmtpr->LinearCoeffCount();
	const int e0_count = p_mlmtpr->TwoLayerResidualE0CoeffCount();
	if (column < 0 || column >= linear_count + e0_count)
		return false;

	switch (two_layer_residual_stage) {
	case kResidualStageE0:
		return column < p_mlmtpr->species_count || column >= linear_count;
	case kResidualStageE1:
		return column >= p_mlmtpr->species_count && column < linear_count;
	case kResidualStageFull:
	default:
		return true;
	}
}

void MTPR_trainer::PrepareLinearSolveColumns(bool readout_residual_only)
{
	if (p_mlmtpr == nullptr)
		p_mlmtpr = dynamic_cast<MLMTPR*>(p_mlip);
	if (p_mlmtpr == nullptr)
		ERROR("MTPR_trainer::PrepareLinearSolveColumns requires an MLMTPR potential");

	const int full_n = p_mlmtpr->LinearEquationCount();
	const int readout_begin = p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
	const int readout_count = p_mlmtpr->TwoLayerGateReadoutCoeffCount();
	if (readout_residual_only && readout_count <= 0)
		ERROR("Readout-only linear solve requires an enabled gate readout");
	linear_solve_columns_.clear();
	linear_solve_full_to_active_.assign(full_n, -1);
	for (int column = 0; column < full_n; ++column) {
		if (readout_residual_only
		    && (column < readout_begin
		        || column >= readout_begin + readout_count))
			continue;
		if (!LinearColumnActiveForStage(column))
			continue;
		const int active_index = static_cast<int>(linear_solve_columns_.size());
		linear_solve_columns_.push_back(column);
		linear_solve_full_to_active_[column] = active_index;
	}
	if (linear_solve_columns_.empty())
		ERROR("MTPR_trainer::PrepareLinearSolveColumns found no active linear columns");
}

int MTPR_trainer::LinearSolveColumnCount() const
{
	if (linear_solve_columns_.empty()) {
		if (p_mlmtpr == nullptr)
			return 0;
		return p_mlmtpr->LinearEquationCount();
	}
	return static_cast<int>(linear_solve_columns_.size());
}

int MTPR_trainer::LinearSolveActiveIndex(int column) const
{
	if (column < 0 || column >= static_cast<int>(linear_solve_full_to_active_.size()))
		return -1;
	return linear_solve_full_to_active_[column];
}

bool MTPR_trainer::LinearSolveColumnActive(int column) const
{
	return LinearSolveActiveIndex(column) >= 0;
}

bool MTPR_trainer::CoeffActiveForStage(int coeff_index) const
{
	if (p_mlmtpr == nullptr || !p_mlmtpr->TwoLayerResidualEnabled())
		return true;

	switch (two_layer_residual_stage) {
	case kResidualStageE0: {
		const int base_end = p_mlmtpr->BaseNonlinearCoeffCount();
		const int e0_begin = p_mlmtpr->TwoLayerResidualE0CoeffOffset();
		const int e0_end = e0_begin + p_mlmtpr->TwoLayerResidualE0CoeffCount();
		const int linear_begin = p_mlmtpr->LinearCoeffOffset();
		const int species_linear_end = linear_begin + p_mlmtpr->species_count;
		return coeff_index < base_end
		    || (coeff_index >= e0_begin && coeff_index < e0_end)
		    || (coeff_index >= linear_begin && coeff_index < species_linear_end);
	}
		case kResidualStageE1: {
			const int shared_radial_scal_begin = p_mlmtpr->species_count;
			const int shared_radial_scal_end = p_mlmtpr->RadialCoeffOffset();
			const int gate_radial_begin = p_mlmtpr->TwoLayerGateRadialCoeffOffset();
			const int gate_radial_end = gate_radial_begin + p_mlmtpr->TwoLayerGateRadialCoeffCount();
			const int gate_additive_begin = p_mlmtpr->TwoLayerGateAdditiveCoeffOffset();
		const int gate_additive_end =
			gate_additive_begin + p_mlmtpr->TwoLayerGateAdditiveCoeffCount();
		const int gate_type_begin = p_mlmtpr->TwoLayerGateTypeCoeffOffset();
		const int gate_type_end =
			gate_type_begin + p_mlmtpr->TwoLayerGateTypeCoeffCount();
		const int gate_weight_begin = p_mlmtpr->TwoLayerGateWeightOffset();
		const int gate_weight_end = gate_weight_begin + p_mlmtpr->TwoLayerGateWeightCount();
			const int gate_body_mix_begin = p_mlmtpr->TwoLayerGateBodyMixWeightOffset();
			const int gate_body_mix_end =
				gate_body_mix_begin + p_mlmtpr->TwoLayerGateBodyMixWeightCount();
			const int gate_edge_l1_begin = p_mlmtpr->TwoLayerGateEdgeL1WeightOffset();
			const int gate_edge_l1_end =
				gate_edge_l1_begin + p_mlmtpr->TwoLayerGateEdgeL1WeightCount();
			const int second_gate_radial_begin = p_mlmtpr->SecondLayerGateRadialCoeffOffset();
			const int second_gate_radial_end =
				second_gate_radial_begin + p_mlmtpr->SecondLayerGateRadialCoeffCount();
			const int second_gate_additive_begin = p_mlmtpr->SecondLayerGateAdditiveCoeffOffset();
			const int second_gate_additive_end =
				second_gate_additive_begin + p_mlmtpr->SecondLayerGateAdditiveCoeffCount();
			const int second_gate_type_begin = p_mlmtpr->SecondLayerGateTypeCoeffOffset();
			const int second_gate_type_end =
				second_gate_type_begin + p_mlmtpr->SecondLayerGateTypeCoeffCount();
			const int second_gate_weight_begin = p_mlmtpr->SecondLayerGateWeightOffset();
			const int second_gate_weight_end =
				second_gate_weight_begin + p_mlmtpr->SecondLayerGateWeightCount();
			const int second_gate_body_mix_begin =
				p_mlmtpr->SecondLayerGateBodyMixWeightOffset();
			const int second_gate_body_mix_end =
				second_gate_body_mix_begin + p_mlmtpr->SecondLayerGateBodyMixWeightCount();
			const int linear_begin = p_mlmtpr->LinearCoeffOffset();
			const int linear_end = linear_begin + p_mlmtpr->LinearCoeffCount();
			return (coeff_index >= shared_radial_scal_begin
			        && coeff_index < shared_radial_scal_end)
			    || (coeff_index >= gate_radial_begin && coeff_index < gate_radial_end)
			    || (coeff_index >= gate_additive_begin && coeff_index < gate_additive_end)
			    || (coeff_index >= gate_type_begin && coeff_index < gate_type_end)
			    || (coeff_index >= gate_weight_begin && coeff_index < gate_weight_end)
			    || (coeff_index >= gate_body_mix_begin && coeff_index < gate_body_mix_end)
			    || (coeff_index >= gate_edge_l1_begin && coeff_index < gate_edge_l1_end)
			    || (coeff_index >= second_gate_radial_begin
			        && coeff_index < second_gate_radial_end)
			    || (coeff_index >= second_gate_additive_begin
			        && coeff_index < second_gate_additive_end)
			    || (coeff_index >= second_gate_type_begin
			        && coeff_index < second_gate_type_end)
			    || (coeff_index >= second_gate_weight_begin
			        && coeff_index < second_gate_weight_end)
			    || (coeff_index >= second_gate_body_mix_begin
			        && coeff_index < second_gate_body_mix_end)
			    || (coeff_index >= linear_begin + p_mlmtpr->species_count && coeff_index < linear_end);
		}
	case kResidualStageFull:
	default:
		return true;
	}
}

double MTPR_trainer::LinearColumnCurrentCoeff(int column) const
{
	const int linear_count = p_mlmtpr->LinearCoeffCount();
	const int linear_begin = p_mlmtpr->LinearCoeffOffset();
	if (column < linear_count) {
		if (column < p_mlmtpr->species_count) {
			return p_mlmtpr->regression_coeffs[column]
			     + p_mlmtpr->regression_coeffs[linear_begin + column];
		}
		const int scalar_index = column - p_mlmtpr->species_count;
		double value = p_mlmtpr->regression_coeffs[linear_begin + column];
		if (scalar_index >= 0 && scalar_index < static_cast<int>(p_mlmtpr->linear_mults.size()))
			value *= p_mlmtpr->linear_mults[scalar_index];
		return value;
	}

	const int e0_index = column - linear_count;
	const int e0_begin = p_mlmtpr->TwoLayerResidualE0CoeffOffset();
	if (e0_index >= 0 && e0_index < p_mlmtpr->TwoLayerResidualE0CoeffCount()) {
		double value = p_mlmtpr->regression_coeffs[e0_begin + e0_index];
		if (e0_index < static_cast<int>(p_mlmtpr->linear_mults.size()))
			value *= p_mlmtpr->linear_mults[e0_index];
		return value;
	}
	const int readout_index =
		column - p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
	if (readout_index >= 0
	    && readout_index < p_mlmtpr->TwoLayerGateReadoutCoeffCount())
		return p_mlmtpr->TwoLayerGateReadoutCoeff(readout_index);
	ERROR("MTPR linear column is out of range");
	return 0.0;
}

void MTPR_trainer::ProjectLinearRowToActive(const double* full_row,
                                            int full_stride,
                                            double& rhs,
                                            double* active_row)
{
	const int active_n = LinearSolveColumnCount();
	for (int active_index = 0; active_index < active_n; ++active_index) {
		const int full_column = linear_solve_columns_[active_index];
		active_row[active_index] = full_row[full_column * full_stride];
	}

	const int full_n = p_mlmtpr->LinearEquationCount();
	for (int column = 0; column < full_n; ++column) {
		if (LinearSolveColumnActive(column))
			continue;
		const double value = full_row[column * full_stride];
		if (value != 0.0)
			rhs -= value * LinearColumnCurrentCoeff(column);
	}
}

void MTPR_trainer::BuildReadoutResidualRow(
	const double* full_row,
	int full_stride,
	double reference_value,
	double predicted_value,
	double& rhs,
	double* active_row) const
{
	rhs = reference_value - predicted_value;
	const int readout_begin = p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
	const int readout_end =
		readout_begin + p_mlmtpr->TwoLayerGateReadoutCoeffCount();
	const int active_n = LinearSolveColumnCount();
	for (int active_index = 0; active_index < active_n; ++active_index) {
		const int column = linear_solve_columns_[active_index];
		if (column < readout_begin || column >= readout_end)
			ERROR("Readout residual row contains a non-readout active column");
		const double value = full_row[column * full_stride];
		active_row[active_index] = value;
		// P_current = P_frozen + X * phi_current.  Move the current
		// readout contribution back to the right-hand side so the solve
		// returns the absolute optimum phi, not merely a correction.
		rhs += value * LinearColumnCurrentCoeff(column);
	}
}



void MTPR_trainer::ClearSLAE(bool readout_residual_only)
{
	if (p_mlmtpr == nullptr)
		p_mlmtpr = dynamic_cast<MLMTPR*>(p_mlip);
	if (p_mlmtpr == nullptr)
		ERROR("MTPR_trainer::ClearSLAE requires an MLMTPR potential");

	PrepareLinearSolveColumns(readout_residual_only);
	int n = LinearSolveColumnCount();	// Active matrix size

	if (quad_opt_allocated_n != n || quad_opt_vec == nullptr || quad_opt_matr == nullptr) {
		delete[] quad_opt_vec;
		delete[] quad_opt_matr;
		quad_opt_vec = new double[n];
		quad_opt_matr = new double[n*n];
		quad_opt_allocated_n = n;
	}

	quad_opt_eqn_count = 0;
	quad_opt_scalar = 0.0;

	memset(quad_opt_vec, 0, n * sizeof(double));
	memset(quad_opt_matr, 0, n * n * sizeof(double));
}

MTPR_trainer::~MTPR_trainer()
{
#ifdef MLIP_MPI
	if (train_comm_owned_ && train_comm_ != MPI_COMM_NULL)
		MPI_Comm_free(&train_comm_);
#endif
	delete[] quad_opt_vec;
	delete[] quad_opt_matr;
	quad_opt_vec = nullptr;
	quad_opt_matr = nullptr;
	quad_opt_allocated_n = 0;
}

void MTPR_trainer::SymmetrizeSLAE()
{
	int n = LinearSolveColumnCount();		// Active matrix size

	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			quad_opt_matr[j*n + i] = quad_opt_matr[i*n + j];
}

void MTPR_trainer::SolveSLAE()
{

	SymmetrizeSLAE();

	double gammareg = 1e-13;

	int n = LinearSolveColumnCount();		// Active matrix size

	for (int i = 0; i < n; i++)
		quad_opt_matr[i*n + i] += gammareg*(1 + quad_opt_matr[i*n + i]);
	ApplyFixedAtomicEnergyGauge(n);

	p_mlmtpr->LinCoeff();	/// TO MOVE TO MLMTPR

#ifdef MLIP_INTEL_MKL
	if (EnvFlagEnabled("SUS2_SH_FORCE_GAUSSIAN_SOLVE")) {
		SolveSLAEGaussian(n, quad_opt_matr, quad_opt_vec);
	} else {
	std::vector<double> matrix_work(quad_opt_matr, quad_opt_matr + n * n);
	std::vector<double> rhs_work(quad_opt_vec, quad_opt_vec + n);
	const int solve_threads = SuggestedLinearSolveThreads(n);
	int previous_threads = 0;
	if (solve_threads > 1)
		previous_threads = mkl_set_num_threads_local(solve_threads);

	int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', n, 1,
							 matrix_work.data(), n,
							 rhs_work.data(), 1);
	if (info != 0) {
		matrix_work.assign(quad_opt_matr, quad_opt_matr + n * n);
		rhs_work.assign(quad_opt_vec, quad_opt_vec + n);
		std::vector<int> ipiv(n);
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1,
							 matrix_work.data(), n,
							 ipiv.data(),
							 rhs_work.data(), 1);
	}

	if (info != 0) {
		matrix_work.assign(quad_opt_matr, quad_opt_matr + n * n);
		rhs_work.assign(quad_opt_vec, quad_opt_vec + n);
		SolveSLAEGaussian(n, matrix_work.data(), rhs_work.data());
	}

	memcpy(quad_opt_matr, matrix_work.data(), n * n * sizeof(double));
	memcpy(quad_opt_vec, rhs_work.data(), n * sizeof(double));
	if (solve_threads > 1)
		mkl_set_num_threads_local(previous_threads);
	}
#else
	SolveSLAEGaussian(n, quad_opt_matr, quad_opt_vec);
#endif

	const int linear_count = p_mlmtpr->LinearCoeffCount();
	const int e0_count = p_mlmtpr->TwoLayerResidualE0CoeffCount();
	const int readout_count = p_mlmtpr->TwoLayerGateReadoutCoeffCount();
	if (static_cast<int>(p_mlmtpr->linear_coeffs.size()) != linear_count)
		p_mlmtpr->linear_coeffs.resize(linear_count);
	for (int active_index = 0; active_index < n; ++active_index) {
		const int column = linear_solve_columns_[active_index];
		if (column < linear_count)
			p_mlmtpr->linear_coeffs[column] = quad_opt_vec[active_index];
	}
	if (e0_count > 0) {
		if (static_cast<int>(p_mlmtpr->two_layer_residual_e0_coeffs_.size()) != e0_count)
			p_mlmtpr->two_layer_residual_e0_coeffs_.assign(e0_count, 0.0);
		for (int active_index = 0; active_index < n; ++active_index) {
			const int column = linear_solve_columns_[active_index];
			if (column >= linear_count && column < linear_count + e0_count)
				p_mlmtpr->two_layer_residual_e0_coeffs_[column - linear_count] =
					quad_opt_vec[active_index];
		}
	}
	if (readout_count > 0) {
		if (static_cast<int>(p_mlmtpr->two_layer_gate_readout_coeffs_.size())
		    != readout_count)
			p_mlmtpr->two_layer_gate_readout_coeffs_.assign(readout_count, 0.0);
		const int readout_column_offset =
			p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
		for (int active_index = 0; active_index < n; ++active_index) {
			const int readout_index =
				linear_solve_columns_[active_index] - readout_column_offset;
			if (readout_index >= 0 && readout_index < readout_count)
				p_mlmtpr->two_layer_gate_readout_coeffs_[readout_index] =
					quad_opt_vec[active_index];
		}
	}
	//double e0_ = 0;
	//for (int i = 0; i < p_mlmtpr->species_count; i++)
	//{
	//	e0_ += p_mlmtpr->regression_coeffs[i] / p_mlmtpr->species_count;
	//}

//	double e_0 = 0;
//	for (int i = 0; i < p_mlmtpr->species_count; i++) {
//		e_0 += p_mlmtpr->linear_coeffs[i] / p_mlmtpr->species_count;
//	}
//	std::random_device rand_device;
//	std::default_random_engine generator(rand_device());
//	std::uniform_real_distribution<> uniform(-0.05, 0.05);
	for (int i = 0; i < p_mlmtpr->species_count; i++) {
		if (LinearSolveColumnActive(i)) {
			if (HasFixedAtomicEnergies()) {
				p_mlmtpr->linear_coeffs[i] = 1.0;
				p_mlmtpr->regression_coeffs[i] = fixed_atomic_energies[i] - p_mlmtpr->linear_coeffs[i];
			} else {
				p_mlmtpr->regression_coeffs[i] = p_mlmtpr->linear_coeffs[i] - 1.0;
				p_mlmtpr->linear_coeffs[i] = 1.0;
			}
		}
	}
	const int e0_offset = p_mlmtpr->TwoLayerResidualE0CoeffOffset();
	for (int i = 0; i < e0_count; ++i) {
		if (!LinearSolveColumnActive(linear_count + i))
			continue;
		if (i < static_cast<int>(p_mlmtpr->linear_mults.size()))
			p_mlmtpr->two_layer_residual_e0_coeffs_[i] /=
				p_mlmtpr->linear_mults[i] * 1.0;
		p_mlmtpr->regression_coeffs[e0_offset + i] =
			p_mlmtpr->two_layer_residual_e0_coeffs_[i];
	}
	const int readout_offset = p_mlmtpr->TwoLayerGateReadoutCoeffOffset();
	const int readout_column_offset =
		p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
	for (int i = 0; i < readout_count; ++i) {
		if (!LinearSolveColumnActive(readout_column_offset + i))
			continue;
		p_mlmtpr->regression_coeffs[readout_offset + i] =
			p_mlmtpr->two_layer_gate_readout_coeffs_[i];
	}
	const int linear_offset = p_mlmtpr->LinearCoeffOffset();
	for (int i = 0; i < linear_count; i++) {
		if (!LinearSolveColumnActive(i))
			continue;
		p_mlmtpr->regression_coeffs[linear_offset + i] = p_mlmtpr->linear_coeffs[i];
	}
	for (int i = linear_offset + p_mlmtpr->species_count;
	     i < linear_offset + linear_count; i++) {
		if (!LinearSolveColumnActive(i - linear_offset))
			continue;
		p_mlmtpr->regression_coeffs[i] /= p_mlmtpr->linear_mults[i - (linear_offset + p_mlmtpr->species_count)] * 1.0;
	}
	p_mlmtpr->LinCoeff();


}


void MTPR_trainer::AddToSLAE(Configuration& cfg,
	                         double weight,
	                         const Neighborhoods* neighborhoods,
	                         bool readout_residual_only)
{
	if (cfg.size() == 0)				// 
		return;

	const int full_n = p_mlmtpr->LinearEquationCount();		// Full linear component count
	const int n = LinearSolveColumnCount();		// Active matrix size
      //  {std::cout<<n<<" "<<std::endl;}
     //   {std::cout<<n<<" "<<  (int)p_mlmtpr->energy_cmpnts.size() <<std::endl;}
	double wgt_energy = wgt_eqtn_energy / cfg.size();
	double wgt_forces = wgt_eqtn_forces;
	double wgt_stress = wgt_eqtn_stress / cfg.size();
	const bool need_forces = (wgt_eqtn_forces > 0) && cfg.has_forces();
	const bool need_stress = (wgt_eqtn_stress > 0) && cfg.has_stresses();
	Configuration readout_prediction;
	if (readout_residual_only) {
		if (!p_mlmtpr->TwoLayerGateReadoutEnabled())
			ERROR("Readout residual linear solve requires an enabled gate readout");
		if (force_loss_kind != ForceLossKind::L2)
			ERROR("Readout residual linear solve requires --force-loss=l2");
		readout_prediction = cfg;
		if (need_forces || need_stress) {
			if (neighborhoods != nullptr)
				p_mlmtpr->CalcEFS(readout_prediction, *neighborhoods);
			else
				p_mlmtpr->CalcEFS(readout_prediction);
		} else {
			if (neighborhoods != nullptr)
				p_mlmtpr->CalcEnergyAndSiteEnergies(
					readout_prediction, *neighborhoods);
			else
				p_mlmtpr->CalcEnergyAndSiteEnergies(readout_prediction);
		}
	}

	if (neighborhoods != nullptr) {
		if (readout_residual_only)
			p_mlmtpr->CalcEFSComponents(
				cfg, *neighborhoods, need_forces, need_stress, true);
		else
			p_mlmtpr->CalcEFSComponents(
				cfg, *neighborhoods, need_forces, need_stress);
	} else {
		if (readout_residual_only) {
			Neighborhoods local_neighborhoods(cfg, p_mlmtpr->CutOff());
			p_mlmtpr->CalcEFSComponents(
				cfg, local_neighborhoods, need_forces, need_stress, true);
		} else
			p_mlmtpr->CalcEFSComponents(cfg, need_forces, need_stress);
	}
	if (HasFixedAtomicEnergies())
		ValidateFixedAtomicEnergies();
//        int w=p_mlmtpr->energy_cmpnts.size();
      //  {std::cout<<n<<" "<<std::endl;}

	if (weighting == "structures")
	{
		wgt_energy /= cfg.size();
		wgt_stress /= cfg.size();

		wgt_forces /= cfg.size();
	}
	else if (weighting == "molecules")
	{
		wgt_energy *= cfg.size();
		wgt_stress *= cfg.size();
	}

	cout.precision(15);

	int fn = norm_by_forces;
	double d = 0.1;
	double avef = 0;

	if (cfg.has_forces())
		for (int ind = 0; ind < cfg.size(); ind++)
			avef += cfg.force(ind).NormSq() / cfg.size();


	if (cfg.has_energy())
	{
		const double alpha = weight * wgt_energy * d / (d + fn*avef);
		const double* energy_cmpnts = p_mlmtpr->energy_cmpnts;
		double energy_rhs = 0.0;
		if (readout_residual_only) {
			lin_active_row_.resize(n);
			BuildReadoutResidualRow(energy_cmpnts,
			                        1,
			                        cfg.energy,
			                        readout_prediction.energy,
			                        energy_rhs,
			                        lin_active_row_.data());
			energy_cmpnts = lin_active_row_.data();
		} else if (HasFixedAtomicEnergies()) {
			energy_rhs = cfg.energy;
			lin_energy_cmpnts_.assign(p_mlmtpr->energy_cmpnts, p_mlmtpr->energy_cmpnts + full_n);
			for (int type = 0; type < p_mlmtpr->species_count; ++type)
				lin_energy_cmpnts_[type] = 0.0;
			energy_cmpnts = lin_energy_cmpnts_.data();
			energy_rhs -= FixedAtomicEnergySum(cfg);
		} else {
			energy_rhs = cfg.energy;
		}
		if (!readout_residual_only) {
			lin_active_row_.resize(n);
			ProjectLinearRowToActive(
				energy_cmpnts, 1, energy_rhs, lin_active_row_.data());
			energy_cmpnts = lin_active_row_.data();
		}
		cblas_dsyr(CBLAS_ORDER::CblasRowMajor,
			CBLAS_UPLO::CblasUpper,
			n,
			alpha,
			energy_cmpnts, 1,
			quad_opt_matr, n);
		cblas_daxpy(n, alpha * energy_rhs, energy_cmpnts, 1, quad_opt_vec, 1);
		quad_opt_scalar += alpha * energy_rhs * energy_rhs;

		quad_opt_eqn_count += (weight > 0) ? 1 : ((weight < 0) ? -1 : 0);
	}

	if ((wgt_eqtn_forces > 0) && (cfg.has_forces()))
	{
		const double alpha = weight * wgt_forces * d / (d + fn*avef);
			const int force_rows = 3 * cfg.size();
			const bool use_force_block = (n >= 1000 && n <= 5000 && force_rows > 0
				&& !EnvFlagEnabled("SUS2_SH_DISABLE_LINEAR_BLOCK_ACCUM"));
			if (use_force_block) {
				lin_force_block_.resize(static_cast<size_t>(force_rows) * n);
				lin_force_rhs_.resize(force_rows);
			for (int ind = 0; ind < cfg.size(); ind++) {
				for (int a = 0; a < 3; a++) {
					const int row = 3 * ind + a;
					if (readout_residual_only)
						BuildReadoutResidualRow(
							&p_mlmtpr->forces_cmpnts(ind, 0, a),
							3,
							cfg.force(ind, a),
							readout_prediction.force(ind, a),
							lin_force_rhs_[row],
							&lin_force_block_[static_cast<size_t>(row) * n]);
					else {
						lin_force_rhs_[row] = cfg.force(ind, a);
						ProjectLinearRowToActive(
							&p_mlmtpr->forces_cmpnts(ind, 0, a),
							3,
							lin_force_rhs_[row],
							&lin_force_block_[static_cast<size_t>(row) * n]);
					}
				}
			}
			cblas_dsyrk(CBLAS_ORDER::CblasRowMajor,
				CBLAS_UPLO::CblasUpper,
				CBLAS_TRANSPOSE::CblasTrans,
				n, force_rows,
				alpha,
				lin_force_block_.data(), n,
				1.0,
				quad_opt_matr, n);
			cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				force_rows, n,
				alpha,
				lin_force_block_.data(), n,
				lin_force_rhs_.data(), 1,
				1.0,
				quad_opt_vec, 1);
			for (double force_value : lin_force_rhs_)
				quad_opt_scalar += alpha * force_value * force_value;
			quad_opt_eqn_count += force_rows * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
		} else {
			for (int ind = 0; ind < cfg.size(); ind++)
			{
				double force_rhs_sq_sum = 0.0;
				for (int a = 0; a < 3; a++) {
					double* force_cmp = &p_mlmtpr->forces_cmpnts(ind, 0, a);
					double force_rhs = 0.0;
					lin_projected_row_.resize(n);
					if (readout_residual_only)
						BuildReadoutResidualRow(force_cmp,
						                        3,
						                        cfg.force(ind, a),
						                        readout_prediction.force(ind, a),
						                        force_rhs,
						                        lin_projected_row_.data());
					else {
						force_rhs = cfg.force(ind, a);
						ProjectLinearRowToActive(
							force_cmp, 3, force_rhs, lin_projected_row_.data());
					}
					const double* force_row = lin_projected_row_.data();
					const int force_stride = 1;
					cblas_dsyr(CBLAS_ORDER::CblasRowMajor,
						CBLAS_UPLO::CblasUpper,
						n,
						alpha,
						force_row, force_stride,
						quad_opt_matr, n);
					cblas_daxpy(n, alpha * force_rhs, force_row, force_stride, quad_opt_vec, 1);
					force_rhs_sq_sum += force_rhs * force_rhs;
				}

				quad_opt_scalar += alpha * force_rhs_sq_sum;

				quad_opt_eqn_count += 3 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
			}
		}
	}

	if ((wgt_eqtn_stress > 0) && (cfg.has_stresses()))
	{
		const double alpha = weight * wgt_stress;
			const int stress_rows = 9;
			const bool use_stress_block = (n >= 1000 && n <= 5000
				&& !EnvFlagEnabled("SUS2_SH_DISABLE_LINEAR_BLOCK_ACCUM"));
			if (use_stress_block) {
				lin_stress_block_.resize(static_cast<size_t>(stress_rows) * n);
				lin_stress_rhs_.resize(stress_rows);
			int row = 0;
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++) {
					double* stress_cmp = &p_mlmtpr->stress_cmpnts[0][a][b];
					if (readout_residual_only)
						BuildReadoutResidualRow(
							stress_cmp,
							9,
							cfg.stresses[a][b],
							readout_prediction.stresses[a][b],
							lin_stress_rhs_[row],
							&lin_stress_block_[static_cast<size_t>(row) * n]);
					else {
						lin_stress_rhs_[row] = cfg.stresses[a][b];
						ProjectLinearRowToActive(
							stress_cmp,
							9,
							lin_stress_rhs_[row],
							&lin_stress_block_[static_cast<size_t>(row) * n]);
					}
					row++;
				}
			cblas_dsyrk(CBLAS_ORDER::CblasRowMajor,
				CBLAS_UPLO::CblasUpper,
				CBLAS_TRANSPOSE::CblasTrans,
				n, stress_rows,
				alpha,
				lin_stress_block_.data(), n,
				1.0,
				quad_opt_matr, n);
			cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				stress_rows, n,
				alpha,
				lin_stress_block_.data(), n,
				lin_stress_rhs_.data(), 1,
				1.0,
				quad_opt_vec, 1);
			for (double stress_value : lin_stress_rhs_)
				quad_opt_scalar += alpha * stress_value * stress_value;
		} else {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++) {
					double* stress_cmp = &p_mlmtpr->stress_cmpnts[0][a][b];
					double stress_rhs = 0.0;
					lin_projected_row_.resize(n);
					if (readout_residual_only)
						BuildReadoutResidualRow(stress_cmp,
						                        9,
						                        cfg.stresses[a][b],
						                        readout_prediction.stresses[a][b],
						                        stress_rhs,
						                        lin_projected_row_.data());
					else {
						stress_rhs = cfg.stresses[a][b];
						ProjectLinearRowToActive(
							stress_cmp, 9, stress_rhs, lin_projected_row_.data());
					}
					const double* stress_row = lin_projected_row_.data();
					const int stress_stride = 1;
					cblas_dsyr(CBLAS_ORDER::CblasRowMajor,
						CBLAS_UPLO::CblasUpper,
						n,
						alpha,
						stress_row, stress_stride,
						quad_opt_matr, n);
					cblas_daxpy(n, alpha * stress_rhs, stress_row, stress_stride, quad_opt_vec, 1);
					quad_opt_scalar += alpha * stress_rhs * stress_rhs;
				}
		}

		quad_opt_eqn_count += 6 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
	}
}


double* MTPR_trainer::ConstructLinHessian()
{
	ERROR("MTPR_trainer::ConstructLinHessian() requires revision and refactoring!");

	//for (auto& cfg : training_set)
	//	AddForTrain(cfg);

	SymmetrizeSLAE();

	int linsize = p_mlmtpr->LinearEquationCount();

	//int m = (int)training_set.size();
	int M = 1;

	//MPI_Allreduce(&m, &M, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	//cout << "M" << M << endl;

	double* Hess = new double[linsize*linsize];

	for (int i = 0; i < linsize; i++)
		for (int j = 0; j < linsize; j++)
			Hess[i*linsize + j] = 2 * quad_opt_matr[i*linsize + j] / M;


	return Hess;
}

void MTPR_trainer::TrainLinear(int prank,
								   vector<Configuration>& training_set,
								   const std::vector<Neighborhoods>* neighborhoods,
								   const std::string& context,
								   bool readout_residual_only)
{
#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	const int mpi_rank = train_rank_;
	const MPI_Comm train_comm = train_comm_;
#ifdef MLIP_DEV
	int inject_nonroot_radial_ulp = 0;
	if (mpi_rank == 0)
		inject_nonroot_radial_ulp =
			EnvFlagEnabled(
				"SUS2_SH_TEST_NONROOT_RADIAL_ULP_BEFORE_TRAINLINEAR")
			? 1 : 0;
	MPI_Bcast(&inject_nonroot_radial_ulp, 1, MPI_INT, 0, train_comm);
	if (inject_nonroot_radial_ulp != 0
	    && radial_parameterization == "weight-norm"
	    && context == "bfgs refresh step 1") {
		int injected = 0;
		if (mpi_rank != 0) {
			const int coeff_index = p_mlmtpr->RadialCoeffOffset();
			p_mlmtpr->Coeff()[coeff_index] = std::nextafter(
				p_mlmtpr->Coeff()[coeff_index],
				std::numeric_limits<double>::infinity());
			injected = 1;
		}
		int injected_total = 0;
		MPI_Allreduce(
			&injected, &injected_total, 1, MPI_INT, MPI_SUM, train_comm);
		if (mpi_rank == 0)
			std::cout << "[test] injected non-root radial one-ULP drift"
			          << " ranks=" << injected_total << std::endl;
	}
#endif
	// TrainLinear is a fixed-feature solve.  All ranks must therefore build
	// their local normal equations from one bitwise-identical physical model.
	// This matters for weight-normalized radial rows: independently unpacking
	// the same optimizer coordinates can differ in the last bit on heterogeneous
	// CPUs because the chart uses transcendental functions.
	MPI_Bcast(
		&p_mlmtpr->Coeff()[0],
		p_mlmtpr->CoeffCount(),
		MPI_DOUBLE,
		0,
		train_comm);
	RequireFiniteArray(
		p_mlmtpr->Coeff(),
		p_mlmtpr->CoeffCount(),
		(context.empty() ? "TrainLinear" : "TrainLinear[" + context + "]")
			+ " canonical coefficients");
#else
	const int mpi_rank = prank;
#endif
	const long long local_cfg_count = static_cast<long long>(training_set.size());
	long long local_atom_count = 0;
	for (const Configuration& cfg : training_set)
		local_atom_count += cfg.size();
	const std::string prefix = context.empty() ? "TrainLinear" : "TrainLinear[" + context + "]";
	const int full_n = p_mlmtpr->LinearEquationCount();		// Full linear component count
	const std::vector<double> coeffs_before =
		readout_residual_only ? p_mlmtpr->regression_coeffs
		                      : std::vector<double>();
	std::vector<int> preserved_radial_indices;
	std::vector<double> preserved_radial_values;
	if (radial_parameterization == "weight-norm") {
		const int row_size = p_mlmtpr->p_RadialBasis->rb_size;
		const int outer_begin = p_mlmtpr->RadialCoeffOffset();
		const int outer_stride = p_mlmtpr->RadialCoeffBlockSize();
		preserved_radial_indices.reserve(
			2 * p_mlmtpr->radial_func_count * row_size);
		for (int mu = 0; mu < p_mlmtpr->radial_func_count; ++mu)
			for (int xi = 0; xi < row_size; ++xi)
				preserved_radial_indices.push_back(
					outer_begin + mu * outer_stride + xi);
		if (p_mlmtpr->TwoLayerGateUsesSharedRadial()) {
			const int gate1_begin = p_mlmtpr->TwoLayerGateRadialCoeffOffset();
			for (int mu = 0; mu < p_mlmtpr->radial_func_count; ++mu)
				for (int xi = 0; xi < row_size; ++xi)
					preserved_radial_indices.push_back(
						gate1_begin + mu * row_size + xi);
		}
		preserved_radial_values.reserve(preserved_radial_indices.size());
		for (int coeff_index : preserved_radial_indices)
			preserved_radial_values.push_back(p_mlmtpr->Coeff()[coeff_index]);
	}

	if (!readout_residual_only && radial_parameterization != "weight-norm") {
#ifdef MLIP_MPI
		if (mpi_rank == 0)
			p_mlmtpr->Orthogonalize();
		MPI_Bcast(
			&p_mlmtpr->Coeff()[0],
			p_mlmtpr->CoeffCount(),
			MPI_DOUBLE,
			0,
			train_comm);
#else
		p_mlmtpr->Orthogonalize();
#endif
	}

	ClearSLAE(readout_residual_only);
	int n = LinearSolveColumnCount();		// Active matrix size
	if (readout_residual_only
	    && n != p_mlmtpr->TwoLayerGateReadoutCoeffCount())
		ERROR(prefix + " did not select exactly the readout coefficient block");

	double build_start = 0.0;
#ifdef MLIP_MPI
	build_start = MPI_Wtime();
#else
	build_start = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
	for (size_t i = 0; i < training_set.size(); ++i)
		AddToSLAE(training_set[i],
		          1.0,
		          neighborhoods == nullptr ? nullptr : &(*neighborhoods)[i],
		          readout_residual_only);
	double build_seconds = 0.0;
#ifdef MLIP_MPI
	build_seconds = MPI_Wtime() - build_start;
#else
	build_seconds = static_cast<double>(clock()) / CLOCKS_PER_SEC - build_start;
#endif
	long long global_cfg_count = local_cfg_count;
#ifdef MLIP_MPI
	// The scalar-head regularizer is defined once per global training
	// configuration.  Its normal-equation diagonal therefore needs the global
	// count for the full linear solve too.  Keep the historical zero-L2 full
	// solve free of this extra collective.
	if (readout_residual_only || scalar_head_l2_regularization != 0.0)
		MPI_Allreduce(&local_cfg_count,
		              &global_cfg_count,
		              1,
		              MPI_LONG_LONG_INT,
		              MPI_SUM,
		              train_comm);
#endif
	double readout_ridge_diagonal = 0.0;
	std::vector<double> readout_objective_matrix;
	std::vector<double> readout_objective_rhs;
	double readout_objective_before = 0.0;
	auto add_scalar_head_ridge = [&]() {
		if (!std::isfinite(scalar_head_l2_regularization)
		    || scalar_head_l2_regularization < 0.0)
			ERROR("--scalar-head-l2 should be finite and >= 0");
		if (scalar_head_l2_regularization == 0.0)
			return;

		const int scalar_begin = p_mlmtpr->species_count;
		const int scalar_end = scalar_begin + p_mlmtpr->alpha_scalar_moments;
		const int readout_count = p_mlmtpr->TwoLayerGateReadoutCoeffCount();
		const int readout_begin =
			p_mlmtpr->TwoLayerGateReadoutLinearColumnOffset();
		const double global_scale = static_cast<double>(global_cfg_count)
			* scalar_head_l2_regularization;
		for (int active_index = 0; active_index < n; ++active_index) {
			const int column = linear_solve_columns_[active_index];
			double ridge = 0.0;
			if (column >= scalar_begin && column < scalar_end) {
				// The full normal equation solves for the physical scalar-head
				// coefficient.  linear_mults is applied only when the solved value
				// is stored back into regression_coeffs, so it must not appear here.
				ridge = global_scale
					/ static_cast<double>(p_mlmtpr->alpha_scalar_moments);
			} else if (readout_count > 0
			           && column >= readout_begin
			           && column < readout_begin + readout_count) {
				ridge = global_scale / static_cast<double>(readout_count);
				readout_ridge_diagonal = ridge;
			}
			quad_opt_matr[active_index * n + active_index] += ridge;
		}
	};
	auto readout_quadratic_value = [&](const std::vector<double>& phi) {
		double linear = 0.0;
		double quadratic = 0.0;
		for (int i = 0; i < n; ++i) {
			linear += readout_objective_rhs[i] * phi[i];
			for (int j = 0; j < n; ++j)
				quadratic += phi[i]
					* readout_objective_matrix[i * n + j] * phi[j];
		}
		return quad_opt_scalar - 2.0 * linear + quadratic;
	};
	auto capture_readout_objective = [&]() {
		if (!readout_residual_only)
			return;
		SymmetrizeSLAE();
		readout_objective_matrix.assign(
			quad_opt_matr, quad_opt_matr + static_cast<size_t>(n) * n);
		readout_objective_rhs.assign(quad_opt_vec, quad_opt_vec + n);
		std::vector<double> phi_before(n, 0.0);
		for (int i = 0; i < n; ++i)
			phi_before[i] = LinearColumnCurrentCoeff(linear_solve_columns_[i]);
		readout_objective_before = readout_quadratic_value(phi_before);
	};
	auto validate_readout_objective = [&]() {
		if (!readout_residual_only)
			return true;
		std::vector<double> phi_after(n, 0.0);
		for (int i = 0; i < n; ++i)
			phi_after[i] = LinearColumnCurrentCoeff(linear_solve_columns_[i]);
		const double objective_after = readout_quadratic_value(phi_after);
		const double tolerance =
			1.0e-10 * std::max(1.0, std::abs(readout_objective_before));
		if (mpi_rank == 0) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "readout-only linear objective"
			          << " cfg_global=" << global_cfg_count
			          << " active_columns=" << n
			          << " full_columns=" << full_n
			          << " head_l2=" << scalar_head_l2_regularization
			          << " ridge_diagonal=" << readout_ridge_diagonal
			          << " before=" << readout_objective_before
			          << " after=" << objective_after
			          << " decrease="
			          << (readout_objective_before - objective_after)
			          << std::endl;
		}
		return std::isfinite(readout_objective_before)
		    && std::isfinite(objective_after)
		    && objective_after <= readout_objective_before + tolerance;
	};

#ifdef MLIP_MPI

	int linear_status = 0;
	std::string linear_error;
	double reduce_start = MPI_Wtime();
	const std::size_t packed_matrix_size = UpperTrianglePackedSize(n);
	if (packed_matrix_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
		ERROR(prefix + " normal matrix is too large for packed MPI reduction.");
	std::vector<double> packed_upper_matrix;
	PackUpperTriangle(quad_opt_matr, n, packed_upper_matrix);
	if (mpi_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear build done"
		          << " ctx=" << prefix
		          << " scope="
		          << (readout_residual_only ? "readout-residual-only" : "full")
		          << " n=" << n
		          << " full_n=" << full_n
		          << " cfg_local=" << local_cfg_count
		          << " atoms_local=" << local_atom_count
		          << " eq_local=" << quad_opt_eqn_count
		          << " build_s=" << build_seconds
		          << std::endl;
	}
	if (mpi_rank == 0) {
		MPI_Reduce(MPI_IN_PLACE,
		           packed_upper_matrix.data(),
		           static_cast<int>(packed_matrix_size),
		           MPI_DOUBLE,
		           MPI_SUM,
		           0,
		           train_comm);
		UnpackUpperTriangle(packed_upper_matrix, n, quad_opt_matr);
		MPI_Reduce(MPI_IN_PLACE, quad_opt_vec, n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(MPI_IN_PLACE, &quad_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		const double reduce_seconds = MPI_Wtime() - reduce_start;
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear reduce done"
		          << " ctx=" << prefix
		          << " reduce_s=" << reduce_seconds << std::endl;
		add_scalar_head_ridge();
		capture_readout_objective();
		const double solve_start = MPI_Wtime();
		const int bad_matrix = FirstNonFinite(quad_opt_matr, n*n);
		const int bad_rhs = FirstNonFinite(quad_opt_vec, n);
		if (bad_matrix >= 0) {
			linear_status = 1;
			linear_error = NonFiniteMessage(prefix + " normal matrix", bad_matrix);
		} else if (bad_rhs >= 0) {
			linear_status = 1;
			linear_error = NonFiniteMessage(prefix + " right-hand side", bad_rhs);
		} else {
			try {
				SolveSLAE();
			}
			catch (const MlipException& exc) {
				linear_status = 1;
				linear_error = exc.What();
			}
			catch (const std::exception& exc) {
				linear_status = 1;
				linear_error = exc.what();
			}
			catch (...) {
				linear_status = 1;
				linear_error = prefix + " linear solve failed with an unknown exception";
			}
		}
		if (linear_status == 0) {
			const int bad_coeff = FirstNonFinite(p_mlmtpr->Coeff(), p_mlmtpr->CoeffCount());
			if (bad_coeff >= 0) {
				linear_status = 1;
				linear_error = NonFiniteMessage(prefix + " solved coefficients", bad_coeff);
			}
		}
		if (linear_status == 0 && !validate_readout_objective()) {
			linear_status = 1;
			linear_error = prefix + " increased the fixed-feature readout objective";
			std::copy(coeffs_before.begin(),
			          coeffs_before.end(),
			          p_mlmtpr->regression_coeffs.begin());
			p_mlmtpr->LinCoeff();
		}
		if (linear_status == 0) {
			for (std::size_t i = 0; i < preserved_radial_indices.size(); ++i) {
				const int coeff_index = preserved_radial_indices[i];
				if (std::memcmp(
				        &p_mlmtpr->Coeff()[coeff_index],
				        &preserved_radial_values[i],
				        sizeof(double)) == 0)
					continue;
				std::ostringstream message;
				message << prefix
				        << " changed a weight-normalized physical radial coefficient"
				        << " index=" << coeff_index
				        << " before=" << std::setprecision(17)
				        << preserved_radial_values[i]
				        << " after=" << p_mlmtpr->Coeff()[coeff_index];
				linear_status = 1;
				linear_error = message.str();
				break;
			}
		}
		const double solve_seconds = MPI_Wtime() - solve_start;
		if (linear_status == 0) {
			std::cout << "[" << CurrentTimestamp() << "] TrainLinear solve done"
			          << " ctx=" << prefix
			          << " solve_s=" << solve_seconds << std::endl;
		} else {
			std::cerr << "[" << CurrentTimestamp() << "] TrainLinear solve failed"
			          << " ctx=" << prefix
			          << " solve_s=" << solve_seconds
			          << " reason=" << linear_error << std::endl;
		}
	} else {
		MPI_Reduce(packed_upper_matrix.data(),
		           packed_upper_matrix.data(),
		           static_cast<int>(packed_matrix_size),
		           MPI_DOUBLE,
		           MPI_SUM,
		           0,
		           train_comm);
		MPI_Reduce(quad_opt_vec, quad_opt_vec, n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(&quad_opt_scalar, &quad_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
	}

#else
	add_scalar_head_ridge();
	capture_readout_objective();
	RequireFiniteArray(quad_opt_matr, n*n, prefix + " normal matrix");
	RequireFiniteArray(quad_opt_vec, n, prefix + " right-hand side");
	SolveSLAE();
	RequireFiniteArray(p_mlmtpr->Coeff(), p_mlmtpr->CoeffCount(), prefix + " solved coefficients");
	if (!validate_readout_objective()) {
		std::copy(coeffs_before.begin(),
		          coeffs_before.end(),
		          p_mlmtpr->regression_coeffs.begin());
		p_mlmtpr->LinCoeff();
		ERROR(prefix + " increased the fixed-feature readout objective");
	}
#endif
#ifdef MLIP_MPI
	double bcast_start = MPI_Wtime();
	MPI_Bcast(&linear_status, 1, MPI_INT, 0, train_comm);
	if (linear_status != 0)
		ERROR(prefix + " failed; see rank-0 log for the first non-finite linear-solve detail.");
	MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
	RequireFiniteArray(p_mlmtpr->Coeff(), p_mlmtpr->CoeffCount(), prefix + " broadcast coefficients");
	double bcast_seconds = MPI_Wtime() - bcast_start;
	if (mpi_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear bcast done"
		          << " ctx=" << prefix
		          << " bcast_s=" << bcast_seconds
		          << std::endl;
	}
#endif

#ifndef MLIP_MPI
	for (std::size_t i = 0; i < preserved_radial_indices.size(); ++i) {
		const int coeff_index = preserved_radial_indices[i];
		if (std::memcmp(
		        &p_mlmtpr->Coeff()[coeff_index],
		        &preserved_radial_values[i],
		        sizeof(double)) != 0)
			ERROR(prefix
			      + " changed a weight-normalized physical radial coefficient at index "
			      + std::to_string(coeff_index));
	}
#endif
	if (readout_residual_only) {
		const int readout_begin = p_mlmtpr->TwoLayerGateReadoutCoeffOffset();
		const int readout_end =
			readout_begin + p_mlmtpr->TwoLayerGateReadoutCoeffCount();
		if (coeffs_before.size() != p_mlmtpr->regression_coeffs.size())
			ERROR(prefix + " changed the coefficient layout");
		for (int i = 0; i < p_mlmtpr->CoeffCount(); ++i) {
			if (i >= readout_begin && i < readout_end)
				continue;
			if (std::memcmp(&coeffs_before[i],
			                &p_mlmtpr->regression_coeffs[i],
			                sizeof(double)) != 0)
				ERROR(prefix + " changed a frozen non-readout coefficient");
		}
	}


}

void MTPR_trainer::random_sample(int prank, std::vector<Configuration>& training_set, int max_step, const std::vector<Neighborhoods>* neighborhoods) {
//	int n_coeffe= p_mlip->CoeffCount();
//	double* x = p_mlip->Coeff();
#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	const int mpi_rank = train_rank_;
	const MPI_Comm train_comm = train_comm_;
#else
	const int mpi_rank = prank;
#endif
	int n_coeffe= p_mlmtpr->CoeffCount();
	double* x = p_mlmtpr->Coeff();
	double c_l = 0.0;
	double std_l = 0.0;
	double p_l = 1e10;
	std::vector<double> _x(n_coeffe);
	for (int i = 0; i < n_coeffe; i++)
		_x[i] = x[i];

	int num_step = 0;
	int m = (int)training_set.size();
	int K = m;

#ifdef MLIP_MPI
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, train_comm);
#endif
	TrainLinear(prank, training_set, neighborhoods, "random_sample init");
	if (mpi_rank == 0) { std::cout << _x.size() << " " << n_coeffe << std::endl; }
	ObjectiveFunction(training_set, neighborhoods);
	loss_ /= K;
	std_ /= K;
	if (mpi_rank == 0) { std::cout << "__________....__________ " << std::endl; }
#ifdef MLIP_MPI
	MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
	MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
#else
	c_l = loss_;
	std_l = std_;
	p_l = loss_ + std_;
#endif
	if (mpi_rank == 0) { std::cout << "__________....__________ " << std::endl; }
	while (num_step < max_step) {
		if (mpi_rank == 0) {
			std::random_device rand_device;
			std::mt19937_64 generator(rand_device());

			std::cout << "Random sample of nonlinear coefficients" << std::endl;
			p_mlmtpr->RandomizeNonlinearCoeffs(generator, 0.5, true, 0.20);
		}
#ifdef MLIP_MPI
		MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
#endif
		TrainLinear(prank, training_set, neighborhoods, "random_sample step " + std::to_string(num_step));
		ObjectiveFunction(training_set, neighborhoods);
		loss_ /= K;
		std_ /= K;
#ifdef MLIP_MPI
		MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
#else
		c_l = loss_;
		std_l = std_;
#endif
		if (mpi_rank == 0) {
			if (std_l + c_l < p_l) {
				p_l = std_l + c_l;
				for (int i = 0; i < n_coeffe; i++)
					_x[i] = x[i];
				std::cout << "num_step: " << num_step << " f= " << c_l << "   std^2= " << std_l/std_scaling << "\t (*opt)"
					<< std::endl;
				num_step += 1;
			}
			else {
				std::cout << "num_step: " << num_step << " f= " << c_l << "   std^2= " << std_l/std_scaling << std::endl;
				num_step += 1;
			}
		}
#ifdef MLIP_MPI
		MPI_Bcast(&num_step, 1, MPI_INT, 0, train_comm);
		MPI_Bcast(&_x[0], n_coeffe, MPI_DOUBLE, 0, train_comm);
#endif
	}
	for (int i = 0; i < n_coeffe; i++)
		x[i] = _x[i];

#ifdef MLIP_MPI
	MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
#endif
	std::vector<double>().swap(_x);
}


#ifndef ALGLIB
void MTPR_trainer::Train(std::vector<Configuration>& training_set) //with Shapeev bfgs
{

	//cout << max_step_count << endl;

	int n = p_mlip->CoeffCount();
	double *x = p_mlip->Coeff();

	int nlin = p_mlmtpr->LinearCoeffCount();
	const bool use_lbfgs_optimizer = optimizer_kind == "lbfgs";
	const bool use_radial_weight_norm = radial_parameterization == "weight-norm";
	const bool use_bfgs_coordinate_normalization =
		bfgs_coordinate_normalization == "efs-gn";
	const std::string optimizer_log_name = use_lbfgs_optimizer ? "L-BFGS" : "BFGS";
	if (!use_lbfgs_optimizer && optimizer_kind != "bfgs")
		ERROR("MTPR_trainer::Train(): unknown optimizer " + optimizer_kind);
	if (!use_radial_weight_norm && radial_parameterization != "physical")
		ERROR("MTPR_trainer::Train(): unknown radial parameterization "
		      + radial_parameterization);
	if (!use_bfgs_coordinate_normalization
	    && bfgs_coordinate_normalization != "none")
		ERROR("MTPR_trainer::Train(): unknown BFGS coordinate normalization "
		      + bfgs_coordinate_normalization);
	if (use_bfgs_coordinate_normalization && use_lbfgs_optimizer)
		ERROR("MTPR_trainer::Train(): --bfgs-coordinate-normalization requires --optimizer=bfgs");
	if (bfgs_coordinate_probes <= 0)
		ERROR("MTPR_trainer::Train(): BFGS coordinate probe count should be positive");
	if (!std::isfinite(bfgs_coordinate_floor)
	    || bfgs_coordinate_floor < 0.0
	    || bfgs_coordinate_floor > 1.0)
		ERROR("MTPR_trainer::Train(): BFGS coordinate floor should be finite and in [0,1]");
	auto require_live_coeff_pointer = [&](const std::string& context) {
		if (p_mlmtpr->CoeffCount() != n || p_mlmtpr->Coeff() != x)
			ERROR("MTPR_trainer::Train(): coefficient storage changed during " + context);
	};
	if (use_lbfgs_optimizer && bfgs_line_search != "armijo-value")
		ERROR("MTPR_trainer::Train(): L-BFGS requires armijo-value line search");
	if (use_lbfgs_optimizer) {
		lbfgs.ResetStatistics();
		lbfgs.SetMaxHistory(lbfgs_history_size);
		lbfgs.relative_curvature_tol = lbfgs_curvature_tolerance;
		lbfgs.powell_damping_curvature = lbfgs_damping_threshold;
		lbfgs.armijo_c1 = lbfgs_armijo_c1;
		lbfgs.SetInitialStep(lbfgs_initial_step);
		lbfgs.initial_step_growth = lbfgs_step_growth;
	}
	struct ResidualEvalStageScope {
		MLMTPR* mtpr;
		int saved_stage;
		ResidualEvalStageScope(MLMTPR* mtpr_, int stage)
			: mtpr(mtpr_),
			  saved_stage(mtpr_ == nullptr ? 0 : mtpr_->two_layer_residual_eval_stage_)
		{
			if (mtpr != nullptr)
				mtpr->two_layer_residual_eval_stage_ =
					mtpr->TwoLayerResidualEnabled() ? stage : 0;
		}
		~ResidualEvalStageScope()
		{
			if (mtpr != nullptr)
				mtpr->two_layer_residual_eval_stage_ = saved_stage;
		}
	} residual_eval_stage_scope(p_mlmtpr, two_layer_residual_stage);

	p_mlmtpr->max_radial.resize(p_mlmtpr->species_count*p_mlmtpr->species_count*p_mlmtpr->radial_func_count);

	for (int i = 0; i < p_mlmtpr->max_radial.size(); i++)
		p_mlmtpr->max_radial[i] = 1e-10;

	int prank = 0;
	int psize = 1;
	std::stringstream logstrm1;

#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	prank = train_rank_;
	psize = train_size_;
	MPI_Comm train_comm = train_comm_;
		const bool allow_distributed_bfgs = train_comm_is_world_;
		if (!use_lbfgs_optimizer)
			bfgs.UseDistributedDense(prank, allow_distributed_bfgs ? psize : 1, train_comm);
	if (prank == 0) {
		logstrm1 << "MTPR parallel training started" << endl;
		// 		if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
	}
#else
	if (prank == 0) {

		logstrm1 << "MTPR serial(?!?) training started" << endl;
		// 		if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");

	}
#endif

	int m = (int)training_set.size(); // train set size on the current core
	int K = 0;                     // train set size over all cores

	K = m;

#ifdef MLIP_MPI												   
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, train_comm);
#endif

	std::vector<Neighborhoods> training_neighborhoods;
	const long long local_atom_count = [&training_set]() {
		long long total_atoms = 0;
		for (const Configuration& cfg : training_set)
			total_atoms += cfg.size();
		return total_atoms;
	}();
	std::size_t estimated_cache_bytes = 0;
	bool cache_training_neighborhoods =
		ShouldCacheNeighborhoods(training_set, p_mlmtpr->CutOff(), &estimated_cache_bytes);
	if (cache_training_neighborhoods) {
		const bool built_ok = TryBuildNeighborhoods(training_set, p_mlmtpr->CutOff(), training_neighborhoods);
		if (built_ok) {
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] optimizer neighborhood cache enabled"
				          << " estimated_bytes=" << estimated_cache_bytes << std::endl;
		} else {
			cache_training_neighborhoods = false;
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] optimizer neighborhood cache fallback"
				          << " estimated_bytes=" << estimated_cache_bytes
				          << " reason=bad_alloc" << std::endl;
		}
	} else if (prank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] optimizer neighborhood cache disabled"
		          << " estimated_bytes=" << estimated_cache_bytes
		          << " budget_bytes=" << kNeighborhoodCacheBudgetBytesPerRank
		          << std::endl;
	}

	// L-BFGS is cheap enough to keep a deterministic replicated history on all
	// active ranks. Dense BFGS keeps its existing distributed-matrix path.
	const bool optimizer_replicated = use_lbfgs_optimizer || bfgs.UsingDistributedDense();
	const bool need_std_terms = NeedStdTerms();

	const int scal_coeff_begin = p_mlmtpr->species_count;
	const int scal_coeff_end = p_mlmtpr->RadialCoeffOffset();
	std::vector<int> active_coeff_indices;
	p_mlmtpr->BuildActiveCoeffIndices(active_coeff_indices, freeze_scal_coeffs);
	if (p_mlmtpr->TwoLayerResidualEnabled()
	    && two_layer_residual_stage != kResidualStageFull) {
		std::vector<int> stage_active_coeff_indices;
		stage_active_coeff_indices.reserve(active_coeff_indices.size());
		for (int coeff_index : active_coeff_indices)
			if (CoeffActiveForStage(coeff_index))
				stage_active_coeff_indices.push_back(coeff_index);
		active_coeff_indices.swap(stage_active_coeff_indices);
	}
	std::vector<int> full_to_active_index(n, -1);
	for (int active_idx = 0; active_idx < static_cast<int>(active_coeff_indices.size()); ++active_idx)
		full_to_active_index[active_coeff_indices[active_idx]] = active_idx;
	const int opt_n = static_cast<int>(active_coeff_indices.size());
	if (opt_n <= 0)
		ERROR("MTPR_trainer::Train(): no active optimizer coordinates.");
	if (prank == 0) {
		const char* execution_mode = use_lbfgs_optimizer
			? "replicated-lbfgs"
			: (bfgs.UsingDistributedDense()
				? "distributed-dense-bfgs"
				: (psize == 1
					? "single-rank-dense-bfgs"
					: "root-only-dense-bfgs"));
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "optimizer execution mode=" << execution_mode
		          << " optimizer_collective="
		          << (optimizer_replicated ? "true" : "false")
		          << " train_ranks=" << psize
		          << " active_coordinates=" << opt_n
#ifdef MLIP_MPI
		          << " train_comm_is_world="
		          << (train_comm_is_world_ ? "true" : "false")
#endif
		          << std::endl;
	}
	struct RadialWeightNormRow {
		int full_begin;
		int active_begin;
		int pivot;
		double pivot_sign;
		bool is_gate1;
		int mu;
		RadialWeightNormRow(int full_begin_,
		                    int active_begin_,
		                    int pivot_,
		                    double pivot_sign_,
		                    bool is_gate1_,
		                    int mu_)
			: full_begin(full_begin_),
			  active_begin(active_begin_),
			  pivot(pivot_),
			  pivot_sign(pivot_sign_),
			  is_gate1(is_gate1_),
			  mu(mu_) {}
	};
	std::vector<RadialWeightNormRow> radial_weight_norm_rows;
	std::vector<int> radial_weight_norm_role(n, -1);
	int outer_radial_weight_norm_row_count = 0;
	int gate1_radial_weight_norm_row_count = 0;
	int outer_radial_weight_norm_zero_fallback_count = 0;
	int gate1_radial_weight_norm_zero_fallback_count = 0;
	double radial_weight_norm_roundtrip_max_abs = 0.0;
	if (use_radial_weight_norm && p_mlmtpr->p_RadialBasis == nullptr)
		ERROR("MTPR_trainer::Train(): radial weight normalization requires an initialized radial basis.");
	const int radial_weight_norm_row_size =
		p_mlmtpr->p_RadialBasis == nullptr ? 0 : p_mlmtpr->p_RadialBasis->rb_size;
	// Keep accepted points well inside the signed-pivot chart. For the mapping
	// theta = pi*tanh(r/2), r=8 still has a finite angular response, while the
	// trial limit leaves room for line search without approaching tanh
	// saturation (the old failure appeared only near r ~= 38).
	const double radial_rechart_direction_radius = 8.0;
	const double radial_trial_direction_radius_limit = 12.0;
	if (use_radial_weight_norm) {
		std::vector<double> initial_coordinates(
			radial_weight_norm_row_size, 0.0);
		std::vector<double> reconstructed_physical(
			radial_weight_norm_row_size, 0.0);
		auto add_radial_weight_norm_row = [&](int row_begin, bool is_gate1, int mu) {
			int active_count = 0;
			for (int xi = 0; xi < radial_weight_norm_row_size; ++xi)
				active_count += full_to_active_index[row_begin + xi] >= 0 ? 1 : 0;
			if (active_count == 0)
				return;
			if (active_count != radial_weight_norm_row_size)
				ERROR("MTPR_trainer::Train(): radial weight normalization requires a complete active radial row.");
			const int active_begin = full_to_active_index[row_begin];
			double max_abs = 0.0;
			for (int xi = 0; xi < radial_weight_norm_row_size; ++xi) {
				if (full_to_active_index[row_begin + xi] != active_begin + xi)
					ERROR("MTPR_trainer::Train(): radial weight-normalization coordinates are not contiguous.");
				const double value = x[row_begin + xi];
				if (!std::isfinite(value))
					ERROR("MTPR_trainer::Train(): radial weight-normalization row contains a non-finite coefficient.");
				max_abs = std::max(max_abs, std::abs(value));
			}
			if (max_abs == 0.0) {
				if (!is_gate1 && !p_mlmtpr->OuterKRankUsesDirectRadial())
					ERROR("MTPR_trainer::Train(): cannot select a nonzero radial weight-normalization pivot.");
				// The norm/log-direction chart is undefined at the origin.  This
				// occurs legitimately when a legacy rank-1 W(l,k)=0 is absorbed.
				// Leave the complete row in ordinary physical coordinates so BFGS
				// can move it away from zero without injecting an arbitrary epsilon.
				if (is_gate1)
					++gate1_radial_weight_norm_zero_fallback_count;
				else
					++outer_radial_weight_norm_zero_fallback_count;
				return;
			}

			int pivot = 0;
			double pivot_sign = 1.0;
			if (!sus2::RadialWeightNormChart::SelectSignedPivot(
			        x + row_begin,
			        radial_weight_norm_row_size,
			        pivot,
			        pivot_sign))
				ERROR("MTPR_trainer::Train(): cannot select a nonzero radial weight-normalization pivot.");
			for (int xi = 0; xi < radial_weight_norm_row_size; ++xi)
				radial_weight_norm_role[row_begin + xi] = xi;

			if (!sus2::RadialWeightNormChart::PackWithPivot(
			        x + row_begin,
			        radial_weight_norm_row_size,
			        pivot,
			        pivot_sign,
			        initial_coordinates.data())
			    || !sus2::RadialWeightNormChart::UnpackWithPivot(
			        initial_coordinates.data(),
			        radial_weight_norm_row_size,
			        pivot,
			        pivot_sign,
			        reconstructed_physical.data()))
				ERROR("MTPR_trainer::Train(): cannot convert a physical radial row to weight-normalization coordinates.");
			for (int xi = 0; xi < radial_weight_norm_row_size; ++xi) {
				const double physical_value = x[row_begin + xi];
				const double abs_error = std::abs(
					reconstructed_physical[xi] - physical_value);
				radial_weight_norm_roundtrip_max_abs = std::max(
					radial_weight_norm_roundtrip_max_abs, abs_error);
				const double tolerance = 64.0
					* std::numeric_limits<double>::epsilon()
					* std::max(1.0, std::abs(physical_value));
				if (abs_error > tolerance)
					ERROR("MTPR_trainer::Train(): radial weight-normalization changed a loaded physical coefficient.");
			}
			radial_weight_norm_rows.push_back(
				RadialWeightNormRow(
					row_begin, active_begin, pivot, pivot_sign, is_gate1, mu));
			if (is_gate1)
				++gate1_radial_weight_norm_row_count;
			else
				++outer_radial_weight_norm_row_count;
		};

		const int radial_begin = p_mlmtpr->RadialCoeffOffset();
		const int radial_stride = p_mlmtpr->RadialCoeffBlockSize();
		for (int mu = 0; mu < p_mlmtpr->radial_func_count; ++mu)
			add_radial_weight_norm_row(
				radial_begin + mu * radial_stride, false, mu);

		if (p_mlmtpr->TwoLayerGateUsesSharedRadial()) {
			const int gate1_radial_begin = p_mlmtpr->TwoLayerGateRadialCoeffOffset();
			for (int mu = 0; mu < p_mlmtpr->radial_func_count; ++mu)
				add_radial_weight_norm_row(
					gate1_radial_begin + mu * radial_weight_norm_row_size,
					true,
					mu);
		}
	}
	const int redundant_radial_species_count =
		n - p_mlmtpr->ActiveCoeffCount(false);

	struct CoordinateNormalizationBlock {
		std::string name;
		std::vector<int> full_indices;
	};
	std::vector<CoordinateNormalizationBlock> coordinate_normalization_blocks;
	std::vector<int> coordinate_normalization_block_by_full(n, -1);
	std::vector<double> coordinate_scale_by_full(n, 1.0);
	std::vector<double> coordinate_reference_by_full(n, 0.0);
	std::vector<double> coordinate_physical_reference;
	bool coordinate_normalization_initialized =
		!use_bfgs_coordinate_normalization;
	auto add_coordinate_normalization_block =
		[&](const std::string& name, int begin, int count) {
			if (count <= 0)
				return;
			if (begin < 0 || begin + count > n)
				ERROR("MTPR_trainer::Train(): invalid coordinate-normalization block " + name);
			CoordinateNormalizationBlock block;
			block.name = name;
			for (int local = 0; local < count; ++local) {
				const int full_idx = begin + local;
				if (full_to_active_index[full_idx] < 0
				    || radial_weight_norm_role[full_idx] >= 0)
					continue;
				if (coordinate_normalization_block_by_full[full_idx] >= 0)
					ERROR("MTPR_trainer::Train(): overlapping coordinate-normalization blocks.");
				block.full_indices.push_back(full_idx);
			}
			if (block.full_indices.empty())
				return;
			const int block_index =
				static_cast<int>(coordinate_normalization_blocks.size());
			for (int full_idx : block.full_indices)
				coordinate_normalization_block_by_full[full_idx] = block_index;
			coordinate_normalization_blocks.push_back(block);
		};
	if (use_bfgs_coordinate_normalization) {
		const int linear_begin = p_mlmtpr->LinearCoeffOffset();
		add_coordinate_normalization_block(
			"species-prefactor", linear_begin, p_mlmtpr->species_count);
		add_coordinate_normalization_block(
			"scalar-head",
			linear_begin + p_mlmtpr->species_count,
			p_mlmtpr->alpha_scalar_moments);
		add_coordinate_normalization_block(
			"residual-e0-head",
			p_mlmtpr->TwoLayerResidualE0CoeffOffset(),
			p_mlmtpr->TwoLayerResidualE0CoeffCount());
		add_coordinate_normalization_block(
			"gate-readout-head",
			p_mlmtpr->TwoLayerGateReadoutCoeffOffset(),
			p_mlmtpr->TwoLayerGateReadoutCoeffCount());
		add_coordinate_normalization_block(
			"gate1-additive",
			p_mlmtpr->TwoLayerGateAdditiveCoeffOffset(),
			p_mlmtpr->TwoLayerGateAdditiveCoeffCount());
		add_coordinate_normalization_block(
			"gate1-weight",
			p_mlmtpr->TwoLayerGateWeightOffset(),
			p_mlmtpr->TwoLayerGateWeightCount());
		add_coordinate_normalization_block(
			"gate1-body-mix",
			p_mlmtpr->TwoLayerGateBodyMixWeightOffset(),
			p_mlmtpr->TwoLayerGateBodyMixWeightCount());
		add_coordinate_normalization_block(
			"gate1-edge-l1",
			p_mlmtpr->TwoLayerGateEdgeL1WeightOffset(),
			p_mlmtpr->TwoLayerGateEdgeL1WeightCount());
		add_coordinate_normalization_block(
			"gate2-additive",
			p_mlmtpr->SecondLayerGateAdditiveCoeffOffset(),
			p_mlmtpr->SecondLayerGateAdditiveCoeffCount());
		add_coordinate_normalization_block(
			"gate2-weight",
			p_mlmtpr->SecondLayerGateWeightOffset(),
			p_mlmtpr->SecondLayerGateWeightCount());
		add_coordinate_normalization_block(
			"gate2-body-mix",
			p_mlmtpr->SecondLayerGateBodyMixWeightOffset(),
			p_mlmtpr->SecondLayerGateBodyMixWeightCount());
		add_coordinate_normalization_block(
			"gate-k-rank",
			p_mlmtpr->TwoLayerGateKRankWeightOffset(),
			p_mlmtpr->TwoLayerGateKRankWeightCount());
		add_coordinate_normalization_block(
			"outer-k-rank",
			p_mlmtpr->OuterKRank1WeightOffset(),
			p_mlmtpr->OuterKRank1WeightCount());
		if (coordinate_normalization_blocks.empty())
			ERROR("MTPR_trainer::Train(): no active parameter block supports BFGS coordinate normalization.");
	}

	std::vector<double> active_coeffs(opt_n, 0.0);
	auto full_to_optimizer_value = [&](int full_idx) {
		double base_value = x[full_idx];
		if (p_mlmtpr->IsRadialFirstCoeff(full_idx))
			base_value = p_mlmtpr->RadialFirstCoeffValueToRaw(base_value);
		const double scale = coordinate_scale_by_full[full_idx];
		if (!sus2::DiagonalCoordinateScaling::IsValidScale(scale))
			ERROR("MTPR_trainer::Train(): invalid optimizer coordinate scale.");
		return sus2::DiagonalCoordinateScaling::PackCentered(
			base_value,
			coordinate_reference_by_full[full_idx],
			scale);
	};
	auto optimizer_to_full_value = [&](int full_idx, double opt_value) {
		const double scale = coordinate_scale_by_full[full_idx];
		if (!sus2::DiagonalCoordinateScaling::IsValidScale(scale))
			ERROR("MTPR_trainer::Train(): invalid optimizer coordinate scale.");
		const double base_value =
			sus2::DiagonalCoordinateScaling::UnpackCentered(
				opt_value,
				coordinate_reference_by_full[full_idx],
				scale);
		if (p_mlmtpr->IsRadialFirstCoeff(full_idx))
			return p_mlmtpr->RadialFirstCoeffRawToValue(base_value);
		return base_value;
	};
	auto physical_coordinate_gradient_value = [&](int full_idx) {
		double grad = loss_grad_[full_idx];
		if (p_mlmtpr->IsRadialFirstCoeff(full_idx))
			grad *= p_mlmtpr->RadialFirstCoeffDerivativeFromValue(x[full_idx]);
		const double scale = coordinate_scale_by_full[full_idx];
		if (!sus2::DiagonalCoordinateScaling::IsValidScale(scale))
			ERROR("MTPR_trainer::Train(): invalid optimizer coordinate scale.");
		return sus2::DiagonalCoordinateScaling::PullbackGradient(grad, scale);
	};
	auto pack_active_coeffs = [&]() {
		for (int active_idx = 0; active_idx < opt_n; ++active_idx) {
			const int full_idx = active_coeff_indices[active_idx];
			if (radial_weight_norm_role[full_idx] < 0)
				active_coeffs[active_idx] = full_to_optimizer_value(full_idx);
		}
		for (const RadialWeightNormRow& row : radial_weight_norm_rows) {
			if (!sus2::RadialWeightNormChart::PackWithPivot(
			        x + row.full_begin,
			        radial_weight_norm_row_size,
			        row.pivot,
			        row.pivot_sign,
			        active_coeffs.data() + row.active_begin))
				ERROR("MTPR_trainer::Train(): cannot pack radial weight-normalization coordinates.");
		}
	};
	auto optimizer_set_x = [&](const double* values, int count) {
		if (use_lbfgs_optimizer)
			lbfgs.Set_x(values, count);
		else
			bfgs.Set_x(values, count);
	};
	auto optimizer_data = [&]() -> const double* {
		return use_lbfgs_optimizer ? lbfgs.Data() : bfgs.Data();
	};
	auto optimizer_x = [&](int index) {
		return use_lbfgs_optimizer ? lbfgs.x(index) : bfgs.x(index);
	};
	auto optimizer_set_inv_hess_diag = [&](const Array1D& diag) {
		if (use_lbfgs_optimizer)
			lbfgs.SetInvHessDiagonal(diag);
		else
			bfgs.SetInvHessDiagonal(diag);
	};
	auto optimizer_mask_coordinates = [&](const std::vector<int>& indices) {
		if (use_lbfgs_optimizer)
			lbfgs.MaskCoordinates(indices);
		else
			bfgs.MaskCoordinates(indices);
	};
	auto optimizer_restart = [&]() {
		if (use_lbfgs_optimizer)
			lbfgs.Restart();
		else
			bfgs.Restart();
	};
	auto optimizer_iter_step = [&]() -> int& {
		return use_lbfgs_optimizer ? lbfgs.iter_step : bfgs.iter_step;
	};
	auto optimizer_armijo_trial_ready = [&]() {
		return use_lbfgs_optimizer
			? lbfgs.ArmijoValueOnlyTrialReady()
			: bfgs.ArmijoValueOnlyTrialReady();
	};
	auto optimizer_armijo_accepts = [&](double value) {
		return use_lbfgs_optimizer
			? lbfgs.ArmijoValueOnlyAccepts(value)
			: bfgs.ArmijoValueOnlyAccepts(value);
	};
	auto optimizer_armijo_backtrack = [&](double value) {
		if (use_lbfgs_optimizer)
			lbfgs.BacktrackArmijoValueOnly(value);
		else
			bfgs.BacktrackArmijoValueOnly(value);
	};
	auto optimizer_reduce_step = [&](double ratio) {
		if (use_lbfgs_optimizer)
			lbfgs.ReduceStep(ratio);
		else
			bfgs.ReduceStep(ratio);
	};
	auto limit_radial_direction_trial = [&]() {
		if (radial_weight_norm_rows.empty())
			return;
		int reductions = 0;
		while (true) {
			double max_radius = 0.0;
			int max_row_idx = -1;
			for (size_t row_idx = 0;
			     row_idx < radial_weight_norm_rows.size();
			     ++row_idx) {
				const RadialWeightNormRow& row =
					radial_weight_norm_rows[row_idx];
				double radius = 0.0;
				if (!sus2::RadialWeightNormChart::GetDirectionRadius(
				        optimizer_data() + row.active_begin,
				        radial_weight_norm_row_size,
				        radius))
					ERROR("MTPR_trainer::Train(): optimizer proposed an invalid radial direction coordinate.");
				if (radius > max_radius) {
					max_radius = radius;
					max_row_idx = static_cast<int>(row_idx);
				}
			}
			if (max_radius <= radial_trial_direction_radius_limit)
				break;
			optimizer_reduce_step(0.25);
			if (++reductions > 64) {
				const RadialWeightNormRow& row =
					radial_weight_norm_rows[max_row_idx];
				ERROR(optimizer_log_name
				      + " could not reduce radial direction step below the safety limit"
				      + " block=" + (row.is_gate1 ? "gate1" : "outer")
				      + " mu=" + std::to_string(row.mu));
			}
		}
		if (reductions > 0 && prank == 0) {
			std::cerr << "[" << CurrentTimestamp() << "] "
			          << optimizer_log_name
			          << " reduced trial to keep radial direction charts finite"
			          << " reductions=" << reductions
			          << " direction_radius_limit="
			          << radial_trial_direction_radius_limit << std::endl;
		}
	};
	auto optimizer_linesearch_stagnated = [&]() {
		return use_lbfgs_optimizer
			? lbfgs.linesearch_stagnated()
			: bfgs.linesearch_stagnated();
	};
	auto optimizer_is_in_linesearch = [&]() {
		return use_lbfgs_optimizer
			? lbfgs.is_in_linesearch()
			: bfgs.is_in_linesearch();
	};
	auto set_optimizer_x_from_full_coeffs = [&]() {
		pack_active_coeffs();
		optimizer_set_x(active_coeffs.data(), opt_n);
	};
	auto set_optimizer_x_from_full_coeffs_collective = [&]() {
#ifdef MLIP_MPI
		if (optimizer_replicated) {
			int pack_status = 0;
			if (prank == 0) {
				try {
					pack_active_coeffs();
				}
				catch (const MlipException& exc) {
					pack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root optimizer-coordinate pack failed: "
					          << exc.What() << std::endl;
				}
				catch (const std::exception& exc) {
					pack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root optimizer-coordinate pack failed: "
					          << exc.what() << std::endl;
				}
				catch (...) {
					pack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root optimizer-coordinate pack failed"
					          << std::endl;
				}
			}
			MPI_Bcast(&pack_status, 1, MPI_INT, 0, train_comm);
			if (pack_status != 0)
				ERROR("MTPR_trainer::Train(): canonical optimizer-coordinate pack failed on root.");
			MPI_Bcast(
				active_coeffs.data(), opt_n, MPI_DOUBLE, 0, train_comm);
			optimizer_set_x(active_coeffs.data(), opt_n);
			return;
		}
#endif
		set_optimizer_x_from_full_coeffs();
	};
	auto copy_optimizer_x_to_full_coeffs_local = [&]() {
		const double* opt_x = optimizer_data();
		for (int active_idx = 0; active_idx < opt_n; ++active_idx) {
			const int full_idx = active_coeff_indices[active_idx];
			if (radial_weight_norm_role[full_idx] < 0)
				x[full_idx] = optimizer_to_full_value(full_idx, opt_x[active_idx]);
		}
		for (const RadialWeightNormRow& row : radial_weight_norm_rows) {
			if (!sus2::RadialWeightNormChart::UnpackWithPivot(
			        opt_x + row.active_begin,
			        radial_weight_norm_row_size,
			        row.pivot,
			        row.pivot_sign,
			        x + row.full_begin))
				ERROR("MTPR_trainer::Train(): invalid radial weight-normalization trial coordinates.");
		}
	};
	auto copy_optimizer_x_to_full_coeffs = [&]() {
#ifdef MLIP_MPI
		if (optimizer_replicated) {
			int unpack_status = 0;
			if (prank == 0) {
				try {
					copy_optimizer_x_to_full_coeffs_local();
				}
				catch (const MlipException& exc) {
					unpack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root physical-model unpack failed: "
					          << exc.What() << std::endl;
				}
				catch (const std::exception& exc) {
					unpack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root physical-model unpack failed: "
					          << exc.what() << std::endl;
				}
				catch (...) {
					unpack_status = 1;
					std::cerr << "MTPR_trainer::Train(): root physical-model unpack failed"
					          << std::endl;
				}
			}
			MPI_Bcast(&unpack_status, 1, MPI_INT, 0, train_comm);
			if (unpack_status != 0)
				ERROR("MTPR_trainer::Train(): canonical physical-model unpack failed on root.");
			// Every local objective and gradient pullback must use root's exact
			// physical image of the shared optimizer coordinates.
			MPI_Bcast(x, n, MPI_DOUBLE, 0, train_comm);
			return;
		}
#endif
		copy_optimizer_x_to_full_coeffs_local();
	};
	auto verify_optimizer_coordinate_roundtrip = [&]() {
		const std::vector<double> physical_reference(x, x + n);
		set_optimizer_x_from_full_coeffs_collective();
		copy_optimizer_x_to_full_coeffs();
		double max_abs_error = 0.0;
		for (int full_idx = 0; full_idx < n; ++full_idx) {
			const double physical_value = physical_reference[full_idx];
			const double abs_error = std::abs(x[full_idx] - physical_value);
			max_abs_error = std::max(max_abs_error, abs_error);
			const double tolerance = 64.0
				* std::numeric_limits<double>::epsilon()
				* std::max(1.0, std::abs(physical_value));
			if (abs_error > tolerance)
				ERROR("MTPR_trainer::Train(): optimizer coordinate round trip changed physical coefficient "
				      + std::to_string(full_idx));
		}
		std::copy(physical_reference.begin(), physical_reference.end(), x);
		set_optimizer_x_from_full_coeffs_collective();
		if (prank == 0) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "optimizer coordinate physical roundtrip verified"
			          << " active_count=" << opt_n
			          << " full_count=" << n
			          << " max_abs_error=" << max_abs_error
			          << " model_format=physical"
			          << std::endl;
		}
	};
	std::vector<double> optimizer_gradient_cache(opt_n, 0.0);
	std::vector<double> radial_log_norm_reference(radial_weight_norm_rows.size(), 0.0);
	auto pack_optimizer_gradient = [&]() {
		for (int active_idx = 0; active_idx < opt_n; ++active_idx) {
			const int full_idx = active_coeff_indices[active_idx];
			if (radial_weight_norm_role[full_idx] < 0)
				optimizer_gradient_cache[active_idx] =
					physical_coordinate_gradient_value(full_idx);
		}
		const double* opt_x = optimizer_data();
		for (const RadialWeightNormRow& row : radial_weight_norm_rows) {
			if (!sus2::RadialWeightNormChart::PullbackGradientWithPivot(
			        x + row.full_begin,
			        opt_x + row.active_begin,
			        &loss_grad_[row.full_begin],
			        radial_weight_norm_row_size,
			        row.pivot,
			        row.pivot_sign,
			        optimizer_gradient_cache.data() + row.active_begin))
				ERROR("MTPR_trainer::Train(): cannot pull back radial weight-normalization gradient.");
		}
	};
	auto initialize_bfgs_coordinate_normalization = [&]() {
		if (coordinate_normalization_initialized)
			return;
		if (K <= 0)
			ERROR("MTPR_trainer::Train(): BFGS coordinate normalization requires training configurations.");
		const std::vector<double> physical_coeff_reference(x, x + n);
		coordinate_physical_reference = physical_coeff_reference;
		const std::vector<double> linear_coeffs_reference = p_mlmtpr->linear_coeffs;
		const std::vector<double> linear_mults_reference = p_mlmtpr->linear_mults;
		const std::vector<double> max_linear_reference = p_mlmtpr->max_linear;
		std::vector<int> selected_full_indices;
		for (const CoordinateNormalizationBlock& block :
		     coordinate_normalization_blocks)
			selected_full_indices.insert(
				selected_full_indices.end(),
				block.full_indices.begin(),
				block.full_indices.end());
		if (selected_full_indices.empty())
			ERROR("MTPR_trainer::Train(): BFGS coordinate normalization has no active coordinates.");

		double build_start = 0.0;
#ifdef MLIP_MPI
		build_start = MPI_Wtime();
#else
		build_start = static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
#endif

		Array1D probe_gradient(n);
		std::vector<double> selected_diagonal_local(
			selected_full_indices.size(), 0.0);
		std::vector<double> selected_diagonal_global(
			selected_full_indices.size(), 0.0);

		const double inv_global_count = 1.0 / static_cast<double>(K);
		auto sqrt_curvature = [&](double curvature,
		                          const std::string& output_name) {
			if (!std::isfinite(curvature) || curvature < 0.0)
				ERROR("MTPR_trainer::Train(): invalid " + output_name
				      + " curvature in BFGS coordinate normalization.");
			return std::sqrt(curvature * inv_global_count);
		};

		for (std::size_t cfg_index = 0;
		     cfg_index < training_set.size();
		     ++cfg_index) {
			const Configuration& original = training_set[cfg_index];
			if (original.size() == 0)
				continue;
			Configuration evaluated = original;
			const Neighborhoods* cfg_neighborhoods =
				cache_training_neighborhoods
					? &training_neighborhoods[cfg_index]
					: nullptr;
			EvaluateTrainingConfiguration(
				original, evaluated, cfg_neighborhoods, true);

			double average_force_sq = 0.0;
			if (original.has_forces()) {
				for (int atom = 0; atom < original.size(); ++atom)
					average_force_sq += original.force(atom).NormSq()
						/ static_cast<double>(original.size());
			}
			const double force_normalization =
				0.1 / (0.1 + norm_by_forces * average_force_sq);

			double energy_weight = wgt_eqtn_energy;
			double stress_weight = wgt_eqtn_stress;
			if (weighting == "structures") {
				energy_weight /= original.size();
				stress_weight /= original.size();
			} else if (weighting == "molecules") {
				energy_weight *= original.size();
				stress_weight *= original.size();
			}

			for (int probe = 0; probe < bfgs_coordinate_probes; ++probe) {
				std::fill(
					&probe_gradient[0], &probe_gradient[0] + n, 0.0);
				std::vector<double> energy_weights(original.size(), 0.0);
				std::vector<Vector3> force_weights(
					original.size(), Vector3(0.0, 0.0, 0.0));
				Matrix3 stress_weights(
					0.0, 0.0, 0.0,
					0.0, 0.0, 0.0,
					0.0, 0.0, 0.0);
				bool has_nonzero_output_weight = false;
				std::uint64_t output_index = 0;

				if (energy_weight != 0.0 && original.has_energy()) {
					const double magnitude = sqrt_curvature(
						2.0 * energy_weight * force_normalization
							/ static_cast<double>(original.size()),
						"energy");
					const double weight = magnitude * CoordinateProbeSign(
						probe, prank, cfg_index, output_index);
					std::fill(
						energy_weights.begin(), energy_weights.end(), weight);
					has_nonzero_output_weight =
						has_nonzero_output_weight || magnitude != 0.0;
				}
				++output_index;

				if (wgt_eqtn_forces != 0.0 && original.has_forces()) {
					for (int atom = 0; atom < original.size(); ++atom) {
						double force_weight = wgt_rel_forces <= 0.0
							? wgt_eqtn_forces
							: wgt_eqtn_forces * wgt_rel_forces
								/ (original.force(atom).NormSq()
								   + wgt_rel_forces);
						if (weighting == "structures")
							force_weight /= original.size();
						const double force_scale =
							force_weight * force_normalization;
						for (int component = 0; component < 3; ++component) {
							const double residual =
								evaluated.force(atom, component)
								- original.force(atom, component);
							const double magnitude = sqrt_curvature(
								force_scale * ForceResidualCurvature(
									residual,
									force_loss_kind,
									force_log_cosh_scale),
								"force");
							force_weights[atom][component] =
								magnitude * CoordinateProbeSign(
									probe,
									prank,
									cfg_index,
									output_index);
							has_nonzero_output_weight =
								has_nonzero_output_weight || magnitude != 0.0;
							++output_index;
						}
					}
				} else {
					output_index +=
						static_cast<std::uint64_t>(3 * original.size());
				}

				if (stress_weight != 0.0 && original.has_stresses()) {
					const double magnitude = sqrt_curvature(
						2.0 * stress_weight
							/ static_cast<double>(original.size()),
						"stress");
					for (int a = 0; a < 3; ++a) {
						for (int b = 0; b < 3; ++b) {
							stress_weights[a][b] =
								magnitude * CoordinateProbeSign(
									probe,
									prank,
									cfg_index,
									output_index);
							++output_index;
						}
					}
					has_nonzero_output_weight =
						has_nonzero_output_weight || magnitude != 0.0;
				}

				if (!has_nonzero_output_weight)
					continue;
				if (NeedPositionDerivativeTerms(original)) {
					if (cfg_neighborhoods != nullptr) {
						p_mlmtpr->AccumulateEFSCombinationGrad(
							evaluated,
							energy_weights,
							force_weights,
							stress_weights,
							probe_gradient,
							*cfg_neighborhoods);
					} else {
						p_mlmtpr->AccumulateEFSCombinationGrad(
							evaluated,
							energy_weights,
							force_weights,
							stress_weights,
							probe_gradient);
					}
				} else if (cfg_neighborhoods != nullptr) {
					p_mlmtpr->AccumulateEnergyCombinationGrad(
						evaluated,
						energy_weights,
						probe_gradient,
						*cfg_neighborhoods);
				} else {
					p_mlmtpr->AccumulateEnergyCombinationGrad(
						evaluated,
						energy_weights,
						probe_gradient);
				}
				for (std::size_t selected = 0;
				     selected < selected_full_indices.size();
				     ++selected) {
					const double value =
						probe_gradient[selected_full_indices[selected]];
					if (!std::isfinite(value))
						ERROR("MTPR_trainer::Train(): non-finite BFGS coordinate probe gradient.");
					selected_diagonal_local[selected] +=
						value * value
						/ static_cast<double>(bfgs_coordinate_probes);
				}
			}
		}

		std::vector<double> diagonal(n, 0.0);
#ifdef MLIP_MPI
		MPI_Allreduce(
			selected_diagonal_local.data(),
			selected_diagonal_global.data(),
			static_cast<int>(selected_diagonal_local.size()),
			MPI_DOUBLE,
			MPI_SUM,
			train_comm);
#else
		selected_diagonal_global = selected_diagonal_local;
#endif
		for (std::size_t selected = 0;
		     selected < selected_full_indices.size();
		     ++selected) {
			const double value = selected_diagonal_global[selected];
			if (!std::isfinite(value) || value < 0.0)
				ERROR("MTPR_trainer::Train(): invalid reduced BFGS coordinate diagonal.");
			diagonal[selected_full_indices[selected]] = value;
		}

		if (!fixed_atomic_energies.empty()
		    && fixed_atomic_energy_weight != 0.0) {
			const double curvature =
				2.0 * fixed_atomic_energy_weight
				/ static_cast<double>(p_mlmtpr->species_count);
			const int species_begin = p_mlmtpr->LinearCoeffOffset();
			for (int type = 0; type < p_mlmtpr->species_count; ++type)
				diagonal[species_begin + type] += curvature;
		}
		if (scalar_head_l2_regularization != 0.0
		    && p_mlmtpr->alpha_scalar_moments > 0) {
			const int scalar_begin =
				p_mlmtpr->LinearCoeffOffset() + p_mlmtpr->species_count;
			const double base_curvature =
				2.0 * scalar_head_l2_regularization
				/ static_cast<double>(p_mlmtpr->alpha_scalar_moments);
			for (int q = 0; q < p_mlmtpr->alpha_scalar_moments; ++q) {
				const double mult =
					q < static_cast<int>(p_mlmtpr->linear_mults.size())
						? p_mlmtpr->linear_mults[q]
						: 1.0;
				diagonal[scalar_begin + q] +=
					base_curvature * mult * mult;
			}
		}
		if (scalar_head_l2_regularization != 0.0
		    && p_mlmtpr->TwoLayerGateReadoutCoeffCount() > 0) {
			const int readout_count =
				p_mlmtpr->TwoLayerGateReadoutCoeffCount();
			const double curvature =
				2.0 * scalar_head_l2_regularization
				/ static_cast<double>(readout_count);
			const int readout_begin =
				p_mlmtpr->TwoLayerGateReadoutCoeffOffset();
			for (int q = 0; q < readout_count; ++q)
				diagonal[readout_begin + q] += curvature;
		}
		if (p_mlmtpr->TwoLayerGateUsesBodyLinearCombo()) {
			const int scalar_count = p_mlmtpr->TwoLayerGateWeightCount();
			if (gate_scalar_l2_regularization != 0.0
			    && scalar_count > 0) {
				const double curvature =
					2.0 * gate_scalar_l2_regularization
					/ static_cast<double>(scalar_count);
				const int offset = p_mlmtpr->TwoLayerGateWeightOffset();
				for (int i = 0; i < scalar_count; ++i)
					diagonal[offset + i] += curvature;
			}
			const int mix_count =
				p_mlmtpr->TwoLayerGateBodyMixWeightCount();
			if (gate_mix_l2_regularization != 0.0 && mix_count > 0) {
				const double curvature =
					2.0 * gate_mix_l2_regularization
					/ static_cast<double>(mix_count);
				const int offset =
					p_mlmtpr->TwoLayerGateBodyMixWeightOffset();
				for (int i = 0; i < mix_count; ++i)
					diagonal[offset + i] += curvature;
			}
		} else if (p_mlmtpr->TwoLayerGateUsesFullScalarWeights()) {
			const int count = p_mlmtpr->TwoLayerGateWeightCount();
			if (gate_full_l2_regularization != 0.0 && count > 0) {
				const double curvature =
					2.0 * gate_full_l2_regularization
					/ static_cast<double>(count);
				const int offset = p_mlmtpr->TwoLayerGateWeightOffset();
				for (int i = 0; i < count; ++i)
					diagonal[offset + i] += curvature;
			}
		}

		if (p_mlmtpr->linear_mults != linear_mults_reference)
			ERROR("MTPR_trainer::Train(): BFGS coordinate scale construction modified linear multipliers.");
		p_mlmtpr->max_linear = max_linear_reference;
		p_mlmtpr->linear_coeffs = linear_coeffs_reference;
		require_live_coeff_pointer("BFGS coordinate scale construction");
		if (p_mlmtpr->linear_coeffs != linear_coeffs_reference)
			ERROR("MTPR_trainer::Train(): BFGS coordinate scale construction modified the linear coefficient cache.");

		for (int full_idx = 0; full_idx < n; ++full_idx) {
			if (x[full_idx] != physical_coeff_reference[full_idx])
				ERROR("MTPR_trainer::Train(): BFGS coordinate scale construction "
				      "modified physical coefficient " + std::to_string(full_idx));
		}

		const double scale_min_limit = 1.0e-8;
		const double scale_max_limit = 1.0e8;
		for (const CoordinateNormalizationBlock& block :
		     coordinate_normalization_blocks) {
			std::vector<double> positive_diagonal;
			positive_diagonal.reserve(block.full_indices.size());
			for (int full_idx : block.full_indices) {
				const double value = diagonal[full_idx];
				if (!std::isfinite(value) || value < 0.0)
					ERROR("MTPR_trainer::Train(): invalid BFGS coordinate diagonal.");
				if (value > 0.0)
					positive_diagonal.push_back(value);
			}
			const double median = MedianPositive(positive_diagonal);
			double raw_min = std::numeric_limits<double>::infinity();
			double raw_max = 0.0;
			double scale_min = std::numeric_limits<double>::infinity();
			double scale_max = 0.0;
			int floored_count = 0;
			int identity_count = 0;
			const double block_floor = bfgs_coordinate_floor * median;
			const double reference_curvature =
				median > 0.0 ? std::min(median, 1.0) : 1.0;
			for (int full_idx : block.full_indices) {
				const double raw = diagonal[full_idx];
				raw_min = std::min(raw_min, raw);
				raw_max = std::max(raw_max, raw);
				if (median > 0.0 && raw < block_floor)
					++floored_count;
				double scale =
					sus2::DiagonalCoordinateScaling::ScaleFromCurvature(
						raw, median, bfgs_coordinate_floor);
				if (median == 0.0
				    || (raw == 0.0 && bfgs_coordinate_floor == 0.0))
					++identity_count;
				scale = std::max(
					scale_min_limit,
					std::min(scale, scale_max_limit));
				if (!sus2::DiagonalCoordinateScaling::IsValidScale(scale))
					ERROR("MTPR_trainer::Train(): invalid stabilized BFGS coordinate scale.");
				coordinate_scale_by_full[full_idx] = scale;
				coordinate_reference_by_full[full_idx] = x[full_idx];
				scale_min = std::min(scale_min, scale);
				scale_max = std::max(scale_max, scale);
			}
			if (prank == 0) {
				std::cout << "[" << CurrentTimestamp() << "] "
				          << "BFGS coordinate normalization block"
				          << " name=" << block.name
				          << " count=" << block.full_indices.size()
				          << " h_min=" << raw_min
				          << " h_median=" << median
				          << " h_max=" << raw_max
				          << " h_reference=" << reference_curvature
				          << " scale_min=" << scale_min
				          << " scale_max=" << scale_max
				          << " floored=" << floored_count
				          << " identity=" << identity_count
				          << std::endl;
			}
		}
		coordinate_normalization_initialized = true;

		double build_seconds = 0.0;
#ifdef MLIP_MPI
		build_seconds = MPI_Wtime() - build_start;
#else
		build_seconds =
			static_cast<double>(std::clock()) / CLOCKS_PER_SEC - build_start;
#endif
		if (prank == 0) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "BFGS coordinate normalization initialized"
			          << " mode=efs-gn"
			          << " probes=" << bfgs_coordinate_probes
			          << " selected_count=" << selected_full_indices.size()
			          << " floor=" << bfgs_coordinate_floor
			          << " seconds=" << build_seconds
			          << " stage=fixed"
			          << " physical_reference=exact"
			          << " coordinate_chart=centered"
			          << " model_format=physical"
			          << std::endl;
		}
	};
	auto full_inv_hess_diag_value = [&](int full_idx) {
		double value = 1.0;
		if (coordinate_normalization_initialized
		    && coordinate_normalization_block_by_full[full_idx] >= 0)
			return value;
		if (p_mlmtpr->TwoLayerResidualEnabled()) {
			const int e0_begin = p_mlmtpr->TwoLayerResidualE0CoeffOffset();
			const int e0_end = e0_begin + p_mlmtpr->TwoLayerResidualE0CoeffCount();
			if (full_idx >= e0_begin && full_idx < e0_end) {
				const double mult = p_mlmtpr->linear_mults[full_idx - e0_begin];
				return value / (mult * mult);
			}
		}
		const int linear_moment_begin = n - nlin + p_mlmtpr->species_count;
		if (full_idx >= linear_moment_begin && full_idx < n) {
			const double mult = p_mlmtpr->linear_mults[full_idx - linear_moment_begin];
			value /= mult * mult;
		}
		return value;
	};
	auto run_linear_refresh = [&](int step) {
		const std::vector<Neighborhoods>* linear_neighborhoods =
			cache_training_neighborhoods ? &training_neighborhoods : nullptr;
		if (do_lin_rescale) {
			if (prank == 0) {
				std::cout << "[" << CurrentTimestamp() << "] "
				          << "optimizer do-lin rescale before full linear solve"
				          << " step=" << step << std::endl;
			}
			Rescale(*this, *p_mlmtpr, linear_neighborhoods);
		}
		TrainLinear(
			prank,
			training_set,
			linear_neighborhoods,
			"bfgs refresh step " + std::to_string(step));
		require_live_coeff_pointer("TrainLinear refresh");
	};
	const bool initial_train_linear_completed =
		do_lin && do_lin_step_limit > 0;
	if (initial_train_linear_completed)
		run_linear_refresh(0);
	if (!coordinate_normalization_initialized) {
		initialize_bfgs_coordinate_normalization();
		verify_optimizer_coordinate_roundtrip();
	} else {
		set_optimizer_x_from_full_coeffs_collective();
	}
	Array1D inv_hess_diag(opt_n, 1.0);
	auto refresh_inv_hess_diag = [&]() {
		for (int active_idx = 0; active_idx < opt_n; ++active_idx)
			inv_hess_diag[active_idx] =
				full_inv_hess_diag_value(active_coeff_indices[active_idx]);
	};
	refresh_inv_hess_diag();
	optimizer_set_inv_hess_diag(inv_hess_diag);
	const int species_coeff_begin = n - nlin;
	std::vector<int> species_coeff_active_indices;
	for (int i = 0; i < p_mlmtpr->species_count; ++i) {
		const int active_idx = full_to_active_index[species_coeff_begin + i];
		if (active_idx >= 0)
			species_coeff_active_indices.push_back(active_idx);
	}
	auto mask_frozen_coordinates = [&](bool freeze_species) {
		std::vector<int> frozen_indices;
		if (freeze_species)
			frozen_indices.insert(frozen_indices.end(), species_coeff_active_indices.begin(), species_coeff_active_indices.end());
		if (!frozen_indices.empty())
			optimizer_mask_coordinates(frozen_indices);
	};
		auto reset_optimizer_state = [&]() {
			optimizer_restart();
			optimizer_iter_step() = 0;
			set_optimizer_x_from_full_coeffs();
			optimizer_set_inv_hess_diag(inv_hess_diag);
		};
		auto reset_optimizer_state_collective = [&]() {
			optimizer_restart();
			optimizer_iter_step() = 0;
			set_optimizer_x_from_full_coeffs_collective();
			optimizer_set_inv_hess_diag(inv_hess_diag);
		};
	bool freeze_species_coeffs = do_lin && do_lin_step_limit > 0;
	sus2::OptimizerAcceptedState accepted_state(n);
	const std::string requested_best_pot_name = best_pot_name.empty()
		? sus2::DefaultBestCheckpointPath(curr_pot_name)
		: best_pot_name;
	if (resolved_best_checkpoint_path_.empty())
		resolved_best_checkpoint_path_ = requested_best_pot_name;
	else if (resolved_best_checkpoint_path_ != requested_best_pot_name)
		ERROR("Best checkpoint path changed between optimizer stages");
	if (sus2::CheckpointDestinationsAlias(
	        resolved_best_checkpoint_path_, curr_pot_name))
		ERROR("Best and current checkpoint paths must differ");
	auto save_potential_atomically_root = [&](const std::string& path) {
		if (path.empty())
			return;
		sus2::AtomicWriteCheckpoint(path, [&](const std::string& temporary) {
			p_mlmtpr->Save(temporary);
		});
	};
	auto prepare_best_checkpoint = [&]() {
		if (existing_best_checkpoint_prepared_
		    || resolved_best_checkpoint_path_.empty())
			return;
		const std::string archive = sus2::ArchiveExistingCheckpoint(
			resolved_best_checkpoint_path_,
			".previous-" + CurrentFilenameTimestamp());
		if (!archive.empty() && prank == 0) {
			std::cerr << "[" << CurrentTimestamp() << "] "
			          << optimizer_log_name
			          << " archived existing best checkpoint"
			          << " source=" << resolved_best_checkpoint_path_
			          << " archive=" << archive << std::endl;
		}
		existing_best_checkpoint_prepared_ = true;
	};
	auto save_last_accepted_coeffs = [&]() {
		accepted_state.SetBoundaryBase(x);
	};
	auto save_evaluated_accepted_coeffs = [&](const double* evaluated_x) {
		accepted_state.SetEvaluatedAccepted(evaluated_x);
	};
	auto restore_last_accepted_coeffs = [&]() {
		accepted_state.Restore(x);
	};
	auto rechart_accepted_radial_rows = [&](int step) {
		if (radial_weight_norm_rows.empty())
			return false;

		const size_t row_count = radial_weight_norm_rows.size();
		std::vector<int> next_pivots(row_count, 0);
		std::vector<double> next_pivot_signs(row_count, 1.0);
		std::vector<double> old_radii(row_count, 0.0);
		std::vector<double> new_radii(row_count, 0.0);
		std::vector<int> reasons(row_count, 0);
		std::vector<int> old_pivots(row_count, 0);
		std::vector<double> old_pivot_signs(row_count, 1.0);
		int rechart_state[2] = {0, -1};
		double max_roundtrip_error = 0.0;

		if (prank == 0) {
			std::vector<double> coordinates(
				radial_weight_norm_row_size, 0.0);
			std::vector<double> reconstructed(
				radial_weight_norm_row_size, 0.0);
			for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
				const RadialWeightNormRow& row =
					radial_weight_norm_rows[row_idx];
				old_pivots[row_idx] = row.pivot;
				old_pivot_signs[row_idx] = row.pivot_sign;
				int next_pivot = row.pivot;
				double next_pivot_sign = row.pivot_sign;
				const sus2::RadialWeightNormChart::RechartResult result =
					sus2::RadialWeightNormChart::RechartIfNeeded(
						x + row.full_begin,
						radial_weight_norm_row_size,
						radial_rechart_direction_radius,
						next_pivot,
						next_pivot_sign,
						coordinates.data());
				if (!result.success) {
					rechart_state[1] = static_cast<int>(row_idx);
					break;
				}
				next_pivots[row_idx] = next_pivot;
				next_pivot_signs[row_idx] = next_pivot_sign;
				old_radii[row_idx] = result.old_direction_radius;
				new_radii[row_idx] = result.new_direction_radius;
				reasons[row_idx] = static_cast<int>(result.reason);
				if (!result.recharted)
					continue;

				if (!sus2::RadialWeightNormChart::UnpackWithPivot(
				        coordinates.data(),
				        radial_weight_norm_row_size,
				        next_pivot,
				        next_pivot_sign,
				        reconstructed.data())) {
					rechart_state[1] = static_cast<int>(row_idx);
					break;
				}
				for (int xi = 0; xi < radial_weight_norm_row_size; ++xi) {
					const double physical_value = x[row.full_begin + xi];
					const double abs_error = std::abs(
						reconstructed[xi] - physical_value);
					max_roundtrip_error = std::max(
						max_roundtrip_error, abs_error);
					const double tolerance = 64.0
						* std::numeric_limits<double>::epsilon()
						* std::max(1.0, std::abs(physical_value));
					if (abs_error > tolerance) {
						rechart_state[1] = static_cast<int>(row_idx);
						break;
					}
				}
				if (rechart_state[1] >= 0)
					break;
				++rechart_state[0];
			}
		}

#ifdef MLIP_MPI
		MPI_Bcast(rechart_state, 2, MPI_INT, 0, train_comm);
#endif
		if (rechart_state[1] >= 0) {
			const RadialWeightNormRow& row =
				radial_weight_norm_rows[rechart_state[1]];
			ERROR("MTPR_trainer::Train(): radial weight-normalization rechart failed"
			      + std::string(" block=")
			      + (row.is_gate1 ? "gate1" : "outer")
			      + " mu=" + std::to_string(row.mu));
		}
		if (rechart_state[0] == 0)
			return false;

#ifdef MLIP_MPI
		MPI_Bcast(next_pivots.data(),
		          static_cast<int>(row_count),
		          MPI_INT,
		          0,
		          train_comm);
		MPI_Bcast(next_pivot_signs.data(),
		          static_cast<int>(row_count),
		          MPI_DOUBLE,
		          0,
		          train_comm);
#endif
		for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
			radial_weight_norm_rows[row_idx].pivot = next_pivots[row_idx];
			radial_weight_norm_rows[row_idx].pivot_sign =
				next_pivot_signs[row_idx];
		}
		reset_optimizer_state_collective();
		mask_frozen_coordinates(freeze_species_coeffs);

		if (prank == 0) {
			for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
				if (old_pivots[row_idx] == next_pivots[row_idx]
				    && old_pivot_signs[row_idx] == next_pivot_signs[row_idx])
					continue;
				const RadialWeightNormRow& row =
					radial_weight_norm_rows[row_idx];
				std::cerr << "[" << CurrentTimestamp() << "] "
				          << optimizer_log_name
				          << " radial weight-normalization rechart"
				          << " step=" << step
				          << " block=" << (row.is_gate1 ? "gate1" : "outer")
				          << " mu=" << row.mu
				          << " reason="
				          << (reasons[row_idx]
				                  == sus2::RadialWeightNormChart::kRechartPackFailed
				              ? "pack-failed"
				              : "direction-radius")
				          << " old_pivot=" << old_pivots[row_idx]
				          << " old_sign=" << old_pivot_signs[row_idx]
				          << " new_pivot=" << next_pivots[row_idx]
				          << " new_sign=" << next_pivot_signs[row_idx]
				          << " old_radius=" << old_radii[row_idx]
				          << " new_radius=" << new_radii[row_idx]
				          << std::endl;
			}
			std::cerr << "[" << CurrentTimestamp() << "] "
			          << optimizer_log_name
			          << " reset after radial rechart"
			          << " step=" << step
			          << " rows=" << rechart_state[0]
			          << " max_physical_roundtrip_error="
			          << max_roundtrip_error << std::endl;
		}
		return true;
	};
	auto optimizer_state_is_finite = [&]() {
		return std::isfinite(bfgs_f)
			&& IsFiniteArray(&bfgs_g[0], opt_n)
			&& FirstNonFinite(p_mlmtpr->Coeff(), n) < 0;
	};
	auto is_ascent_direction_failure = [](const std::string& reason) {
		return reason.find("stepping in accend direction") != std::string::npos
			|| reason.find("stepping in ascend direction") != std::string::npos
			|| reason.find("failed to produce a descent direction") != std::string::npos;
	};
	auto is_linesearch_stagnation_failure = [](const std::string& reason) {
		return reason.find("Linesearch stagnated") != std::string::npos;
	};
	auto recover_optimizer_state_from_current_coeffs = [&](const std::string& reason, int step) {
			reset_optimizer_state();
			mask_frozen_coordinates(freeze_species_coeffs);
			if (prank == 0) {
			std::cerr << "[" << CurrentTimestamp() << "] "
			          << optimizer_log_name << " state reset"
			          << " step=" << step
			          << " reason=" << reason << std::endl;
		}
	};
	mask_frozen_coordinates(freeze_species_coeffs);
	save_last_accepted_coeffs();
	if (prank == 0
	    && (!curr_pot_name.empty() || !resolved_best_checkpoint_path_.empty())) {
		std::cout << "[" << CurrentTimestamp() << "] "
		          << optimizer_log_name << " checkpoint safety enabled"
		          << " current="
		          << (curr_pot_name.empty() ? "disabled" : curr_pot_name)
		          << " best="
		          << (resolved_best_checkpoint_path_.empty()
		                  ? "disabled"
		                  : resolved_best_checkpoint_path_)
		          << " writes=atomic accepted-only" << std::endl;
	}
	if (prank == 0 && freeze_scal_coeffs) {
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "scal_coeffs excluded from " << optimizer_log_name << " coordinates"
		          << " frozen_count=" << (scal_coeff_end - scal_coeff_begin)
		          << " active_count=" << opt_n
		          << " full_count=" << n
		          << " redundant_radial_species_count=" << redundant_radial_species_count
		          << std::endl;
	} else if (prank == 0 && redundant_radial_species_count > 0) {
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "unused radial species coefficients excluded from " << optimizer_log_name << " coordinates"
		          << " redundant_count=" << redundant_radial_species_count
		          << " active_count=" << opt_n
		          << " full_count=" << n << std::endl;
	}
	if (prank == 0 && use_radial_weight_norm) {
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "outer radial weight-normalization coordinates enabled"
		          << " outer_rows=" << outer_radial_weight_norm_row_count
		          << " gate1_rows=" << gate1_radial_weight_norm_row_count
		          << " outer_zero_fallback_rows="
		          << outer_radial_weight_norm_zero_fallback_count
		          << " gate1_zero_fallback_rows="
		          << gate1_radial_weight_norm_zero_fallback_count
		          << " total_rows=" << radial_weight_norm_rows.size()
		          << " row_size=" << radial_weight_norm_row_size
		          << " rechart_radius=" << radial_rechart_direction_radius
		          << " trial_radius_limit="
		          << radial_trial_direction_radius_limit
		          << " physical_load=roundtrip-verified"
		          << " max_abs_error=" << radial_weight_norm_roundtrip_max_abs
		          << " model_format=physical"
		          << std::endl;
	}
	if (prank == 0 && p_mlmtpr->TwoLayerResidualEnabled()
	    && two_layer_residual_stage != kResidualStageFull) {
		const char* stage_name =
			(two_layer_residual_stage == kResidualStageE0) ? "E0" : "E1";
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "residual two-layer optimizer stage filter"
		          << " stage=" << stage_name
		          << " active_count=" << opt_n
		          << " full_count=" << n << std::endl;
	}
	bfgs_g.resize(opt_n);
	std::vector<double> optimizer_reduce_local(opt_n + kOptimizerReducePrefix, 0.0);
	std::vector<double> optimizer_reduce_global(opt_n + kOptimizerReducePrefix, 0.0);
	std::array<double, kAcceptedDiagDoubleCount> accepted_diag_local{};
	std::array<double, kAcceptedDiagDoubleCount> accepted_diag_global{};
	std::array<long long, kAcceptedDiagCountCount> accepted_count_local{};
	std::array<long long, kAcceptedDiagCountCount> accepted_count_global{};

		int num_step = 0;
		bool startup_objective_logged = false;
		int total_linesearch_restarts = 0;
		const int max_total_linesearch_restarts = 3;

		double linf = 9e99;
		double loss_reduced_by = 0.0;
	double loss_prev = 9e99;
	double accepted_loss_reference = 0.0;
	bool accepted_loss_reference_valid = false;
	double std_l = 0.0;
	double stdd_l = 0.0;
	double mean_1_l = 0.0;
	double mean_2_l = 0.0;
	double mean_3_l = 0.0;
	double energy_mae_mev_atom_l = 0.0;
	double energy_rmse_mev_atom_l = 0.0;
	double force_mae_mev_a_l = 0.0;
	double force_rmse_mev_a_l = 0.0;
	double stress_mae_ev_l = 0.0;
	double stress_rmse_ev_l = 0.0;



		bool converge = false;
		bool restored_rejected_linesearch = false;

		double max_shift = 0.1*random_perturb;
		double cooling_rate = 0.2;
		bool linesearch = false;
		const bool use_armijo_value_line_search = (bfgs_line_search == "armijo-value");
		auto abort_if_any_rank_reports = [&](int local_status, const std::string& message) {
		int global_status = local_status;
#ifdef MLIP_MPI
		MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, train_comm);
#endif
		if (global_status != 0)
			ERROR(message);
	};
	auto run_root_checkpoint_io_collective = [&]
		(const std::string& context, const std::function<void()>& root_action) {
		int io_failed = 0;
		std::string io_error;
		if (prank == 0) {
			try {
				root_action();
			} catch (const MlipException& exception) {
				io_failed = 1;
				io_error = exception.What();
			} catch (const std::exception& exception) {
				io_failed = 1;
				io_error = exception.what();
			} catch (...) {
				io_failed = 1;
				io_error = "unknown checkpoint I/O exception";
			}
		}
#ifdef MLIP_MPI
		MPI_Bcast(&io_failed, 1, MPI_INT, 0, train_comm);
		int error_length = prank == 0
			? static_cast<int>(io_error.size())
			: 0;
		MPI_Bcast(&error_length, 1, MPI_INT, 0, train_comm);
		if (prank != 0)
			io_error.resize(error_length);
		if (error_length > 0)
			MPI_Bcast(&io_error[0], error_length, MPI_CHAR, 0, train_comm);
#endif
		if (io_failed != 0)
			ERROR("Checkpoint I/O failed during " + context + ": " + io_error);
	};
	const bool trace_checkpoint_updates =
		EnvFlagEnabled("SUS2_SH_CHECKPOINT_UPDATE_TRACE");
	auto update_best_checkpoint_collective = [&]
		(double root_loss, int step, const std::string& point_kind) {
		int update_best = 0;
		if (prank == 0 && !resolved_best_checkpoint_path_.empty())
			update_best = best_accepted_checkpoint_.Observe(root_loss, step) ? 1 : 0;
#ifdef MLIP_MPI
		MPI_Bcast(&update_best, 1, MPI_INT, 0, train_comm);
#endif
		if (update_best == 0)
			return;
		run_root_checkpoint_io_collective(
			"best accepted model at step " + std::to_string(step),
			[&]() {
				prepare_best_checkpoint();
				save_potential_atomically_root(resolved_best_checkpoint_path_);
			});
		if (prank == 0 && trace_checkpoint_updates) {
			std::cerr << "[" << CurrentTimestamp() << "] "
			          << optimizer_log_name
			          << " best checkpoint updated"
			          << " kind=" << point_kind
			          << " step=" << step
			          << " total_loss=" << std::setprecision(17)
			          << root_loss
			          << " path=" << resolved_best_checkpoint_path_
			          << std::endl;
		}
	};
	auto require_finite_coeffs_all = [&](const std::string& context) {
		const int local_status = FirstNonFinite(p_mlmtpr->Coeff(), n) >= 0 ? 1 : 0;
		abort_if_any_rank_reports(local_status, context + ": non-finite MTPR coefficients detected.");
	};

	std::random_device random_device;
	std::default_random_engine eng(random_device());
	std::uniform_real_distribution<double> distr(-1, 1);
	std::ofstream bfgs_trace_stream;
	if (prank == 0 && !bfgs_trace_file.empty()) {
		bfgs_trace_stream.open(bfgs_trace_file, std::ios::out | std::ios::trunc);
		if (!bfgs_trace_stream.is_open())
			ERROR("Can't open optimizer trace file " + bfgs_trace_file + " for writing");
		bfgs_trace_stream << "step,total_loss,efs_loss,energy_mae_mev_atom,force_mae_mev_a,stress_mae_ev";
		if (use_lbfgs_optimizer)
			bfgs_trace_stream << ",lbfgs_history,lbfgs_step,lbfgs_gamma,lbfgs_curvature,lbfgs_relative_curvature,lbfgs_backtracks,lbfgs_damped_updates,lbfgs_skipped_updates";
		bfgs_trace_stream << '\n';
		bfgs_trace_stream << std::scientific << std::setprecision(17);
	}
	int trace_lines_since_flush = 0;
	// Reuse this buffer for every objective evaluation.  An optimizer step can
	// contain many line-search trials, so allocating an n-sized vector in the
	// inner loop is unnecessary overhead for large readout models.
	std::vector<double> evaluated_coeffs(n, 0.0);

		while (!converge)
		{
			bool restarted_rejected_linesearch_this_iteration = false;
			bool accepted_evaluated_point_this_iteration = false;
			bool evaluated_boundary_base_this_iteration = false;
			//if (prank==0)
			//cout << "itr" << endl;

		if (!linesearch)
		{
			// A successful reverse-communication BFGS call leaves optimizer.x at
			// the first *unevaluated* trial of the next line search even though
			// is_in_linesearch() is false.  Boundary work must start from the
			// separately saved evaluated/accepted physical point.
			const bool boundary_had_pending_trial =
				accepted_state.RestoreAcceptedForBoundary(x);

			bool external_x_modified = false;
			bool external_x_needs_broadcast = false;
			bool linear_refresh_applied = false;
			const bool random_perturb_applied = (max_shift != 0.0);
			if (prank == 0) {
				for (int i = 0; i < n - nlin; i++) {
					if (full_to_active_index[i] < 0)
						continue;
					if (p_mlmtpr->IsRadialFirstCoeff(i)) {
						const double raw_value = p_mlmtpr->RadialFirstCoeffValueToRaw(x[i])
							+ distr(eng) * max_shift;
						x[i] = p_mlmtpr->RadialFirstCoeffRawToValue(raw_value);
					} else {
						x[i] += distr(eng)*max_shift;
					}
				}
			}
			external_x_modified = random_perturb_applied;
			external_x_needs_broadcast = random_perturb_applied;

			const bool should_freeze_species_coeffs = do_lin && num_step < do_lin_step_limit;
			const bool freeze_species_state_changed =
				should_freeze_species_coeffs != freeze_species_coeffs;
			if (should_freeze_species_coeffs != freeze_species_coeffs) {
				freeze_species_coeffs = should_freeze_species_coeffs;
				if (prank == 0) {
					std::cout << "[" << CurrentTimestamp() << "] "
					          << "species_coeffs optimizer freeze "
					          << (freeze_species_coeffs ? "enabled" : "disabled")
					          << " at step=" << num_step << std::endl;
				}
			} else if (freeze_species_coeffs) {
				mask_frozen_coordinates(freeze_species_coeffs);
			}

			const bool run_train_linear = do_lin
				&& num_step < do_lin_step_limit
				&& (num_step % do_lin_frequency == 0)
				&& !(initial_train_linear_completed && num_step == 0);
			if (run_train_linear) {
				run_linear_refresh(num_step);
				linear_refresh_applied = true;
				external_x_modified = true;
				external_x_needs_broadcast = false; // TrainLinear already broadcasts coeffs.
			}

#ifdef MLIP_MPI
			if (external_x_needs_broadcast)
				MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, train_comm);
#endif
			// Rechart before every path that packs externally modified physical
			// coefficients. Otherwise an external perturbation can hit the old
			// chart's excluded antipode before the automatic repair gets a chance
			// to run.
			const bool radial_recharted =
				rechart_accepted_radial_rows(num_step);
			if (radial_recharted) {
				linf = 9e99;
				loss_reduced_by = 0.0;
				loss_prev = 9e99;
				if (prank == 0)
					std::cerr << "[" << CurrentTimestamp() << "] "
					          << optimizer_log_name
					          << " convergence window reset after radial rechart"
					          << " step=" << num_step << std::endl;
			} else if (freeze_species_state_changed) {
				reset_optimizer_state_collective();
			} else if (linear_refresh_applied) {
				// An exact linear refresh changes the optimizer base point without
				// supplying a secant pair. Curvature accumulated at the old base is
				// no longer valid.
				reset_optimizer_state_collective();
			} else if (external_x_modified) {
				// Preserve the historical random-perturbation policy: rebase the
				// transient line-search state but retain the inverse-Hessian estimate.
				set_optimizer_x_from_full_coeffs_collective();
			}
			const bool boundary_rebased_optimizer =
				radial_recharted || freeze_species_state_changed
				|| external_x_modified;
			if (external_x_modified)
				accepted_loss_reference_valid = false;
			if (!(boundary_had_pending_trial && !boundary_rebased_optimizer)) {
				// Initial state, or an intentional boundary reset/refresh.  This is
				// now the base to which a rejected line search must return.
				save_last_accepted_coeffs();
				evaluated_boundary_base_this_iteration = true;
			}
			// Otherwise no accepted-point operation changed the optimizer state.
			// The common objective-materialization path immediately below restores
			// the pending trial, avoiding a duplicate physical-model MPI broadcast.
			mask_frozen_coordinates(freeze_species_coeffs);
		}

				require_live_coeff_pointer("optimizer coordinate unpack");
				copy_optimizer_x_to_full_coeffs();
				std::copy(x, x + n, evaluated_coeffs.begin());
				if (!startup_objective_logged
				    && !coordinate_physical_reference.empty()) {
					double max_abs_delta = 0.0;
					int max_abs_delta_index = -1;
					for (int full_idx = 0; full_idx < n; ++full_idx) {
						const double delta = std::abs(
							x[full_idx] - coordinate_physical_reference[full_idx]);
						if (delta > max_abs_delta) {
							max_abs_delta = delta;
							max_abs_delta_index = full_idx;
						}
					}
					if (prank == 0) {
						std::cout << "[" << CurrentTimestamp() << "] "
						          << "optimizer pre-objective physical delta"
						          << " max_abs=" << std::scientific
						          << std::setprecision(17) << max_abs_delta
						          << " index=" << max_abs_delta_index
						          << std::defaultfloat << std::endl;
					}
				}
			require_finite_coeffs_all(optimizer_log_name + " trial step " + std::to_string(num_step));

			const std::vector<Neighborhoods>* objective_neighborhoods =
				cache_training_neighborhoods ? &training_neighborhoods : nullptr;
			bool armijo_value_trial = false;
			if (use_armijo_value_line_search
			    && (optimizer_replicated || prank == 0))
				armijo_value_trial = optimizer_armijo_trial_ready();
	#ifdef MLIP_MPI
			// In root-only dense BFGS, non-root optimizers are synchronized with
			// Set_x(), which deliberately clears their local line-search state.
			// Broadcast root's protocol state before choosing an objective path so
			// every rank enters matching MPI collectives.
			if (use_armijo_value_line_search && !optimizer_replicated) {
				int armijo_value_trial_state = armijo_value_trial ? 1 : 0;
				MPI_Bcast(
					&armijo_value_trial_state, 1, MPI_INT, 0, train_comm);
				armijo_value_trial = (armijo_value_trial_state != 0);
			}
	#endif
			bool armijo_value_trial_accepted = false;
			bool armijo_value_trial_rejected = false;
			double armijo_value_only_loss = std::numeric_limits<double>::quiet_NaN();
			if (armijo_value_trial) {
				ObjectiveFunctionAndCachePredictions(training_set, objective_neighborhoods);
				loss_ /= K;
				std_ /= K;
				stdd_ /= K;
				mean_1 /= K;
				mean_2 /= K;
				mean_3 /= K;
	#ifdef MLIP_MPI
				optimizer_reduce_local[0] = loss_;
				MPI_Reduce(optimizer_reduce_local.data(),
				           prank == 0 ? optimizer_reduce_global.data() : optimizer_reduce_local.data(),
				           1,
				           MPI_DOUBLE,
				           MPI_SUM,
				           0,
				           train_comm);
				if (prank == 0)
					bfgs_f = optimizer_reduce_global[0];
				if (optimizer_replicated)
					MPI_Bcast(&bfgs_f, 1, MPI_DOUBLE, 0, train_comm);
		#else
				bfgs_f = loss_;
		#endif
				armijo_value_only_loss = bfgs_f;
				int armijo_accept_state = 0;
				if (optimizer_replicated || prank == 0)
					armijo_accept_state = optimizer_armijo_accepts(bfgs_f) ? 1 : 0;
	#ifdef MLIP_MPI
				MPI_Bcast(&armijo_accept_state, 1, MPI_INT, 0, train_comm);
	#endif
				armijo_value_trial_accepted = (armijo_accept_state != 0);
				armijo_value_trial_rejected = !armijo_value_trial_accepted;
				if (armijo_value_trial_rejected
				    && (optimizer_replicated || prank == 0)) {
					optimizer_armijo_backtrack(bfgs_f);
					mask_frozen_coordinates(freeze_species_coeffs);
				}
			}

			if (!armijo_value_trial_rejected) {
				if (armijo_value_trial)
					CalcObjectiveFunctionGradFromCachedPredictions(training_set, objective_neighborhoods);
				else
					CalcObjectiveFunctionGrad(training_set, objective_neighborhoods);

				loss_ /= K;
				std_ /= K;
				stdd_/=K;
				mean_1 /= K;
				mean_2 /= K;
				mean_3 /= K;
				for (int i = 0; i < n; i++)
					loss_grad_[i] /= K;
				pack_optimizer_gradient();

	#ifdef MLIP_MPI
				optimizer_reduce_local[0] = loss_;
				for (int active_idx = 0; active_idx < opt_n; ++active_idx)
					optimizer_reduce_local[kOptimizerReducePrefix + active_idx] =
						optimizer_gradient_cache[active_idx];
				MPI_Reduce(optimizer_reduce_local.data(),
				           prank == 0 ? optimizer_reduce_global.data() : optimizer_reduce_local.data(),
				           opt_n + kOptimizerReducePrefix,
				           MPI_DOUBLE,
				           MPI_SUM,
				           0,
				           train_comm);
				if (prank == 0) {
					bfgs_f = optimizer_reduce_global[0];
					std::memcpy(&bfgs_g[0],
					            optimizer_reduce_global.data() + kOptimizerReducePrefix,
					            opt_n * sizeof(double));
				}

	#else
				bfgs_f = loss_;
				for (int active_idx = 0; active_idx < opt_n; ++active_idx)
					bfgs_g[active_idx] = optimizer_gradient_cache[active_idx];
		#endif
				if (!startup_objective_logged
				    && !optimizer_is_in_linesearch()) {
					if (prank == 0) {
						std::ostringstream value;
						value << std::scientific << std::setprecision(17)
						      << bfgs_f;
						std::cout << "[" << CurrentTimestamp() << "] "
						          << "optimizer physical startup objective"
						          << " total_loss=" << value.str()
						          << " coordinate_mode="
						          << bfgs_coordinate_normalization
						          << " radial_parameterization="
						          << radial_parameterization
						          << " stage=before-first-proposal"
						          << std::endl;
					}
					startup_objective_logged = true;
				}
				if (freeze_species_coeffs) {
					for (int idx : species_coeff_active_indices)
						bfgs_g[idx] = 0.0;
				}

	#ifdef MLIP_MPI
				if (optimizer_replicated) {
					if (prank == 0) {
						optimizer_reduce_global[0] = bfgs_f;
						std::memcpy(optimizer_reduce_global.data() + kOptimizerReducePrefix,
						            &bfgs_g[0],
						            opt_n * sizeof(double));
					}
					MPI_Bcast(optimizer_reduce_global.data(),
					          opt_n + kOptimizerReducePrefix,
					          MPI_DOUBLE,
					          0,
					          train_comm);
					if (prank != 0) {
						bfgs_f = optimizer_reduce_global[0];
						std::memcpy(&bfgs_g[0],
						            optimizer_reduce_global.data() + kOptimizerReducePrefix,
						            opt_n * sizeof(double));
					}
				}
		#endif
			}

			// The cached value-only path is an optimization. The full gradient path
			// remains authoritative for the accepted total loss, so recheck Armijo
			// before passing an accepted-point gradient to either optimizer.
			if (armijo_value_trial && armijo_value_trial_accepted) {
				int gradient_armijo_accept_state = 0;
				if (optimizer_replicated || prank == 0)
					gradient_armijo_accept_state = optimizer_armijo_accepts(bfgs_f) ? 1 : 0;
		#ifdef MLIP_MPI
				MPI_Bcast(&gradient_armijo_accept_state, 1, MPI_INT, 0, train_comm);
		#endif
				if (gradient_armijo_accept_state == 0) {
					armijo_value_trial_accepted = false;
					armijo_value_trial_rejected = true;
					if (optimizer_replicated || prank == 0) {
						optimizer_armijo_backtrack(bfgs_f);
						mask_frozen_coordinates(freeze_species_coeffs);
					}
					if (prank == 0) {
						std::cerr << "[" << CurrentTimestamp() << "] "
						          << optimizer_log_name
						          << " Armijo trial rechecked with gradient loss"
						          << " value_only_loss=" << armijo_value_only_loss
						          << " gradient_loss=" << bfgs_f << std::endl;
					}
				}
			}

			if (evaluated_boundary_base_this_iteration
			    && !armijo_value_trial_rejected) {
				int boundary_status = 0;
				if (prank == 0) {
					boundary_status =
						!std::isfinite(bfgs_f)
						|| FirstNonFinite(&bfgs_g[0], opt_n) >= 0
						|| FirstNonFinite(
							p_mlmtpr->Coeff(), p_mlmtpr->CoeffCount()) >= 0;
				}
				abort_if_any_rank_reports(
					boundary_status,
					"Non-finite evaluated optimizer boundary at step "
						+ std::to_string(num_step)
						+ "; refusing to write best potential.");
				update_best_checkpoint_collective(
					prank == 0 ? bfgs_f : 0.0,
					num_step,
					"evaluated-boundary");
			}

			if (!armijo_value_trial_rejected
			    && (optimizer_replicated || prank == 0) && !converge) {
			const int bad_grad_active_idx = FirstNonFinite(&bfgs_g[0], opt_n);
			if (bad_grad_active_idx >= 0 && prank == 0) {
				const int bad_grad_full_idx = active_coeff_indices[bad_grad_active_idx];
				const std::string dump_name = curr_pot_name.empty()
					? "nonfinite-gradient-step-" + std::to_string(num_step) + ".mtp"
					: curr_pot_name + ".nonfinite-gradient-step-" + std::to_string(num_step) + ".mtp";
				std::cerr << "[" << CurrentTimestamp() << "] "
				          << optimizer_log_name << " non-finite gradient diagnostic"
				          << " step=" << num_step
				          << " active_index=" << bad_grad_active_idx
				          << " full_index=" << bad_grad_full_idx
				          << " gradient=" << bfgs_g[bad_grad_active_idx]
				          << " coeff=" << x[bad_grad_full_idx]
				          << " dump=" << dump_name << std::endl;
				try {
					save_potential_atomically_root(dump_name);
				} catch (const MlipException& exception) {
					std::cerr << "[" << CurrentTimestamp() << "] "
					          << "non-finite diagnostic checkpoint failed: "
					          << exception.What() << std::endl;
				} catch (const std::exception& exception) {
					std::cerr << "[" << CurrentTimestamp() << "] "
					          << "non-finite diagnostic checkpoint failed: "
					          << exception.what() << std::endl;
				}
			}
		}

				auto restart_or_finish_rejected_linesearch =
					[&](const std::string& final_message,
					    const std::string& detail) {
					restore_last_accepted_coeffs();
	#ifdef MLIP_MPI
						if (!optimizer_replicated) {
						MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, train_comm);
					}
	#endif
						reset_optimizer_state_collective();
					accepted_state.DiscardPendingTrial();
					mask_frozen_coordinates(freeze_species_coeffs);
					if (total_linesearch_restarts
					    < max_total_linesearch_restarts) {
						++total_linesearch_restarts;
						linesearch = false;
						restarted_rejected_linesearch_this_iteration = true;
						linf = 9e99;
						loss_reduced_by = 0.0;
						if (prank == 0) {
							logstrm1 << "[" << CurrentTimestamp() << "] "
							         << optimizer_log_name << " restored last accepted coefficients, "
							         << "reset optimizer state, and will retry after rejected "
							         << "line-search trial"
							         << " step=" << num_step
							         << " restart="
							         << total_linesearch_restarts << "/"
							         << max_total_linesearch_restarts;
							if (!detail.empty())
								logstrm1 << " " << detail;
							logstrm1 << endl;
							MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
						}
						if (!curr_pot_name.empty()) {
							run_root_checkpoint_io_collective(
								"restored accepted model at step "
									+ std::to_string(num_step),
								[&]() {
									save_potential_atomically_root(curr_pot_name);
								});
						}
						return;
					}
					restored_rejected_linesearch = true;
					converge = true;
					linesearch = true;
					if (prank == 0) {
						logstrm1 << final_message << endl;
						logstrm1 << "[" << CurrentTimestamp() << "] "
						         << optimizer_log_name << " restored last accepted coefficients after "
						         << "rejected line-search trial; maximum optimizer "
						         << "restart count reached"
						         << " step=" << num_step
						         << " restarts="
						         << total_linesearch_restarts << "/"
						         << max_total_linesearch_restarts;
						if (!detail.empty())
							logstrm1 << " " << detail;
						logstrm1 << endl;
						MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
					}
					if (!curr_pot_name.empty()) {
						run_root_checkpoint_io_collective(
							"final restored accepted model at step "
								+ std::to_string(num_step),
							[&]() {
								save_potential_atomically_root(curr_pot_name);
							});
					}
				};

				const bool optimizer_had_active_trial_before_iterate =
					(optimizer_replicated || prank == 0)
						? (use_armijo_value_line_search
							? armijo_value_trial
							: optimizer_is_in_linesearch())
						: false;
				int bfgs_status = 0;
				int rejected_linesearch_from_exception = 0;
				int recovered_accepted_point_from_ascent = 0;
				std::string rejected_linesearch_exception_message;
				std::string rejected_linesearch_exception_detail;
			if (!armijo_value_trial_rejected
				    && (optimizer_replicated || prank == 0) && !converge) {
					auto run_optimizer_iterate = [&]() {
						for (size_t row_idx = 0;
						     row_idx < radial_weight_norm_rows.size();
						     ++row_idx) {
							const RadialWeightNormRow& row = radial_weight_norm_rows[row_idx];
							const double current_log_norm =
								optimizer_x(row.active_begin);
							if (!std::isfinite(current_log_norm))
								ERROR("MTPR_trainer::Train(): current radial log norm is non-finite.");
							radial_log_norm_reference[row_idx] = current_log_norm;
						}
						mask_frozen_coordinates(freeze_species_coeffs);
						if (use_lbfgs_optimizer)
							lbfgs.IterateArmijoWithGradient(bfgs_f, bfgs_g);
						else if (use_armijo_value_line_search)
							bfgs.IterateArmijoWithGradient(bfgs_f, bfgs_g);
					else
						bfgs.Iterate(bfgs_f, bfgs_g);
					mask_frozen_coordinates(freeze_species_coeffs);

				int scaling_step_reductions = 0;
				bool scaling_step_too_large = true;
				while (scaling_step_too_large) {
					scaling_step_too_large = false;
					for (int full_idx = scal_coeff_begin; full_idx < scal_coeff_end; ++full_idx) {
						const int active_idx = full_to_active_index[full_idx];
						if (active_idx < 0)
							continue;
							if (std::abs(optimizer_x(active_idx) - x[full_idx]) > 0.5) {
							scaling_step_too_large = true;
							break;
						}
					}
						if (scaling_step_too_large) {
								optimizer_reduce_step(0.25);
							if (++scaling_step_reductions > 64)
									ERROR(optimizer_log_name + " could not reduce scaling-coefficient step below the safety limit.");
						}
					}

					const double radial_log_norm_min =
						std::log(std::numeric_limits<double>::min()) + 2.0;
					const double radial_log_norm_max =
						std::log(std::numeric_limits<double>::max()) - 2.0;
					int radial_step_reductions = 0;
					bool radial_step_too_large = true;
					while (radial_step_too_large) {
						radial_step_too_large = false;
						for (size_t row_idx = 0;
						     row_idx < radial_weight_norm_rows.size();
						     ++row_idx) {
							const double trial_log_norm = optimizer_x(
								radial_weight_norm_rows[row_idx].active_begin);
							if (!std::isfinite(trial_log_norm)
							    || trial_log_norm < radial_log_norm_min
							    || trial_log_norm > radial_log_norm_max
							    || std::abs(trial_log_norm
							                - radial_log_norm_reference[row_idx]) > 1.0) {
								radial_step_too_large = true;
								break;
							}
						}
						if (radial_step_too_large) {
							optimizer_reduce_step(0.25);
							if (++radial_step_reductions > 64)
								ERROR(optimizer_log_name
								      + " could not reduce radial log-norm step below the safety limit.");
						}
					}
						RequireFiniteArray(optimizer_data(), opt_n, "optimizer proposed coefficients");
					};
				auto try_recover_optimizer_from_ascent_direction_local =
					[&](const std::string& first_reason) {
					if (!is_ascent_direction_failure(first_reason) || !optimizer_state_is_finite())
						return false;
					try {
						recover_optimizer_state_from_current_coeffs(first_reason, num_step);
						run_optimizer_iterate();
						// An ascent failure after an active Wolfe/accepted Armijo trial
						// occurs only after that evaluated trial passed its acceptance
						// conditions.  The retry has already generated the next trial, so
						// its current is_in_linesearch() value cannot classify the point.
						// The initial base-point gradient has no active trial and remains
						// outside accepted-step accounting.
						recovered_accepted_point_from_ascent =
							sus2::AscentRecoveryPointWasAccepted(
								optimizer_had_active_trial_before_iterate,
								use_armijo_value_line_search,
								armijo_value_trial_accepted)
								? 1 : 0;
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name << " recovered after state reset"
						          << " step=" << num_step << std::endl;
					return true;
				}
				catch (const MlipException& retry_exc) {
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name << " recovery failed"
						          << " step=" << num_step
						          << " reason=" << retry_exc.What() << std::endl;
					return false;
				}
				catch (const std::exception& retry_exc) {
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name << " recovery failed"
						          << " step=" << num_step
						          << " reason=" << retry_exc.what() << std::endl;
					return false;
				}
				catch (...) {
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name << " recovery failed"
						          << " step=" << num_step
						          << " reason=unknown exception" << std::endl;
					return false;
					}
					};
#ifdef MLIP_MPI
				int ascent_recovery_requested = 0;
#endif
				auto handle_optimizer_failure = [&](const std::string& reason) {
					if (is_linesearch_stagnation_failure(reason)
					    && optimizer_state_is_finite()) {
						rejected_linesearch_from_exception = 1;
						rejected_linesearch_exception_message =
							optimizer_log_name
							+ " converged due to linesearch stagnation";
						rejected_linesearch_exception_detail = "reason=" + reason;
						return;
					}
					if (is_ascent_direction_failure(reason)
					    && optimizer_state_is_finite()) {
#ifdef MLIP_MPI
						if (optimizer_replicated) {
							ascent_recovery_requested = 1;
							return;
						}
#endif
						if (try_recover_optimizer_from_ascent_direction_local(reason))
							return;
					}
					bfgs_status = 1;
				};
					try {
						run_optimizer_iterate();
					}
				catch (const MlipException& exc) {
					const std::string reason = exc.What();
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] " << optimizer_log_name << " failed"
					          << " step=" << num_step
					          << " reason=" << reason << std::endl;
					handle_optimizer_failure(reason);
				}
				catch (const std::exception& exc) {
					const std::string reason = exc.what();
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] " << optimizer_log_name << " failed"
					          << " step=" << num_step
					          << " reason=" << reason << std::endl;
					handle_optimizer_failure(reason);
				}
				catch (...) {
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] " << optimizer_log_name << " failed"
					          << " step=" << num_step
					          << " reason=unknown exception" << std::endl;
					handle_optimizer_failure("unknown exception");
				}
	#ifdef MLIP_MPI
			if (optimizer_replicated) {
				int local_outcomes[3] = {
					bfgs_status != 0 ? 1 : 0,
					rejected_linesearch_from_exception != 0 ? 1 : 0,
					ascent_recovery_requested != 0 ? 1 : 0
				};
				int global_outcomes[3] = {0, 0, 0};
				MPI_Allreduce(
					local_outcomes,
					global_outcomes,
					3,
					MPI_INT,
					MPI_MAX,
					train_comm);
				const int outcome_kinds = global_outcomes[0]
					+ global_outcomes[1] + global_outcomes[2];
				if (outcome_kinds > 1) {
					bfgs_status = 1;
					rejected_linesearch_from_exception = 0;
					ascent_recovery_requested = 0;
					if (prank == 0)
						std::cerr << "[" << CurrentTimestamp() << "] "
						          << optimizer_log_name
						          << " inconsistent replicated optimizer outcomes"
						          << " step=" << num_step
						          << " fatal=" << global_outcomes[0]
						          << " stagnation=" << global_outcomes[1]
						          << " ascent=" << global_outcomes[2]
						          << std::endl;
				} else if (global_outcomes[0] != 0) {
					bfgs_status = 1;
				} else if (global_outcomes[1] != 0) {
					rejected_linesearch_from_exception = 1;
					if (prank == 0
					    && rejected_linesearch_exception_message.empty()) {
						rejected_linesearch_exception_message =
							optimizer_log_name
							+ " converged due to linesearch stagnation";
						rejected_linesearch_exception_detail =
							"reason=reported by a non-root replicated optimizer";
					}
				} else if (global_outcomes[2] != 0) {
					// Do not retry inside a rank-local catch: distributed dense BFGS
					// uses MPI collectives.  First agree on recovery, then reset and
					// retry exactly once on every active rank from root's canonical
					// physical model and optimizer coordinates.
					reset_optimizer_state_collective();
					mask_frozen_coordinates(freeze_species_coeffs);
					int retry_status = 0;
					std::string retry_reason;
					try {
						run_optimizer_iterate();
					}
					catch (const MlipException& retry_exc) {
						retry_status = 1;
						retry_reason = retry_exc.What();
					}
					catch (const std::exception& retry_exc) {
						retry_status = 1;
						retry_reason = retry_exc.what();
					}
					catch (...) {
						retry_status = 1;
						retry_reason = "unknown exception";
					}
					int retry_status_global = 0;
					MPI_Allreduce(
						&retry_status,
						&retry_status_global,
						1,
						MPI_INT,
						MPI_MAX,
						train_comm);
					if (retry_status_global != 0) {
						bfgs_status = 1;
						if (prank == 0)
							std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name
							          << " collective recovery failed"
							          << " step=" << num_step
							          << " reason="
							          << (retry_reason.empty()
							                  ? "reported by a non-root optimizer"
							                  : retry_reason)
							          << std::endl;
					} else {
						int recovered_accepted = 0;
						if (prank == 0) {
							recovered_accepted =
								sus2::AscentRecoveryPointWasAccepted(
									optimizer_had_active_trial_before_iterate,
									use_armijo_value_line_search,
									armijo_value_trial_accepted)
									? 1 : 0;
							std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name
							          << " recovered after collective state reset"
							          << " step=" << num_step
							          << std::endl;
						}
						MPI_Bcast(
							&recovered_accepted,
							1,
							MPI_INT,
							0,
							train_comm);
						recovered_accepted_point_from_ascent =
							recovered_accepted;
					}
				}
			}
	#endif
		}
	#ifdef MLIP_MPI
			{
				int rejected_linesearch_global = 0;
			MPI_Allreduce(&rejected_linesearch_from_exception,
			              &rejected_linesearch_global,
			              1,
			              MPI_INT,
			              MPI_MAX,
			              train_comm);
			rejected_linesearch_from_exception = rejected_linesearch_global;
			int message_len =
				static_cast<int>(rejected_linesearch_exception_message.size());
			int detail_len =
				static_cast<int>(rejected_linesearch_exception_detail.size());
			MPI_Bcast(&message_len, 1, MPI_INT, 0, train_comm);
			if (prank != 0)
				rejected_linesearch_exception_message.resize(message_len);
			if (message_len > 0)
				MPI_Bcast(&rejected_linesearch_exception_message[0],
				          message_len,
				          MPI_CHAR,
				          0,
				          train_comm);
			MPI_Bcast(&detail_len, 1, MPI_INT, 0, train_comm);
			if (prank != 0)
				rejected_linesearch_exception_detail.resize(detail_len);
			if (detail_len > 0)
				MPI_Bcast(&rejected_linesearch_exception_detail[0],
				          detail_len,
				          MPI_CHAR,
				          0,
				          train_comm);
		}
#endif
		if (rejected_linesearch_from_exception != 0) {
			restart_or_finish_rejected_linesearch(
				rejected_linesearch_exception_message,
				rejected_linesearch_exception_detail);
		}
			abort_if_any_rank_reports(bfgs_status,
					optimizer_log_name + " failed at step " + std::to_string(num_step)
				+ "; refusing to continue with non-finite optimizer state.");
			if ((optimizer_replicated || prank == 0) && !converge)
				limit_radial_direction_trial();

#ifdef MLIP_MPI
			if (!optimizer_replicated) {
				if (prank == 0)
					copy_optimizer_x_to_full_coeffs();
			MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, train_comm);
			if (prank != 0) {
					set_optimizer_x_from_full_coeffs();
				mask_frozen_coordinates(freeze_species_coeffs);
			}
		}
#endif
			require_finite_coeffs_all(optimizer_log_name + " synchronized step " + std::to_string(num_step));

		int linesearch_stop_reason = 0;
			if ((optimizer_replicated || prank == 0) && !converge) {
				if (optimizer_linesearch_stagnated())
				linesearch_stop_reason = 1;
				else if (optimizer_iter_step() > 30)
				linesearch_stop_reason = 2;
		}
#ifdef MLIP_MPI
		MPI_Bcast(&linesearch_stop_reason, 1, MPI_INT, 0, train_comm);
#endif
			if (linesearch_stop_reason != 0) {
				std::string final_message;
				std::string detail;
				if (prank == 0) {
					if (linesearch_stop_reason == 1) {
							final_message = optimizer_log_name + " ended due to linesearch stagnation";
					} else {
							final_message = optimizer_log_name + " ended due to linesearch more than 30 iterations";
						std::ostringstream oss;
						oss << "d_x=" << optimizer_x(0) - x[0];
						detail = oss.str();
					}
				}
	#ifdef MLIP_MPI
				int final_message_len = static_cast<int>(final_message.size());
				int detail_len = static_cast<int>(detail.size());
				MPI_Bcast(&final_message_len, 1, MPI_INT, 0, train_comm);
				if (prank != 0)
					final_message.resize(final_message_len);
				if (final_message_len > 0)
					MPI_Bcast(&final_message[0], final_message_len, MPI_CHAR, 0, train_comm);
				MPI_Bcast(&detail_len, 1, MPI_INT, 0, train_comm);
				if (prank != 0)
					detail.resize(detail_len);
				if (detail_len > 0)
					MPI_Bcast(&detail[0], detail_len, MPI_CHAR, 0, train_comm);
	#endif
				restart_or_finish_rejected_linesearch(final_message, detail);
					} else if (optimizer_replicated || prank == 0) {
						const bool evaluated_point_accepted =
							sus2::EvaluatedPointWasAccepted(
								recovered_accepted_point_from_ascent != 0,
								use_armijo_value_line_search,
								armijo_value_trial_accepted,
								optimizer_is_in_linesearch());
						linesearch = !evaluated_point_accepted;
				}

	#ifdef MLIP_MPI
				if (!optimizer_replicated) {
					int linesearch_state = linesearch ? 1 : 0;
					MPI_Bcast(&linesearch_state, 1, MPI_INT, 0, train_comm);
					linesearch = (linesearch_state != 0);
				}
			{
				int restarted_state =
					restarted_rejected_linesearch_this_iteration ? 1 : 0;
				MPI_Bcast(&restarted_state, 1, MPI_INT, 0, train_comm);
				restarted_rejected_linesearch_this_iteration =
					(restarted_state != 0);
			}
			#endif

					if (!linesearch && !restarted_rejected_linesearch_this_iteration)
				{
					accepted_evaluated_point_this_iteration = true;
					// Optimizer.x is already the next trial.  Diagnostics and model
					// checkpoints must instead see the point whose objective and
					// gradient were evaluated and accepted above.
					std::copy(evaluated_coeffs.begin(), evaluated_coeffs.end(), x);
					int accepted_loss_increased = 0;
					if (prank == 0 && accepted_loss_reference_valid) {
						const double tolerance = 1.0e-8 * std::max(
							1.0,
							std::max(
								std::abs(accepted_loss_reference),
								std::abs(bfgs_f)));
						if (bfgs_f > accepted_loss_reference + tolerance) {
							accepted_loss_increased = 1;
							std::cerr << "[" << CurrentTimestamp() << "] "
							          << optimizer_log_name
							          << " accepted-loss invariant failed"
							          << " step=" << num_step
							          << " previous=" << accepted_loss_reference
							          << " current=" << bfgs_f
							          << " tolerance=" << tolerance
							          << std::endl;
						}
					}
					abort_if_any_rank_reports(
						accepted_loss_increased,
						optimizer_log_name
							+ " accepted an objective increase at step "
							+ std::to_string(num_step)
							+ "; refusing to lose the last evaluated accepted point.");
					if (prank == 0) {
						accepted_loss_reference = bfgs_f;
						accepted_loss_reference_valid = true;
					}
					accepted_diag_local[0] = metric_energy_abs_sum_;
			accepted_diag_local[1] = metric_energy_sq_weighted_sum_;
			accepted_diag_local[2] = metric_force_abs_component_sum_;
			accepted_diag_local[3] = metric_force_sq_component_sum_;
			accepted_diag_local[4] = metric_stress_abs_component_sum_;
			accepted_diag_local[5] = metric_stress_sq_component_sum_;
			accepted_diag_local[6] = std_;
			accepted_diag_local[7] = stdd_;
			accepted_diag_local[8] = mean_1;
			accepted_diag_local[9] = mean_2;
			accepted_diag_local[10] = mean_3;
			accepted_count_local[0] = metric_energy_atom_count_;
			accepted_count_local[1] = metric_force_component_count_;
			accepted_count_local[2] = metric_stress_component_count_;

#ifdef MLIP_MPI
			MPI_Request accepted_reduce_requests[2];
			MPI_Ireduce(accepted_diag_local.data(),
			            prank == 0 ? accepted_diag_global.data() : accepted_diag_local.data(),
			            kAcceptedDiagDoubleCount,
			            MPI_DOUBLE,
			            MPI_SUM,
			            0,
			            train_comm,
			            &accepted_reduce_requests[0]);
			MPI_Ireduce(accepted_count_local.data(),
			            prank == 0 ? accepted_count_global.data() : accepted_count_local.data(),
			            kAcceptedDiagCountCount,
			            MPI_LONG_LONG_INT,
			            MPI_SUM,
			            0,
			            train_comm,
			            &accepted_reduce_requests[1]);
#else
			accepted_diag_global = accepted_diag_local;
			accepted_count_global = accepted_count_local;
#endif

			if (prank == 0)
			{
				if (loss_prev < bfgs_f)
				{
					max_shift *= (1 - cooling_rate);

					logstrm1 << "*" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}

				if (abs(loss_prev - bfgs_f) < 1e-13)
				{
					converge = true;
						logstrm1 << optimizer_log_name << " ended due to small decr. for 1 iteration" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}

				loss_prev = bfgs_f;
			}

#ifdef MLIP_MPI
			MPI_Waitall(2, accepted_reduce_requests, MPI_STATUSES_IGNORE);
#endif

			int accepted_status = 0;
			double efs_loss = 0.0;
			if (prank == 0)
			{
				std_l = accepted_diag_global[6];
				stdd_l = accepted_diag_global[7];
				mean_1_l = accepted_diag_global[8];
				mean_2_l = accepted_diag_global[9];
				mean_3_l = accepted_diag_global[10];
				const long long energy_mae_count_global = accepted_count_global[0];
				const long long force_mae_count_global = accepted_count_global[1];
				const long long stress_mae_count_global = accepted_count_global[2];
				energy_mae_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * accepted_diag_global[0] / static_cast<double>(energy_mae_count_global) : 0.0;
				energy_rmse_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * std::sqrt(accepted_diag_global[1] / static_cast<double>(energy_mae_count_global)) : 0.0;
				force_mae_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * accepted_diag_global[2] / static_cast<double>(force_mae_count_global) : 0.0;
				force_rmse_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * std::sqrt(accepted_diag_global[3] / static_cast<double>(force_mae_count_global)) : 0.0;
				stress_mae_ev_l = stress_mae_count_global > 0 ?
					accepted_diag_global[4] / static_cast<double>(stress_mae_count_global) : 0.0;
				stress_rmse_ev_l = stress_mae_count_global > 0 ?
					std::sqrt(accepted_diag_global[5] / static_cast<double>(stress_mae_count_global)) : 0.0;

				last_train_error_summary_.energy_mae_mev_atom = energy_mae_mev_atom_l;
				last_train_error_summary_.energy_rmse_mev_atom = energy_rmse_mev_atom_l;
				last_train_error_summary_.force_mae_mev_a = force_mae_mev_a_l;
				last_train_error_summary_.force_rmse_mev_a = force_rmse_mev_a_l;
				last_train_error_summary_.stress_mae_ev = stress_mae_ev_l;
				last_train_error_summary_.stress_rmse_ev = stress_rmse_ev_l;
				have_last_train_error_summary_ = true;

				efs_loss = bfgs_f - std_l - stdd_l;
				const double accepted_values[] = {
					bfgs_f, std_l, stdd_l, efs_loss,
					energy_mae_mev_atom_l, energy_rmse_mev_atom_l,
					force_mae_mev_a_l, force_rmse_mev_a_l,
					stress_mae_ev_l, stress_rmse_ev_l,
					mean_1_l, mean_2_l, mean_3_l
				};
				if (!IsFiniteArray(accepted_values, static_cast<int>(sizeof(accepted_values) / sizeof(accepted_values[0]))))
					accepted_status = 1;
				else if (FirstNonFinite(p_mlmtpr->Coeff(), p_mlmtpr->CoeffCount()) >= 0)
					accepted_status = 1;
			}
			abort_if_any_rank_reports(accepted_status,
					"Non-finite accepted " + optimizer_log_name + " state at step " + std::to_string(num_step)
					+ "; refusing to write current potential.");
			save_evaluated_accepted_coeffs(evaluated_coeffs.data());
			update_best_checkpoint_collective(
				prank == 0 ? bfgs_f : 0.0,
				num_step,
				"accepted-trial");
			if (!curr_pot_name.empty()) {
				run_root_checkpoint_io_collective(
					"latest accepted model at step " + std::to_string(num_step),
					[&]() {
						save_potential_atomically_root(curr_pot_name);
					});
			}

			if (prank == 0)
			{
				if (bfgs_trace_stream.is_open()) {
					bfgs_trace_stream << num_step << ','
					                  << bfgs_f << ','
						                  << efs_loss << ','
						                  << energy_mae_mev_atom_l << ','
						                  << force_mae_mev_a_l << ','
						                  << stress_mae_ev_l;
						if (use_lbfgs_optimizer) {
							bfgs_trace_stream << ',' << lbfgs.history_size()
							                  << ',' << lbfgs.last_accepted_step()
							                  << ',' << lbfgs.gamma()
							                  << ',' << lbfgs.last_curvature()
							                  << ',' << lbfgs.last_relative_curvature()
							                  << ',' << lbfgs.total_backtracks()
							                  << ',' << lbfgs.damped_update_count()
							                  << ',' << lbfgs.skipped_update_count();
						}
						bfgs_trace_stream << '\n';
					if (++trace_lines_since_flush >= kTraceFlushInterval) {
						bfgs_trace_stream.flush();
						trace_lines_since_flush = 0;
					}
				}
				logstrm1 << "[" << CurrentTimestamp() << "] "
						 << "step=" << num_step
						 << " total_loss=" << std::fixed << std::setprecision(3) << bfgs_f
						 << " efs_loss=" << std::fixed << std::setprecision(3) << efs_loss
						 << " energy_MAE(meV/atom)=" << std::fixed << std::setprecision(3) << energy_mae_mev_atom_l
						 << " force_MAE(meV/A)=" << std::fixed << std::setprecision(3) << force_mae_mev_a_l
						 << " stress_MAE(eV)=" << std::fixed << std::setprecision(3) << stress_mae_ev_l
						 << endl;
				MLP_LOG("dev", logstrm1.str()); logstrm1.str("");

				num_step++;

				if (num_step % 60 == 1) linf = bfgs_f;
				if (num_step % 60 == 0)
				{
					if ((linf - bfgs_f) / bfgs_f < linstop && (linf - bfgs_f) / bfgs_f < loss_reduced_by && num_step > 120)
					{
						converge = true;
							logstrm1 << optimizer_log_name << " ended due to small decr. in 60 iterations" << endl;
						MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
					}
					loss_reduced_by = (linf - bfgs_f) / bfgs_f;
				}

				if (num_step >= max_step_count)
				{
					converge = true;

					logstrm1 << "step limit reached" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}
			}
		}

	#ifdef MLIP_MPI
				int iteration_state[5] = {
					converge ? 1 : 0,
					linesearch ? 1 : 0,
					num_step,
					restored_rejected_linesearch ? 1 : 0,
					accepted_evaluated_point_this_iteration ? 1 : 0
				};
				MPI_Bcast(iteration_state, 5, MPI_INT, 0, train_comm);
				converge = (iteration_state[0] != 0);
				linesearch = (iteration_state[1] != 0);
				num_step = iteration_state[2];
				restored_rejected_linesearch = (iteration_state[3] != 0);
				accepted_evaluated_point_this_iteration =
					(iteration_state[4] != 0);
		#endif
			if (accepted_evaluated_point_this_iteration && converge) {
				// The final live/saved model must remain at the last evaluated
				// accepted point, not at BFGS's unused next proposal.  Otherwise the
				// next loop's common materialization restores the pending trial once.
				accepted_state.DiscardPendingTrial();
			}
		}

			if (prank == 0 && use_lbfgs_optimizer) {
				std::cout << "[" << CurrentTimestamp() << "] L-BFGS summary"
				          << " history=" << lbfgs.history_size()
				          << " accepted_pairs=" << lbfgs.accepted_update_count()
				          << " damped_pairs=" << lbfgs.damped_update_count()
				          << " skipped_pairs=" << lbfgs.skipped_update_count()
				          << " backtracks=" << lbfgs.total_backtracks()
				          << " restarts=" << lbfgs.restart_count()
				          << " non_descent_resets=" << lbfgs.non_descent_reset_count()
				          << std::endl;
			}
			p_mlmtpr->inited = true;
		have_hess = true;
		if (bfgs_trace_stream.is_open())
			bfgs_trace_stream.flush();

		if (restored_rejected_linesearch)
		{
			CalcObjectiveFunctionGrad(training_set, cache_training_neighborhoods ? &training_neighborhoods : nullptr);
			loss_ /= K;
			std_ /= K;
			stdd_ /= K;
			mean_1 /= K;
			mean_2 /= K;
			mean_3 /= K;

			std::array<double, kAcceptedDiagDoubleCount> restored_diag_local{};
			std::array<double, kAcceptedDiagDoubleCount> restored_diag_global{};
			std::array<long long, kAcceptedDiagCountCount> restored_count_local{};
			std::array<long long, kAcceptedDiagCountCount> restored_count_global{};
			restored_diag_local[0] = metric_energy_abs_sum_;
			restored_diag_local[1] = metric_energy_sq_weighted_sum_;
			restored_diag_local[2] = metric_force_abs_component_sum_;
			restored_diag_local[3] = metric_force_sq_component_sum_;
			restored_diag_local[4] = metric_stress_abs_component_sum_;
			restored_diag_local[5] = metric_stress_sq_component_sum_;
			restored_diag_local[6] = std_;
			restored_diag_local[7] = stdd_;
			restored_diag_local[8] = mean_1;
			restored_diag_local[9] = mean_2;
			restored_diag_local[10] = mean_3;
			restored_count_local[0] = metric_energy_atom_count_;
			restored_count_local[1] = metric_force_component_count_;
			restored_count_local[2] = metric_stress_component_count_;

			double restored_loss_global = loss_;
#ifdef MLIP_MPI
			MPI_Reduce(&loss_, &restored_loss_global, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
			MPI_Reduce(restored_diag_local.data(),
			           prank == 0 ? restored_diag_global.data() : restored_diag_local.data(),
			           kAcceptedDiagDoubleCount,
			           MPI_DOUBLE,
			           MPI_SUM,
			           0,
			           train_comm);
			MPI_Reduce(restored_count_local.data(),
			           prank == 0 ? restored_count_global.data() : restored_count_local.data(),
			           kAcceptedDiagCountCount,
			           MPI_LONG_LONG_INT,
			           MPI_SUM,
			           0,
			           train_comm);
#else
			restored_diag_global = restored_diag_local;
			restored_count_global = restored_count_local;
#endif
			if (prank == 0)
			{
				bfgs_f = restored_loss_global;
				std_l = restored_diag_global[6];
				stdd_l = restored_diag_global[7];
				mean_1_l = restored_diag_global[8];
				mean_2_l = restored_diag_global[9];
				mean_3_l = restored_diag_global[10];
				const long long energy_mae_count_global = restored_count_global[0];
				const long long force_mae_count_global = restored_count_global[1];
				const long long stress_mae_count_global = restored_count_global[2];
				energy_mae_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * restored_diag_global[0] / static_cast<double>(energy_mae_count_global) : 0.0;
				energy_rmse_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * std::sqrt(restored_diag_global[1] / static_cast<double>(energy_mae_count_global)) : 0.0;
				force_mae_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * restored_diag_global[2] / static_cast<double>(force_mae_count_global) : 0.0;
				force_rmse_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * std::sqrt(restored_diag_global[3] / static_cast<double>(force_mae_count_global)) : 0.0;
				stress_mae_ev_l = stress_mae_count_global > 0 ?
					restored_diag_global[4] / static_cast<double>(stress_mae_count_global) : 0.0;
				stress_rmse_ev_l = stress_mae_count_global > 0 ?
					std::sqrt(restored_diag_global[5] / static_cast<double>(stress_mae_count_global)) : 0.0;
				last_train_error_summary_.energy_mae_mev_atom = energy_mae_mev_atom_l;
				last_train_error_summary_.energy_rmse_mev_atom = energy_rmse_mev_atom_l;
				last_train_error_summary_.force_mae_mev_a = force_mae_mev_a_l;
				last_train_error_summary_.force_rmse_mev_a = force_rmse_mev_a_l;
				last_train_error_summary_.stress_mae_ev = stress_mae_ev_l;
				last_train_error_summary_.stress_rmse_ev = stress_rmse_ev_l;
				have_last_train_error_summary_ = true;
				logstrm1 << "[" << CurrentTimestamp() << "] "
					         << optimizer_log_name << " recomputed restored final objective after rejected line-search trial"
				         << endl;
				MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
			}
		}

		if (prank == 0)
		{
			const double joint_std_report = need_std_terms ? (std_l / std_scaling) : 0.0;
		const double center_std_report = need_std_terms ? (stdd_l / stdd_scaling) : 0.0;
		logstrm1 << "MTPR training ended:" << "\t joint_std^2:" << joint_std_report <<  "\t center_std^2:"  << center_std_report <<"   " << mean_1_l <<"   " << mean_2_l << "   " << mean_3_l << "\t efs:" << bfgs_f - std_l - 0 * stdd_l << endl;
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
	}
}
#else
#ifdef MLIP_MPI
void MTPR_trainer::Train2(vector<Configuration>& train_set)
{
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (mpi_rank == 0) cout << "Parallel training started (using AlgLib BFGS)" << endl;

	int size = p_mlmtpr->CoeffCount();
	double *coeffs = p_mlmtpr->Coeff();

	alglib::real_1d_array x;
	x.setcontent(size, coeffs);

	//double prev_func, curr_func;
	int needNextIterInt;
	int curr_iter = 0;
	alglib::ae_int_t m = x.length();
	alglib::minlbfgsstate state;
	alglib::minlbfgsreport rep;

	int isPrintFunc = 1;
	double epsx = 0.0;
	double epsg = 0.0;
	double epsf = 1e-13;
	int maxits = 1500;

	int mm = (int)train_set.size(); // train set size on the current core
	int K = 0;                     // train set size over all cores

	K = mm;

	MPI_Allreduce(&mm, &K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	alglib::minlbfgscreate(x.length(), m, x, state);
	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);

	alglib_impl::ae_state _alglib_env_state;
	alglib_impl::ae_state_init(&_alglib_env_state);
	try {
		if (mpi_rank == 0) {
			if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits)) needNextIterInt = 1;
			else needNextIterInt = 0;
		}
		MPI_Bcast(&needNextIterInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
		state.needfg = true;
		while (needNextIterInt == 1) {
			if (state.needfg) {
				if (mpi_rank == 0)
					memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
				MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				CalcObjectiveFunctionGrad(train_set);
				loss_ /= K;
				for (int i = 0; i < size; i++)
					loss_grad_[i] /= K;
				MPI_Reduce(&loss_, &state.f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				std::cout.precision(12);
				if (mpi_rank == 0 && isPrintFunc == 1) {
					std::cout << "curr_iter = " << curr_iter << ", func = " << state.f << std::endl;
				}
				MPI_Reduce(&loss_grad_[0], &state.g[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				//if (mpi_rank == 0) {
				//	for (int i = 0; i < size; i++)
				//		std::cout << state.g[i] << " ";
				//	std::cout << std::endl;
				//}
				if (mpi_rank == 0) {
					if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits))
						needNextIterInt = 1;
					else {
						needNextIterInt = 0;
						//std::cout << "first stop criterion" << std::endl;	
					}
				}
				//check progress of functional minimization each 100 iterations
				//if (mpi_rank == 0 && needNextIterInt == 1) {
				//	if (curr_iter == 0) prev_func = loss_;
				//	if (curr_iter % 100 == 0 && curr_iter != 0) {
				//		curr_func = loss_;
						//std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ";
						//std::cout << fabs((curr_func - prev_func) / prev_func) << std::endl;
				//		if (fabs((curr_func - prev_func) / prev_func) < 1E-5) {
				//			needNextIterInt = 0;
				//			std::cout << "second stop criterion" << std::endl;
				//		}
				//		else {
				//			prev_func = curr_func;
				//		}
				//	}					
				//}
				MPI_Bcast(&needNextIterInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
				if (mpi_rank == 0) {
					curr_iter++;
					//std::cout << curr_iter << std::endl;
				}
				/*if (mpi_rank == 0) {
					for (int i = 0; i < size; i++)
						std::cout << coeffs[i] << ", ";
					std::cout << std::endl;
				}*/
				if (mpi_rank == 0) {
					p_mlmtpr->Save(curr_pot_name);
				}
				continue;
			}
			if (state.xupdated) {
				if (mpi_rank == 0)
					memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
				MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				double loss = ObjectiveFunction(train_set);
				//state.f=f;
				MPI_Allreduce(&loss, &state.f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				if (mpi_rank == 0) {
					if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits)) needNextIterInt = 1;
					else needNextIterInt = 0;
				}
				continue;
			}
			throw alglib::ap_error("ALGLIB: error in 'minlbfgsoptimize' (some derivatives were not provided?)");
		}
		alglib_impl::ae_state_clear(&_alglib_env_state);
	}
	catch (alglib_impl::ae_error_type) {
		throw alglib::ap_error(_alglib_env_state.error_msg);
	}

	if (mpi_rank == 0) {
		alglib::minlbfgsresults(state, x, rep);
		memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
	}
	MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//printf("%d\n", int(rep.terminationtype)); 
	//printf("%d\n", int(rep.iterationscount));
	//printf("%d\n", int(rep.inneriterationscount));
	//printf("%s\n", x.tostring(x.length()).c_str());
}
#endif
#endif
