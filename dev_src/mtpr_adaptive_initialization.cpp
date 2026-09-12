/* Data-balanced fresh initialization for environment-conditioned radial rows.
 *
 * The persisted model remains physical:
 *   d_jlon = c_lon + a_lon * sum_q U_lonq S_jq
 * The optional optimizer-scale metadata records natural physical coordinate
 * scales only; it is not part of prediction and can default to one on load.
 */
#include "mtpr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#ifdef MLIP_MPI
#include <mpi.h>
#endif

namespace {

uint64_t SplitMix64(uint64_t x)
{
	x += UINT64_C(0x9e3779b97f4a7c15);
	x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
	return x ^ (x >> 31);
}

double CenteredUnit(uint64_t seed, int row, int q)
{
	const uint64_t mixed = SplitMix64(
		seed
		^ (UINT64_C(0x6a09e667f3bcc909) * static_cast<uint64_t>(row + 1))
		^ (UINT64_C(0xbb67ae8584caa73b) * static_cast<uint64_t>(q + 1)));
	const double u = static_cast<double>(mixed >> 11)
		* (1.0 / 9007199254740992.0);
	return std::sqrt(3.0) * (2.0 * u - 1.0);
}

bool PositiveFinite(double x)
{
	return std::isfinite(x) && x > 0.0;
}

bool AllFiniteNonnegative(const std::vector<double>& values)
{
	for (double value : values)
		if (!std::isfinite(value) || value < 0.0)
			return false;
	return true;
}

void UpdateRange(double value, double& min_value, double& max_value)
{
	if (!std::isfinite(value))
		return;
	min_value = std::min(min_value, value);
	max_value = std::max(max_value, value);
}

bool RankZero()
{
#ifdef MLIP_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank == 0;
#else
	return true;
#endif
}

void AllreduceInPlace(std::vector<double>& values)
{
#ifdef MLIP_MPI
	if (!values.empty())
		MPI_Allreduce(
			MPI_IN_PLACE,
			values.data(),
			static_cast<int>(values.size()),
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD);
#else
	(void)values;
#endif
}

void AllreduceInPlace(unsigned long long& value)
{
#ifdef MLIP_MPI
	MPI_Allreduce(
		MPI_IN_PLACE,
		&value,
		1,
		MPI_UNSIGNED_LONG_LONG,
		MPI_SUM,
		MPI_COMM_WORLD);
#else
	(void)value;
#endif
}

} // namespace

void MLMTPR::AccumulateAdaptiveRadialNormalizationSums(
	std::vector<Configuration>& training_set,
	const std::vector<Neighborhoods>* training_neighborhoods,
	MLMTPR::AdaptiveRadialNormalizationSums& sums)
{
	if (!OuterUsesAdaptiveRadialCoefficients())
		ERROR("Adaptive radial normalization statistics require adaptive-coefficients outer mode");
	if (!TwoLayerGateEnabled() || !TwoLayerGateUsesFullScalarWeights())
		ERROR("Adaptive radial normalization statistics require a mu-scalar-full two-layer gate");
	if (training_neighborhoods != nullptr
	    && training_neighborhoods->size() != training_set.size())
		ERROR("Adaptive radial normalization statistics received inconsistent neighborhood cache size");
	ValidateOuterModeConfiguration();

	const int Q = TwoLayerGateScalarCount();
	if (Q <= 0 || alpha_scalar_moments <= 0)
		ERROR("Adaptive radial normalization statistics found inconsistent scalar dimensions");
	if (sums.gate_scalar_sq.size() != static_cast<std::size_t>(Q))
		sums.gate_scalar_sq.assign(Q, 0.0);
	if (sums.outer_scalar_sq.size() != static_cast<std::size_t>(alpha_scalar_moments))
		sums.outer_scalar_sq.assign(alpha_scalar_moments, 0.0);
	if (JointReadoutEnabled()) {
		if (sums.joint_scalar_sq.size() != static_cast<std::size_t>(JointReadoutScalarCount()))
			sums.joint_scalar_sq.assign(JointReadoutScalarCount(), 0.0);
	} else {
		sums.joint_scalar_sq.clear();
	}
	if (TensorMessageEnabled()) {
		const std::size_t irreps = static_cast<std::size_t>(tensor_message_lmax_)
			* static_cast<std::size_t>(two_layer_gate_k_rank_);
		if (sums.tensor_irrep_sq.size() != irreps)
			sums.tensor_irrep_sq.assign(irreps, 0.0);
	} else {
		sums.tensor_irrep_sq.clear();
	}

	for (std::size_t cfg_index = 0; cfg_index < training_set.size(); ++cfg_index) {
		std::unique_ptr<Neighborhoods> owned_neighborhoods;
		const Neighborhoods* neighborhoods_ptr = nullptr;
		if (training_neighborhoods != nullptr) {
			neighborhoods_ptr = &(*training_neighborhoods)[cfg_index];
		} else {
			owned_neighborhoods.reset(
				new Neighborhoods(training_set[cfg_index], CutOff()));
			neighborhoods_ptr = owned_neighborhoods.get();
		}
		const Neighborhoods& neighborhoods = *neighborhoods_ptr;
		PrepareAdaptiveRadial(neighborhoods, false, false);
		// Reuse the post-rank gate cache prepared above. Sum every magnetic
		// component together; a component-dependent scale would break rotation
		// equivariance. gate_samples below supplies the common atom count.
		if (TensorMessageEnabled())
			AccumulateTensorMessageIrrepSquares(sums.tensor_irrep_sq);
		for (int atom = 0; atom < static_cast<int>(neighborhoods.size()); ++atom) {
			const double* gate = two_layer_gate_scalar_values_cache_.data()
				+ static_cast<std::size_t>(atom) * Q;
			for (int q = 0; q < Q; ++q) {
				if (!std::isfinite(gate[q]))
					ERROR("Adaptive radial normalization statistics found non-finite gate scalar");
				sums.gate_scalar_sq[q] += gate[q] * gate[q];
			}
			++sums.gate_samples;

			ForwardAdaptiveRadialSite(neighborhoods[atom], atom, nullptr);
			for (int q = 0; q < alpha_scalar_moments; ++q) {
				const int node = alpha_moment_mapping[q];
				const double value = adaptive_radial_execution_.values[node];
				if (!std::isfinite(value))
					ERROR("Adaptive radial normalization statistics found non-finite outer scalar");
				sums.outer_scalar_sq[q] += value * value;
			}
			if (JointReadoutEnabled())
				AccumulateJointReadoutScalarSquares(sums.joint_scalar_sq);
			++sums.outer_samples;
		}
	}
	adaptive_radial_execution_ = AdaptiveRadialExecution();
	two_layer_edge_cache_ready_ = false;
	two_layer_forward_final_moment_cache_ready_ = false;
}

void MLMTPR::InitializeAdaptiveRadialFromData(
	std::vector<Configuration>& training_set,
	const std::vector<Neighborhoods>* training_neighborhoods,
	double relative_rms,
	unsigned seed)
{
	if (!OuterUsesAdaptiveRadialCoefficients())
		ERROR("Balanced adaptive radial initialization requires adaptive-coefficients outer mode");
	if (!TwoLayerGateEnabled() || !TwoLayerGateUsesFullScalarWeights())
		ERROR("Balanced adaptive radial initialization requires a mu-scalar-full two-layer gate");
	if (training_neighborhoods != nullptr
	    && training_neighborhoods->size() != training_set.size())
		ERROR("Balanced adaptive radial initialization received inconsistent neighborhood cache size");
	if (!std::isfinite(relative_rms) || relative_rms <= 0.0)
		ERROR("--adaptive-radial-init-relative-rms should be finite and positive");

	ValidateOuterModeConfiguration();
	const int D = AdaptiveRadialRowCount();
	const int projection_rows = AdaptiveRadialProjectionRowCount();
	const int amplitude_count = AdaptiveRadialAmplitudeCount();
	const int Q = TwoLayerGateScalarCount();
	const int R = p_RadialBasis->rb_size;
	const int blocks = (sh_l_max_ + 1) * outer_k_rank_;
	if (D <= 0 || Q <= 0 || R <= 0 || D != blocks * R)
		ERROR("Balanced adaptive radial initialization found inconsistent adaptive dimensions");

	std::vector<double> scalar_sq(Q, 0.0);
	unsigned long long local_sites = 0;
	for (std::size_t cfg_index = 0; cfg_index < training_set.size(); ++cfg_index) {
		std::unique_ptr<Neighborhoods> owned_neighborhoods;
		const Neighborhoods* neighborhoods_ptr = nullptr;
		if (training_neighborhoods != nullptr) {
			neighborhoods_ptr = &(*training_neighborhoods)[cfg_index];
		} else {
			owned_neighborhoods.reset(
				new Neighborhoods(training_set[cfg_index], CutOff()));
			neighborhoods_ptr = owned_neighborhoods.get();
		}
		const Neighborhoods& neighborhoods = *neighborhoods_ptr;
		PrepareAdaptiveRadial(neighborhoods, false, false);
		for (int atom = 0; atom < static_cast<int>(neighborhoods.size()); ++atom) {
			const double* s = two_layer_gate_scalar_values_cache_.data()
				+ static_cast<size_t>(atom) * Q;
			for (int q = 0; q < Q; ++q)
				scalar_sq[q] += s[q] * s[q];
		}
		local_sites += static_cast<unsigned long long>(neighborhoods.size());
	}

	AllreduceInPlace(scalar_sq);
	if (!AllFiniteNonnegative(scalar_sq))
		ERROR("Balanced adaptive radial initialization reduced non-finite gate scalar moments");
	unsigned long long global_sites = local_sites;
	AllreduceInPlace(global_sites);
	if (global_sites == 0)
		ERROR("Balanced adaptive radial initialization needs a nonempty training set");

	std::vector<double> scalar_rms(Q, 1.0);
	adaptive_radial_weight_optimizer_scales_.assign(Q, 1.0);
	double scalar_rms_min = std::numeric_limits<double>::infinity();
	double scalar_rms_max = 0.0;
	for (int q = 0; q < Q; ++q) {
		const double second = scalar_sq[q] / static_cast<double>(global_sites);
		if (!std::isfinite(second) || second < 0.0)
			ERROR("Balanced adaptive radial initialization found non-finite gate scalar RMS");
		const double rms = std::sqrt(second);
		scalar_rms[q] = rms > 1.0e-12 ? rms : 1.0;
		UpdateRange(rms, scalar_rms_min, scalar_rms_max);
		// Natural physical U scale. Existing optimizer chart uses its inverse.
		adaptive_radial_weight_optimizer_scales_[q] =
			std::max(1.0e-12, std::min(1.0e12, 1.0 / scalar_rms[q]));
	}

	double* amplitudes = regression_coeffs.data() + AdaptiveRadialAmplitudeOffset();
	double* weights = regression_coeffs.data() + AdaptiveRadialWeightOffset();
	const double inv_sqrt_q = 1.0 / std::sqrt(static_cast<double>(Q));
	std::fill(amplitudes, amplitudes + amplitude_count, 1.0);
	for (int row = 0; row < projection_rows; ++row) {
		for (int q = 0; q < Q; ++q)
			weights[static_cast<size_t>(row) * Q + q] =
				CenteredUnit(seed, AdaptiveRadialSharedN() ? row * R : row, q) * inv_sqrt_q
				* adaptive_radial_weight_optimizer_scales_[q];
	}

	std::vector<double> base_sq(blocks, 0.0);
	std::vector<double> response_sq(blocks, 0.0);
	unsigned long long local_edges = 0;
	for (std::size_t cfg_index = 0; cfg_index < training_set.size(); ++cfg_index) {
		std::unique_ptr<Neighborhoods> owned_neighborhoods;
		const Neighborhoods* neighborhoods_ptr = nullptr;
		if (training_neighborhoods != nullptr) {
			neighborhoods_ptr = &(*training_neighborhoods)[cfg_index];
		} else {
			owned_neighborhoods.reset(
				new Neighborhoods(training_set[cfg_index], CutOff()));
			neighborhoods_ptr = owned_neighborhoods.get();
		}
		const Neighborhoods& neighborhoods = *neighborhoods_ptr;
		PrepareAdaptiveRadial(neighborhoods, false, false);
		for (int atom = 0; atom < static_cast<int>(neighborhoods.size()); ++atom) {
			const Neighborhood& nbh = neighborhoods[atom];
			for (int j = 0; j < nbh.count; ++j) {
				const size_t edge = two_layer_edge_offsets_cache_[atom] + j;
				const double* rb = two_layer_edge_radial_vals_cache_.data()
					+ edge * adaptive_radial_execution_.radial_stride;
				const double* signal = adaptive_radial_execution_.signals.data()
					+ static_cast<size_t>(nbh.inds[j]) * projection_rows;
				for (int l = 0; l <= sh_l_max_; ++l) {
					for (int o = 0; o < outer_k_rank_; ++o) {
						const int block = l * outer_k_rank_ + o;
						const int row0 = block * R;
						const int mu = o * (sh_l_max_ + 1) + l;
						const int rb_start = mu_to_radial_eval_block_[mu] * R;
						const double* c = regression_coeffs.data()
							+ RadialCoeffOffset() + mu * RadialCoeffBlockSize();
						double base = 0.0;
						double response = 0.0;
						for (int n = 0; n < R; ++n) {
							base += c[n] * rb[rb_start + n];
							response += signal[AdaptiveRadialSharedN() ? block : row0 + n] * rb[rb_start + n];
						}
						base_sq[block] += base * base;
						response_sq[block] += response * response;
					}
				}
				++local_edges;
			}
		}
	}
	AllreduceInPlace(base_sq);
	AllreduceInPlace(response_sq);
	if (!AllFiniteNonnegative(base_sq) || !AllFiniteNonnegative(response_sq))
		ERROR("Balanced adaptive radial initialization reduced non-finite radial calibration moments");
	unsigned long long global_edges = local_edges;
	AllreduceInPlace(global_edges);
	if (global_edges == 0)
		ERROR("Balanced adaptive radial initialization needs at least one neighbor edge");

	double base_rms_min = std::numeric_limits<double>::infinity();
	double base_rms_max = 0.0;
	double response_rms_min = std::numeric_limits<double>::infinity();
	double response_rms_max = 0.0;
	double initial_a_min = std::numeric_limits<double>::infinity();
	double initial_a_max = 0.0;
	int degenerate_base_blocks = 0;
	int degenerate_response_blocks = 0;
	for (int l = 0; l <= sh_l_max_; ++l) {
		for (int o = 0; o < outer_k_rank_; ++o) {
			const int block = l * outer_k_rank_ + o;
			const int row0 = block * R;
			const double base_rms =
				std::sqrt(base_sq[block] / static_cast<double>(global_edges));
			const double response_rms =
				std::sqrt(response_sq[block] / static_cast<double>(global_edges));
			UpdateRange(base_rms, base_rms_min, base_rms_max);
			UpdateRange(response_rms, response_rms_min, response_rms_max);
			const bool usable_base = PositiveFinite(base_rms);
			const bool usable_response = PositiveFinite(response_rms);
			if (!usable_base)
				++degenerate_base_blocks;
			if (!usable_response)
				++degenerate_response_blocks;
			double a = relative_rms * (usable_base ? base_rms : 1.0);
			if (usable_response)
				a /= response_rms;
			a = std::max(1.0e-12, std::min(1.0e12, a));
			UpdateRange(a, initial_a_min, initial_a_max);
			for (int type = 0; type < (AdaptiveRadialSpeciesAmplitude() ? species_count : 1); ++type)
				for (int n = 0; n < R; ++n)
					amplitudes[type * D + row0 + n] = a;
		}
	}

	std::vector<double> moment_sq(blocks, 0.0);
	local_sites = 0;
	for (std::size_t cfg_index = 0; cfg_index < training_set.size(); ++cfg_index) {
		std::unique_ptr<Neighborhoods> owned_neighborhoods;
		const Neighborhoods* neighborhoods_ptr = nullptr;
		if (training_neighborhoods != nullptr) {
			neighborhoods_ptr = &(*training_neighborhoods)[cfg_index];
		} else {
			owned_neighborhoods.reset(
				new Neighborhoods(training_set[cfg_index], CutOff()));
			neighborhoods_ptr = owned_neighborhoods.get();
		}
		const Neighborhoods& neighborhoods = *neighborhoods_ptr;
		PrepareAdaptiveRadial(neighborhoods, false, false);
		for (int atom = 0; atom < static_cast<int>(neighborhoods.size()); ++atom) {
			ForwardAdaptiveRadialSite(neighborhoods[atom], atom, nullptr);
			for (int l = 0; l <= sh_l_max_; ++l) {
				for (int o = 0; o < outer_k_rank_; ++o) {
					const int block = l * outer_k_rank_ + o;
					double avg_m = 0.0;
					for (int m = -l; m <= l; ++m) {
						const int h = l * l + l + m;
						const int node = adaptive_radial_execution_.outer_nodes[
							o * adaptive_radial_execution_.harmonics + h];
						const double value = adaptive_radial_execution_.values[node];
						avg_m += value * value;
					}
					moment_sq[block] += avg_m / static_cast<double>(2 * l + 1);
				}
			}
			++local_sites;
		}
	}
	AllreduceInPlace(moment_sq);
	if (!AllFiniteNonnegative(moment_sq))
		ERROR("Balanced adaptive radial initialization reduced non-finite outer moment moments");
	global_sites = local_sites;
	AllreduceInPlace(global_sites);
	if (global_sites == 0)
		ERROR("Balanced adaptive radial initialization lost all training sites");

	double moment_rms_min = std::numeric_limits<double>::infinity();
	double moment_rms_max = 0.0;
	double moment_scale_min = std::numeric_limits<double>::infinity();
	double moment_scale_max = 0.0;
	int degenerate_moment_blocks = 0;
	for (int l = 0; l <= sh_l_max_; ++l) {
		for (int o = 0; o < outer_k_rank_; ++o) {
			const int block = l * outer_k_rank_ + o;
			const double rms =
				std::sqrt(moment_sq[block] / static_cast<double>(global_sites));
			UpdateRange(rms, moment_rms_min, moment_rms_max);
			double factor = 1.0;
			if (PositiveFinite(rms)) {
				factor = std::max(1.0e-6, std::min(1.0e6, 1.0 / rms));
			} else {
				++degenerate_moment_blocks;
			}
			UpdateRange(factor, moment_scale_min, moment_scale_max);
			const int mu = o * (sh_l_max_ + 1) + l;
			double* c = regression_coeffs.data()
				+ RadialCoeffOffset() + mu * RadialCoeffBlockSize();
			for (int n = 0; n < R; ++n) {
				c[n] *= factor;
				for (int type = 0; type < (AdaptiveRadialSpeciesAmplitude() ? species_count : 1); ++type)
					amplitudes[type * D + block * R + n] *= factor;
			}
		}
	}

	adaptive_radial_amplitudes_.assign(amplitudes, amplitudes + amplitude_count);
	adaptive_radial_weights_.assign(weights, weights + static_cast<size_t>(projection_rows) * Q);
	adaptive_radial_amplitude_optimizer_scales_.assign(amplitude_count, 1.0);
	for (int row = 0; row < amplitude_count; ++row)
		adaptive_radial_amplitude_optimizer_scales_[row] =
			std::max(1.0e-12, std::abs(amplitudes[row]));

	if (RankZero()) {
		std::cout << "[adaptive-radial-init] sites=" << global_sites
		          << " edges=" << global_edges
		          << " scalar_rms_range=[" << scalar_rms_min
		          << "," << scalar_rms_max << "]"
		          << " baseline_rms_range=[" << base_rms_min
		          << "," << base_rms_max << "]"
		          << " correction_rms_range=[" << response_rms_min
		          << "," << response_rms_max << "]"
		          << " initial_a_range=[" << initial_a_min
		          << "," << initial_a_max << "]"
		          << " moment_rms_range=[" << moment_rms_min
		          << "," << moment_rms_max << "]"
		          << " moment_scale_range=[" << moment_scale_min
		          << "," << moment_scale_max << "]"
		          << " degenerate_blocks(base,response,moment)="
		          << degenerate_base_blocks << ","
		          << degenerate_response_blocks << ","
		          << degenerate_moment_blocks
		          << " optimizer_scale(weight,amplitude)=physical"
		          << std::endl;
	}

	adaptive_radial_execution_ = AdaptiveRadialExecution();
	two_layer_edge_cache_ready_ = false;
	two_layer_forward_final_moment_cache_ready_ = false;
}
