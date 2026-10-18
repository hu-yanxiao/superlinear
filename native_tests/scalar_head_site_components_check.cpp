#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dev_src/mtpr.h"
#include "src/neighborhoods.h"

#ifdef MLIP_MPI
#include <mpi.h>
#endif

// Usage:
//   scalar_head_site_components_check PURE_W_MODEL CFG_FILE
//
// This test deliberately uses nonuniform species-center factors and signed,
// non-unit linear multipliers.  It checks that the atom-major component table
// is the derivative with respect to the stored scalar-head coefficients, not
// merely the unscaled outer moments.
int main(int argc, char** argv)
{
	#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
	int world_rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	#else
	const int world_rank = 0;
	#endif
	if (argc != 3) {
	#ifdef MLIP_MPI
		MPI_Finalize();
	#endif
		return 2;
	}
	MLMTPR model(argv[1]);
	Configuration cfg;
	std::ifstream input(argv[2]);
	if (!cfg.Load(input) || cfg.size() == 0) {
	#ifdef MLIP_MPI
		MPI_Finalize();
	#endif
		return 3;
	}
	Neighborhoods neighborhoods(cfg, model.CutOff());

	const int q_count = model.alpha_scalar_moments;
	const int linear_offset = model.LinearCoeffOffset();
	const int head_offset = linear_offset + model.species_count;
	if (model.species_count < 2 || q_count <= 0) {
	#ifdef MLIP_MPI
		MPI_Finalize();
	#endif
		return 4;
	}
	for (int type = 0; type < model.species_count; ++type)
		model.Coeff()[linear_offset + type] = 0.65 + 0.9 * type;
	for (int q = 0; q < q_count; ++q) {
		model.linear_mults[q] = (q % 2 == 0 ? 1.0 : -1.0)
		                        * (0.35 + 0.017 * q);
		model.Coeff()[head_offset + q] = 0.02 * std::cos(q + 0.3);
	}
	model.LinCoeff();

	// Seed every transient evaluator context with a sentinel.  The helper must
	// restore a caller's context even though ordinary evaluation is non-reentrant.
	std::vector<double> gate_sentinel(1, 3.0), gate_adjoint_sentinel(1, 4.0);
	std::vector<double> second_sentinel(1, 5.0), second_adjoint_sentinel(1, 6.0);
	model.active_two_layer_gate_values_ = &gate_sentinel;
	model.active_two_layer_gate_adjoints_ = &gate_adjoint_sentinel;
	model.active_second_layer_gate_values_ = &second_sentinel;
	model.active_second_layer_gate_adjoints_ = &second_adjoint_sentinel;
	model.active_two_layer_edge_cache_atom_index_ = 71;

	const Configuration cfg_before = cfg;
	std::vector<double> site_components;
	model.CalcScalarHeadSiteComponents(cfg, neighborhoods, site_components);
	const bool context_ok =
		model.active_two_layer_gate_values_ == &gate_sentinel
		&& model.active_two_layer_gate_adjoints_ == &gate_adjoint_sentinel
		&& model.active_second_layer_gate_values_ == &second_sentinel
		&& model.active_second_layer_gate_adjoints_ == &second_adjoint_sentinel
		&& model.active_two_layer_edge_cache_atom_index_ == 71;
	const bool cfg_unchanged = cfg == cfg_before;
	if (site_components.size()
	    != static_cast<std::size_t>(cfg.size()) * q_count) {
	#ifdef MLIP_MPI
		MPI_Finalize();
	#endif
		return 5;
	}

	// End the artificial nested context before ordinary standalone inference.
	model.active_two_layer_gate_values_ = nullptr;
	model.active_two_layer_gate_adjoints_ = nullptr;
	model.active_second_layer_gate_values_ = nullptr;
	model.active_second_layer_gate_adjoints_ = nullptr;
	model.active_two_layer_edge_cache_atom_index_ = -1;

	for (int q = 0; q < q_count; ++q)
		model.Coeff()[head_offset + q] = 0.0;
	Configuration zero_head = cfg;
	model.CalcEnergyAndSiteEnergies(zero_head, neighborhoods);
	for (int q = 0; q < q_count; ++q)
		model.Coeff()[head_offset + q] = 1.0;
	Configuration one_head = cfg;
	model.CalcEnergyAndSiteEnergies(one_head, neighborhoods);

	double max_ones_error = 0.0;
	for (int atom = 0; atom < cfg.size(); ++atom) {
		double row_sum = 0.0;
		for (int q = 0; q < q_count; ++q)
			row_sum += site_components[static_cast<std::size_t>(atom) * q_count + q];
		const double actual = one_head.cal_se[atom] - zero_head.cal_se[atom];
		max_ones_error = std::max(
			max_ones_error,
			std::abs(actual - row_sum) / std::max(1.0, std::abs(row_sum)));
	}

	std::vector<double> direction(q_count, 0.0);
	for (int q = 0; q < q_count; ++q) {
		model.Coeff()[head_offset + q] = 0.03 * std::sin(q + 0.1);
		direction[q] = std::cos(0.7 * q + 0.2);
	}
	const double fd_step = 2.0e-5;
	for (int q = 0; q < q_count; ++q)
		model.Coeff()[head_offset + q] += fd_step * direction[q];
	Configuration plus = cfg;
	model.CalcEnergyAndSiteEnergies(plus, neighborhoods);
	for (int q = 0; q < q_count; ++q)
		model.Coeff()[head_offset + q] -= 2.0 * fd_step * direction[q];
	Configuration minus = cfg;
	model.CalcEnergyAndSiteEnergies(minus, neighborhoods);

	double max_site_fd_error = 0.0;
	double expected_total = 0.0;
	for (int atom = 0; atom < cfg.size(); ++atom) {
		double expected = 0.0;
		for (int q = 0; q < q_count; ++q)
			expected += site_components[static_cast<std::size_t>(atom) * q_count + q]
			            * direction[q];
		const double actual =
			(plus.cal_se[atom] - minus.cal_se[atom]) / (2.0 * fd_step);
		max_site_fd_error = std::max(
			max_site_fd_error,
			std::abs(actual - expected) / std::max(1.0, std::abs(expected)));
		expected_total += expected;
	}
	const double actual_total =
		(plus.energy - minus.energy) / (2.0 * fd_step);
	const double total_fd_error =
		std::abs(actual_total - expected_total)
		/ std::max(1.0, std::abs(expected_total));

	if (world_rank == 0)
		std::cout << std::setprecision(15)
		          << "scalar_head_site_components_check atoms=" << cfg.size()
		          << " heads=" << q_count
		          << " centers=" << model.Coeff()[linear_offset]
		          << ',' << model.Coeff()[linear_offset + 1]
		          << " signed_linear_mults=true"
		          << " cfg_unchanged=" << cfg_unchanged
		          << " context_restored=" << context_ok
		          << " max_all_heads_one_error=" << max_ones_error
		          << " max_site_directional_fd_error=" << max_site_fd_error
		          << " total_directional_fd_error=" << total_fd_error << '\n';

	int status = cfg_unchanged && context_ok
		&& max_ones_error < 2.0e-11
		&& max_site_fd_error < 5.0e-8
		&& total_fd_error < 5.0e-8
		? 0 : 1;
	#ifdef MLIP_MPI
	MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Finalize();
	#endif
	return status;
}
