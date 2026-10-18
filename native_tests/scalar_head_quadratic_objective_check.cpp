#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Pre-include every dependency so the test-only access macro applies only to
// MTPR_trainer's class definition, never to libc++ or another project class.
#include "dev_src/mtpr.h"
#include "dev_src/non_linear_regression.h"
#include "dev_src/optimizer_checkpoint.h"
#include "dev_src/efs_head_coordinate_scaling.h"
#include "src/common/bfgs.h"
#include "src/common/lbfgs.h"
#define private public
#include "dev_src/mtpr_trainer.h"
#undef private

struct InspectTrainer : MTPR_trainer {
	explicit InspectTrainer(MLMTPR* model)
		: MTPR_trainer(model, 1.3, 0.4, 0.07, 0.2, 0.0) {}
	double Loss() const { return loss_; }
	const std::vector<double>& Gradient() const { return loss_grad_; }
};

static double QuadraticValue(
	const MTPR_trainer::ScalarHeadQuadratic& quadratic,
	const std::vector<double>& head)
{
	const int n = static_cast<int>(head.size());
	double value = quadratic.constant;
	for (int i = 0; i < n; ++i) {
		value -= 2.0 * quadratic.rhs[i] * head[i];
		for (int j = 0; j < n; ++j)
			value += head[i]
			       * quadratic.matrix[static_cast<std::size_t>(i) * n + j]
			       * head[j];
	}
	return value;
}

static int RunCheck(
	int argc, char** argv, int world_rank, int world_size)
{
	if (argc != 3)
		return 2;
	MLMTPR model(argv[1]);
	std::ifstream input(argv[2]);
	std::vector<Configuration> all_configs;
	for (int i = 0; i < 2; ++i) {
		Configuration cfg;
		if (!cfg.Load(input) || cfg.size() == 0) {
			// A one-configuration fixture is sufficient for the algebraic
			// identity; duplicate it so the same test still exercises the
			// global configuration-count scaling of physical head L2.
			if (i == 1 && all_configs.size() == 1) {
				all_configs.push_back(all_configs.front());
				break;
			}
			return 3;
		}
		all_configs.push_back(cfg);
	}
	std::vector<Configuration> training_set;
#ifdef MLIP_MPI
	// With three or more ranks this deliberately leaves at least one active
	// communicator rank with an empty local partition while ranks 0 and 1 own
	// one configuration each (distinct when the fixture provides two).
	if (world_rank < 2)
		training_set.push_back(all_configs[world_rank]);
#else
	training_set = all_configs;
#endif

	const int count = model.alpha_scalar_moments;
	const int linear_begin = model.LinearCoeffOffset();
	const int head_begin = linear_begin + model.species_count;
	if (count <= 0 || model.species_count <= 0)
		return 4;
	for (int type = 0; type < model.species_count; ++type)
		model.Coeff()[linear_begin + type] = 1.7;
	for (int q = 0; q < count; ++q) {
		model.linear_mults[q] = (q % 2 == 0 ? 1.0 : -1.0)
		                        * (0.6 + 0.13 * q);
		model.Coeff()[head_begin + q] = 0.03 * std::sin(q + 1.0);
	}
	model.LinCoeff();

	InspectTrainer trainer(&model);
#ifdef MLIP_MPI
	// Keep the empty local partition in the same active communicator so this
	// test exercises the zero-row contribution to every global reduction.
	trainer.ConfigureTrainComm(true, world_rank, world_size);
#endif
	trainer.norm_by_forces = 1;
	trainer.weighting = "structures";
	trainer.std_scaling = 2.0;
	trainer.stdd_scaling = 0.05;
	trainer.radial_smooth_regularization = 0.0;
	trainer.fixed_atomic_energy_weight = 0.0;
	trainer.scalar_head_l2_regularization = 3.0e-4;
	trainer.gate_scalar_l2_regularization = 0.0;
	trainer.gate_mix_l2_regularization = 0.0;
	trainer.gate_full_l2_regularization = 0.0;
	trainer.gate_x_l2_regularization = 0.0;

	const MTPR_trainer::ScalarHeadQuadratic quadratic =
		trainer.BuildScalarHeadQuadratic(training_set, nullptr, "objective-audit");
	if (quadratic.global_config_count != 2)
		return 5;

	double max_gradient_relative_error = 0.0;
	double max_loss_relative_error = 0.0;
	for (int point = 0; point < 3; ++point) {
		std::vector<double> head(count, 0.0);
		for (int q = 0; q < count; ++q) {
			head[q] = point == 0
				? model.Coeff()[head_begin + q]
				: 0.04 * std::sin((point + 0.3) * (q + 0.7));
			model.Coeff()[head_begin + q] = head[q];
		}
		model.LinCoeff();
		trainer.CalcObjectiveFunctionGrad(training_set);
		std::vector<double> actual_gradient(count, 0.0);
		for (int q = 0; q < count; ++q)
			actual_gradient[q] = trainer.Gradient()[head_begin + q];
		double actual_loss = trainer.Loss();
#ifdef MLIP_MPI
		MPI_Allreduce(MPI_IN_PLACE,
		              actual_gradient.data(),
		              count,
		              MPI_DOUBLE,
		              MPI_SUM,
		              MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE,
		              &actual_loss,
		              1,
		              MPI_DOUBLE,
		              MPI_SUM,
		              MPI_COMM_WORLD);
#endif
		for (int i = 0; i < count; ++i) {
			double expected = -2.0 * quadratic.rhs[i];
			for (int j = 0; j < count; ++j)
				expected += 2.0
				          * quadratic.matrix[static_cast<std::size_t>(i) * count + j]
				          * head[j];
			const double actual = actual_gradient[i];
			max_gradient_relative_error = std::max(
				max_gradient_relative_error,
				std::abs(expected - actual)
				/ std::max(1.0e-10, std::max(std::abs(expected), std::abs(actual))));
		}
		const double expected_loss = QuadraticValue(quadratic, head);
		max_loss_relative_error = std::max(
			max_loss_relative_error,
			std::abs(expected_loss - actual_loss)
			/ std::max(1.0, std::max(std::abs(expected_loss), std::abs(actual_loss))));
	}

	// Exercise the dense optimizer chart against the real physical objective,
	// not a synthetic quadratic owned by the coordinate helper.  H=2A/K is
	// exactly the averaged scalar-head Hessian.  If H+sI=LL^T and
	// z=L^T(h-h_ref), the chain rule requires grad_z=L^{-1}grad_h.
	std::vector<double> metric(quadratic.matrix.size(), 0.0);
	const double metric_scale =
		2.0 / static_cast<double>(quadratic.global_config_count);
	for (std::size_t i = 0; i < metric.size(); ++i)
		metric[i] = metric_scale * quadratic.matrix[i];
	std::vector<double> lower;
	double diagonal_shift = 0.0;
	if (!sus2::FullHeadCoordinateScaling::FactorMetric(
	        metric, count, 1.0e-6, lower, diagonal_shift))
		return 6;
	std::vector<double> reference(count, 0.0);
	for (int q = 0; q < count; ++q) {
		reference[q] = 0.08 * std::cos(0.43 * q + 0.17);
		model.Coeff()[head_begin + q] = reference[q];
	}
	model.LinCoeff();
	trainer.CalcObjectiveFunctionGrad(training_set);
	std::vector<double> physical_gradient(count, 0.0);
	for (int q = 0; q < count; ++q)
		physical_gradient[q] = trainer.Gradient()[head_begin + q];
#ifdef MLIP_MPI
	MPI_Allreduce(MPI_IN_PLACE,
	              physical_gradient.data(),
	              count,
	              MPI_DOUBLE,
	              MPI_SUM,
	              MPI_COMM_WORLD);
#endif
	std::vector<double> coordinate_gradient;
	if (!sus2::FullHeadCoordinateScaling::PullbackGradient(
	        physical_gradient, lower, count, coordinate_gradient))
		return 7;
	double coordinate_gradient_norm_sq = 0.0;
	for (double value : coordinate_gradient)
		coordinate_gradient_norm_sq += value * value;
	const double coordinate_gradient_norm =
		std::sqrt(coordinate_gradient_norm_sq);
	if (!(coordinate_gradient_norm > 0.0)
	    || !std::isfinite(coordinate_gradient_norm))
		return 8;
	std::vector<double> direction(count, 0.0);
	for (int q = 0; q < count; ++q)
		direction[q] = coordinate_gradient[q] / coordinate_gradient_norm;
	const double coordinate_step = 1.0e-3;
	double chart_losses[2] = {0.0, 0.0};
	for (int side = 0; side < 2; ++side) {
		const double sign = side == 0 ? 1.0 : -1.0;
		std::vector<double> coordinate(count, 0.0);
		for (int q = 0; q < count; ++q)
			coordinate[q] = sign * coordinate_step * direction[q];
		std::vector<double> physical;
		if (!sus2::FullHeadCoordinateScaling::UnpackCentered(
		        coordinate, reference, lower, count, physical))
			return 9;
		for (int q = 0; q < count; ++q)
			model.Coeff()[head_begin + q] = physical[q];
		model.LinCoeff();
		trainer.CalcObjectiveFunctionGrad(training_set);
		chart_losses[side] = trainer.Loss();
#ifdef MLIP_MPI
		MPI_Allreduce(MPI_IN_PLACE,
		              &chart_losses[side],
		              1,
		              MPI_DOUBLE,
		              MPI_SUM,
		              MPI_COMM_WORLD);
#endif
	}
	const double chart_fd =
		(chart_losses[0] - chart_losses[1]) / (2.0 * coordinate_step);
	const double chart_expected = coordinate_gradient_norm;
	const double chart_gradient_relative_error =
		std::abs(chart_fd - chart_expected)
		/ std::max(1.0e-10, std::max(std::abs(chart_fd), std::abs(chart_expected)));

	if (world_rank == 0)
		std::cout << std::setprecision(17)
		          << "scalar_head_quadratic_objective_check ranks=" << world_size
		          << " cfgs_global=2 heads=" << count
		          << " std=2 stdd=.05 scalar_head_l2=3e-4"
		          << " structures=true scale_by_force=true relative_force=.2"
		          << " max_gradient_relative_error=" << max_gradient_relative_error
		          << " max_loss_relative_error=" << max_loss_relative_error
		          << " chart_shift=" << diagonal_shift
		          << " dense_chart_directional_fd_error="
		          << chart_gradient_relative_error << '\n';
	const int status = max_gradient_relative_error < 2.0e-9
	    && max_loss_relative_error < 2.0e-11
	    && chart_gradient_relative_error < 2.0e-7 ? 0 : 1;
	return status;
}

int main(int argc, char** argv)
{
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int status = RunCheck(argc, argv, world_rank, world_size);
	int global_status = 0;
	MPI_Allreduce(
		&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Finalize();
	return global_status;
#else
	return RunCheck(argc, argv, 0, 1);
#endif
}
