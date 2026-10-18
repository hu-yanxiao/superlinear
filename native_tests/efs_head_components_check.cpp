#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dev_src/mtpr.h"
#include "dev_src/non_linear_regression.h"
#include "dev_src/diagonal_coordinate_scaling.h"
#include "dev_src/efs_head_coordinate_scaling.h"

struct InspectRegression : NonLinearRegression {
	explicit InspectRegression(MLMTPR* model)
		: NonLinearRegression(model, 1.0, 0.3, 0.1, 0.2, 0.0) {}
	double Loss() const { return loss_; }
	const std::vector<double>& Gradient() const { return loss_grad_; }
};

// Link against the ordinary serial model objects. No training/model files are
// changed: all altered center prefactors, multipliers and heads stay in memory.
int main(int argc, char** argv)
{
	if (argc != 3) return 2;
	MLMTPR model(argv[1]);
	Configuration cfg;
	std::ifstream input(argv[2]);
	if (!cfg.Load(input) || cfg.size() == 0) return 3;
	const int count = model.alpha_scalar_moments;
	const int head_begin = model.LinearCoeffOffset() + model.species_count;
	const double center = 1.7;
	for (int t = 0; t < model.species_count; ++t) {
		model.Coeff()[t] = 0.0;
		model.Coeff()[model.LinearCoeffOffset() + t] = center;
	}
	model.scaling = 0.4;
	for (int q = 0; q < count; ++q) {
		model.linear_mults[q] = (q % 2 ? -1.0 : 1.0) * (0.6 + 0.13 * q);
		model.Coeff()[head_begin + q] = 0.03 * std::sin(q + 1.0);
	}
	model.LinCoeff();
	model.CalcEFSComponents(cfg, true, true);
	const int outputs = 1 + 3 * cfg.size() + 9;
	std::vector<double> columns(static_cast<size_t>(count) * outputs, 0.0);
	for (int q = 0; q < count; ++q) {
		const int col = model.species_count + q;
		const double mult = center * model.linear_mults[q];
		double* values = columns.data() + static_cast<size_t>(q) * outputs;
		values[0] = mult * model.energy_cmpnts[col];
		for (int atom = 0; atom < cfg.size(); ++atom)
			for (int a = 0; a < 3; ++a)
				values[1 + 3 * atom + a] = mult * model.forces_cmpnts(atom, col, a);
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				values[1 + 3 * cfg.size() + 3 * a + b] = mult * model.stress_cmpnts[col][a][b];
	}
	std::vector<double> energy_weights(cfg.size(), 0.43);
	std::vector<Vector3> force_weights(cfg.size(), Vector3(0.0, 0.0, 0.0));
	Matrix3 stress_weights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	std::vector<double> weights(outputs, 0.0);
	weights[0] = energy_weights[0];
	for (int atom = 0; atom < cfg.size(); ++atom)
		for (int a = 0; a < 3; ++a)
			weights[1 + 3 * atom + a] = force_weights[atom][a] = std::sin(3 * atom + a + 0.2);
	for (int a = 0; a < 3; ++a)
		for (int b = 0; b < 3; ++b)
			// The combination-gradient API takes the opposite of the saved
			// PlusStress coefficient (it adds +weight * edge displacement).
			weights[1 + 3 * cfg.size() + 3 * a + b] =
				-(stress_weights[a][b] = 0.1 * std::cos(3 * a + b + 0.3));
	Array1D gradient(model.CoeffCount());
	std::fill(&gradient[0], &gradient[0] + model.CoeffCount(), 0.0);
	Configuration evaluated = cfg;
	model.CalcEFS(evaluated);
	model.AccumulateEFSCombinationGrad(evaluated, energy_weights, force_weights, stress_weights, gradient);
	double max_vjp_error = 0.0, max_fd_error = 0.0;
	for (int q = 0; q < count; ++q) {
		const double* values = columns.data() + static_cast<size_t>(q) * outputs;
		double expected = 0.0;
		for (int j = 0; j < outputs; ++j) expected += weights[j] * values[j];
		max_vjp_error = std::max(max_vjp_error,
			std::abs(expected - gradient[head_begin + q]) / std::max(1.0e-12, std::abs(expected)));
		const double base = model.Coeff()[head_begin + q], step = 0.02;
		Configuration plus = cfg, minus = cfg;
		model.Coeff()[head_begin + q] = base + step;
		model.CalcEFS(plus);
		model.Coeff()[head_begin + q] = base - step;
		model.CalcEFS(minus);
		model.Coeff()[head_begin + q] = base;
		std::vector<double> differences(outputs, 0.0);
		differences[0] = (plus.energy - minus.energy) / (2.0 * step);
		for (int atom = 0; atom < cfg.size(); ++atom)
			for (int a = 0; a < 3; ++a)
				differences[1 + 3 * atom + a] = (plus.force(atom, a) - minus.force(atom, a)) / (2.0 * step);
		for (int a = 0; a < 3; ++a)
			for (int b = 0; b < 3; ++b)
				differences[1 + 3 * cfg.size() + 3 * a + b] = (plus.stresses[a][b] - minus.stresses[a][b]) / (2.0 * step);
		for (int j = 0; j < outputs; ++j)
			max_fd_error = std::max(max_fd_error,
				std::abs(differences[j] - values[j]) / std::max(1.0e-10, std::abs(values[j])));
	}
	// The fixed coordinate metric is not an alternative objective. Check its
	// pullback using the complete physical objective, including std/stdd and
	// physical scalar-head L2, with nontrivial force/structure weights.
	InspectRegression regression(&model);
	regression.norm_by_forces = 1;
	regression.weighting = "structures";
	regression.std_scaling = 2.0;
	regression.stdd_scaling = 0.05;
	regression.radial_smooth_regularization = 1.0e-6;
	regression.scalar_head_l2_regularization = 1.0e-5;
	regression.gate_x_l2_regularization = 0.0;
	std::vector<Configuration> training_set(1, cfg);
	regression.CalcObjectiveFunctionGrad(training_set);
	const double physical_loss = regression.Loss();
	const std::vector<double> physical_gradient = regression.Gradient();
	double average_force_sq = 0.0;
	for (int atom = 0; atom < cfg.size(); ++atom)
		average_force_sq += cfg.force(atom).NormSq() / cfg.size();
	const double force_normalization = 0.1 / (0.1 + average_force_sq);
	std::vector<double> curvature(count, 0.0);
	for (int q = 0; q < count; ++q) {
		const double* values = columns.data() + static_cast<size_t>(q) * outputs;
		curvature[q] = 2.0 * force_normalization * values[0] * values[0]
			/ (cfg.size() * cfg.size());
		for (int atom = 0; atom < cfg.size(); ++atom) {
			const double force_weight = 0.3 * 0.2
				/ (cfg.force(atom).NormSq() + 0.2) / cfg.size();
			for (int a = 0; a < 3; ++a) {
				const double response = values[1 + 3 * atom + a];
				curvature[q] += 2.0 * force_weight * force_normalization * response * response;
			}
		}
		for (int a = 0; a < 9; ++a) {
			const double response = values[1 + 3 * cfg.size() + a];
			curvature[q] += 2.0 * 0.1 * response * response / (cfg.size() * cfg.size());
		}
		curvature[q] += 2.0e-5 * model.linear_mults[q] * model.linear_mults[q] / count;
	}
	std::vector<double> sorted_curvature = curvature;
	std::sort(sorted_curvature.begin(), sorted_curvature.end());
	const double median = sorted_curvature[count / 2];
	double max_loss_gradient_error = 0.0, max_loss_value_error = 0.0;
	for (int q = 0; q < count; ++q) {
		const double scale = std::max(1.0e-8, std::min(1.0e8,
			sus2::ExactHeadCoordinateScaling::ScaleFromCurvature(curvature[q], median, 1.0e-6)));
		const double reference = model.Coeff()[head_begin + q];
		const double step = 1.0e-3;
		model.Coeff()[head_begin + q] = sus2::DiagonalCoordinateScaling::UnpackCentered(step, reference, scale);
		regression.CalcObjectiveFunctionGrad(training_set);
		const double plus = regression.Loss();
		model.Coeff()[head_begin + q] = sus2::DiagonalCoordinateScaling::UnpackCentered(-step, reference, scale);
		regression.CalcObjectiveFunctionGrad(training_set);
		const double minus = regression.Loss();
		model.Coeff()[head_begin + q] = reference;
		const double difference = (plus - minus) / (2.0 * step);
		const double expected = physical_gradient[head_begin + q] / scale;
		max_loss_gradient_error = std::max(max_loss_gradient_error,
			std::abs(difference - expected) / std::max(1.0e-9, std::abs(expected)));
		regression.CalcObjectiveFunctionGrad(training_set);
		max_loss_value_error = std::max(max_loss_value_error,
			std::abs(regression.Loss() - physical_loss) / std::max(1.0, std::abs(physical_loss)));
	}
	std::cout << std::setprecision(15) << "efs_head_components_check heads=" << count
	          << " common_center=" << center << " nontrivial_signed_linear_mults=true"
	          << " max_vjp_relative_error=" << max_vjp_error
	          << " max_fd_relative_error=" << max_fd_error
	          << " std=2 stdd=.05 relative_force_weight=.2 structures=true"
	          << " max_physical_loss_roundtrip_error=" << max_loss_value_error
	          << " max_full_objective_coordinate_fd_error=" << max_loss_gradient_error << '\n';
	return max_vjp_error < 2.0e-8 && max_fd_error < 2.0e-6
		&& max_loss_value_error < 2.0e-12 && max_loss_gradient_error < 2.0e-7 ? 0 : 1;
}
