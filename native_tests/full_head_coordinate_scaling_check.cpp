#include "../dev_src/efs_head_coordinate_scaling.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

double MaxAbs(const std::vector<double>& values)
{
	double maximum = 0.0;
	for (double value : values)
		maximum = std::max(maximum, std::abs(value));
	return maximum;
}

double Dot(const std::vector<double>& left, const std::vector<double>& right)
{
	double value = 0.0;
	for (std::size_t i = 0; i < left.size(); ++i)
		value += left[i] * right[i];
	return value;
}

} // namespace

int main()
{
	using sus2::FullHeadCoordinateScaling;
	const int size = 3;
	const std::vector<double> metric = {
		4.0, 1.0, 0.5,
		1.0, 3.0, -0.2,
		0.5, -0.2, 2.0
	};
	std::vector<double> lower;
	double shift = 0.0;
	if (!FullHeadCoordinateScaling::FactorMetric(
	        metric, size, 1.0e-8, lower, shift))
		return 1;
	std::vector<double> factor_error(size * size, 0.0);
	for (int row = 0; row < size; ++row)
		for (int col = 0; col < size; ++col) {
			double reconstructed = 0.0;
			for (int inner = 0; inner < size; ++inner)
				reconstructed += lower[row * size + inner]
				               * lower[col * size + inner];
			factor_error[row * size + col] = reconstructed
				- metric[row * size + col]
				- (row == col ? shift : 0.0);
		}
	if (MaxAbs(factor_error) > 2.0e-14)
		return 2;

	const std::vector<double> reference = {0.4, -0.7, 1.2};
	const std::vector<double> physical = {0.9, -0.1, -0.3};
	std::vector<double> packed;
	std::vector<double> reconstructed;
	if (!FullHeadCoordinateScaling::PackCentered(
	        physical, reference, lower, size, packed)
	    || !FullHeadCoordinateScaling::UnpackCentered(
	        packed, reference, lower, size, reconstructed))
		return 3;
	for (int i = 0; i < size; ++i)
		if (std::abs(reconstructed[i] - physical[i]) > 2.0e-14)
			return 4;

	const std::vector<double> physical_gradient = {1.3, -0.8, 0.45};
	std::vector<double> packed_gradient;
	if (!FullHeadCoordinateScaling::PullbackGradient(
	        physical_gradient, lower, size, packed_gradient))
		return 5;
	const std::vector<double> direction = {0.2, -0.35, 0.41};
	const double step = 1.0e-6;
	std::vector<double> plus = packed;
	std::vector<double> minus = packed;
	for (int i = 0; i < size; ++i) {
		plus[i] += step * direction[i];
		minus[i] -= step * direction[i];
	}
	std::vector<double> physical_plus;
	std::vector<double> physical_minus;
	if (!FullHeadCoordinateScaling::UnpackCentered(
	        plus, reference, lower, size, physical_plus)
	    || !FullHeadCoordinateScaling::UnpackCentered(
	        minus, reference, lower, size, physical_minus))
		return 6;
	const double finite_difference =
		(Dot(physical_gradient, physical_plus)
		 - Dot(physical_gradient, physical_minus)) / (2.0 * step);
	const double analytic = Dot(packed_gradient, direction);
	if (std::abs(finite_difference - analytic)
	    > 2.0e-9 * std::max(1.0, std::abs(analytic)))
		return 7;

	const std::vector<double> expected_solution = {0.3, -1.1, 0.75};
	std::vector<double> rhs(size, 0.0);
	for (int row = 0; row < size; ++row)
		for (int col = 0; col < size; ++col)
			rhs[row] += metric[row * size + col] * expected_solution[col];
	std::vector<double> solution;
	if (!FullHeadCoordinateScaling::SolvePhysicalPositiveDefinite(
	        metric, rhs, size, solution))
		return 8;
	for (int i = 0; i < size; ++i)
		if (std::abs(solution[i] - expected_solution[i]) > 2.0e-14)
			return 9;

	const std::vector<double> singular = {
		1.0, 1.0,
		1.0, 1.0
	};
	std::vector<double> singular_lower;
	double singular_shift = 0.0;
	std::vector<double> singular_solution;
	if (!FullHeadCoordinateScaling::FactorMetric(
	        singular, 2, 1.0e-6, singular_lower, singular_shift)
	    || !(singular_shift > 0.0)
	    || FullHeadCoordinateScaling::SolvePhysicalPositiveDefinite(
	        singular, std::vector<double>{1.0, 1.0}, 2, singular_solution))
		return 10;

	std::cout << "full_head_coordinate_scaling_check_ok"
	          << " shift=" << shift
	          << " directional_abs=" << std::abs(finite_difference - analytic)
	          << std::endl;
	return 0;
}
