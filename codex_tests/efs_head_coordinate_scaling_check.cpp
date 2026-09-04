#include <algorithm>
#include <cmath>
#include <iostream>

#include "dev_src/diagonal_coordinate_scaling.h"
#include "dev_src/efs_head_coordinate_scaling.h"

int main()
{
	using sus2::DiagonalCoordinateScaling;
	using sus2::ExactHeadCoordinateScaling;
	const double curvature[] = {1.0e-4, 4.0, 4.0, 0.0};
	const double physical[] = {0.3, -2.0, 1.5, -0.1};
	const double reference[] = {0.2, -2.1, 1.4, 0.0};
	double scale[4], coordinate[4];
	for (int i = 0; i < 4; ++i) {
		scale[i] = ExactHeadCoordinateScaling::ScaleFromCurvature(curvature[i], 4.0, 1.0e-6);
		coordinate[i] = DiagonalCoordinateScaling::PackCentered(physical[i], reference[i], scale[i]);
		const double recovered = DiagonalCoordinateScaling::UnpackCentered(coordinate[i], reference[i], scale[i]);
		if (std::abs(recovered - physical[i]) > 2.0e-15) return 1;
	}
	if (std::abs(scale[0] - 0.01) > 1.0e-15 || scale[1] != scale[2]
	    || std::abs(scale[3] - 0.002) > 1.0e-15
	    || ExactHeadCoordinateScaling::ScaleFromCurvature(0.0, 0.0, 0.0) != 1.0
	    || ExactHeadCoordinateScaling::ScaleFromCurvature(-1.0, 1.0, 0.0) != 0.0)
		return 2;
	// Include a non-diagonal data term and physical L2; the chart must not
	// replace physical L2 by an unweighted coordinate-space penalty.
	const double lambda = 0.17;
	auto loss = [&](const double* z) {
		double h[4], value = 0.0, response = -0.6;
		for (int i = 0; i < 4; ++i) {
			h[i] = DiagonalCoordinateScaling::UnpackCentered(z[i], reference[i], scale[i]);
			response += (i + 1) * h[i];
			value += lambda * h[i] * h[i] / 4.0;
		}
		return value + 0.5 * response * response;
	};
	double response = -0.6;
	for (int i = 0; i < 4; ++i) response += (i + 1) * physical[i];
	for (int varied = 0; varied < 4; ++varied) {
		double plus[4], minus[4];
		std::copy(coordinate, coordinate + 4, plus);
		std::copy(coordinate, coordinate + 4, minus);
		const double step = 1.0e-6 * scale[varied];
		plus[varied] += step;
		minus[varied] -= step;
		const double fd = (loss(plus) - loss(minus)) / (2.0 * step);
		const double gp = (varied + 1) * response + 2.0 * lambda * physical[varied] / 4.0;
		const double gz = DiagonalCoordinateScaling::PullbackGradient(gp, scale[varied]);
		if (std::abs(fd - gz) > 3.0e-9 * std::max(1.0, std::abs(gz))) return 3;
	}
	std::cout << "efs_head_coordinate_scaling_check passed\n";
	return 0;
}
