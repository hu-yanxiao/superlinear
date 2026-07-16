#ifndef MLIP_DIAGONAL_COORDINATE_SCALING_H
#define MLIP_DIAGONAL_COORDINATE_SCALING_H

#include <algorithm>
#include <cmath>

namespace sus2 {

// Fixed centered diagonal optimizer coordinates:
//   z = d * (x - x_ref), x = x_ref + z / d,
//   dL/dz = (dL/dx) / d.
// The physical parameter x and the objective are unchanged.
class DiagonalCoordinateScaling
{
public:
	static bool IsValidScale(double scale)
	{
		return std::isfinite(scale) && scale > 0.0;
	}

	static double Pack(double physical_value, double scale)
	{
		return scale * physical_value;
	}

	static double PackCentered(double physical_value,
	                           double reference_value,
	                           double scale)
	{
		return scale * (physical_value - reference_value);
	}

	static double Unpack(double optimizer_value, double scale)
	{
		return optimizer_value / scale;
	}

	static double UnpackCentered(double optimizer_value,
	                             double reference_value,
	                             double scale)
	{
		return reference_value + optimizer_value / scale;
	}

	static double PullbackGradient(double physical_gradient, double scale)
	{
		return physical_gradient / scale;
	}

	static double ScaleFromCurvature(double curvature,
	                                 double positive_block_median,
	                                 double relative_floor)
	{
		if (!std::isfinite(curvature) || curvature < 0.0
		    || !std::isfinite(positive_block_median)
		    || positive_block_median < 0.0
		    || !std::isfinite(relative_floor)
		    || relative_floor < 0.0)
			return 0.0;
		if (positive_block_median == 0.0)
			return 1.0;

		const double stabilized =
			std::max(curvature, relative_floor * positive_block_median);
		if (stabilized == 0.0)
			return 1.0;
		const double reference_curvature =
			std::min(positive_block_median, 1.0);
		return std::sqrt(stabilized / reference_curvature);
	}
};

} // namespace sus2

#endif
