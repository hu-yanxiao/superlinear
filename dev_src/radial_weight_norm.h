#ifndef MLIP_RADIAL_WEIGHT_NORM_H
#define MLIP_RADIAL_WEIGHT_NORM_H

#include <algorithm>
#include <cmath>

namespace sus2 {

// A non-redundant norm/direction chart for radial rows whose first coefficient
// is constrained to remain positive. Coordinate 0 is log(||c||); the remaining
// coordinates are c_i / c_0.
class RadialWeightNormChart
{
private:
	static double StableNorm(const double* values, int count)
	{
		double scale = 0.0;
		double sumsq = 1.0;
		for (int i = 0; i < count; ++i) {
			const double value = std::abs(values[i]);
			if (value == 0.0)
				continue;
			if (scale < value) {
				const double ratio = scale / value;
				sumsq = 1.0 + sumsq * ratio * ratio;
				scale = value;
			} else {
				const double ratio = value / scale;
				sumsq += ratio * ratio;
			}
		}
		return scale == 0.0 ? 0.0 : scale * std::sqrt(sumsq);
	}

	static bool DirectionScale(const double* coordinates,
	                           int count,
	                           double& inverse_direction_norm)
	{
		double scale = 1.0;
		for (int i = 1; i < count; ++i) {
			if (!std::isfinite(coordinates[i]))
				return false;
			scale = std::max(scale, std::abs(coordinates[i]));
		}

		double scaled_sum = 1.0 / (scale * scale);
		for (int i = 1; i < count; ++i) {
			const double value = coordinates[i] / scale;
			scaled_sum += value * value;
		}
		inverse_direction_norm = 1.0 / (scale * std::sqrt(scaled_sum));
		return std::isfinite(inverse_direction_norm)
		    && inverse_direction_norm > 0.0;
	}

public:
	static bool Pack(const double* physical, int count, double* coordinates)
	{
		if (physical == nullptr || coordinates == nullptr || count <= 0)
			return false;
		if (!std::isfinite(physical[0]) || physical[0] <= 0.0)
			return false;

		const double norm = StableNorm(physical, count);
		if (!std::isfinite(norm) || norm <= 0.0)
			return false;
		coordinates[0] = std::log(norm);
		for (int i = 1; i < count; ++i) {
			coordinates[i] = physical[i] / physical[0];
			if (!std::isfinite(coordinates[i]))
				return false;
		}
		return std::isfinite(coordinates[0]);
	}

	static bool Unpack(const double* coordinates, int count, double* physical)
	{
		if (coordinates == nullptr || physical == nullptr || count <= 0
		    || !std::isfinite(coordinates[0]))
			return false;

		double inverse_direction_norm = 0.0;
		if (!DirectionScale(coordinates, count, inverse_direction_norm))
			return false;
		const double norm = std::exp(coordinates[0]);
		physical[0] = norm * inverse_direction_norm;
		if (!std::isfinite(physical[0]) || physical[0] <= 0.0)
			return false;
		for (int i = 1; i < count; ++i) {
			physical[i] = physical[0] * coordinates[i];
			if (!std::isfinite(physical[i]))
				return false;
		}
		return true;
	}

	static bool PullbackGradient(const double* physical,
	                             const double* coordinates,
	                             const double* physical_gradient,
	                             int count,
	                             double* coordinate_gradient)
	{
		if (physical == nullptr || coordinates == nullptr
		    || physical_gradient == nullptr || coordinate_gradient == nullptr
		    || count <= 0 || !std::isfinite(physical[0]) || physical[0] <= 0.0)
			return false;

		double inverse_direction_norm = 0.0;
		if (!DirectionScale(coordinates, count, inverse_direction_norm))
			return false;
		const double inverse_direction_norm_sq =
			inverse_direction_norm * inverse_direction_norm;

		double radial_dot = 0.0;
		for (int i = 0; i < count; ++i) {
			if (!std::isfinite(physical[i]) || !std::isfinite(physical_gradient[i]))
				return false;
			radial_dot += physical_gradient[i] * physical[i];
		}
		coordinate_gradient[0] = radial_dot;
		for (int i = 1; i < count; ++i) {
			coordinate_gradient[i] =
				physical[0] * physical_gradient[i]
				- coordinates[i] * inverse_direction_norm_sq * radial_dot;
		}
		for (int i = 0; i < count; ++i)
			if (!std::isfinite(coordinate_gradient[i]))
				return false;
		return true;
	}
};

}  // namespace sus2

#endif  // MLIP_RADIAL_WEIGHT_NORM_H
