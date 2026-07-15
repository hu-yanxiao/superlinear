#ifndef MLIP_RADIAL_WEIGHT_NORM_H
#define MLIP_RADIAL_WEIGHT_NORM_H

#include <algorithm>
#include <cmath>

namespace sus2 {

// A non-redundant norm/direction chart for radial rows whose first coefficient
// is constrained to remain positive. Coordinate 0 is log(||c||); the remaining
// coordinates map R^(n-1) smoothly onto the open positive hemisphere.
class RadialWeightNormChart
{
private:
	static double HalfPi()
	{
		return 1.57079632679489661923;
	}

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

	static bool DirectionRadius(const double* coordinates,
	                            int count,
	                            double& radius)
	{
		for (int i = 1; i < count; ++i)
			if (!std::isfinite(coordinates[i]))
				return false;
		radius = StableNorm(coordinates + 1, count - 1);
		return std::isfinite(radius);
	}

public:
	static bool Pack(const double* physical, int count, double* coordinates)
	{
		if (physical == nullptr || coordinates == nullptr || count <= 0)
			return false;
		if (!std::isfinite(physical[0]) || physical[0] <= 0.0)
			return false;
		for (int i = 1; i < count; ++i)
			if (!std::isfinite(physical[i]))
				return false;

		const double norm = StableNorm(physical, count);
		if (!std::isfinite(norm) || norm <= 0.0)
			return false;
		const double perpendicular_norm = StableNorm(physical + 1, count - 1);
		const double theta = std::atan2(perpendicular_norm, physical[0]);
		const double hemisphere_fraction = theta / HalfPi();
		if (!std::isfinite(hemisphere_fraction)
		    || hemisphere_fraction < 0.0
		    || hemisphere_fraction >= 1.0)
			return false;
		const double direction_radius = 0.5 * (
			std::log1p(hemisphere_fraction)
			- std::log1p(-hemisphere_fraction));
		if (!std::isfinite(direction_radius))
			return false;

		coordinates[0] = std::log(norm);
		if (perpendicular_norm == 0.0) {
			for (int i = 1; i < count; ++i)
				coordinates[i] = 0.0;
		} else {
			for (int i = 1; i < count; ++i) {
				coordinates[i] = direction_radius * physical[i] / perpendicular_norm;
				if (!std::isfinite(coordinates[i]))
					return false;
			}
		}
		return std::isfinite(coordinates[0]);
	}

	static bool Unpack(const double* coordinates, int count, double* physical)
	{
		if (coordinates == nullptr || physical == nullptr || count <= 0
		    || !std::isfinite(coordinates[0]))
			return false;

		double direction_radius = 0.0;
		if (!DirectionRadius(coordinates, count, direction_radius))
			return false;
		const double norm = std::exp(coordinates[0]);
		const double theta = HalfPi() * std::tanh(direction_radius);
		physical[0] = norm * std::cos(theta);
		if (!std::isfinite(physical[0]) || physical[0] <= 0.0)
			return false;
		if (direction_radius == 0.0) {
			for (int i = 1; i < count; ++i)
				physical[i] = 0.0;
		} else {
			const double perpendicular_scale =
				norm * std::sin(theta) / direction_radius;
			if (!std::isfinite(perpendicular_scale))
				return false;
			for (int i = 1; i < count; ++i) {
				physical[i] = perpendicular_scale * coordinates[i];
				if (!std::isfinite(physical[i]))
					return false;
			}
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

		double direction_radius = 0.0;
		if (!DirectionRadius(coordinates, count, direction_radius))
			return false;

		double radial_dot = 0.0;
		for (int i = 0; i < count; ++i) {
			if (!std::isfinite(physical[i]) || !std::isfinite(physical_gradient[i]))
				return false;
			radial_dot += physical_gradient[i] * physical[i];
		}
		coordinate_gradient[0] = radial_dot;

		const double norm = StableNorm(physical, count);
		if (!std::isfinite(norm) || norm <= 0.0)
			return false;
		const double alpha = HalfPi();
		if (direction_radius < 1.0e-4) {
			double direction_dot = 0.0;
			for (int i = 1; i < count; ++i)
				direction_dot += physical_gradient[i] * coordinates[i];
			const double perpendicular_scale = alpha;
			const double first_response_scale = alpha * alpha;
			const double curvature_scale =
				-alpha * (2.0 + alpha * alpha) / 3.0;
			for (int i = 1; i < count; ++i) {
				coordinate_gradient[i] = norm * (
					perpendicular_scale * physical_gradient[i]
					- physical_gradient[0] * first_response_scale * coordinates[i]
					+ curvature_scale * direction_dot * coordinates[i]);
			}
		} else {
			const double tanh_radius = std::tanh(direction_radius);
			const double theta = alpha * tanh_radius;
			const double sin_theta = std::sin(theta);
			const double cos_theta = std::cos(theta);
			const double exp_minus_two_radius =
				std::exp(-2.0 * direction_radius);
			const double one_plus_exp = 1.0 + exp_minus_two_radius;
			const double theta_derivative = alpha * 4.0 * exp_minus_two_radius
				/ (one_plus_exp * one_plus_exp);
			double perpendicular_gradient_dot_direction = 0.0;
			for (int i = 1; i < count; ++i)
				perpendicular_gradient_dot_direction +=
					physical_gradient[i] * coordinates[i] / direction_radius;
			const double radial_response =
				-physical_gradient[0] * sin_theta
				+ perpendicular_gradient_dot_direction * cos_theta;
			const double tangential_scale = sin_theta / direction_radius;
			for (int i = 1; i < count; ++i) {
				const double direction_component =
					coordinates[i] / direction_radius;
				coordinate_gradient[i] = norm * (
					theta_derivative * radial_response * direction_component
					+ tangential_scale * (
						physical_gradient[i]
						- perpendicular_gradient_dot_direction * direction_component));
			}
		}
		for (int i = 0; i < count; ++i)
			if (!std::isfinite(coordinate_gradient[i]))
				return false;
		return true;
	}
};

}  // namespace sus2

#endif  // MLIP_RADIAL_WEIGHT_NORM_H
