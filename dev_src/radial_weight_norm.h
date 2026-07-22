#ifndef MLIP_RADIAL_WEIGHT_NORM_H
#define MLIP_RADIAL_WEIGHT_NORM_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace sus2 {

// A non-redundant norm/direction chart for radial rows. Coordinate 0 is
// log(||c||); the remaining coordinates map R^(n-1) smoothly onto a sphere
// chart selected by a fixed pivot and pivot sign. The default wrappers retain
// the legacy positive-first-coefficient hemisphere.
class RadialWeightNormChart
{
private:
	static double HalfPi()
	{
		return 1.57079632679489661923;
	}

	static double Pi()
	{
		return 3.14159265358979323846;
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

	static double StableNormExcluding(const double* values,
	                                  int count,
	                                  int excluded_index)
	{
		double scale = 0.0;
		double sumsq = 1.0;
		for (int i = 0; i < count; ++i) {
			if (i == excluded_index)
				continue;
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

	static bool ValidPivot(int count, int pivot, double pivot_sign)
	{
		return pivot >= 0 && pivot < count
		    && (pivot_sign == 1.0 || pivot_sign == -1.0);
	}

	static int DirectionPhysicalIndex(int coordinate_index, int pivot)
	{
		const int direction_index = coordinate_index - 1;
		return direction_index < pivot ? direction_index : direction_index + 1;
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

	static bool PackChart(const double* physical,
	                      int count,
	                      int pivot,
	                      double pivot_sign,
	                      double angular_limit,
	                      double radius_scale,
	                      bool require_positive_pivot,
	                      double* coordinates)
	{
		if (physical == nullptr || coordinates == nullptr || count <= 0
		    || !ValidPivot(count, pivot, pivot_sign))
			return false;
		for (int i = 0; i < count; ++i)
			if (!std::isfinite(physical[i]))
				return false;
		const double axis_component = pivot_sign * physical[pivot];
		if (!std::isfinite(axis_component)
		    || (require_positive_pivot && axis_component <= 0.0))
			return false;

		const double norm = StableNorm(physical, count);
		if (!std::isfinite(norm) || norm <= 0.0)
			return false;
		const double perpendicular_norm =
			StableNormExcluding(physical, count, pivot);
		const double theta = std::atan2(perpendicular_norm, axis_component);
		const double angular_fraction = theta / angular_limit;
		if (!std::isfinite(angular_fraction)
		    || angular_fraction < 0.0
		    || angular_fraction >= 1.0)
			return false;
		const double direction_radius = 0.5 * radius_scale * (
			std::log1p(angular_fraction)
			- std::log1p(-angular_fraction));
		if (!std::isfinite(direction_radius))
			return false;

		coordinates[0] = std::log(norm);
		if (perpendicular_norm == 0.0) {
			for (int i = 1; i < count; ++i)
				coordinates[i] = 0.0;
		} else {
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				coordinates[coordinate_index] =
					direction_radius * physical[physical_index] / perpendicular_norm;
				if (!std::isfinite(coordinates[coordinate_index]))
					return false;
			}
		}
		return std::isfinite(coordinates[0]);
	}

	static bool UnpackChart(const double* coordinates,
	                        int count,
	                        int pivot,
	                        double pivot_sign,
	                        double angular_limit,
	                        double radius_scale,
	                        bool require_positive_pivot,
	                        double* physical)
	{
		if (coordinates == nullptr || physical == nullptr || count <= 0
		    || !std::isfinite(coordinates[0])
		    || !ValidPivot(count, pivot, pivot_sign))
			return false;

		double direction_radius = 0.0;
		if (!DirectionRadius(coordinates, count, direction_radius))
			return false;
		const double norm = std::exp(coordinates[0]);
		const double theta = angular_limit
		                   * std::tanh(direction_radius / radius_scale);
		physical[pivot] = pivot_sign * norm * std::cos(theta);
		if (!std::isfinite(physical[pivot])
		    || (require_positive_pivot
		        && pivot_sign * physical[pivot] <= 0.0))
			return false;
		if (direction_radius == 0.0) {
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				physical[physical_index] = 0.0;
			}
		} else {
			const double perpendicular_scale =
				norm * std::sin(theta) / direction_radius;
			if (!std::isfinite(perpendicular_scale))
				return false;
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				physical[physical_index] =
					perpendicular_scale * coordinates[coordinate_index];
				if (!std::isfinite(physical[physical_index]))
					return false;
			}
		}
		return true;
	}

	static bool PullbackGradientChart(const double* physical,
	                                  const double* coordinates,
	                                  const double* physical_gradient,
	                                  int count,
	                                  int pivot,
	                                  double pivot_sign,
	                                  double angular_limit,
	                                  double radius_scale,
	                                  bool require_positive_pivot,
	                                  double* coordinate_gradient)
	{
		if (physical == nullptr || coordinates == nullptr
		    || physical_gradient == nullptr || coordinate_gradient == nullptr
		    || count <= 0 || !ValidPivot(count, pivot, pivot_sign)
		    || !std::isfinite(physical[pivot])
		    || (require_positive_pivot
		        && pivot_sign * physical[pivot] <= 0.0))
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
		const double angular_slope = angular_limit / radius_scale;
		const double axis_gradient = pivot_sign * physical_gradient[pivot];
		if (direction_radius < 1.0e-4) {
			double direction_dot = 0.0;
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				direction_dot += physical_gradient[physical_index]
				               * coordinates[coordinate_index];
			}
			const double first_response_scale =
				angular_slope * angular_slope;
			const double curvature_scale = -angular_slope * (
				2.0 / (radius_scale * radius_scale)
				+ first_response_scale) / 3.0;
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				coordinate_gradient[coordinate_index] = norm * (
					angular_slope * physical_gradient[physical_index]
					- axis_gradient * first_response_scale
					  * coordinates[coordinate_index]
					+ curvature_scale * direction_dot
					  * coordinates[coordinate_index]);
			}
		} else {
			const double scaled_radius = direction_radius / radius_scale;
			const double tanh_radius = std::tanh(scaled_radius);
			const double theta = angular_limit * tanh_radius;
			const double sin_theta = std::sin(theta);
			const double cos_theta = std::cos(theta);
			const double exp_minus_two_radius =
				std::exp(-2.0 * scaled_radius);
			const double one_plus_exp = 1.0 + exp_minus_two_radius;
			const double theta_derivative = angular_slope
				* 4.0 * exp_minus_two_radius
				/ (one_plus_exp * one_plus_exp);
			double perpendicular_gradient_dot_direction = 0.0;
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				perpendicular_gradient_dot_direction +=
					physical_gradient[physical_index]
					* coordinates[coordinate_index] / direction_radius;
			}
			const double radial_response =
				-axis_gradient * sin_theta
				+ perpendicular_gradient_dot_direction * cos_theta;
			const double tangential_scale = sin_theta / direction_radius;
			for (int coordinate_index = 1;
			     coordinate_index < count;
			     ++coordinate_index) {
				const int physical_index =
					DirectionPhysicalIndex(coordinate_index, pivot);
				const double direction_component =
					coordinates[coordinate_index] / direction_radius;
				coordinate_gradient[coordinate_index] = norm * (
					theta_derivative * radial_response * direction_component
					+ tangential_scale * (
						physical_gradient[physical_index]
						- perpendicular_gradient_dot_direction
						  * direction_component));
			}
		}
		for (int i = 0; i < count; ++i)
			if (!std::isfinite(coordinate_gradient[i]))
				return false;
		return true;
	}

public:
	enum RechartReason {
		kRechartNone = 0,
		kRechartDirectionRadiusExceeded = 1,
		kRechartPackFailed = 2
	};

	struct RechartResult {
		bool success;
		bool recharted;
		RechartReason reason;
		double old_direction_radius;
		double new_direction_radius;

		RechartResult()
			: success(false),
			  recharted(false),
			  reason(kRechartNone),
			  old_direction_radius(
				  std::numeric_limits<double>::quiet_NaN()),
			  new_direction_radius(
				  std::numeric_limits<double>::quiet_NaN()) {}
	};

	static bool SelectSignedPivot(const double* physical,
	                              int count,
	                              int& pivot,
	                              double& pivot_sign)
	{
		if (physical == nullptr || count <= 0)
			return false;
		pivot = -1;
		double largest_abs = 0.0;
		for (int i = 0; i < count; ++i) {
			if (!std::isfinite(physical[i]))
				return false;
			const double value_abs = std::abs(physical[i]);
			if (value_abs > largest_abs) {
				largest_abs = value_abs;
				pivot = i;
			}
		}
		if (pivot < 0 || largest_abs == 0.0)
			return false;
		pivot_sign = physical[pivot] > 0.0 ? 1.0 : -1.0;
		return true;
	}

	static bool Pack(const double* physical, int count, double* coordinates)
	{
		return PackChart(
			physical, count, 0, 1.0, HalfPi(), 1.0, true, coordinates);
	}

	static bool PackWithPivot(const double* physical,
	                          int count,
	                          int pivot,
	                          double pivot_sign,
	                          double* coordinates)
	{
		return PackChart(
			physical, count, pivot, pivot_sign, Pi(), 2.0, false, coordinates);
	}

	static bool GetDirectionRadius(const double* coordinates,
	                               int count,
	                               double& radius)
	{
		if (coordinates == nullptr || count <= 0)
			return false;
		return DirectionRadius(coordinates, count, radius);
	}

	// Re-select the signed pivot before the fixed chart reaches its excluded
	// antipode. This function never writes the physical row: it only updates
	// the chart metadata and returns coordinates for the same physical values.
	static RechartResult RechartIfNeeded(const double* physical,
	                                     int count,
	                                     double max_direction_radius,
	                                     int& pivot,
	                                     double& pivot_sign,
	                                     double* coordinates)
	{
		RechartResult result;
		if (physical == nullptr || coordinates == nullptr || count <= 0
		    || !std::isfinite(max_direction_radius)
		    || max_direction_radius <= 0.0
		    || !ValidPivot(count, pivot, pivot_sign))
			return result;

		const int old_pivot = pivot;
		const double old_pivot_sign = pivot_sign;
		if (PackWithPivot(
		        physical, count, old_pivot, old_pivot_sign, coordinates)) {
			if (!DirectionRadius(coordinates, count, result.old_direction_radius))
				return result;
			if (result.old_direction_radius <= max_direction_radius) {
				result.success = true;
				result.new_direction_radius = result.old_direction_radius;
				return result;
			}
			result.reason = kRechartDirectionRadiusExceeded;
		} else {
			result.reason = kRechartPackFailed;
			result.old_direction_radius =
				std::numeric_limits<double>::infinity();
		}

		int new_pivot = -1;
		double new_pivot_sign = 0.0;
		if (!SelectSignedPivot(
		        physical, count, new_pivot, new_pivot_sign)
		    || (new_pivot == old_pivot
		        && new_pivot_sign == old_pivot_sign)
		    || !PackWithPivot(
		        physical, count, new_pivot, new_pivot_sign, coordinates)
		    || !DirectionRadius(
		        coordinates, count, result.new_direction_radius)
		    || result.new_direction_radius > max_direction_radius)
			return result;

		pivot = new_pivot;
		pivot_sign = new_pivot_sign;
		result.success = true;
		result.recharted = true;
		return result;
	}

	static bool Unpack(const double* coordinates, int count, double* physical)
	{
		return UnpackChart(
			coordinates, count, 0, 1.0, HalfPi(), 1.0, true, physical);
	}

	static bool UnpackWithPivot(const double* coordinates,
	                            int count,
	                            int pivot,
	                            double pivot_sign,
	                            double* physical)
	{
		return UnpackChart(
			coordinates, count, pivot, pivot_sign, Pi(), 2.0, false, physical);
	}

	static bool PullbackGradient(const double* physical,
	                             const double* coordinates,
	                             const double* physical_gradient,
	                             int count,
	                             double* coordinate_gradient)
	{
		return PullbackGradientChart(
			physical, coordinates, physical_gradient, count, 0, 1.0,
			HalfPi(), 1.0, true, coordinate_gradient);
	}

	static bool PullbackGradientWithPivot(const double* physical,
	                                      const double* coordinates,
	                                      const double* physical_gradient,
	                                      int count,
	                                      int pivot,
	                                      double pivot_sign,
	                                      double* coordinate_gradient)
	{
		return PullbackGradientChart(
			physical, coordinates, physical_gradient, count, pivot, pivot_sign,
			Pi(), 2.0, false, coordinate_gradient);
	}
};

}  // namespace sus2

#endif  // MLIP_RADIAL_WEIGHT_NORM_H
