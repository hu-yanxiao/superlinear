#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "dev_src/radial_weight_norm.h"

namespace {

void Require(bool condition, const char* message)
{
	if (!condition) {
		std::cerr << message << '\n';
		std::exit(1);
	}
}

double Loss(const double* physical, int count)
{
	double value = 0.0;
	for (int i = 0; i < count; ++i)
		value += 0.5 * (i + 1.0) * physical[i] * physical[i]
		       + std::sin(physical[i]);
	return value;
}

}  // namespace

int main()
{
	const int count = 5;
	const double rechart_radius = 8.0;
	double coordinates[count] = {};
	double reconstructed[count] = {};

	// A regular row must stay in the same chart and round-trip exactly within
	// floating-point tolerance.
	double stable_physical[count] = {1.4, -0.2, 0.1, 0.05, -0.03};
	int stable_pivot = 0;
	double stable_sign = 1.0;
	const sus2::RadialWeightNormChart::RechartResult stable_result =
		sus2::RadialWeightNormChart::RechartIfNeeded(
			stable_physical,
			count,
			rechart_radius,
			stable_pivot,
			stable_sign,
			coordinates);
	Require(stable_result.success && !stable_result.recharted
	            && stable_result.reason
	               == sus2::RadialWeightNormChart::kRechartNone,
	        "stable chart unexpectedly recharted");
	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            coordinates,
	            count,
	            stable_pivot,
	            stable_sign,
	            reconstructed),
	        "stable chart roundtrip unpack failed");
	for (int i = 0; i < count; ++i) {
		const double tolerance = 64.0
			* std::numeric_limits<double>::epsilon()
			* std::max(1.0, std::abs(stable_physical[i]));
		Require(std::abs(reconstructed[i] - stable_physical[i]) <= tolerance,
		        "stable chart roundtrip changed physical values");
	}

	// The old chart cannot represent its exact antipode, while the same
	// physical row is regular after selecting the signed pivot again.
	double exact_antipode[count] = {-2.0, 0.0, 0.0, 0.0, 0.0};
	double exact_before[count];
	std::copy(exact_antipode, exact_antipode + count, exact_before);
	int pivot = 0;
	double pivot_sign = 1.0;
	Require(!sus2::RadialWeightNormChart::PackWithPivot(
	            exact_antipode, count, pivot, pivot_sign, coordinates),
	        "old chart unexpectedly packed its exact antipode");
	const sus2::RadialWeightNormChart::RechartResult exact_result =
		sus2::RadialWeightNormChart::RechartIfNeeded(
			exact_antipode,
			count,
			rechart_radius,
			pivot,
			pivot_sign,
			coordinates);
	Require(exact_result.success && exact_result.recharted,
	        "exact-antipode rechart failed");
	Require(exact_result.reason
	            == sus2::RadialWeightNormChart::kRechartPackFailed,
	        "exact-antipode rechart reported the wrong reason");
	Require(pivot == 0 && pivot_sign == -1.0,
	        "exact-antipode rechart selected the wrong signed pivot");
	Require(std::equal(
	            exact_antipode, exact_antipode + count, exact_before),
	        "exact-antipode rechart modified physical values");

	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            coordinates,
	            count,
	            pivot,
	            pivot_sign,
	            reconstructed),
	        "exact-antipode rechart roundtrip failed");
	for (int i = 0; i < count; ++i) {
		const double tolerance = 64.0
			* std::numeric_limits<double>::epsilon()
			* std::max(1.0, std::abs(exact_antipode[i]));
		Require(std::abs(reconstructed[i] - exact_antipode[i]) <= tolerance,
		        "exact-antipode roundtrip changed physical values");
	}

	// Radius 9 is deliberately above the accepted-point rechart threshold,
	// but far below the double-precision tanh saturation wall near radius 38.
	double old_coordinates[count] = {};
	old_coordinates[0] = std::log(1.7);
	old_coordinates[1] = 9.0;
	pivot = 0;
	pivot_sign = 1.0;
	double physical[count] = {};
	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            old_coordinates,
	            count,
	            pivot,
	            pivot_sign,
	            physical),
	        "near-antipode setup failed");
	double physical_before[count];
	std::copy(physical, physical + count, physical_before);
	const double loss_before = Loss(physical, count);
	const sus2::RadialWeightNormChart::RechartResult radius_result =
		sus2::RadialWeightNormChart::RechartIfNeeded(
			physical,
			count,
			rechart_radius,
			pivot,
			pivot_sign,
			coordinates);
	Require(radius_result.success && radius_result.recharted,
	        "direction-radius rechart failed");
	Require(radius_result.reason
	            == sus2::RadialWeightNormChart::kRechartDirectionRadiusExceeded,
	        "direction-radius rechart reported the wrong reason");
	Require(radius_result.old_direction_radius >= rechart_radius
	            && radius_result.new_direction_radius < rechart_radius,
	        "direction-radius rechart did not move to a safe chart");
	Require(std::equal(physical, physical + count, physical_before),
	        "direction-radius rechart modified physical values");
	Require(Loss(physical, count) == loss_before,
	        "direction-radius rechart modified the physical objective");
	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            coordinates,
	            count,
	            pivot,
	            pivot_sign,
	            reconstructed),
	        "direction-radius rechart roundtrip failed");
	for (int i = 0; i < count; ++i) {
		const double tolerance = 64.0
			* std::numeric_limits<double>::epsilon()
			* std::max(1.0, std::abs(physical_before[i]));
		Require(std::abs(reconstructed[i] - physical_before[i]) <= tolerance,
		        "direction-radius roundtrip changed physical values");
	}

	double physical_gradient[count] = {};
	for (int i = 0; i < count; ++i)
		physical_gradient[i] =
			(i + 1.0) * physical[i] + std::cos(physical[i]);
	double coordinate_gradient[count] = {};
	Require(sus2::RadialWeightNormChart::PullbackGradientWithPivot(
	            physical,
	            coordinates,
	            physical_gradient,
	            count,
	            pivot,
	            pivot_sign,
	            coordinate_gradient),
	        "recharted pullback failed");
	const double h = 1.0e-6;
	for (int coordinate = 0; coordinate < count; ++coordinate) {
		double plus_coordinates[count];
		double minus_coordinates[count];
		std::copy(coordinates, coordinates + count, plus_coordinates);
		std::copy(coordinates, coordinates + count, minus_coordinates);
		plus_coordinates[coordinate] += h;
		minus_coordinates[coordinate] -= h;
		double plus_physical[count] = {};
		double minus_physical[count] = {};
		Require(sus2::RadialWeightNormChart::UnpackWithPivot(
		            plus_coordinates,
		            count,
		            pivot,
		            pivot_sign,
		            plus_physical)
		            && sus2::RadialWeightNormChart::UnpackWithPivot(
		                minus_coordinates,
		                count,
		                pivot,
		                pivot_sign,
		                minus_physical),
		        "finite-difference unpack failed");
		const double fd =
			(Loss(plus_physical, count) - Loss(minus_physical, count))
			/ (2.0 * h);
		const double scale = std::max(1.0, std::abs(fd));
		Require(std::isfinite(coordinate_gradient[coordinate])
		            && std::abs(fd - coordinate_gradient[coordinate])
		               <= 2.0e-8 * scale,
		        "recharted pullback finite difference mismatch");
	}

	// Exercise the actual numerical failure chain: a huge optimizer direction
	// saturates tanh, Unpack lands on the excluded antipode, and the old fixed
	// chart can no longer Pack it. Automatic rechart must still recover the
	// exact physical row.
	double saturated_coordinates[count] = {};
	saturated_coordinates[0] = std::log(1.3);
	saturated_coordinates[1] = 80.0;
	pivot = 0;
	pivot_sign = 1.0;
	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            saturated_coordinates,
	            count,
	            pivot,
	            pivot_sign,
	            physical),
	        "saturated direction setup failed");
	std::copy(physical, physical + count, physical_before);
	Require(!sus2::RadialWeightNormChart::PackWithPivot(
	            physical,
	            count,
	            pivot,
	            pivot_sign,
	            coordinates),
	        "saturated old chart unexpectedly packed its antipode");
	const sus2::RadialWeightNormChart::RechartResult saturated_result =
		sus2::RadialWeightNormChart::RechartIfNeeded(
			physical,
			count,
			rechart_radius,
			pivot,
			pivot_sign,
			coordinates);
	Require(saturated_result.success && saturated_result.recharted
	            && saturated_result.reason
	               == sus2::RadialWeightNormChart::kRechartPackFailed,
	        "saturated antipode automatic rechart failed");
	Require(sus2::RadialWeightNormChart::UnpackWithPivot(
	            coordinates,
	            count,
	            pivot,
	            pivot_sign,
	            reconstructed),
	        "saturated antipode rechart roundtrip failed");
	for (int i = 0; i < count; ++i) {
		const double tolerance = 64.0
			* std::numeric_limits<double>::epsilon()
			* std::max(1.0, std::abs(physical_before[i]));
		Require(std::abs(reconstructed[i] - physical_before[i]) <= tolerance,
		        "saturated antipode roundtrip changed physical values");
	}

	// Repeated chart changes must not accumulate any physical modification.
	pivot = 0;
	pivot_sign = 1.0;
	for (int cycle = 0; cycle < 8; ++cycle) {
		double far_coordinates[count] = {};
		far_coordinates[0] = std::log(1.0 + 0.1 * cycle);
		far_coordinates[1] = 9.0;
		Require(sus2::RadialWeightNormChart::UnpackWithPivot(
		            far_coordinates,
		            count,
		            pivot,
		            pivot_sign,
		            physical),
		        "repeated rechart setup failed");
		std::copy(physical, physical + count, physical_before);
		const double old_sign = pivot_sign;
		const sus2::RadialWeightNormChart::RechartResult result =
			sus2::RadialWeightNormChart::RechartIfNeeded(
				physical,
				count,
				rechart_radius,
				pivot,
				pivot_sign,
				coordinates);
		Require(result.success && result.recharted,
		        "repeated rechart failed");
		Require(result.reason
		            == sus2::RadialWeightNormChart::kRechartDirectionRadiusExceeded
		            && result.old_direction_radius > rechart_radius
		            && result.new_direction_radius <= rechart_radius,
		        "repeated rechart used the wrong trigger or unsafe radius");
		Require(pivot == 0 && pivot_sign == -old_sign,
		        "repeated rechart selected the wrong signed pivot");
		Require(std::equal(physical, physical + count, physical_before),
		        "repeated rechart modified physical values");
		Require(sus2::RadialWeightNormChart::UnpackWithPivot(
		            coordinates,
		            count,
		            pivot,
		            pivot_sign,
		            reconstructed),
		        "repeated rechart roundtrip unpack failed");
		for (int i = 0; i < count; ++i) {
			const double tolerance = 64.0
				* std::numeric_limits<double>::epsilon()
				* std::max(1.0, std::abs(physical_before[i]));
			Require(std::abs(reconstructed[i] - physical_before[i]) <= tolerance,
			        "repeated rechart accumulated physical error");
		}
	}

	std::cout << "radial_weight_norm_rechart_check passed\n";
	return 0;
}
