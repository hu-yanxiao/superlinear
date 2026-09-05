#ifndef MLIP_EFS_HEAD_COORDINATE_SCALING_H
#define MLIP_EFS_HEAD_COORDINATE_SCALING_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sus2 {

// Exact linear-head coordinates z = d * (h - h_ref). Unlike the conservative
// efs-gn chart, this opt-in chart can amplify weak physical directions: its
// purpose is to normalize the weighted E/F/stress column curvature to one.
// The caller clamps finite scales and keeps the objective/regularizer physical.
struct ExactHeadCoordinateScaling {
	static double ScaleFromCurvature(double curvature,
	                                 double positive_block_median,
	                                 double relative_floor)
	{
		if (!std::isfinite(curvature) || curvature < 0.0
		    || !std::isfinite(positive_block_median) || positive_block_median < 0.0
		    || !std::isfinite(relative_floor) || relative_floor < 0.0)
			return 0.0;
		const double stabilized =
			std::max(curvature, relative_floor * positive_block_median);
		return stabilized > 0.0 ? std::sqrt(stabilized) : 1.0;
	}
};

// Dense fixed linear-head coordinates.  If H + shift I = L L^T, the chart is
//
//     z = L^T (h - h_ref),       h = h_ref + L^{-T} z,
//     grad_z = L^{-1} grad_h.
//
// Consequently an identity inverse Hessian in z is the inverse of the
// stabilized physical head metric.  The diagonal shift belongs only to this
// optimizer chart; callers must continue to evaluate the objective and every
// regularizer in physical h.
struct FullHeadCoordinateScaling {
	static bool Cholesky(const std::vector<double>& matrix,
	                     int size,
	                     double diagonal_shift,
	                     std::vector<double>& lower)
	{
		if (size <= 0
		    || matrix.size() != static_cast<std::size_t>(size) * size
		    || !std::isfinite(diagonal_shift) || diagonal_shift < 0.0)
			return false;
		lower.assign(static_cast<std::size_t>(size) * size, 0.0);
		for (int row = 0; row < size; ++row) {
			for (int col = 0; col <= row; ++col) {
				double value = 0.5 * (
					matrix[static_cast<std::size_t>(row) * size + col]
					+ matrix[static_cast<std::size_t>(col) * size + row]);
				if (row == col)
					value += diagonal_shift;
				for (int inner = 0; inner < col; ++inner)
					value -= lower[static_cast<std::size_t>(row) * size + inner]
					       * lower[static_cast<std::size_t>(col) * size + inner];
				if (row == col) {
					if (!(value > 0.0) || !std::isfinite(value))
						return false;
					lower[static_cast<std::size_t>(row) * size + col] =
						std::sqrt(value);
				} else {
					const double pivot =
						lower[static_cast<std::size_t>(col) * size + col];
					if (!(pivot > 0.0) || !std::isfinite(pivot))
						return false;
					const double entry = value / pivot;
					if (!std::isfinite(entry))
						return false;
					lower[static_cast<std::size_t>(row) * size + col] = entry;
				}
			}
		}
		return true;
	}

	static bool FactorMetric(const std::vector<double>& metric,
	                         int size,
	                         double relative_floor,
	                         std::vector<double>& lower,
	                         double& diagonal_shift)
	{
		if (size <= 0
		    || metric.size() != static_cast<std::size_t>(size) * size
		    || !std::isfinite(relative_floor) || relative_floor < 0.0)
			return false;
		std::vector<double> positive_diagonal;
		positive_diagonal.reserve(size);
		double maximum_diagonal = 0.0;
		for (int i = 0; i < size; ++i) {
			const double value = metric[static_cast<std::size_t>(i) * size + i];
			if (!std::isfinite(value) || value < 0.0)
				return false;
			maximum_diagonal = std::max(maximum_diagonal, value);
			if (value > 0.0)
				positive_diagonal.push_back(value);
		}
		if (positive_diagonal.empty())
			return false;
		const std::size_t middle = positive_diagonal.size() / 2;
		std::nth_element(
			positive_diagonal.begin(),
			positive_diagonal.begin() + middle,
			positive_diagonal.end());
		double median = positive_diagonal[middle];
		if ((positive_diagonal.size() & 1U) == 0U) {
			const double upper = median;
			std::nth_element(
				positive_diagonal.begin(),
				positive_diagonal.begin() + middle - 1,
				positive_diagonal.begin() + middle);
			median = 0.5 * (positive_diagonal[middle - 1] + upper);
		}
		const double roundoff_floor =
			std::numeric_limits<double>::epsilon()
			* std::max(1.0, maximum_diagonal)
			* std::max(1, size);
		diagonal_shift = std::max(relative_floor * median, roundoff_floor);
		if (Cholesky(metric, size, diagonal_shift, lower))
			return true;

		// A Gram assembled in floating point can have a tiny negative eigenvalue.
		// Increasing a chart-only shift is safe, but keep the process bounded and
		// observable through the returned shift.
		for (int attempt = 0; attempt < 8; ++attempt) {
			diagonal_shift *= 10.0;
			if (Cholesky(metric, size, diagonal_shift, lower))
				return true;
		}
		return false;
	}

	static bool SolvePhysicalPositiveDefinite(
		const std::vector<double>& matrix,
		const std::vector<double>& rhs,
		int size,
		std::vector<double>& solution)
	{
		std::vector<double> lower;
		// No diagonal floor is permitted here: this solves the physical
		// quadratic.  A singular physical problem must be regularized explicitly
		// by its physical head penalty or rejected by the caller.
		if (rhs.size() != static_cast<std::size_t>(size)
		    || !Cholesky(matrix, size, 0.0, lower))
			return false;
		solution = rhs;
		for (int row = 0; row < size; ++row) {
			double value = solution[row];
			for (int col = 0; col < row; ++col)
				value -= lower[static_cast<std::size_t>(row) * size + col]
				       * solution[col];
			const double pivot = lower[static_cast<std::size_t>(row) * size + row];
			solution[row] = value / pivot;
		}
		for (int row = size - 1; row >= 0; --row) {
			double value = solution[row];
			for (int col = row + 1; col < size; ++col)
				value -= lower[static_cast<std::size_t>(col) * size + row]
				       * solution[col];
			const double pivot = lower[static_cast<std::size_t>(row) * size + row];
			solution[row] = value / pivot;
			if (!std::isfinite(solution[row]))
				return false;
		}
		return true;
	}

	static bool PackCentered(const std::vector<double>& physical,
	                         const std::vector<double>& reference,
	                         const std::vector<double>& lower,
	                         int size,
	                         std::vector<double>& packed)
	{
		if (!ValidSizes(physical, reference, lower, size))
			return false;
		packed.assign(size, 0.0);
		for (int col = 0; col < size; ++col) {
			double value = 0.0;
			for (int row = col; row < size; ++row)
				value += lower[static_cast<std::size_t>(row) * size + col]
				       * (physical[row] - reference[row]);
			if (!std::isfinite(value))
				return false;
			packed[col] = value;
		}
		return true;
	}

	static bool UnpackCentered(const std::vector<double>& packed,
	                           const std::vector<double>& reference,
	                           const std::vector<double>& lower,
	                           int size,
	                           std::vector<double>& physical)
	{
		if (!ValidSizes(packed, reference, lower, size))
			return false;
		physical.assign(size, 0.0);
		for (int row = size - 1; row >= 0; --row) {
			double value = packed[row];
			for (int col = row + 1; col < size; ++col)
				value -= lower[static_cast<std::size_t>(col) * size + row]
				       * (physical[col] - reference[col]);
			const double pivot = lower[static_cast<std::size_t>(row) * size + row];
			if (!(pivot > 0.0) || !std::isfinite(pivot))
				return false;
			physical[row] = reference[row] + value / pivot;
			if (!std::isfinite(physical[row]))
				return false;
		}
		return true;
	}

	static bool PullbackGradient(const std::vector<double>& physical_gradient,
	                             const std::vector<double>& lower,
	                             int size,
	                             std::vector<double>& packed_gradient)
	{
		if (size <= 0
		    || physical_gradient.size() != static_cast<std::size_t>(size)
		    || lower.size() != static_cast<std::size_t>(size) * size)
			return false;
		packed_gradient.assign(size, 0.0);
		for (int row = 0; row < size; ++row) {
			double value = physical_gradient[row];
			for (int col = 0; col < row; ++col)
				value -= lower[static_cast<std::size_t>(row) * size + col]
				       * packed_gradient[col];
			const double pivot = lower[static_cast<std::size_t>(row) * size + row];
			if (!(pivot > 0.0) || !std::isfinite(pivot))
				return false;
			packed_gradient[row] = value / pivot;
			if (!std::isfinite(packed_gradient[row]))
				return false;
		}
		return true;
	}

private:
	static bool ValidSizes(const std::vector<double>& values,
	                       const std::vector<double>& reference,
	                       const std::vector<double>& lower,
	                       int size)
	{
		return size > 0
		    && values.size() == static_cast<std::size_t>(size)
		    && reference.size() == static_cast<std::size_t>(size)
		    && lower.size() == static_cast<std::size_t>(size) * size;
	}
};

} // namespace sus2

#endif
