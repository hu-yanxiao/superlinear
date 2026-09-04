#ifndef MLIP_EFS_HEAD_COORDINATE_SCALING_H
#define MLIP_EFS_HEAD_COORDINATE_SCALING_H

#include <algorithm>
#include <cmath>

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

} // namespace sus2

#endif
