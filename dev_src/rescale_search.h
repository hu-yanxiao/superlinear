#pragma once

#include <cmath>

// Select the first bracket-expansion direction from explicit probes on both
// sides of the seed.  Returning -1 selects the lower-scaling probe, +1 the
// upper-scaling probe, and 0 keeps the seed because neither probe improves
// |center_log|.  This avoids assuming a fixed sign convention between model
// scaling and the fitted moment-coefficient scale.
inline int SelectRescaleExpansionDirection(
	double seed_center_log,
	bool lower_valid,
	double lower_center_log,
	bool upper_valid,
	double upper_center_log)
{
	const double seed_error = std::abs(seed_center_log);
	double best_error = seed_error;
	int direction = 0;
	if (lower_valid && std::abs(lower_center_log) < best_error) {
		best_error = std::abs(lower_center_log);
		direction = -1;
	}
	if (upper_valid && std::abs(upper_center_log) < best_error) {
		direction = 1;
	}
	return direction;
}
