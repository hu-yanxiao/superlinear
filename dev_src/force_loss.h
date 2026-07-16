#ifndef MLIP_FORCE_LOSS_H
#define MLIP_FORCE_LOSS_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

enum class ForceLossKind {
	L2,
	LogCosh
};

inline const char* ForceLossKindName(ForceLossKind kind)
{
	switch (kind) {
	case ForceLossKind::L2:
		return "l2";
	case ForceLossKind::LogCosh:
		return "log-cosh";
	}
	return "unknown";
}

inline ForceLossKind ParseForceLossKind(const std::string& value)
{
	if (value == "l2")
		return ForceLossKind::L2;
	if (value == "log-cosh")
		return ForceLossKind::LogCosh;
	throw std::invalid_argument("unknown force loss kind: " + value);
}

inline double StableLogCosh(double x)
{
	const double ax = std::abs(x);
	return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
}

inline void ValidateForceLogCoshScale(double scale)
{
	if (!std::isfinite(scale) || scale <= 0.0)
		throw std::invalid_argument("--force-log-cosh-scale should be a finite positive value");
}

inline double ForceResidualLoss(double residual, ForceLossKind kind, double log_cosh_scale)
{
	switch (kind) {
	case ForceLossKind::L2:
		return residual * residual;
	case ForceLossKind::LogCosh:
		ValidateForceLogCoshScale(log_cosh_scale);
		return log_cosh_scale * log_cosh_scale *
			StableLogCosh(residual / log_cosh_scale);
	}
	throw std::invalid_argument("unknown force loss kind");
}

inline double ForceResidualGrad(double residual, ForceLossKind kind, double log_cosh_scale)
{
	switch (kind) {
	case ForceLossKind::L2:
		return 2.0 * residual;
	case ForceLossKind::LogCosh:
		ValidateForceLogCoshScale(log_cosh_scale);
		return log_cosh_scale * std::tanh(residual / log_cosh_scale);
	}
	throw std::invalid_argument("unknown force loss kind");
}

inline double ForceResidualCurvature(double residual, ForceLossKind kind, double log_cosh_scale)
{
	switch (kind) {
	case ForceLossKind::L2:
		return 2.0;
	case ForceLossKind::LogCosh: {
		ValidateForceLogCoshScale(log_cosh_scale);
		const double tangent = std::tanh(residual / log_cosh_scale);
		return std::max(0.0, 1.0 - tangent * tangent);
	}
	}
	throw std::invalid_argument("unknown force loss kind");
}

#endif
