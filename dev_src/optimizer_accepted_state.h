#ifndef MLIP_OPTIMIZER_ACCEPTED_STATE_H
#define MLIP_OPTIMIZER_ACCEPTED_STATE_H

#include <algorithm>
#include <vector>

namespace sus2 {

inline bool AscentRecoveryPointWasAccepted(
	bool optimizer_had_active_trial,
	bool use_armijo_value_line_search,
	bool armijo_value_trial_accepted)
{
	if (!optimizer_had_active_trial)
		return false;
	return !use_armijo_value_line_search || armijo_value_trial_accepted;
}

// Convert the optimizer's reverse-communication state into the trainer's
// evaluated-point state.  A successful recovery from an ascent-direction
// failure has already accepted the evaluated point, even though the retry has
// immediately placed the optimizer back in line search at its next trial.
inline bool EvaluatedPointWasAccepted(
	bool recovered_accepted_point_from_ascent,
	bool use_armijo_value_line_search,
	bool armijo_value_trial_accepted,
	bool optimizer_is_in_linesearch)
{
	if (recovered_accepted_point_from_ascent)
		return true;
	if (use_armijo_value_line_search)
		return armijo_value_trial_accepted;
	return !optimizer_is_in_linesearch;
}

// BFGS uses a reverse-communication interface.  After it accepts the point
// whose objective was just evaluated, its x already contains the first trial
// for the next line search.  Keep the evaluated/accepted physical point in a
// separate object so checkpoint, restart, and coordinate-chart operations can
// never accidentally consume that unevaluated trial.
class OptimizerAcceptedState {
public:
	explicit OptimizerAcceptedState(int count)
		: coefficients_(count, 0.0), has_pending_trial_(false) {}

	void SetBoundaryBase(const double* coefficients)
	{
		std::copy(
			coefficients,
			coefficients + coefficients_.size(),
			coefficients_.begin());
		has_pending_trial_ = false;
	}

	void SetEvaluatedAccepted(const double* coefficients)
	{
		std::copy(
			coefficients,
			coefficients + coefficients_.size(),
			coefficients_.begin());
		has_pending_trial_ = true;
	}

	void Restore(double* coefficients) const
	{
		std::copy(
			coefficients_.begin(),
			coefficients_.end(),
			coefficients);
	}

	bool HasPendingTrial() const { return has_pending_trial_; }
	bool RestoreAcceptedForBoundary(double* coefficients) const
	{
		if (!has_pending_trial_)
			return false;
		Restore(coefficients);
		return true;
	}
	void DiscardPendingTrial() { has_pending_trial_ = false; }

private:
	std::vector<double> coefficients_;
	bool has_pending_trial_;
};

} // namespace sus2

#endif // MLIP_OPTIMIZER_ACCEPTED_STATE_H
