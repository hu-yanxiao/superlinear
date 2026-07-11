/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "lbfgs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

int FirstNonFinite(const Array1D& values, int size)
{
	for (int i = 0; i < size; ++i)
		if (!std::isfinite(values[i]))
			return i;
	return -1;
}

double ClampDouble(double value, double low, double high)
{
	return std::max(low, std::min(high, value));
}

} // namespace

LBFGS::LBFGS(int max_history)
	: max_history_(std::max(max_history, 1))
{
}

void LBFGS::ClearHistory()
{
	history_.clear();
	gamma_ = 1.0;
}

void LBFGS::ResetLineSearchState()
{
	linesearch_.Reset();
	is_in_linesearch_ = false;
	armijo_value_only_trial_ready_ = false;
	linesearch_stagnated_ = false;
	last_trial_backtracks_ = 0;
	last_trial_rejected_nonfinite_ = false;
	iter_step = 0;
}

void LBFGS::Resize(int _size)
{
	if (_size < 0)
		ERROR("LBFGS::Resize(): invalid size.");
	if (size_ == _size)
		return;

	size_ = _size;
	x_.resize(size_);
	x_start_.resize(size_);
	g_start_.resize(size_);
	p_.resize(size_);
	inv_hess_diag_.assign(size_, 1.0);
	q_.resize(size_);
	r_.resize(size_);
	damped_y_.resize(size_);
	b0s_.resize(size_);
	two_loop_alpha_.resize(max_history_);
	coordinate_mask_.assign(size_, 0);
	ClearHistory();
	has_start_ = false;
	ResetLineSearchState();
}

void LBFGS::ValidateSize(const Array1D& values, const char* name) const
{
	if ((int)values.size() != size_)
		ERROR(std::string("LBFGS::") + name + "(): vector size mismatch.");
}

void LBFGS::ValidateFinite(const Array1D& values, const char* name) const
{
	const int bad_index = FirstNonFinite(values, size_);
	if (bad_index >= 0)
		ERROR(std::string(name) + " contains a non-finite value at index "
		      + std::to_string(bad_index));
}

void LBFGS::ApplyMask(Array1D& values) const
{
	for (int i = 0; i < size_; ++i)
		if (coordinate_mask_[i])
			values[i] = 0.0;
}

double LBFGS::Dot(const Array1D& lhs, const Array1D& rhs) const
{
	double result = 0.0;
	for (int i = 0; i < size_; ++i)
		if (!coordinate_mask_[i])
			result += lhs[i] * rhs[i];
	return result;
}

double LBFGS::BaseHessianQuadratic(const Array1D& values) const
{
	double result = 0.0;
	for (int i = 0; i < size_; ++i) {
		if (coordinate_mask_[i])
			continue;
		result += values[i] * values[i] / (gamma_ * inv_hess_diag_[i]);
	}
	return result;
}

double LBFGS::BaseInverseQuadratic(const Array1D& values) const
{
	double result = 0.0;
	for (int i = 0; i < size_; ++i) {
		if (coordinate_mask_[i])
			continue;
		result += values[i] * values[i] * gamma_ * inv_hess_diag_[i];
	}
	return result;
}

void LBFGS::UpdateHistoryFromAcceptedGradient(const Array1D& g)
{
	last_update_skipped_ = false;
	last_update_damped_ = false;
	last_powell_theta_ = 1.0;
	last_curvature_ = 0.0;
	last_relative_curvature_ = 0.0;
	last_base_sBs_ = 0.0;
	last_base_yHy_ = 0.0;

	if (!has_start_)
		return;

	HistoryEntry entry;
	entry.s.resize(size_);
	entry.y.resize(size_);
	for (int i = 0; i < size_; ++i) {
		entry.s[i] = coordinate_mask_[i] ? 0.0 : x_[i] - x_start_[i];
		entry.y[i] = coordinate_mask_[i] ? 0.0 : g[i] - g_start_[i];
	}

	ValidateFinite(entry.s, "LBFGS accepted displacement");
	ValidateFinite(entry.y, "LBFGS accepted gradient change");

	const double raw_sy = Dot(entry.s, entry.y);
	const double s_b0_s = BaseHessianQuadratic(entry.s);
	const double y_h0_y = BaseInverseQuadratic(entry.y);
	last_base_sBs_ = s_b0_s;
	last_base_yHy_ = y_h0_y;

	const double metric_product = s_b0_s * y_h0_y;
	if (!(s_b0_s > 0.0) || !(y_h0_y > 0.0) || !std::isfinite(metric_product)) {
		last_update_skipped_ = true;
		++skipped_updates_;
		return;
	}

	const double raw_relative_curvature = raw_sy / std::sqrt(metric_product);
	last_relative_curvature_ = raw_relative_curvature;

	for (int i = 0; i < size_; ++i) {
		b0s_[i] = coordinate_mask_[i]
			? 0.0
			: entry.s[i] / (gamma_ * inv_hess_diag_[i]);
		damped_y_[i] = entry.y[i];
	}

	const double damping_floor = powell_damping_curvature * s_b0_s;
	if (raw_sy < damping_floor) {
		const double denominator = s_b0_s - raw_sy;
		if (!(denominator > 0.0) || !std::isfinite(denominator)) {
			last_update_skipped_ = true;
			++skipped_updates_;
			return;
		}
		const double theta = ClampDouble((1.0 - powell_damping_curvature) * s_b0_s
		                                 / denominator,
		                                 0.0, 1.0);
		last_powell_theta_ = theta;
		last_update_damped_ = true;
		for (int i = 0; i < size_; ++i)
			damped_y_[i] = theta * entry.y[i] + (1.0 - theta) * b0s_[i];
		++damped_updates_;
	}

	const double sy = Dot(entry.s, damped_y_);
	const double damped_y_h0_y = BaseInverseQuadratic(damped_y_);
	last_curvature_ = sy;
	last_base_yHy_ = damped_y_h0_y;

	if (!(damped_y_h0_y > 0.0) || !std::isfinite(damped_y_h0_y)) {
		last_update_skipped_ = true;
		++skipped_updates_;
		return;
	}

	const double relative_curvature = sy / std::sqrt(s_b0_s * damped_y_h0_y);
	entry.relative_curvature = relative_curvature;
	last_relative_curvature_ = relative_curvature;

	const double curvature_scale = std::sqrt(s_b0_s * damped_y_h0_y);
	const double abs_floor = 64.0 * std::numeric_limits<double>::epsilon()
		* curvature_scale;
	if (!(sy > abs_floor) || !(relative_curvature >= relative_curvature_tol)
	    || !std::isfinite(relative_curvature)) {
		last_update_skipped_ = true;
		++skipped_updates_;
		return;
	}

	entry.y = damped_y_;
	entry.s_dot_y = sy;
	entry.rho = 1.0 / sy;
	if (!std::isfinite(entry.rho)) {
		last_update_skipped_ = true;
		++skipped_updates_;
		return;
	}

	if ((int)history_.size() == max_history_)
		history_.erase(history_.begin());
	history_.push_back(entry);
	++accepted_updates_;

	const double gamma_denominator = damped_y_h0_y / gamma_;
	const double gamma_candidate = sy / gamma_denominator;
	if (gamma_candidate > 0.0 && std::isfinite(gamma_candidate))
		gamma_ = ClampDouble(gamma_candidate, 1.0e-12, 1.0e12);
}

bool LBFGS::FormDirection(const Array1D& g)
{
	for (int i = 0; i < size_; ++i)
		q_[i] = coordinate_mask_[i] ? 0.0 : g[i];

	const int hist_size = (int)history_.size();
	if ((int)two_loop_alpha_.size() < hist_size)
		two_loop_alpha_.resize(hist_size);

	for (int i = hist_size - 1; i >= 0; --i) {
		const HistoryEntry& entry = history_[i];
		const double alpha = entry.rho * Dot(entry.s, q_);
		two_loop_alpha_[i] = alpha;
		for (int j = 0; j < size_; ++j)
			q_[j] -= alpha * entry.y[j];
	}

	for (int i = 0; i < size_; ++i)
		r_[i] = coordinate_mask_[i] ? 0.0 : gamma_ * inv_hess_diag_[i] * q_[i];

	for (int i = 0; i < hist_size; ++i) {
		const HistoryEntry& entry = history_[i];
		const double beta = entry.rho * Dot(entry.y, r_);
		const double coeff = two_loop_alpha_[i] - beta;
		for (int j = 0; j < size_; ++j)
			r_[j] += coeff * entry.s[j];
	}

	for (int i = 0; i < size_; ++i)
		p_[i] = coordinate_mask_[i] ? 0.0 : -r_[i];

	ValidateFinite(p_, "LBFGS search direction");
	p_dot_g_ = Dot(p_, g);
	if (!std::isfinite(p_dot_g_))
		ERROR("LBFGS produced a non-finite directional derivative.");

	const double grad_h0_grad = BaseInverseQuadratic(g);
	if (p_dot_g_ < 0.0)
		return true;
	if (!(grad_h0_grad > 0.0))
		return false;

	ClearHistory();
	++non_descent_resets_;
	for (int i = 0; i < size_; ++i)
		p_[i] = coordinate_mask_[i] ? 0.0 : -inv_hess_diag_[i] * g[i];
	ValidateFinite(p_, "LBFGS reset search direction");
	p_dot_g_ = Dot(p_, g);
	if (!std::isfinite(p_dot_g_))
		ERROR("LBFGS reset produced a non-finite directional derivative.");
	if (p_dot_g_ < 0.0)
		return true;

	ERROR("LBFGS failed to produce a descent direction.");
	return false;
}

void LBFGS::BuildTrialPoint()
{
	last_trial_step_ = linesearch_.x();
	for (int i = 0; i < size_; ++i)
		x_[i] = x_start_[i] + last_trial_step_ * p_[i];
	ValidateFinite(x_, "LBFGS Armijo trial point");
}

bool LBFGS::ReduceCurrentStep(double ratio)
{
	for (;;) {
		const double old_step = linesearch_.x();
		linesearch_.ReduceStep(ratio);
		const double new_step = linesearch_.x();
		const double step_tol = 64.0 * std::numeric_limits<double>::epsilon()
			* std::max(1.0, std::max(std::abs(old_step), std::abs(new_step)));
		if (std::abs(new_step - old_step) <= step_tol
		    || std::abs(new_step) < min_initial_step) {
			linesearch_stagnated_ = true;
			return false;
		}
		try {
			BuildTrialPoint();
			return true;
		}
		catch (const MlipException&) {
			last_trial_rejected_nonfinite_ = true;
			++last_trial_backtracks_;
			++total_backtracks_;
		}
	}
}

bool LBFGS::StartArmijoTrial(double f, const Array1D& g)
{
	if (!FormDirection(g)) {
		armijo_value_only_trial_ready_ = false;
		is_in_linesearch_ = false;
		linesearch_stagnated_ = true;
		return false;
	}

	std::copy(x_.begin(), x_.end(), x_start_.begin());
	std::copy(g.begin(), g.end(), g_start_.begin());
	f_start_ = f;
	p_dot_g_start_ = p_dot_g_;
	has_start_ = true;

	const double initial_step = ClampDouble(next_initial_step_, min_initial_step, max_initial_step);
	linesearch_.Reset(initial_step);
	linesearch_.Iterate(f, p_dot_g_start_);
	is_in_linesearch_ = true;
	armijo_value_only_trial_ready_ = true;
	linesearch_stagnated_ = false;
	last_trial_backtracks_ = 0;
	last_trial_rejected_nonfinite_ = false;

	for (;;) {
		try {
			BuildTrialPoint();
			break;
		}
		catch (const MlipException&) {
			last_trial_rejected_nonfinite_ = true;
			++last_trial_backtracks_;
			++total_backtracks_;
			if (!ReduceCurrentStep(0.5)) {
				linesearch_stagnated_ = true;
				armijo_value_only_trial_ready_ = false;
				is_in_linesearch_ = false;
				std::copy(x_start_.begin(), x_start_.end(), x_.begin());
				break;
			}
			break;
		}
	}
	return armijo_value_only_trial_ready_;
}

void LBFGS::AdaptNextInitialStepAfterAccept()
{
	last_accepted_step_ = linesearch_.x();
	if (!std::isfinite(last_accepted_step_) || !(last_accepted_step_ > 0.0))
		return;

	if (last_trial_backtracks_ == 0)
		next_initial_step_ = ClampDouble(last_accepted_step_ * initial_step_growth,
		                                 min_initial_step, max_initial_step);
	else
		next_initial_step_ = ClampDouble(last_accepted_step_, min_initial_step, max_initial_step);
}

void LBFGS::Set_x(const double* x, int _size)
{
	Resize(_size);
	if (_size > 0)
		memcpy(x_.data(), x, _size * sizeof(double));
	ValidateFinite(x_, "LBFGS x");
	ClearHistory();
	std::fill(coordinate_mask_.begin(), coordinate_mask_.end(), 0);
	next_initial_step_ = configured_initial_step_;
	f_start_ = HUGE_DOUBLE;
	p_dot_g_start_ = HUGE_DOUBLE;
	has_start_ = false;
	ResetLineSearchState();
}

void LBFGS::Restart()
{
	ClearHistory();
	std::fill(coordinate_mask_.begin(), coordinate_mask_.end(), 0);
	next_initial_step_ = configured_initial_step_;
	has_start_ = false;
	f_start_ = HUGE_DOUBLE;
	p_dot_g_start_ = HUGE_DOUBLE;
	++restart_count_;
	ResetLineSearchState();
}

void LBFGS::SetInvHessDiagonal(const Array1D& diag)
{
	ValidateSize(diag, "SetInvHessDiagonal");
	for (int i = 0; i < size_; ++i) {
		if (!(diag[i] > 0.0) || !std::isfinite(diag[i]))
			ERROR("LBFGS::SetInvHessDiagonal(): diagonal must be finite and positive.");
		inv_hess_diag_[i] = diag[i];
	}
	ClearHistory();
	has_start_ = false;
	next_initial_step_ = configured_initial_step_;
	ResetLineSearchState();
}

void LBFGS::MaskCoordinates(const std::vector<int>& indices)
{
	std::vector<char> new_mask(size_, 0);
	for (int idx : indices) {
		if (idx < 0 || idx >= size_)
			ERROR("LBFGS::MaskCoordinates(): index out of range.");
		new_mask[idx] = 1;
	}
	if (new_mask == coordinate_mask_)
		return;
	coordinate_mask_.swap(new_mask);
	ApplyMask(p_);
	ClearHistory();
	has_start_ = false;
	next_initial_step_ = configured_initial_step_;
	ResetLineSearchState();
}

void LBFGS::SetMaxHistory(int max_history)
{
	if (max_history <= 0)
		ERROR("LBFGS::SetMaxHistory(): history size must be positive.");

	max_history_ = max_history;
	if ((int)two_loop_alpha_.size() < max_history_)
		two_loop_alpha_.resize(max_history_);
	while ((int)history_.size() > max_history_)
		history_.erase(history_.begin());
}

void LBFGS::SetInitialStep(double initial_step)
{
	if (!(initial_step > 0.0) || !std::isfinite(initial_step))
		ERROR("LBFGS::SetInitialStep(): initial step must be finite and positive.");
	configured_initial_step_ = ClampDouble(initial_step, min_initial_step, max_initial_step);
	next_initial_step_ = configured_initial_step_;
}

void LBFGS::ResetStatistics()
{
	last_update_skipped_ = false;
	last_update_damped_ = false;
	last_trial_rejected_nonfinite_ = false;
	last_accepted_step_ = 0.0;
	last_trial_step_ = 0.0;
	last_curvature_ = 0.0;
	last_relative_curvature_ = 0.0;
	last_powell_theta_ = 1.0;
	last_base_sBs_ = 0.0;
	last_base_yHy_ = 0.0;
	last_trial_backtracks_ = 0;
	total_backtracks_ = 0;
	accepted_updates_ = 0;
	skipped_updates_ = 0;
	damped_updates_ = 0;
	restart_count_ = 0;
	non_descent_resets_ = 0;
}

const Array1D& LBFGS::IterateArmijoWithGradient(double f, const Array1D& g)
{
	ValidateSize(g, "IterateArmijoWithGradient");
	if (!std::isfinite(f))
		ERROR("LBFGS Armijo received a non-finite function value at an accepted point.");
	ValidateFinite(g, "LBFGS accepted gradient");

	if (armijo_value_only_trial_ready_) {
		if (!ArmijoValueOnlyAccepts(f))
			ERROR("LBFGS accepted-point gradient was supplied for a point that fails Armijo.");
		AdaptNextInitialStepAfterAccept();
		const int previous_accepted_updates = accepted_updates_;
		UpdateHistoryFromAcceptedGradient(g);
		if (accepted_updates_ > previous_accepted_updates)
			next_initial_step_ = configured_initial_step_;
	}

	iter_step = 0;
	return StartArmijoTrial(f, g) ? x_ : x_;
}

bool LBFGS::ArmijoValueOnlyTrialReady() const
{
	return armijo_value_only_trial_ready_;
}

bool LBFGS::ArmijoValueOnlyAccepts(double f) const
{
	if (!armijo_value_only_trial_ready_)
		return false;
	if (!std::isfinite(f))
		return false;
	return f <= f_start_ + armijo_c1 * linesearch_.x() * p_dot_g_start_;
}

const Array1D& LBFGS::BacktrackArmijoValueOnly(double f)
{
	if (!armijo_value_only_trial_ready_)
		ERROR("LBFGS Armijo value-only backtrack requested without an active trial.");
	if (ArmijoValueOnlyAccepts(f))
		return x_;

	last_trial_rejected_nonfinite_ = !std::isfinite(f);
	is_in_linesearch_ = true;
	++iter_step;
	++last_trial_backtracks_;
	++total_backtracks_;
	if (!ReduceCurrentStep(0.5)) {
		armijo_value_only_trial_ready_ = false;
		is_in_linesearch_ = false;
		std::copy(x_start_.begin(), x_start_.end(), x_.begin());
		return x_;
	}
	return x_;
}

const Array1D& LBFGS::ReduceStep(double coeff)
{
	if (!armijo_value_only_trial_ready_)
		ERROR("LBFGS::ReduceStep(): no active line search.");
	if (!(coeff > 0.0) || !(coeff < 1.0) || !std::isfinite(coeff))
		ERROR("LBFGS::ReduceStep(): reduction coefficient must be finite and in (0, 1).");

	is_in_linesearch_ = true;
	++iter_step;
	++last_trial_backtracks_;
	++total_backtracks_;
	if (!ReduceCurrentStep(coeff)) {
		armijo_value_only_trial_ready_ = false;
		is_in_linesearch_ = false;
		std::copy(x_start_.begin(), x_start_.end(), x_.begin());
		ERROR("LBFGS Linesearch stagnated while reducing the trial step.");
	}
	return x_;
}
