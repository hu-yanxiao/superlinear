/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_LBFGS_H
#define MLIP_LBFGS_H

#include <vector>
#include <cstddef>
#include "bfgs.h"

//! Limited-memory BFGS with value-only Armijo line search.
//!
//! The line-search protocol is intentionally compatible with the BFGS Armijo
//! path used by the trainer: trial objective values can be checked without a
//! gradient, and gradients are supplied only for accepted points.
class LBFGS
{
private:
	struct HistoryEntry {
		Array1D s;
		Array1D y;
		double rho = 0.0;
		double s_dot_y = 0.0;
		double relative_curvature = 0.0;
	};

	int size_ = 0;
	int max_history_ = 20;
	Array1D x_;
	Array1D x_start_;
	Array1D g_start_;
	Array1D p_;
	Array1D inv_hess_diag_;
	Array1D q_;
	Array1D r_;
	Array1D damped_y_;
	Array1D b0s_;
	Array1D two_loop_alpha_;
	std::vector<char> coordinate_mask_;
	std::vector<HistoryEntry> history_;
	Linesearch linesearch_;

	bool is_in_linesearch_ = false;
	bool armijo_value_only_trial_ready_ = false;
	bool has_start_ = false;
	bool linesearch_stagnated_ = false;
	bool last_update_skipped_ = false;
	bool last_update_damped_ = false;
	bool last_trial_rejected_nonfinite_ = false;

	double f_start_ = HUGE_DOUBLE;
	double p_dot_g_ = HUGE_DOUBLE;
	double p_dot_g_start_ = HUGE_DOUBLE;
	double gamma_ = 1.0;
	double configured_initial_step_ = 1.0;
	double next_initial_step_ = 1.0;
	double last_accepted_step_ = 0.0;
	double last_trial_step_ = 0.0;
	double last_curvature_ = 0.0;
	double last_relative_curvature_ = 0.0;
	double last_powell_theta_ = 1.0;
	double last_base_sBs_ = 0.0;
	double last_base_yHy_ = 0.0;

	int last_trial_backtracks_ = 0;
	int total_backtracks_ = 0;
	int accepted_updates_ = 0;
	int skipped_updates_ = 0;
	int damped_updates_ = 0;
	int restart_count_ = 0;
	int non_descent_resets_ = 0;

	void Resize(int _size);
	void ClearHistory();
	void ResetLineSearchState();
	void ValidateSize(const Array1D& values, const char* name) const;
	void ValidateFinite(const Array1D& values, const char* name) const;
	void ApplyMask(Array1D& values) const;
	double Dot(const Array1D& lhs, const Array1D& rhs) const;
	double BaseHessianQuadratic(const Array1D& values) const;
	double BaseInverseQuadratic(const Array1D& values) const;
	void UpdateHistoryFromAcceptedGradient(const Array1D& g);
	bool FormDirection(const Array1D& g);
	bool StartArmijoTrial(double f, const Array1D& g);
	void BuildTrialPoint();
	void AdaptNextInitialStepAfterAccept();
	bool ReduceCurrentStep(double ratio);

public:
	double armijo_c1 = 1.0e-4;
	double relative_curvature_tol = 1.0e-10;
	double powell_damping_curvature = 0.2;
	double min_initial_step = 1.0e-20;
	double max_initial_step = 1.0e20;
	double initial_step_growth = 1.5;
	int iter_step = 0;

	explicit LBFGS(int max_history = 20);

	void Set_x(const double* x, int _size);
	void Set_x(Array1D x) { Set_x(x.data(), (int)x.size()); }
	const double* Data() const { return x_.data(); }
	double x(int i) const { return x_[i]; }
	void Restart();
	void SetInvHessDiagonal(const Array1D& diag);
	void MaskCoordinates(const std::vector<int>& indices);
	void SetMaxHistory(int max_history);
	void SetInitialStep(double initial_step);
	void ResetStatistics();

	const Array1D& IterateArmijoWithGradient(double f, const Array1D& g);
	bool ArmijoValueOnlyTrialReady() const;
	bool ArmijoValueOnlyAccepts(double f) const;
	const Array1D& BacktrackArmijoValueOnly(double f);
	const Array1D& ReduceStep(double coeff = 0.25);

	bool is_in_linesearch() const { return is_in_linesearch_; }
	bool linesearch_stagnated() const { return linesearch_stagnated_ || linesearch_.stagnated(); }

	int size() const { return size_; }
	int history_size() const { return (int)history_.size(); }
	int max_history() const { return max_history_; }
	double gamma() const { return gamma_; }
	double last_step() const { return last_trial_step_; }
	double last_accepted_step() const { return last_accepted_step_; }
	double next_initial_step() const { return next_initial_step_; }
	double last_directional_derivative() const { return p_dot_g_start_; }
	double last_curvature() const { return last_curvature_; }
	double last_relative_curvature() const { return last_relative_curvature_; }
	double last_powell_theta() const { return last_powell_theta_; }
	double last_base_sBs() const { return last_base_sBs_; }
	double last_base_yHy() const { return last_base_yHy_; }
	int last_trial_backtracks() const { return last_trial_backtracks_; }
	int total_backtracks() const { return total_backtracks_; }
	int accepted_update_count() const { return accepted_updates_; }
	int skipped_update_count() const { return skipped_updates_; }
	int damped_update_count() const { return damped_updates_; }
	int restart_count() const { return restart_count_; }
	int non_descent_reset_count() const { return non_descent_resets_; }
	bool last_update_skipped() const { return last_update_skipped_; }
	bool last_update_damped() const { return last_update_damped_; }
	bool last_trial_rejected_nonfinite() const { return last_trial_rejected_nonfinite_; }
};

#endif // MLIP_LBFGS_H
