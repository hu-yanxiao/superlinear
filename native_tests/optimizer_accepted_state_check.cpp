#include <cmath>
#include <cstdlib>
#include <iostream>

#include "dev_src/optimizer_accepted_state.h"

namespace {

void Require(bool condition, const char* message)
{
	if (!condition) {
		std::cerr << message << '\n';
		std::exit(1);
	}
}

void RequirePoint(const double* actual, double x0, double x1, const char* message)
{
	Require(std::abs(actual[0] - x0) <= 1.0e-15, message);
	Require(std::abs(actual[1] - x1) <= 1.0e-15, message);
}

} // namespace

int main()
{
	Require(!sus2::AscentRecoveryPointWasAccepted(false, false, false),
	        "initial Wolfe base was misclassified as an accepted trial");
	Require(sus2::AscentRecoveryPointWasAccepted(true, false, false),
	        "accepted Wolfe trial was lost during ascent recovery");
	Require(!sus2::AscentRecoveryPointWasAccepted(true, true, false),
	        "unaccepted Armijo trial was promoted during ascent recovery");
	Require(sus2::AscentRecoveryPointWasAccepted(true, true, true),
	        "accepted Armijo trial was lost during ascent recovery");

	Require(!sus2::EvaluatedPointWasAccepted(false, false, false, true),
	        "Wolfe line-search trial was misclassified as accepted");
	Require(sus2::EvaluatedPointWasAccepted(false, false, false, false),
	        "Wolfe accepted point was not recognized");
	Require(sus2::EvaluatedPointWasAccepted(true, false, false, true),
	        "recovered Wolfe accepted point was hidden by the retry trial");
	Require(!sus2::EvaluatedPointWasAccepted(false, true, false, true),
	        "rejected Armijo trial was misclassified as accepted");
	Require(sus2::EvaluatedPointWasAccepted(false, true, true, true),
	        "accepted Armijo trial was not recognized");

	sus2::OptimizerAcceptedState state(2);
	double initial[2] = {1.0, -2.0};
	state.SetBoundaryBase(initial);
	Require(!state.HasPendingTrial(), "boundary base unexpectedly has a pending trial");

	double live[2] = {100.0, 200.0};
	state.Restore(live);
	RequirePoint(live, 1.0, -2.0, "boundary restore mismatch");

	// Simulate reverse-communication BFGS: evaluated is accepted while the
	// optimizer has already generated a distinct, unevaluated next trial.
	double evaluated[2] = {0.25, -0.5};
	double next_trial[2] = {1.0e6, -1.0e6};
	state.SetEvaluatedAccepted(evaluated);
	Require(state.HasPendingTrial(), "accepted point did not record pending trial");
	Require(state.RestoreAcceptedForBoundary(next_trial),
	        "pending trial was not detected at accepted boundary");
	RequirePoint(next_trial, 0.25, -0.5,
	             "unevaluated trial replaced evaluated accepted point");

	state.DiscardPendingTrial();
	Require(!state.HasPendingTrial(), "discard did not clear pending-trial state");
	live[0] = -9.0;
	live[1] = 8.0;
	state.Restore(live);
	RequirePoint(live, 0.25, -0.5, "discard changed accepted snapshot");

	std::cout << "optimizer_accepted_state_check passed\n";
	return 0;
}
