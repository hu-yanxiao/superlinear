#include <cmath>
#include <cstdlib>
#include <iostream>

#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include "src/common/bfgs.h"

namespace {

void Require(bool condition, const char* message)
{
	if (!condition) {
		std::cerr << message << '\n';
		std::exit(1);
	}
}

class RestartProbe : public BFGS {
public:
	void PoisonTransientState()
	{
		p[0] = -1.0e101;
		x_start[0] = 1.0e50;
		g_start[0] = -1.0e75;
		p_dot_g = -1.0e101;
		p_dot_g_start = -1.0e101;
		f_start = 1.0;
	}
};

}  // namespace

int main(int argc, char** argv)
{
	int rank = 0;
	int process_count = 1;
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &process_count);
#else
	(void)argc;
	(void)argv;
#endif

	Array1D base(1);
	base[0] = 2.0;
	Array1D gradient(1);
	gradient[0] = 3.0;

	RestartProbe bfgs;
#ifdef MLIP_MPI
	bfgs.UseDistributedDense(rank, process_count, MPI_COMM_WORLD);
#endif
	bfgs.Set_x(base);
	bfgs.PoisonTransientState();
	bfgs.Restart();
	bfgs.Set_x(base);
	bfgs.Iterate(4.5, gradient);

	const bool local_success =
		bfgs.is_in_linesearch()
		&& std::abs(bfgs.x(0) + 1.0) <= 1.0e-12;

#ifdef MLIP_MPI
	int success = local_success ? 1 : 0;
	int global_success = 0;
	MPI_Allreduce(
		&success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	if (rank == 0) {
		if (global_success != 0)
			std::cout << "bfgs_restart_state_check passed\n";
		else
			std::cerr
				<< "BFGS restart reused a stale coordinate-chart direction\n";
	}
	MPI_Finalize();
	return global_success != 0 ? 0 : 1;
#else
	Require(
		bfgs.is_in_linesearch(),
		"fresh BFGS base did not start a new line search");
	Require(
		std::abs(bfgs.x(0) + 1.0) <= 1.0e-12,
		"BFGS restart reused a stale coordinate-chart direction");

	std::cout << "bfgs_restart_state_check passed\n";
	return 0;
#endif
}
