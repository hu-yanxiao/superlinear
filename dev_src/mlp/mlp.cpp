/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *   This file contributors: Ivan Novikov, Konstantin Gubaev
 */


#ifdef MLIP_MPI
#	include <mpi.h>
#endif
#include "../../src/common/stdafx.h"
#include "dev_self_test.h"
#include "../../src/mlp/self_test.h"
#include "../../src/mlp/mlp.h"
#include "../../src/mlp/train.h"
#include "../../src/mlp/calc_errors.h"
#include "../../src/mlp/mlp_commands.h"
#include "../../src/mlp/grouped_input.h"
#include <stdexcept>
#include "../../dev_src/mlp/dev_mlp_commands.h"

int ExecuteCommand(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	bool is_command_found = false;

	is_command_found = Commands(command, args, opts) || is_command_found;

	if(!is_command_found || command == "list" || (command == "help" && args.size()==0))
		is_command_found = DevCommands(command, args, opts) || is_command_found;

	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	if (!is_command_found)
		if (mpi_rank == 0) std::cout << "Error: command " << command << " does not exist." << std::endl;

	return 1;
}

int main(int argc, char *argv[])
{
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
#endif

	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
	if (mpi_rank == 0)
		PrintSUS2Banner(std::cout,
#ifdef MLIP_MPI
			true,
#else
			false,
#endif
			mpi_size);

	std::vector<std::string> args;
	std::map<std::string, std::string> opts;

	try {
		int iarg = ParseCommonOptions(argc, argv);
		if ( 0 > iarg || false == CheckCommonOptions() ) return 2;
		// parse argv to extract args and opts
		std::string command = "help";
		if ( argc > iarg ) command = argv[iarg];

		auto normalized = superlinear_input::NormalizeArguments(argc - iarg, argv + iarg);
		std::vector<char*> pointers;
		for (auto& token : normalized) pointers.push_back(&token[0]);
		ParseOptions(static_cast<int>(pointers.size()), pointers.data(), args, opts);
		std::string run_token;
		if (mpi_rank == 0) run_token = superlinear_input::GenerateRunToken();
#ifdef MLIP_MPI
		int token_size = static_cast<int>(run_token.size());
		MPI_Bcast(&token_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		run_token.resize(token_size);
		MPI_Bcast(&run_token[0], token_size, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
		const bool check_only = superlinear_input::Resolve(command, args, opts, mpi_rank == 0 ? &std::cout : nullptr, run_token);
#ifdef MLIP_MPI
		// All ranks must finish grouped output-reuse checks before rank 0 can
		// create the inline model. Otherwise a late rank mistakes the new
		// initial.sus2 for output left by an earlier run and aborts the world.
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if (!check_only)
			ExecuteCommand(command, args, opts);
		if (mpi_rank == 0 && command == "help" &&
		    (args.empty() || args[0] == "train" || args[0] == "init-sh"))
			std::cout << superlinear_input::Help();
	}
	catch (const MlipException& e) { std::cerr << e.What(); return 1; }
	catch (const std::exception& e) {
		if (mpi_rank != 0) std::cerr << "MPI rank " << mpi_rank << ": ";
		std::cerr << e.what() << std::endl;
#ifdef MLIP_MPI
		MPI_Abort(MPI_COMM_WORLD, 2);
#endif
		return 2;
	}

#ifdef MLIP_MPI
	MPI_Finalize();
#endif

	return 0;
}
