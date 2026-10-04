/* Test-only MPI interposer: delay one rank after MPI initialization. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int MPI_Init(int *argc, char ***argv)
{
    int result = PMPI_Init(argc, argv);
    int rank = -1;
    const char *delay_rank = getenv("GROUPED_TEST_DELAY_RANK");
    const char *delay_ms = getenv("GROUPED_TEST_DELAY_MS");
    if (result != MPI_SUCCESS || !delay_rank || !delay_ms)
        return result;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == atoi(delay_rank)) {
        fprintf(stderr, "test-only MPI startup delay: rank=%d ms=%s\n", rank, delay_ms);
        fflush(stderr);
        usleep((useconds_t)strtoul(delay_ms, NULL, 10) * 1000U);
    }
    return result;
}
