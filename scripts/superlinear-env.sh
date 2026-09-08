# Source this file on phy-weigw before using the installed superlinear programs.
# No module purge: preserve the caller's other, compatible runtime modules.
module load compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0 gcc/11.2.0
export PATH=/work/phy-weigw/app/superlinear:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
