# Directories
PREFIX = .
BIN_DIR = $(PREFIX)/bin
LIB_DIR = $(PREFIX)/lib
FORTRAN_DIR = /usr/local/gfortran/lib

# Compilers for the executable
CC_EXE  = mpiicc
CXX_EXE = mpiicpc
FC_EXE  = mpiifort

# Compilers for the library
CC_LIB  = icc
CXX_LIB = icpc
FC_LIB  = ifort

# Compile and link flags
CPPFLAGS += -O3 
FFLAGS += -O3 
CXXFLAGS += -DMLIP_MPI
LDFLAGS += -L/share/intel/oneapi/mkl/2022.2.0/lib/intel64 -lmkl_rt -lifcore
CPPFLAGS += -I/share/intel/oneapi/mkl/2022.2.0/include
CXXFLAGS += -DMLIP_INTEL_MKL 
CXXFLAGS += -O3 -ipo -no-prec-div -fp-model fast=2 -qoverride-limits -qopt-mem-layout-trans=3  -qopt-streaming-stores=auto -fimf-use-svml=true -std=c++11 -qopt-report=5


# Extra variables
