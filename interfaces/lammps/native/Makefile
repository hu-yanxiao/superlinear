# Isolated native inference library. Never copy native .cpp/.h into LAMMPS src.
SUS2_ROOT ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
BUILD_DIR ?= $(SUS2_ROOT)/.codex_tmp/lammps-superlinear-native
CXX ?= c++
AR ?= ar
CXXFLAGS ?= -O3 -fPIC -std=c++17
CPPFLAGS += -DMLIP_DEV -I$(SUS2_ROOT)/cblas

SOURCES = src/common/utils.cpp src/basic_potentials.cpp src/basic_mlip.cpp \
  src/configuration.cpp src/neighborhoods.cpp src/radial_basis.cpp \
  src/zbl.cpp src/zbl_pair.cpp dev_src/mtpr.cpp dev_src/mtpr_sh.cpp \
  dev_src/mtpr_adaptive_radial.cpp dev_src/mtpr_adaptive_initialization.cpp \
  dev_src/mtpr_pure_w.cpp dev_src/pure_w_execution.cpp \
  dev_src/pure_w_basis_kernels.cpp dev_src/sh_model_init.cpp \
  dev_src/matvec_oper.cpp dev_src/mtpr_distributed.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/obj/,$(SOURCES:.cpp=.o))

.PHONY: all
all: $(BUILD_DIR)/lib/libsus2_superlinear_native.a

$(BUILD_DIR)/lib/libsus2_superlinear_native.a: $(OBJECTS)
	mkdir -p $(@D)
	$(AR) rcs $@ $(OBJECTS)

$(BUILD_DIR)/obj/%.o: $(SUS2_ROOT)/%.cpp $(BUILD_DIR)/native-build-flags
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(OBJECTS:.o=.d)
