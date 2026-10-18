cmake_minimum_required(VERSION 3.18)
project(PySuperlinear LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(BLAS REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(DEFAULT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SUPERLINEAR_SOURCE_DIR "${DEFAULT_SOURCE}" CACHE PATH "Matching superlinear C++ source checkout")
set(SUPERLINEAR_SOURCE_REVISION "unknown" CACHE STRING "Audited source revision")
set(S "${SUPERLINEAR_SOURCE_DIR}")
if(SUPERLINEAR_SOURCE_REVISION STREQUAL "unknown")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${S}" rev-parse HEAD
      OUTPUT_VARIABLE DETECTED_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE REVISION_STATUS)
    if(REVISION_STATUS EQUAL 0)
      set(SUPERLINEAR_SOURCE_REVISION "${DETECTED_REVISION}")
    endif()
  endif()
endif()
if(NOT EXISTS "${S}/dev_src/mtpr_adaptive_radial.cpp")
  message(FATAL_ERROR "SUPERLINEAR_SOURCE_DIR must contain the adaptive superlinear sources")
endif()
pybind11_add_module(_core MODULE NO_EXTRAS
  bindings/core.cpp
  "${S}/src/common/utils.cpp" "${S}/src/common/stdafx.cpp"
  "${S}/src/basic_potentials.cpp" "${S}/src/basic_mlip.cpp"
  "${S}/src/configuration.cpp" "${S}/src/neighborhoods.cpp"
  "${S}/src/radial_basis.cpp" "${S}/src/zbl.cpp" "${S}/src/zbl_pair.cpp"
  "${S}/dev_src/matvec_oper.cpp" "${S}/dev_src/mtpr.cpp"
  "${S}/dev_src/mtpr_sh.cpp" "${S}/dev_src/mtpr_pure_w.cpp"
  "${S}/dev_src/pure_w_execution.cpp" "${S}/dev_src/pure_w_basis_kernels.cpp"
  "${S}/dev_src/mtpr_adaptive_radial.cpp"
  "${S}/dev_src/mtpr_adaptive_initialization.cpp"
  "${S}/dev_src/sh_model_init.cpp")
target_include_directories(_core PRIVATE "${S}" "${S}/src" "${S}/src/common" "${S}/dev_src" "${S}/cblas")
target_link_libraries(_core PRIVATE ${BLAS_LIBRARIES})
target_compile_definitions(_core PRIVATE MLIP_DEV _MLP_NOLOG
  SUPERLINEAR_SOURCE_REVISION="${SUPERLINEAR_SOURCE_REVISION}")
if(NOT MSVC)
  target_compile_options(_core PRIVATE -Wno-unused-parameter -Wno-sign-compare)
  # Never inherit approximate floating-point or interprocedural compilation
  # from the training executable's machine-specific build configuration.
  set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()
install(TARGETS _core LIBRARY DESTINATION pysuperlinear)
