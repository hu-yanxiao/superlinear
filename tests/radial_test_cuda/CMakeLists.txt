cmake_minimum_required(VERSION 3.20)
project(sus2_radial_device_qualification LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SUS2_RADIAL_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "Independent qualification source snapshot")
set(SUS2_KOKKOS_PACKAGE "" CACHE PATH "Existing canonical LAMMPS build/lib/kokkos directory")
option(SUS2_RADIAL_REQUIRE_CUDA "Reject a non-CUDA Kokkos package" ON)
if(NOT SUS2_KOKKOS_PACKAGE)
  message(FATAL_ERROR "Pass -DSUS2_KOKKOS_PACKAGE=<existing build>/lib/kokkos")
endif()
find_package(Kokkos 4.7.1 EXACT CONFIG REQUIRED PATHS "${SUS2_KOKKOS_PACKAGE}" NO_DEFAULT_PATH)
if(SUS2_RADIAL_REQUIRE_CUDA AND NOT "CUDA" IN_LIST Kokkos_DEVICES)
  message(FATAL_ERROR "Existing Kokkos package has no CUDA backend: ${Kokkos_DEVICES}")
endif()
set(cpu "${SUS2_RADIAL_SOURCE}/interfaces/lammps/ML-SUS2")
add_executable(radial_test_device
  "${SUS2_RADIAL_SOURCE}/tests/radial_test_device.cpp"
  "${cpu}/sus2_sl_model.cpp" "${cpu}/sus2_sl_radial.cpp" "${cpu}/sus2_sl_radial_math.cpp")
target_include_directories(radial_test_device PRIVATE "${cpu}" "${SUS2_RADIAL_SOURCE}/interfaces/lammps/ML-SUS2-KK")
target_link_libraries(radial_test_device PRIVATE Kokkos::kokkos)
add_executable(kokkos_table_lookup_test
  "${SUS2_RADIAL_SOURCE}/interfaces/lammps/tests/kokkos_table_lookup_test.cpp"
  "${cpu}/sus2_sl_model.cpp" "${cpu}/sus2_sl_radial.cpp" "${cpu}/sus2_sl_radial_math.cpp")
target_include_directories(kokkos_table_lookup_test PRIVATE "${cpu}" "${SUS2_RADIAL_SOURCE}/interfaces/lammps/ML-SUS2-KK")
target_link_libraries(kokkos_table_lookup_test PRIVATE Kokkos::kokkos)
add_executable(kokkos_kernel_parity
  "${SUS2_RADIAL_SOURCE}/interfaces/lammps/tests/kokkos_kernel_parity.cpp"
  "${cpu}/sus2_sl_model.cpp" "${cpu}/sus2_sl_radial.cpp" "${cpu}/sus2_sl_radial_math.cpp"
  "${cpu}/sus2_sl_kernel.cpp" "${cpu}/sus2_sl_adaptive.cpp" "${cpu}/sus2_sl_joint.cpp")
target_include_directories(kokkos_kernel_parity PRIVATE "${cpu}" "${SUS2_RADIAL_SOURCE}/interfaces/lammps/ML-SUS2-KK")
target_link_libraries(kokkos_kernel_parity PRIVATE Kokkos::kokkos)
message(STATUS "Qualification reuses Kokkos ${Kokkos_VERSION} devices=${Kokkos_DEVICES} package=${Kokkos_DIR}")
