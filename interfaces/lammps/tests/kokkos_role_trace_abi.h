// Minimal C ABI subset from Kokkos 4.7.1 profiling interface (20240906).
// Full upstream source/hash provenance is in kokkos_role_trace_abi.md.
#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
struct Kokkos_Profiling_KokkosPDeviceInfo { std::size_t deviceID; };
struct Kokkos_Tools_ToolSettings { bool requires_global_fencing; bool padding[255]; };
}
static_assert(sizeof(Kokkos_Tools_ToolSettings)==256*sizeof(bool),"Kokkos 4.7.1 ABI");
