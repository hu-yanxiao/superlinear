#pragma once
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>
inline int sus2_device_arch() {
  Kokkos::View<int*> proof("execution proof",1);
  Kokkos::parallel_for("execution proof",1,KOKKOS_LAMBDA(int){
#ifdef __CUDA_ARCH__
    proof(0)=__CUDA_ARCH__;
#else
    proof(0)=0;
#endif
  });
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),proof);
  const int arch=host(0);
  const char* require=std::getenv("SUS2_RADIAL_REQUIRE_CUDA");
  if(require&&std::string(require)!="0"&&arch==0)
    throw std::runtime_error("SUS2_RADIAL_REQUIRE_CUDA requires device __CUDA_ARCH__ proof");
  return arch;
}
