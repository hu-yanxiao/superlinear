#include "sus2_sl_kokkos_kernel.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace LAMMPS_NS::SUS2SL;
using Engine=KokkosKernel<Kokkos::DefaultExecutionSpace>;

int main(int argc,char**argv) {
  if(argc!=2){std::cerr<<"Usage: kokkos-perf-switch-check MODEL\n";return 1;}
  Kokkos::initialize(argc,argv);int code=0;
  try {
    Model m;m.load(argv[1]);
    const char* names[]={"SUS2_SL_KOKKOS_GENERIC","SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE",
                         "SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE","SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE"};
    for(const char* name:names)unsetenv(name);
    {
      Engine engine(m);
      if(engine.uses_generic_reference()||engine.uses_forward_radial_tile()||engine.uses_reverse_center_reduce()||engine.uses_forward_role_specialize())
        throw std::runtime_error("Performance switches must default to disabled");
    }
    int combinations=0;
    for(int generic=0;generic<2;++generic)for(int forward=0;forward<2;++forward)for(int reverse=0;reverse<2;++reverse)for(int role=0;role<2;++role){
      setenv(names[0],generic?"1":"0",1);setenv(names[1],forward?"1":"0",1);setenv(names[2],reverse?"1":"0",1);
      setenv(names[3],role?"1":"0",1);
      Engine engine(m);
      if(engine.uses_generic_reference()!=bool(generic)||
         engine.uses_forward_radial_tile()!=bool(!generic&&forward)||
         engine.uses_reverse_center_reduce()!=bool(!generic&&reverse)||
         engine.uses_forward_role_specialize()!=bool(!generic&&role))
        throw std::runtime_error("Effective performance switch state is incorrect");
      ++combinations;
    }
    int invalid=0;
    for(const char* name:names){
      for(const char* reset:names)setenv(reset,"0",1);
      setenv(name,"invalid",1);bool rejected=false;
      try{Engine engine(m);}catch(const std::runtime_error& error){
        rejected=std::string(error.what())==std::string(name)+" must be 0 or 1";
      }
      if(!rejected)throw std::runtime_error("Malformed performance switch was accepted");
      ++invalid;
    }
    std::cout<<"KOKKOS_PERF_SWITCH_JSON {\"combinations\":"<<combinations
             <<",\"invalid_rejected\":"<<invalid<<",\"status\":\"PASS\"}\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';code=1;}
  Kokkos::finalize();return code;
}
