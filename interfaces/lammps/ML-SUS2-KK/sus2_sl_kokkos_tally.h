#ifndef LMP_SUS2_SL_KOKKOS_TALLY_H
#define LMP_SUS2_SL_KOKKOS_TALLY_H
#include <Kokkos_Core.hpp>
namespace LAMMPS_NS { namespace SUS2SL {
struct GlobalTally { double energy; double virial[6]; };
template<class EnergyView,class VirialView> struct GlobalTallyFunctor {
  using value_type=GlobalTally;
  EnergyView energy;VirialView virial;bool want_energy,want_virial;
  KOKKOS_INLINE_FUNCTION void init(value_type& v)const {v.energy=0.;for(int q=0;q<6;++q)v.virial[q]=0.;}
  KOKKOS_INLINE_FUNCTION void join(value_type& a,const value_type& b)const {a.energy+=b.energy;for(int q=0;q<6;++q)a.virial[q]+=b.virial[q];}
  KOKKOS_INLINE_FUNCTION void operator()(int i,value_type& v)const {if(want_energy)v.energy+=energy(i);if(want_virial)for(int q=0;q<6;++q)v.virial[q]+=virial(i,q);}
};
template<class ExecutionSpace,class EnergyView,class VirialView>
GlobalTally global_tally(int owned,const EnergyView& energy,const VirialView& virial,bool want_energy,bool want_virial){
 GlobalTally value{};
 if(want_energy && !want_virial && owned>0){
  Kokkos::parallel_reduce("sus2_energy",Kokkos::RangePolicy<ExecutionSpace>(0,owned),KOKKOS_LAMBDA(int i,double& sum){sum+=energy(i);},value.energy);
  return value;
 }
 if(want_virial&&owned>0)
  Kokkos::parallel_reduce("sus2_global_tally",Kokkos::RangePolicy<ExecutionSpace>(0,owned),GlobalTallyFunctor<EnergyView,VirialView>{energy,virial,want_energy,want_virial},value);
 return value;
}
} }
#endif
