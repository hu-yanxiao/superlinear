#ifndef LMP_SUS2_SL_KOKKOS_SEED_H
#define LMP_SUS2_SL_KOKKOS_SEED_H
#include <Kokkos_Core.hpp>
namespace LAMMPS_NS { namespace SUS2SL {
template<class Exec,class AdjointView,class SignalView,class Weights,class Indices>
void effect_seed_rows(int owned,int F,int Q,const AdjointView& ga,const SignalView& sa,const Weights& weights,const Indices& destination,const Indices& offsets,const Indices& qs,const Indices& types,const Indices& error,int species_count){
 const int rows=static_cast<int>(destination.size());if(!owned||!rows)return;
 Kokkos::parallel_for("sus2_inner_adjoint_seed_rows",Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>(0,static_cast<std::size_t>(owned)*rows),KOKKOS_LAMBDA(std::size_t index){
  const int i=index%owned,row=index/owned;
  if(types(i)<0 || types(i)>=species_count){Kokkos::atomic_exchange(&error(0),3);return;}
  const int node=destination(row);double value=ga(i,node);
  for(int at=offsets(row);at<offsets(row+1);++at){const int q=qs(at);double z=0.;for(int mu=0;mu<F;++mu)z+=sa(i,mu)*weights(mu*Q+q);value+=z;}ga(i,node)=value;
 });
}
} }
#endif
