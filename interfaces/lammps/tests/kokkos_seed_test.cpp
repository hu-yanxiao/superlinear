#include "sus2_sl_kokkos_seed.h"
#include <cmath>
#include "kokkos_execution_proof.h"
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{using E=Kokkos::DefaultExecutionSpace;const int arch=sus2_device_arch();int checks=0;
 for(int F:{1,7,40})for(int n:{0,1,17})for(int Q:{0,5}){
 Kokkos::View<double**,Kokkos::LayoutLeft,E> ga("ga",n,4),ref("ref",n,4);Kokkos::View<double**,Kokkos::LayoutRight,E> sa("sa",n,F);Kokkos::View<double*,E>w("w",F*Q);Kokkos::View<int*,E>d("dest",Q?3:0),off("off",Q?4:1),qs("q",Q);
 Kokkos::View<int*,E> types("types",n),error("error",1);Kokkos::deep_copy(types,0);Kokkos::deep_copy(error,0);
 auto hg=Kokkos::create_mirror_view(ga);auto hs=Kokkos::create_mirror_view(sa);auto hw=Kokkos::create_mirror_view(w);auto hd=Kokkos::create_mirror_view(d);auto ho=Kokkos::create_mirror_view(off);auto hq=Kokkos::create_mirror_view(qs);auto ht=Kokkos::create_mirror_view(types);Kokkos::deep_copy(ht,0);
 int nodes[5]={2,0,2,3,0};if(Q){hd(0)=0;hd(1)=2;hd(2)=3;ho(0)=0;ho(1)=2;ho(2)=4;ho(3)=5;int order[5]={1,4,0,2,3};for(int q=0;q<Q;++q)hq(q)=order[q];}
 for(int i=0;i<n;++i){for(int j=0;j<4;++j)hg(i,j)=.13*(j+1);for(int mu=0;mu<F;++mu)hs(i,mu)=std::sin(i*.3+mu*.7);}
 for(int mu=0;mu<F;++mu)for(int q=0;q<Q;++q)hw(mu*Q+q)=std::cos(mu*.2+q*.9);
 Kokkos::deep_copy(ga,hg);Kokkos::deep_copy(ref,hg);Kokkos::deep_copy(sa,hs);Kokkos::deep_copy(w,hw);Kokkos::deep_copy(d,hd);Kokkos::deep_copy(off,ho);Kokkos::deep_copy(qs,hq);
 Kokkos::parallel_for("old_seed_reference",Kokkos::RangePolicy<E>(0,n),KOKKOS_LAMBDA(int i){for(int q=0;q<Q;++q){double z=0;for(int mu=0;mu<F;++mu)z+=sa(i,mu)*w(mu*Q+q);int node=q==0||q==2?2:q==1||q==4?0:3;ref(i,node)+=z;}});
 LAMMPS_NS::SUS2SL::effect_seed_rows<E>(n,F,Q,ga,sa,w,d,off,qs,types,error,2);Kokkos::fence();Kokkos::deep_copy(hg,ga);auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),ref);for(int i=0;i<n;++i)for(int j=0;j<4;++j)if(!std::isfinite(hg(i,j))||hg(i,j)!=hr(i,j))throw std::runtime_error("seed mismatch");++checks;
 if(n && Q){
  for(int bad:{-1,2}){ht(0)=bad;Kokkos::deep_copy(types,ht);Kokkos::deep_copy(error,0);double before[4];for(int j=0;j<4;++j)before[j]=hg(0,j);
   LAMMPS_NS::SUS2SL::effect_seed_rows<E>(n,F,Q,ga,sa,w,d,off,qs,types,error,2);Kokkos::fence();Kokkos::deep_copy(hg,ga);auto he=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),error);
   if(he(0)!=3)throw std::runtime_error("invalid type not reported");for(int j=0;j<4;++j)if(hg(0,j)!=before[j])throw std::runtime_error("invalid atom modified");++checks;
  }
 }

 }std::cout<<"PASS execution_space="<<E::name()<<" device_arch="<<arch<<" backend="<<(arch?"CUDA":"HOST")<<" seed checks="<<checks<<" max_abs=0\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";code=1;}Kokkos::finalize();return code;}
