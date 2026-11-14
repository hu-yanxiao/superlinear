#include "sus2_sl_kokkos_tally.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 using E=Kokkos::Serial;Kokkos::View<double*,E> energy("e",513);Kokkos::View<double**,E> virial("v",513,6);
 int checks=0;double maximum=0;
 for(int pattern=0;pattern<3;++pattern){
 for(int i=0;i<513;++i){double v=pattern==0?.125*(i%11):pattern==1?(i%2?-.25:.25):((i%3)==0?1e12:(i%3)==1?-1e12:.5);energy(i)=v;for(int q=0;q<6;++q)virial(i,q)=v*(q+1);}
 for(int n:{0,1,513})for(int e=0;e<2;++e)for(int v=0;v<2;++v){
 // Poison unrequested arrays to expose unconditional reads/propagation.
 double saved_e[513],saved_v[513][6];for(int i=0;i<513;++i){saved_e[i]=energy(i);if(!e)energy(i)=std::numeric_limits<double>::quiet_NaN();for(int q=0;q<6;++q){saved_v[i][q]=virial(i,q);if(!v)virial(i,q)=std::numeric_limits<double>::quiet_NaN();}}
 auto actual=LAMMPS_NS::SUS2SL::global_tally<E>(n,energy,virial,e,v);double expected[7]={};
 if(e)Kokkos::parallel_reduce(Kokkos::RangePolicy<E>(0,n),KOKKOS_LAMBDA(int i,double&s){s+=energy(i);},expected[0]);
 if(v)for(int q=0;q<6;++q)Kokkos::parallel_reduce(Kokkos::RangePolicy<E>(0,n),KOKKOS_LAMBDA(int i,double&s){s+=virial(i,q);},expected[q+1]);
 for(int q=0;q<7;++q){double x=q?actual.virial[q-1]:actual.energy;double d=std::abs(x-expected[q]);if(!std::isfinite(x)||d>1e-9)throw std::runtime_error("tally mismatch");maximum=std::max(maximum,d);}++checks;
 for(int i=0;i<513;++i){energy(i)=saved_e[i];for(int q=0;q<6;++q)virial(i,q)=saved_v[i][q];}
 }}std::cout<<"PASS Serial checks="<<checks<<" max_abs="<<maximum<<" (not CUDA qualification)\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";code=1;}Kokkos::finalize();return code;}
