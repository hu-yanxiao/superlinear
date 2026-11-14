// Stateful synthetic-neighborhood regression; see embedded_angular_cache_check.md.
#include "sus2_sl_kernel.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace LAMMPS_NS::SUS2SL;
static double maximum=0;static int checks=0;
static void equal(double a,double b){double d=std::abs(a-b);if(!std::isfinite(a)||!std::isfinite(b)||d>1e-9)throw std::runtime_error("EFS mismatch");maximum=std::max(maximum,d);}
static void compare(const Result&a,const Result&b){equal(a.energy,b.energy);if(a.force.size()!=b.force.size()||a.site_energy.size()!=b.site_energy.size())throw std::runtime_error("layout mismatch");for(size_t i=0;i<a.force.size();++i)for(int q=0;q<3;++q)equal(a.force[i][q],b.force[i][q]);for(size_t i=0;i<a.site_energy.size();++i)equal(a.site_energy[i],b.site_energy[i]);for(int q=0;q<6;++q)equal(a.virial[q],b.virial[q]);++checks;}
static Result run(Kernel&k,int state,size_t cap){
 int count=state==2?0:state==1?3:int(cap+17),owned=count?1:0;
 std::vector<int> types(count?count+1:0);for(size_t j=0;j<types.size();++j)types[j]=(int(j)+state)%k.species_count();
 std::vector<Edge> edges;const auto&m=k.model();
 for(int j=1;j<=count;++j){
  // Unit directions are noncollinear; shell probes include exact boundaries.
  double x=.31+.01*(j%7),y=.43+.01*(j%5),z=.57+.01*(j%3);double norm=std::sqrt(x*x+y*y+z*z);
  double shell[]={m.angular_max_dist-1e-8,m.angular_max_dist,m.angular_max_dist+1e-8,m.max_dist-1e-8,m.max_dist,m.max_dist+1e-8};
  double r=shell[(j+state)%6];
  edges.push_back({0,j,{{r*x/norm,r*y/norm,r*z/norm}}});
 }
 if(state==3)std::reverse(edges.begin(),edges.end());
 k.begin(owned,types.size(),types,edges);
 // Explicit deterministic received ghost gates: no MPI, no missing ghost data.
 std::vector<double> signal(k.communication_width());for(int j=1;j<int(types.size());++j){for(size_t q=0;q<signal.size();++q)signal[q]=.0001*std::sin(.7*q+.2*j+state);k.unpack_forward(j,signal.data());}
 k.outer();return k.finish();
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: embedded-angular-cache-check EFFECT_MODEL");Model m;m.load(argv[1]);if(!m.boolean("two_layer_gate_enabled")||m.text("outer_radial_mode","legacy-plus-mixer")!="legacy-plus-mixer")throw std::runtime_error("requires Effect model");
 const size_t H=(m.lmax+1)*(m.lmax+1),cap=1024*1024/(4*H*sizeof(double)+1);
 for(int mode=0;mode<2;++mode)for(int radial=0;radial<2;++radial)for(int source=0;source<2;++source)for(int stream=0;stream<2;++stream)for(int budget=0;budget<2;++budget){
 setenv("SUS2_SL_STEP_RADIAL_CACHE",radial?"1":"0",1);setenv("SUS2_SL_EFFECT_SOURCE_CACHE",source?"1":"0",1);setenv("SUS2_SL_EFFECT_FORWARD_STREAM",stream?"1":"0",1);setenv("SUS2_SL_STEP_ANGULAR_CACHE_MB",budget?"1":"0",1);setenv("SUS2_SL_STEP_ANGULAR_CACHE","1",1);
 Kernel reused(m);reused.configure_radial(mode?RadialTableMode::Table:RadialTableMode::Direct);
 for(int state=0;state<4;++state){Result actual=run(reused,state,cap);setenv("SUS2_SL_STEP_ANGULAR_CACHE","0",1);Kernel fresh(m);fresh.configure_radial(mode?RadialTableMode::Table:RadialTableMode::Direct);compare(actual,run(fresh,state,cap));setenv("SUS2_SL_STEP_ANGULAR_CACHE","1",1);}
 }
 std::cout<<"PASS checks="<<checks<<" max_abs="<<maximum<<" prefix_edges="<<cap<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
