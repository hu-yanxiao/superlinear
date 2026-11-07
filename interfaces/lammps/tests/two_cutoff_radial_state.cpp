// Portable state regression: compile with sus2_sl_{model,radial,radial_math}.cpp.
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace LAMMPS_NS::SUS2SL;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::unique_ptr<Radial> make(const Model&m){return std::unique_ptr<Radial>(new Radial(m.text("radial_basis_type"),m.min_dist,m.max_dist,m.radial_size,m.scaling,m.text("radial_cutoff_type","legacy"),m.integer("radial_cutoff_p",5),m.angular_max_dist));}
void close(double actual,double expected,double tol){require(std::isfinite(actual)&&std::abs(actual-expected)<=tol*(1+std::abs(expected)),"direct/table tolerance exceeded");}
void check(const Model&m,Radial&table,Radial&direct,double r,bool gate,RadialInterpolation&context){
 std::vector<double> values,ders,rb,dr;
 require(table.eval_frozen(r,1,gate,values,ders,&context),"missing frozen request");
 const int C=m.species_count,R=m.radial_size,L=m.lmax+1,ti=0,tj=1;
 const auto scale=m.numbers(gate&&!m.numbers("two_layer_gate_scaling_coeffs").empty()?"two_layer_gate_scaling_coeffs":"scal_coeffs");
 const auto gc=m.numbers("two_layer_gate_radial_coeffs");const bool separate=gate&&!gc.empty();
 const auto&coeff=separate?gc:m.radial_coeffs;const auto map=m.text("scaling_map","LK");
 for(int mu=0;mu<m.radial_funcs;++mu){const int l=mu%L,k=mu/L;
  const int sb=map=="K"?k:map=="L"?l:map=="pair"?0:mu,sc=2*sb*C*C+ti*C+tj;
  direct.eval(r,scale.at(sc),scale.at(sc+C*C),k,rb,dr,true,l>0);
  double v=0,d=0;for(int n=0;n<R;++n){const double c=coeff.at(mu*(separate?R:R+C)+n);v+=c*rb[n];d+=c*dr[n];}
  // Same acceptance bounds used by frozen table qualification, never widened.
  close(values.at(mu),v,1e-7);close(ders.at(mu),d,1e-4);
  if(l>0&&r>=m.angular_max_dist)require(values.at(mu)==0&&ders.at(mu)==0,"far angular lane not zero");
 }
}
void sequence(const Model&m){
 auto table=make(m),direct=make(m);direct->configure(RadialTableMode::Direct);
 table->configure(RadialTableMode::Table,.004);table->freeze(m);
 RadialInterpolation near,far;
 for(int gate=0;gate<2;++gate){check(m,*table,*direct,3.,gate,near);check(m,*table,*direct,5.7,gate,far);check(m,*table,*direct,3.,gate,near);}
 const auto first=table->stats();require(first.hits>0&&first.tables>0,"no table hits");
 for(const auto spacing:{.006,.003})for(const std::size_t budget:{std::size_t(0),std::size_t(1000000)}){
  table->configure(RadialTableMode::Table,spacing,budget);
  require(table->stats().bytes==0,"configure retained knot storage");
  near=RadialInterpolation();far=RadialInterpolation();
  check(m,*table,*direct,3.,false,near);check(m,*table,*direct,5.7,true,far);check(m,*table,*direct,3.,true,near);
  require(table->stats().bytes<=budget,"table budget exceeded");
  if(!budget)require(table->stats().memory_fallbacks>0,"zero budget lacked memory fallback");
 }
}
void partial(const Model&m){
 const double spacing=.004,lo=std::max(0.,m.min_dist);
 const auto bytes=[&](double cutoff){return (std::size_t(std::ceil((cutoff-lo)/spacing))+1)*2*sizeof(double);};
 auto direct=make(m);direct->configure(RadialTableMode::Direct);
 for(int shell=0;shell<2;++shell){auto table=make(m);const auto budget=bytes(shell?m.max_dist:m.angular_max_dist);
  table->configure(RadialTableMode::Table,spacing,budget);table->freeze(m);RadialInterpolation c;
  check(m,*table,*direct,3.,false,c);auto stats=table->stats();
  require(stats.hits>0&&stats.memory_fallbacks>0,"partial capacity did not mix hits and fallback");
  require(stats.bytes==budget&&stats.tables==1,"partial budget failed to select exactly one cutoff table");
  std::cout<<"partial shell="<<shell<<" bytes="<<stats.bytes<<" budget="<<budget<<" hits="<<stats.hits<<" memory_fallbacks="<<stats.memory_fallbacks<<'\n';
 }
}
void empty_domain(Model m){m.min_dist=5.2;auto table=make(m),direct=make(m);direct->configure(RadialTableMode::Direct);table->configure(RadialTableMode::Table,.004);table->freeze(m);
 RadialInterpolation below,above;check(m,*table,*direct,4.7,false,below);check(m,*table,*direct,5.7,true,above);
 require(table->stats().domain_fallbacks>0,"empty angular table domain did not fall back");
}
}
int main(int argc,char**argv){try{require(argc==2,"usage: two_cutoff_radial_state model.mtp");Model m;m.load(argv[1]);require(m.species_count>=2&&m.angular_max_dist<m.max_dist,"dual cutoff fixture required");sequence(m);partial(m);empty_domain(m);std::cout<<"TWO_CUTOFF_RADIAL_STATE_PASS\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
