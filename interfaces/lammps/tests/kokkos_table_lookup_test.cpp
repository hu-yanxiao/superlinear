// Standalone native Kokkos test: the same source runs Serial and CUDA.
#include "sus2_sl_kokkos_radial.h"
#include "kokkos_execution_proof.h"
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace LAMMPS_NS::SUS2SL;
using Device=Kokkos::DefaultExecutionSpace::device_type;
using DR=DeviceRadial<Device>;
namespace {
std::size_t cases=0, comparisons=0;
std::size_t context_cases=0, packed_cases=0;
double max_error=0;
std::string list(const std::vector<double>& a) {
  std::ostringstream s;s<<std::setprecision(17)<<"{";
  for(std::size_t i=0;i<a.size();++i)s<<(i?",":"")<<a[i];return s.str()+"}";
}
Model fixture(const std::string& basis,bool adaptive=false,bool gate=false,
              const std::string& map="LK",bool dedup=false) {
  Model m;m.species_count=2;m.lmax=1;m.kmax=3;m.radial_size=12;m.radial_funcs=6;
  m.min_dist=.5;m.max_dist=5;m.scaling=.73;
  m.fields["radial_basis_type"]=basis;m.fields["radial_cutoff_type"]="polynomial";
  m.fields["radial_cutoff_p"]="5";m.fields["scaling_map"]=map;
  m.fields["outer_radial_mode"]=adaptive?"adaptive-coefficients":"legacy-plus-mixer";
  m.fields["outer_k_rank"]="3";
  const int blocks=map=="pair"?1:(map=="L"?2:(map=="K"?3:6));
  std::vector<double> sc(8*blocks);
  for(int mu=0;mu<blocks;++mu)for(int p=0;p<4;++p){
    sc[8*mu+p]=dedup?1.2:1.1+.05*mu+.1*p;
    sc[8*mu+4+p]=dedup?.2:.2+.01*p;
  }
  m.fields["scal_coeffs"]=list(sc);m.radial_coeffs.resize(6*14);
  for(int mu=0;mu<6;++mu)for(int n=0;n<12;++n)
    m.radial_coeffs[14*mu+n]=std::sin(double(n+(dedup?0:mu)+1))/(n+1);
  if(gate){for(double& x:sc)x*=.83;m.fields["two_layer_gate_scaling_coeffs"]=list(sc);
    std::vector<double> gc(72);for(int mu=0;mu<6;++mu)for(int n=0;n<12;++n)
      gc[12*mu+n]=std::cos(double(n+(dedup?0:mu)+1))/(n+1);
    m.fields["two_layer_gate_radial_coeffs"]=list(gc);}
  return m;
}
bool same(double a,double b,double tol) {
  if(std::isnan(a)||std::isnan(b))return std::isnan(a)&&std::isnan(b);
  if(!std::isfinite(a)||!std::isfinite(b))return a==b;
  const double e=std::abs(a-b)/(1+std::abs(b));max_error=std::max(max_error,e);
  return e<=tol;
}
// One thread owns all channels at a radius, as in the production neighbor loop.
// This also tests reuse across independent outer/gate coefficient and warp sets.
std::vector<double> evaluate(DR dev,bool context,bool stale=false) {
  const std::vector<double> rs={0.,.25,.5,.51,.511,.9,1.0137,2.5,4.999999,5.,5.2,
    std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()};
  Kokkos::View<double*,Device> r("lookup radii",rs.size());
  auto hr=Kokkos::create_mirror_view(r);for(std::size_t a=0;a<rs.size();++a)hr(a)=rs[a];Kokkos::deep_copy(r,hr);
  const int width=12*7+12; // 7 pair requests (including invalid) and 12 generic basis entries.
  Kokkos::View<double*,Device> out("lookup results",rs.size()*width*2);
  Kokkos::parallel_for("table lookup qualification",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,int(rs.size())),KOKKOS_LAMBDA(int a){
    for(int pair=0;pair<7;++pair){
      const int ti=pair<4?pair/2:(pair==4?-1:(pair==5?2:0));
      const int tj=pair<4?pair%2:(pair==6?2:0);
      auto c=dev.prepare_context(stale?2.3:r(a),stale?1-ti:ti,tj);
      for(int gate=0;gate<2;++gate)for(int mu=0;mu<6;++mu){double v,d;
        if(context)dev.contract(r(a),ti,tj,mu/2,mu%2,gate,c,v,d);
        else dev.contract(r(a),ti,tj,mu/2,mu%2,gate,v,d);
        const int idx=a*width+pair*12+gate*6+mu;out(2*idx)=v;out(2*idx+1)=d;
      }
    }
    for(int n=0;n<12;++n){double v,d;dev.basis(r(a),1.1,.2,0,n,v,d);
      const int idx=a*width+84+n;out(2*idx)=v;out(2*idx+1)=d;}
  });
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
  std::vector<double> result(h.extent(0));for(std::size_t i=0;i<result.size();++i)result[i]=h(i);return result;
}
void stats_equal(const RadialTableStats&a,const RadialTableStats&b,const std::string& label) {
  if(a.tables!=b.tables||a.bytes!=b.bytes||a.hits!=b.hits||a.direct_fallbacks!=b.direct_fallbacks||
     a.domain_fallbacks!=b.domain_fallbacks||a.memory_fallbacks!=b.memory_fallbacks||a.invalid_fallbacks!=b.invalid_fallbacks)
    throw std::runtime_error(label+" statistics differ");
}
void exercise(const Model&m,RadialTableMode mode=RadialTableMode::Table,
              std::size_t cap=64*1024*1024,const std::vector<int>& species={0,1},
              bool stale=false,bool counting=true) {
  std::vector<double> reference;RadialTableStats stats;std::size_t memory=0;
  for(int variant=0;variant<4;++variant){
    DR dev(m);dev.configure_lookup(variant&1,variant&2);
    // Reconfiguration must remove old tables and counters with the old layout.
    dev.configure(RadialTableMode::Table,.02,64*1024*1024,{0,1});
    dev.configure(mode,.01,cap,species);dev.set_counting(counting);
    context_cases+=dev.uses_context();packed_cases+=dev.uses_packed();
    if(mode==RadialTableMode::Direct&&(dev.uses_context()||dev.uses_packed()))
      throw std::runtime_error("Direct lookup activation");
    const auto values=evaluate(dev,variant&1,stale);const auto s=dev.stats();
    if(variant==0){reference=values;stats=s;memory=dev.memory_usage();}
    else {stats_equal(s,stats,"variant "+std::to_string(variant));
      if(memory!=dev.memory_usage())throw std::runtime_error("layout changed memory allocation size");
      for(std::size_t i=0;i<values.size();++i){++comparisons;
        // Packing preserves exact doubles. Keep the fixed context rounding
        // budget for both Serial and CUDA interpolation arithmetic.
        if(!same(values[i],reference[i],variant&1?3e-14:0.))
          throw std::runtime_error(m.text("radial_basis_type")+" variant="+std::to_string(variant)+" index="+std::to_string(i));}
    }
    if(!counting&&(s.hits||s.direct_fallbacks))throw std::runtime_error("disabled counting");
  }
  ++cases;
}
void structural_checks() {
  const auto m=fixture("RBLaguerre_log1p",false,true);
  DR dev(m);dev.configure_lookup(true,true);
  dev.configure(RadialTableMode::Table,.01,451*2*sizeof(double),{0,1});
  if(dev.stats().tables!=1||dev.stats().bytes!=451*2*sizeof(double)||
     !dev.uses_context()||!dev.uses_packed())throw std::runtime_error("partial one-lane packing not exercised");
  evaluate(dev,true);if(!dev.stats().hits||!dev.stats().memory_fallbacks)
    throw std::runtime_error("partial table hit/cap fallback not exercised");
  dev.configure(RadialTableMode::Table,.01,64*1024*1024,{0,1});const auto all=dev.stats();
  dev.configure(RadialTableMode::Table,.01,64*1024*1024,{1});
  if(!dev.stats().bytes||dev.stats().bytes>=all.bytes)throw std::runtime_error("species pruning not exercised");
  DR dd(fixture("RBLaguerre_log1p",false,true,"LK",true));
  dd.configure_lookup(true,true);dd.configure(RadialTableMode::Table,.01,64*1024*1024,{0,1});
  if(!dd.stats().tables||dd.stats().tables>=all.tables)throw std::runtime_error("dedup not exercised");
  // Switch both flags on the same evaluator, including Direct and back again.
  dev.configure_lookup(false,false);dev.configure(RadialTableMode::Table,.01,64*1024*1024,{0,1});
  const auto reference=evaluate(dev,false);const auto stats=dev.stats();
  dev.configure_lookup(true,true);dev.configure(RadialTableMode::Direct,.01,0,{0,1});
  if(dev.uses_context()||dev.uses_packed()||dev.stats().tables||dev.stats().hits)
    throw std::runtime_error("Direct reconfiguration retained layout/counters");
  dev.configure(RadialTableMode::Table,.01,64*1024*1024,{0,1});const auto result=evaluate(dev,true);
  stats_equal(stats,dev.stats(),"roundtrip");
  for(std::size_t i=0;i<result.size();++i)if(!same(result[i],reference[i],3e-14))
    throw std::runtime_error("lookup flag roundtrip");
}
void generation_check() {
  DR dev(fixture("RBLaguerre_log1p",false,true));
  dev.configure_lookup(true,true);
  dev.configure(RadialTableMode::Table,.01,64*1024*1024,{1});
  Kokkos::View<DR::FrozenContext*,Device> saved("saved old generation",1);
  Kokkos::parallel_for("save generation context",1,KOKKOS_LAMBDA(int){saved(0)=dev.prepare_context(1.37,1,1);});
  Kokkos::fence();
  // The same radius/grid/types now map to different valid active slots.
  // A cached context must fall back rather than index species zero tables.
  dev.configure(RadialTableMode::Table,.01,64*1024*1024,{0,1});
  Kokkos::View<double*,Device> values("generation compare",48);
  Kokkos::parallel_for("reuse old generation context",1,KOKKOS_LAMBDA(int){
    for(int gate=0;gate<2;++gate)for(int mu=0;mu<6;++mu){
      const int at=4*(gate*6+mu);double v,d;
      dev.contract(1.37,1,1,mu/2,mu%2,gate,saved(0),v,d);values(at)=v;values(at+1)=d;
      dev.contract(1.37,1,1,mu/2,mu%2,gate,v,d);values(at+2)=v;values(at+3)=d;
    }
  });
  const auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);
  for(int at=0;at<48;at+=4)if(!same(h(at),h(at+2),0)||!same(h(at+1),h(at+3),0))
    throw std::runtime_error("old generation context bypassed scalar fallback");
  const auto s=dev.stats();
  if(s.hits!=24||s.direct_fallbacks)throw std::runtime_error("generation fallback counters differ");
}
}
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);int status=0;
  try {
    const int device_arch=sus2_device_arch();
    const std::vector<std::string> families={"RBShapeev","RBChebyshev","RBChebyshev_s","RBChebyshev_ss","RBChebyshev_ssw","RBChebyshev_sss","RBChebyshev_sss_rational","RBChebyshev_sssw","RBChebyshev_sss_lmp","RBChebyshev_sss_rational_lmp","RBChebyshev_sssw_lmp","RBChebyshev_ss_lmp","RBChebyshev_ssw_lmp","RBChebyshev_s_lmp","RBChebyshev_ssss","RBChebyshev_sssss","RBChebyshev_sigma","RBLaguerre_log1p","RBLaguerre_log1p_lmp","RBLaguerre_log1p_pos","RBLaguerre_log1p_pos_lmp","RBLaguerre_log1p_noenv","RBLaguerre_log1p_noenv_lmp","RBJacobi_sss","RBJacobi_sss_lmp","RBJacobi_sss_noweight","RBJacobi_sss_noweight_lmp","RBBessel","RBBesselw","RBChebyshev_Tri","RBBessel_sss","RBBessel_sssw","RBChebyshev_tanhexp","RBChebyshev_tanhexp_w","RBChebyshev_repuls","RBTaylor"};
    for(const auto& family:families)exercise(fixture(family));
    for(const auto& family:{"RBChebyshev_sss","RBLaguerre_log1p","RBJacobi_sss"})
      for(const auto& map:{"pair","K","L","LK"})for(bool adaptive:{false,true}){
        const auto m=fixture(family,adaptive,true,map);exercise(m);exercise(m,RadialTableMode::Auto);
      }
    const auto m=fixture("RBLaguerre_log1p",false,true);
    exercise(m,RadialTableMode::Direct);exercise(m,RadialTableMode::Table,0);
    // A one-table cap leaves a partial packed group and rejected table handles.
    exercise(m,RadialTableMode::Table,451*2*sizeof(double));
    exercise(m,RadialTableMode::Table,64*1024*1024,{1});
    exercise(m,RadialTableMode::Auto,64*1024*1024,{});
    exercise(m,RadialTableMode::Table,64*1024*1024,{0,1},true);
    exercise(m,RadialTableMode::Table,64*1024*1024,{0,1},false,false);
    exercise(fixture("RBLaguerre_log1p",false,true,"LK",true));
    auto bad=fixture("RBChebyshev_sssw");auto sc=bad.numbers("scal_coeffs");
    for(int mu=0;mu<6;++mu)for(int pair=0;pair<4;++pair)sc[8*mu+4+pair]=2.;
    bad.fields["scal_coeffs"]=list(sc);exercise(bad);
    structural_checks();
    generation_check();
    std::cout<<"KOKKOS_EXECUTION_JSON {\"execution_space\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"device_cuda_arch\":"<<device_arch<<",\"requested_context\":1,\"active_context\":1,\"requested_packed\":1,\"active_packed\":1,\"role\":0,\"tile\":0}\n";
    std::cout<<std::setprecision(17)<<"TABLE_LOOKUP_JSON {\"execution_space\":\""<<Kokkos::DefaultExecutionSpace::name()
      <<"\",\"device_cuda_arch\":"<<device_arch<<",\"requested_context_variants\":2,\"requested_packed_variants\":2,\"families\":36,\"cases\":"<<cases<<",\"comparisons\":"<<comparisons
      <<",\"active_context_variants\":"<<context_cases<<",\"active_packed_variants\":"<<packed_cases
      <<",\"context_scaled_tolerance\":3e-14,\"pack_tolerance\":0,\"max_scaled_error\":"<<max_error<<",\"pass\":true}\n";
  }catch(const std::exception&e){std::cerr<<"TABLE_LOOKUP_FAIL "<<e.what()<<'\n';status=1;}
  Kokkos::finalize();return status;
}
