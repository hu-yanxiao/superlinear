// Standalone: compile with the installed Kokkos include/library directories and
// ML-SUS2/{sus2_sl_model,sus2_sl_radial,sus2_sl_radial_math}.cpp. The execution
// space follows the Kokkos build, so this same test also runs CUDA/HIP kernels.
#include "sus2_sl_kokkos_radial.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif
using namespace LAMMPS_NS::SUS2SL;
using Device=Kokkos::DefaultExecutionSpace::device_type;
using DR=DeviceRadial<Device>;
static bool collect_numerical_failures=false;
static std::size_t numerical_failures=0;
static int direct_cases=0,table_cases=0,fallback_cases=0;
struct NumericalFailure : std::runtime_error {using std::runtime_error::runtime_error;};
static std::string json_string(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<char(c);
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<char(c);}return out.str()+'"';
}
static std::string json_number(double value) {
  if(!std::isfinite(value))return json_string(std::isnan(value)?"NaN":(value>0?"Infinity":"-Infinity"));
  std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
}
static void backend_test() {
  Kokkos::View<int*,Device> proof("radial:backend_proof",1);
  Kokkos::parallel_for("radial backend proof",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1),KOKKOS_LAMBDA(int){
#ifdef __CUDA_ARCH__
    proof(0)=__CUDA_ARCH__;
#else
    proof(0)=0;
#endif
  });
  const int arch=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),proof)(0);
  bool cuda=false;int ordinal=-1,major=0,minor=0,device_count=0;
  std::string device="none",uuid="none",pci_bus="none";
#ifdef KOKKOS_ENABLE_CUDA
  cuda=std::is_same<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>::value;
  if(cuda){cudaDeviceProp prop{};
    char bus[64]{};
    if(cudaGetDeviceCount(&device_count)!=cudaSuccess||cudaGetDevice(&ordinal)!=cudaSuccess||
       cudaGetDeviceProperties(&prop,ordinal)!=cudaSuccess||cudaDeviceGetPCIBusId(bus,sizeof(bus),ordinal)!=cudaSuccess)
      throw std::runtime_error("CUDA runtime device query failed");
    device=prop.name;major=prop.major;minor=prop.minor;pci_bus=bus;
    std::ostringstream identifier;identifier<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)identifier<<'-';identifier<<std::setw(2)<<static_cast<unsigned>(static_cast<unsigned char>(prop.uuid.bytes[i]));}
    uuid=identifier.str();
    if(arch<=0)throw std::runtime_error("CUDA default execution space did not execute device proof kernel");
  }
#endif
  const char* required=std::getenv("SUS2_RADIAL_REQUIRE_CUDA");
  if(required&&std::strcmp(required,"1")==0&&(!cuda||arch<=0))
    throw std::runtime_error("SUS2_RADIAL_REQUIRE_CUDA=1 but actual default backend is not CUDA");
  std::cout<<"RADIAL_BACKEND {\"kokkos_version\":"<<KOKKOS_VERSION
    <<",\"execution_space\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"cuda\":"<<(cuda?"true":"false")
    <<",\"kernel_cuda_arch\":"<<arch<<",\"device_ordinal\":"<<ordinal<<",\"device_name\":\""<<device
    <<"\",\"compute_major\":"<<major<<",\"compute_minor\":"<<minor<<",\"device_count\":"<<device_count
    <<",\"device_uuid\":"<<json_string(uuid)<<",\"pci_bus_id\":"<<json_string(pci_bus)<<"}\n";
  Kokkos::print_configuration(std::cout,true);
}
static std::string list(const std::vector<double>& a) {
  std::ostringstream s;s<<std::setprecision(17)<<"{";
  for(std::size_t i=0;i<a.size();++i)s<<(i?",":"")<<a[i];return s.str()+"}";
}
static Model model(const std::string& basis,int B,const std::string& cutoff,bool adaptive=false) {
  Model m;m.species_count=2;m.lmax=1;m.kmax=3;m.radial_size=B;m.radial_funcs=6;
  m.min_dist=0.5;m.max_dist=5;m.scaling=0.73;
  m.fields["radial_basis_type"]=basis;m.fields["radial_cutoff_type"]=cutoff;
  m.fields["radial_cutoff_p"]="5";m.fields["scaling_map"]="LK";
  m.fields["outer_radial_mode"]=adaptive?"adaptive-coefficients":"legacy-plus-mixer";
  m.fields["outer_k_rank"]="3";
  std::vector<double> sc(2*4*6);
  for(int mu=0;mu<6;++mu)for(int p=0;p<4;++p){sc[2*mu*4+p]=1.1+0.05*mu+0.1*p;sc[2*mu*4+4+p]=0.2+0.01*p;}
  m.fields["scal_coeffs"]=list(sc);m.radial_coeffs.resize(6*(B+2));
  for(int mu=0;mu<6;++mu)for(int n=0;n<B;++n)m.radial_coeffs[mu*(B+2)+n]=std::sin(double(n+mu+1))/(n+1);
  return m;
}
static bool close(double a,double b,double tol=3e-12) {
  if(std::isnan(a)||std::isnan(b))return std::isnan(a)&&std::isnan(b);
  if(std::isinf(a)||std::isinf(b))return a==b;
  return std::abs(a-b)<=tol*(1+std::abs(b));
}
static void check(double a,double b,const std::string& message,double tol=3e-12) {
  if(!close(a,b,tol)){
    ++numerical_failures;
    std::cout<<"RADIAL_FAILURE {\"message\":"<<json_string(message)<<",\"actual\":"<<json_number(a)
      <<",\"reference\":"<<json_number(b)<<",\"abs_error\":"<<json_number(std::abs(a-b))
      <<",\"scaled_tolerance\":"<<json_number(tol*(1+std::abs(b)))<<"}\n";
    if(!collect_numerical_failures){std::ostringstream s;s<<std::setprecision(17)<<message<<": "<<a<<" != "<<b;throw NumericalFailure(s.str());}
  }
}
static std::vector<double> evaluate(DR dev,const std::vector<double>& r,int B,bool contracted=false,bool gate=false) {
  Kokkos::View<double*,Device> radii("test:r",r.size()),result("test:result",r.size()*B*2);
  auto hr=Kokkos::create_mirror_view(radii);for(std::size_t i=0;i<r.size();++i)hr(i)=r[i];Kokkos::deep_copy(radii,hr);
  Kokkos::parallel_for("radial native qualification",Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,int(r.size()*B)),KOKKOS_LAMBDA(int idx){
    const int at=idx/B,n=idx%B;double v,d;
    if(contracted)dev.contract(radii(at),at%2,(at/2)%2,n/2,n%2,gate,v,d);
    else dev.basis(radii(at),2.4,0.7,at%6,n,v,d);
    result(2*idx)=v;result(2*idx+1)=d;
  });
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
  std::vector<double> out(h.extent(0));for(std::size_t i=0;i<out.size();++i)out[i]=h(i);return out;
}
static void direct_test(const std::string& type,int B,const std::string& cutoff) {
  const auto failures_before=numerical_failures;
  Model m=model(type,B,cutoff);DR dev(m);dev.configure(RadialTableMode::Direct,0.01,0,{0,1});
  Radial cpu(type,m.min_dist,m.max_dist,B,m.scaling,cutoff,5);cpu.configure(RadialTableMode::Direct);
  const std::vector<double> rs={0.25,0.5,0.70001,0.9,1.01,2.5,4.999999,5.0,5.2,0.999,3.33,4.777};
  auto out=evaluate(dev,rs,B);std::vector<double> v,d;double ev=0,ed=0;
  for(std::size_t a=0;a<rs.size();++a){cpu.eval(rs[a],2.4,0.7,int(a%6),v,d);
    for(int n=0;n<B;++n){const auto where=type+" cutoff="+cutoff+" B="+std::to_string(B)+" r="+json_number(rs[a])+" k="+std::to_string(a%6)+" n="+std::to_string(n);
      check(out[2*(a*B+n)],v[n],where+" value");check(out[2*(a*B+n)+1],d[n],where+" derivative");
      if(std::isfinite(v[n]))ev=std::max(ev,std::abs(out[2*(a*B+n)]-v[n]));
      if(std::isfinite(d[n]))ed=std::max(ed,std::abs(out[2*(a*B+n)+1]-d[n]));}}
  std::cout<<std::setprecision(17)<<"RADIAL_CASE {\"mode\":\"Direct\",\"basis\":\""<<type<<"\",\"B\":"<<B
    <<",\"cutoff\":\""<<cutoff<<"\",\"comparisons\":"<<rs.size()*B*2
    <<",\"oracle_value_max_abs\":"<<json_number(ev)<<",\"oracle_derivative_max_abs\":"<<json_number(ed)
    <<",\"numerical_failures\":"<<numerical_failures-failures_before<<",\"pass\":"<<(numerical_failures==failures_before?"true":"false")<<"}\n";
  ++direct_cases;
}
static void table_test(const std::string& type,const std::string& cutoff,bool adaptive,
                       bool gate=false,const std::string& map="LK") {
  const auto failures_before=numerical_failures;
  const int B=12;Model m=model(type,B,cutoff,adaptive);DR dev(m);
  m.fields["scaling_map"]=map;
  auto sc=m.numbers("scal_coeffs");sc.resize(8*(map=="pair"?1:(map=="L"?2:(map=="K"?3:6))));
  m.fields["scal_coeffs"]=list(sc);
  if(gate){for(double& x:sc)x*=0.83;m.fields["two_layer_gate_scaling_coeffs"]=list(sc);
    std::vector<double> gc(6*B);for(std::size_t i=0;i<gc.size();++i)gc[i]=std::cos(double(i+1))/(1+i%B);
    m.fields["two_layer_gate_radial_coeffs"]=list(gc);}
  dev=DR(m);
  dev.configure(RadialTableMode::Table,0.01,64*1024*1024,{0,1});
  Radial cpu(type,m.min_dist,m.max_dist,B,m.scaling,cutoff,5);cpu.configure(RadialTableMode::Table,0.01,64*1024*1024);
  cpu.freeze(m,{0,1});const auto scales=m.numbers(gate?"two_layer_gate_scaling_coeffs":"scal_coeffs");
  const auto coefficients=gate?m.numbers("two_layer_gate_radial_coeffs"):m.radial_coeffs;
  const std::vector<double> rs={0.25,0.5,0.511,0.9,1.0137,2.5,4.999999,5.0,5.2};
  const auto result=evaluate(dev,rs,6,true,gate);const auto resident=dev.stats();std::vector<double> v,d,bv,bd;
  DR directdev(m);directdev.configure(RadialTableMode::Direct,0.01,0,{0,1});
  const auto direct_result=evaluate(directdev,rs,6,true,gate);
  double error=0,error_d=0,oracle_error=0,oracle_error_d=0;
  for(std::size_t a=0;a<rs.size();++a){const int ti=a%2,tj=(a/2)%2,pair=2*ti+tj;
    cpu.eval_frozen(rs[a],pair,gate,v,d);
    for(int mu=0;mu<6;++mu){
      const auto where=type+" cutoff="+cutoff+" adaptive="+std::to_string(adaptive)+" gate="+std::to_string(gate)+" map="+map+" r="+json_number(rs[a])+" pair="+std::to_string(pair)+" mu="+std::to_string(mu);
      check(result[2*(a*6+mu)],v[mu],where+" Table/CPU Table value");
      check(result[2*(a*6+mu)+1],d[mu],where+" Table/CPU Table derivative");
      if(std::isfinite(v[mu]))oracle_error=std::max(oracle_error,std::abs(result[2*(a*6+mu)]-v[mu]));
      if(std::isfinite(d[mu]))oracle_error_d=std::max(oracle_error_d,std::abs(result[2*(a*6+mu)+1]-d[mu]));
      Radial direct(type,m.min_dist,m.max_dist,B,m.scaling,cutoff,5);
      const int sb=map=="pair"?0:(map=="L"?mu%2:(map=="K"?mu/2:mu));
      direct.configure(RadialTableMode::Direct);direct.eval(rs[a],scales[8*sb+pair],scales[8*sb+4+pair],mu/2,bv,bd);
      double exact=0,ed=0;for(int n=0;n<B;++n){exact+=coefficients[mu*(gate?B:B+2)+n]*bv[n];ed+=coefficients[mu*(gate?B:B+2)+n]*bd[n];}
      check(direct_result[2*(a*6+mu)],exact,where+" Direct contraction value");
      check(direct_result[2*(a*6+mu)+1],ed,where+" Direct contraction derivative");
      if(std::isfinite(exact)){error=std::max(error,std::abs(result[2*(a*6+mu)]-exact));error_d=std::max(error_d,std::abs(result[2*(a*6+mu)+1]-ed));}
    }}
  // Capacity, domain, unknown-warp and cache-rebuild fallbacks stay device Direct.
  dev.configure(RadialTableMode::Table,0.01,0,{1});evaluate(dev,rs,6,true);
  auto s=dev.stats();if(s.bytes||s.hits||!s.direct_fallbacks||!s.domain_fallbacks||!s.memory_fallbacks)throw std::runtime_error(type+" capacity counters");
  dev.configure(RadialTableMode::Auto,0.01,64*1024*1024,{});evaluate(dev,rs,B);
  if(!dev.stats().invalid_fallbacks)throw std::runtime_error(type+" empty active species fallback");
  std::cout<<type<<" "<<cutoff<<" dynamic="<<adaptive<<" gate="<<gate<<" map="<<map<<" Table/Direct max_abs="<<error<<" deriv="<<error_d<<"\n";
  std::cout<<std::setprecision(17)<<"RADIAL_CASE {\"mode\":\"Table\",\"basis\":\""<<type<<"\",\"B\":"<<B
    <<",\"cutoff\":\""<<cutoff<<"\",\"adaptive\":"<<(adaptive?"true":"false")<<",\"gate\":"<<(gate?"true":"false")
    <<",\"map\":\""<<map<<"\",\"comparisons\":"<<rs.size()*6*2
    <<",\"oracle_value_max_abs\":"<<json_number(oracle_error)<<",\"oracle_derivative_max_abs\":"<<json_number(oracle_error_d)
    <<",\"direct_value_max_abs\":"<<json_number(error)<<",\"direct_derivative_max_abs\":"<<json_number(error_d)
    <<",\"table_count\":"<<resident.tables<<",\"table_bytes\":"<<resident.bytes<<",\"table_hits\":"<<resident.hits
    <<",\"direct_fallbacks\":"<<resident.direct_fallbacks<<",\"domain_fallbacks\":"<<resident.domain_fallbacks
    <<",\"memory_fallbacks\":"<<resident.memory_fallbacks<<",\"invalid_fallbacks\":"<<resident.invalid_fallbacks
    <<",\"numerical_failures\":"<<numerical_failures-failures_before<<",\"pass\":"<<(numerical_failures==failures_before?"true":"false")<<"}\n";
  ++table_cases;
}
static void fallback_test() {
  Model m=model("RBLaguerre_log1p",12,"polynomial");DR dev(m);
  dev.configure(RadialTableMode::Table,1e-300,1024,{0,1});evaluate(dev,{0.6,0.7},6,true);
  if(dev.stats().bytes||!dev.stats().memory_fallbacks)throw std::runtime_error("absurd spacing must remain device Direct");
  dev.configure(RadialTableMode::Table,0.01,64*1024*1024,{0,1});const auto all=dev.stats().bytes;
  evaluate(dev,{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()},6,true);
  if(dev.stats().domain_fallbacks!=12)throw std::runtime_error("nonfinite radius fallback counter");
  dev.configure(RadialTableMode::Table,0.01,64*1024*1024,{1});
  if(!all||dev.stats().bytes>=all)throw std::runtime_error("active pair pruning");
  m=model("RBChebyshev_sssw",12,"legacy");auto sc=m.numbers("scal_coeffs");
  for(int mu=0;mu<6;++mu)for(int pair=0;pair<4;++pair)sc[8*mu+4+pair]=2.0;
  m.fields["scal_coeffs"]=list(sc);dev=DR(m);dev.configure(RadialTableMode::Table,0.01,64*1024*1024,{0,1});
  evaluate(dev,{0.6,0.7},6,true);if(!dev.stats().invalid_fallbacks)throw std::runtime_error("nonfinite knots must reject table");
  // PureW O may exceed K. Only the K basis channels consume radial tables.
  m=model("RBJacobi_sss",12,"polynomial");m.fields["outer_radial_mode"]="pure-matrix";m.fields["outer_k_rank"]="9";
  dev=DR(m);dev.configure(RadialTableMode::Table,0.01,64*1024*1024,{0,1});
  std::cout<<"fallback checks: nonfinite knots/radius, absurd spacing, active pair pruning, PureW O>K PASS\n";
  std::cout<<"RADIAL_CASE {\"mode\":\"Fallback\",\"pass\":true}\n";
  ++fallback_cases;
}
int main(int argc,char**argv) {
  const char* collect=std::getenv("SUS2_RADIAL_COLLECT_NUMERICAL_FAILURES");
  collect_numerical_failures=collect&&std::strcmp(collect,"1")==0;
  Kokkos::initialize(argc,argv);int result=0;bool fatal=false;
  try {
    backend_test();
    const std::vector<std::string> types={"RBShapeev","RBChebyshev","RBChebyshev_s","RBChebyshev_ss","RBChebyshev_ssw","RBChebyshev_sss","RBChebyshev_sss_rational","RBChebyshev_sssw","RBChebyshev_sss_lmp","RBChebyshev_sss_rational_lmp","RBChebyshev_sssw_lmp","RBChebyshev_ss_lmp","RBChebyshev_ssw_lmp","RBChebyshev_s_lmp","RBChebyshev_ssss","RBChebyshev_sssss","RBChebyshev_sigma","RBLaguerre_log1p","RBLaguerre_log1p_lmp","RBLaguerre_log1p_pos","RBLaguerre_log1p_pos_lmp","RBLaguerre_log1p_noenv","RBLaguerre_log1p_noenv_lmp","RBJacobi_sss","RBJacobi_sss_lmp","RBJacobi_sss_noweight","RBJacobi_sss_noweight_lmp","RBBessel","RBBesselw","RBChebyshev_Tri","RBBessel_sss","RBBessel_sssw","RBChebyshev_tanhexp","RBChebyshev_tanhexp_w","RBChebyshev_repuls","RBTaylor"};
    for(const auto& type:types)for(const std::string cutoff:{"legacy","polynomial"}){
      direct_test(type,12,cutoff);table_test(type,cutoff,false);
      if(type!="RBShapeev")direct_test(type,65,cutoff);
    }
    for(const auto& type:{"RBChebyshev_sss","RBLaguerre_log1p","RBJacobi_sss"})table_test(type,"polynomial",true);
    for(const auto& map:{"pair","K","L","LK"})for(const auto& type:{"RBChebyshev_sss","RBLaguerre_log1p","RBJacobi_sss"})table_test(type,"polynomial",true,true,map);
    for(const auto& type:{"RBChebyshev_sss","RBChebyshev_sss_rational","RBLaguerre_log1p","RBJacobi_sss"})direct_test(type,1,"polynomial");
    fallback_test();
    if(numerical_failures)result=1;
    std::cout<<(result?"FAIL":"PASS")<<": 36 native families, Direct B=1/12/65 k=0..5, Table/Auto/fallback/reconfigure, decoupled gate, pair/K/L/LK warps\n";
  }catch(const NumericalFailure&e){std::cerr<<e.what()<<"\n";result=1;}
  catch(const std::exception&e){std::cerr<<"FATAL: "<<e.what()<<"\n";result=1;fatal=true;}
  std::cout<<"RADIAL_SUMMARY {\"direct_cases\":"<<direct_cases<<",\"table_cases\":"<<table_cases
    <<",\"fallback_cases\":"<<fallback_cases<<",\"families\":36,\"relative_scaled_tolerance\":3e-12";
  if(collect_numerical_failures)std::cout<<",\"numerical_failures\":"<<numerical_failures
    <<",\"collect_numerical_failures\":true,\"fatal\":"<<(fatal?"true":"false");
  std::cout<<",\"pass\":"<<(result?"false":"true")<<"}\n";
  Kokkos::finalize();return result;
}
