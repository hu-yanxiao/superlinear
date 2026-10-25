#include "sus2_sl_kokkos_kernel.h"
#include "sus2_sl_kernel.h"
#include "kokkos_execution_proof.h"
#include <iomanip>
#include <iostream>
#include <memory>
using namespace LAMMPS_NS::SUS2SL;
using KK=KokkosKernel<Kokkos::DefaultExecutionSpace>;
void configure_counting(KK&k){k.configure_radial_counting(std::getenv("SUS2_TEST_NO_RADIAL_COUNTS")==nullptr);}
double dense_compare(const Model&m,bool table){
  const int n=33;std::vector<std::array<double,3>>x(n);std::vector<int>types(n);std::vector<Edge>edges;std::vector<int>offset(1,0),neighbors;
  for(int i=0;i<n;++i){types[i]=i%m.species_count;x[i]={{1.4*(i%3)+.037*std::sin(i),1.4*((i/3)%3)+.021*std::cos(1.7*i),1.4*(i/9)+.031*std::sin(2.1*i)}};}
  int maximum=0;for(int i=0;i<n;++i){for(int j=0;j<n;++j)if(i!=j){std::array<double,3>d;double r2=0.;for(int q=0;q<3;++q){d[q]=x[j][q]-x[i][q];r2+=d[q]*d[q];}if(r2<m.cutoff()*m.cutoff()){edges.push_back({i,j,d});neighbors.push_back(j);}}maximum=std::max(maximum,static_cast<int>(neighbors.size())-offset.back());offset.push_back(neighbors.size());}
  if(maximum<=16)throw std::runtime_error("dense fixture did not cross a shared-neighbor tile");
  Kernel reference(m);reference.configure_radial(table?RadialTableMode::Table:RadialTableMode::Direct,.001);KK device(m);std::vector<int>species;for(int t=0;t<m.species_count;++t)species.push_back(t);device.configure_radial(table?RadialTableMode::Table:RadialTableMode::Direct,.001,std::numeric_limits<std::size_t>::max(),species);device.prepare(n,n,neighbors.size());configure_counting(device);
  auto xx=Kokkos::create_mirror_view(device.positions);auto tt=Kokkos::create_mirror_view(device.types);auto oo=Kokkos::create_mirror_view(device.offsets);auto nn=Kokkos::create_mirror_view(device.neighbors);for(int i=0;i<n;++i){tt(i)=types[i];for(int q=0;q<3;++q)xx(i,q)=x[i][q];}for(int i=0;i<=n;++i)oo(i)=offset[i];for(std::size_t e=0;e<neighbors.size();++e)nn(e)=neighbors[e];Kokkos::deep_copy(device.positions,xx);Kokkos::deep_copy(device.types,tt);Kokkos::deep_copy(device.offsets,oo);Kokkos::deep_copy(device.neighbors,nn);
  double error=0.;auto compare=[&](double a,double b){if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite dense E/F/virial");error=std::max(error,std::abs(a-b)/(1.+std::abs(b)));};
  for(int step=0;step<2;++step){reference.begin(n,n,types,edges);reference.outer();const auto&r=reference.finish();device.begin();device.outer();device.finish();if(device.error_code())throw std::runtime_error("dense device error");auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device.site_energy);auto f=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device.force);auto v=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device.virial);for(int i=0;i<n;++i){compare(e(i),r.site_energy[i]);for(int q=0;q<3;++q)compare(f(i,q),r.force[i][q]);}for(int q=0;q<6;++q){double z=0.;for(int i=0;i<n;++i)z+=v(i,q);compare(z,r.virial[q]);}}
  std::cout<<"dense atoms "<<n<<" maximum_neighbors "<<maximum<<" relative_max "<<error<<'\n';return error;
}
int main(int argc,char**argv){if(argc<2){std::cerr<<"Usage: kokkos-kernel-parity MODEL [direct|table] [zbl|gate-readout]\n";return 1;}Kokkos::initialize(argc,argv);int code=0;try{
  const int device_arch=sus2_device_arch();
  Model m;m.load(argv[1]);const bool table=argc>2&&std::string(argv[2])=="table";
  if(argc>3&&std::string(argv[3])=="zbl"){m.fields["zbl_enabled"]="true";std::string z="{";for(int t=0;t<m.species_count;++t){if(t)z+=",";z+=std::to_string(t?8:1);}m.fields["zbl_atomic_numbers"]=z+"}";m.fields["zbl_inner"]="1.2";m.fields["zbl_outer"]="3.0";m.fields["zbl_typewise_cutoff_enabled"]="false";m.validate();}
  if(argc>3&&std::string(argv[3])=="gate-readout"){m.fields["two_layer_gate_readout_enabled"]="true";const int q=m.integer("two_layer_gate_scalar_count");std::string w="{";for(int i=0;i<q;++i){if(i)w+=",";w+=std::to_string(.03*(i+1));}m.fields["two_layer_gate_readout_coeffs"]=w+"}";m.fields["two_layer_gate_readout_coeff_count"]=std::to_string(q);m.validate();}
  std::cout<<"model "<<argv[1]<<" radial "<<(table?"table":"direct")<<'\n';
  std::vector<std::array<double,3>>x={{{0.,0.,0.}},{{1.7,.3,.2}},{{.2,1.9,.5}},{{1.5,1.8,.9}},{{.5,.6,2.1}}};
  std::vector<int>types;for(int i=0;i<5;++i)types.push_back(i%m.species_count);
  std::vector<Edge>edges;for(int i=0;i<5;++i)for(int j=0;j<5;++j)if(i!=j){std::array<double,3>d;double r2=0;for(int q=0;q<3;++q){d[q]=x[j][q]-x[i][q];r2+=d[q]*d[q];}if(r2<m.cutoff()*m.cutoff())edges.push_back({i,j,d});}
  Kernel cpu(m);std::vector<int>species;for(int t=0;t<m.species_count;++t)species.push_back(t);cpu.configure_radial(table?RadialTableMode::Table:RadialTableMode::Direct,.001);
  KK gpu(m);gpu.configure_radial(table?RadialTableMode::Table:RadialTableMode::Direct,.001,std::numeric_limits<std::size_t>::max(),species);gpu.prepare(5,5,edges.size());configure_counting(gpu);
  auto xx=Kokkos::create_mirror_view(gpu.positions);auto tt=Kokkos::create_mirror_view(gpu.types);auto oo=Kokkos::create_mirror_view(gpu.offsets);auto nn=Kokkos::create_mirror_view(gpu.neighbors);
  int edge=0;for(int i=0;i<5;++i){tt(i)=types[i];for(int q=0;q<3;++q)xx(i,q)=x[i][q];oo(i)=edge;for(const auto&e:edges)if(e.i==i)nn(edge++)=e.j;}oo(5)=edge;
  Kokkos::deep_copy(gpu.positions,xx);Kokkos::deep_copy(gpu.types,tt);Kokkos::deep_copy(gpu.offsets,oo);Kokkos::deep_copy(gpu.neighbors,nn);
  double error=0.;
  for(int step=0;step<2;++step){cpu.begin(5,5,types,edges);cpu.outer();const auto&reference=cpu.finish();gpu.begin();gpu.outer();gpu.finish();Kokkos::fence();
    auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gpu.site_energy);auto f=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gpu.force);auto v=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),gpu.virial);double energy=0.;for(int i=0;i<5;++i){energy+=e(i);error=std::max(error,std::abs(e(i)-reference.site_energy[i])/(1.+std::abs(reference.site_energy[i])));for(int q=0;q<3;++q)error=std::max(error,std::abs(f(i,q)-reference.force[i][q])/(1.+std::abs(reference.force[i][q])));}for(int q=0;q<6;++q){double z=0;for(int i=0;i<5;++i)z+=v(i,q);error=std::max(error,std::abs(z-reference.virial[q])/(1.+std::abs(reference.virial[q])));}if(gpu.error_code())throw std::runtime_error("device geometry error");std::cout<<std::setprecision(17)<<"step "<<step<<" energy "<<energy<<" reference "<<reference.energy<<" relative_max "<<error<<" bytes "<<gpu.memory_usage()<<'\n';
  }
  // Simulate two owner partitions and a rank with no owned atoms. The engine
  // must expose source adjoints on ghost slots before its inner reverse.
  std::vector<std::unique_ptr<KK>>domains;std::vector<std::vector<int>>map(3);std::vector<int>owned={3,2,0};int owner[5]={0,1,0,1,0},owner_local[5]={0,0,1,1,2};
  for(int p=0;p<3;++p){for(int i=0;i<5;++i)if(owner[i]==p)map[p].push_back(i);for(int i=0;i<5;++i)if(owner[i]!=p)map[p].push_back(i);std::vector<int>local(5);for(int a=0;a<5;++a)local[map[p][a]]=a;std::vector<int>nbr,offset(1,0);for(int a=0;a<owned[p];++a){for(const auto&e:edges)if(e.i==map[p][a])nbr.push_back(local[e.j]);offset.push_back(nbr.size());}
    domains.emplace_back(new KK(m));auto&k=*domains.back();k.configure_radial(table?RadialTableMode::Table:RadialTableMode::Direct,.001,std::numeric_limits<std::size_t>::max(),species);k.prepare(owned[p],5,nbr.size());configure_counting(k);auto px=Kokkos::create_mirror_view(k.positions);auto pt=Kokkos::create_mirror_view(k.types);auto po=Kokkos::create_mirror_view(k.offsets);auto pn=Kokkos::create_mirror_view(k.neighbors);for(int a=0;a<5;++a){pt(a)=types[map[p][a]];for(int q=0;q<3;++q)px(a,q)=x[map[p][a]][q];}for(std::size_t a=0;a<offset.size();++a)po(a)=offset[a];for(std::size_t a=0;a<nbr.size();++a)pn(a)=nbr[a];Kokkos::deep_copy(k.positions,px);Kokkos::deep_copy(k.types,pt);Kokkos::deep_copy(k.offsets,po);Kokkos::deep_copy(k.neighbors,pn);
  }
  cpu.begin(5,5,types,edges);cpu.outer();const auto reference=cpu.finish();
  for(int step=0;step<2;++step){for(auto&k:domains)k->begin();std::vector<KK::PublicMatrix::HostMirror>signal;for(auto&k:domains)signal.push_back(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),k->signal));for(int p=0;p<3;++p)for(int a=owned[p];a<5;++a){const int i=map[p][a];for(int q=0;q<gpu.communication_width();++q)signal[p](a,q)=signal[owner[i]](owner_local[i],q);}for(int p=0;p<3;++p){Kokkos::deep_copy(domains[p]->signal,signal[p]);domains[p]->outer();}
    std::vector<KK::PublicMatrix::HostMirror>adj;for(auto&k:domains)adj.push_back(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),k->signal_adjoint));for(int p=0;p<3;++p)for(int a=owned[p];a<5;++a){const int i=map[p][a];for(int q=0;q<gpu.communication_width();++q)adj[owner[i]](owner_local[i],q)+=adj[p](a,q);}for(int p=0;p<3;++p){Kokkos::deep_copy(domains[p]->signal_adjoint,adj[p]);domains[p]->finish();}
    double energies[5]={},forces[5][3]={},virial[6]={};for(int p=0;p<3;++p){auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),domains[p]->site_energy);auto f=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),domains[p]->force);auto v=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),domains[p]->virial);for(int a=0;a<5;++a){const int i=map[p][a];if(a<owned[p]){energies[i]+=e(a);for(int q=0;q<6;++q)virial[q]+=v(a,q);}for(int q=0;q<3;++q)forces[i][q]+=f(a,q);}if(domains[p]->error_code())throw std::runtime_error("partition device error");}
    for(int i=0;i<5;++i){error=std::max(error,std::abs(energies[i]-reference.site_energy[i])/(1.+std::abs(reference.site_energy[i])));for(int q=0;q<3;++q)error=std::max(error,std::abs(forces[i][q]-reference.force[i][q])/(1.+std::abs(reference.force[i][q])));}for(int q=0;q<6;++q)error=std::max(error,std::abs(virial[q]-reference.virial[q])/(1.+std::abs(reference.virial[q])));std::cout<<"partition_step "<<step<<" relative_max "<<error<<'\n';
  }
  KK empty(m);empty.prepare(0,0,0);empty.begin();empty.outer();empty.finish();Kokkos::fence();
  // Isolated invalid source types must set the device status without indexing
  // species/head arrays, even when there is no edge to discover the bad type.
  KK invalid(m);invalid.prepare(1,1,0);auto it=Kokkos::create_mirror_view(invalid.types);it(0)=m.species_count;Kokkos::deep_copy(invalid.types,it);Kokkos::deep_copy(invalid.offsets,0);Kokkos::deep_copy(invalid.positions,0.);invalid.begin();if(invalid.error_code()!=3)throw std::runtime_error("isolated type was not validated during begin");invalid.outer();invalid.finish();if(invalid.error_code()!=3)throw std::runtime_error("invalid type status lost");
  error=std::max(error,dense_compare(m,table));
  const auto stats=gpu.radial_stats();if(std::getenv("SUS2_TEST_NO_RADIAL_COUNTS")&&(stats.hits||stats.direct_fallbacks))throw std::runtime_error("disabled radial counters changed");
  std::cout<<"KOKKOS_EXECUTION_JSON {\"execution_space\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"device_cuda_arch\":"<<device_arch<<",\"requested_context\":"<<gpu.uses_table_context()<<",\"active_context\":"<<gpu.table_context_active()<<",\"requested_packed\":"<<gpu.uses_table_packed()<<",\"active_packed\":"<<gpu.table_packed_active()<<",\"role\":"<<gpu.uses_forward_role_specialize()<<",\"tile\":"<<gpu.uses_forward_radial_tile()<<"}\n";
  std::cout<<"KOKKOS_PARITY_JSON {\"model\":\""<<argv[1]<<"\",\"radial\":\""<<(table?"table":"direct")<<"\",\"generic_reference\":"<<(gpu.uses_generic_reference()?"true":"false")<<",\"normalized_error\":"<<error<<",\"comparisons\":6,\"status\":\""<<(error>2e-10?"FAIL":"PASS")<<"\"}\n";
  if(error>2e-10)code=2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
