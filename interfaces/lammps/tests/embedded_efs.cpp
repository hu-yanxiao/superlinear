// Portable embedded-engine qualification; no native SUS2 or LAMMPS link.
// Build from repo root: c++ -O2 -std=c++17 -Iinterfaces/lammps/ML-SUS2 \
// interfaces/lammps/tests/embedded_efs.cpp interfaces/lammps/ML-SUS2/sus2_sl_*.cpp -o embedded-efs
// Usage: embedded-efs MODEL CFG [domains=1] [direct|table] [tabstep=.001]
#include "sus2_sl_kernel.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
using namespace LAMMPS_NS::SUS2SL;
using V=std::array<double,3>;
using Matrix=std::array<V,3>;
struct Geometry { Matrix cell{}; std::vector<V> x; std::vector<int> type; };
Geometry read_cfg(const char* path)
{
  std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open CFG");
  Geometry g; std::string word; int n=0;
  while(in>>word) {
    if(word=="Size") { in>>n; if(n<=0) throw std::runtime_error("Invalid CFG size"); }
    else if(word=="Supercell") for(auto& row:g.cell) for(auto& x:row) in>>x;
    else if(word=="AtomData:") {
      std::string line; std::getline(in,line); std::istringstream header(line);
      std::vector<std::string> keys; while(header>>word) keys.push_back(word);
      const auto index=[&](const char* key){auto k=std::find(keys.begin(),keys.end(),key);
        if(k==keys.end()) throw std::runtime_error("Missing CFG coordinate/type column"); return k-keys.begin();};
      const auto ti=index("type"),xi=index("cartes_x"),yi=index("cartes_y"),zi=index("cartes_z");
      for(int i=0;i<n;++i) {std::getline(in,line); std::istringstream row(line); std::vector<double> v(keys.size());
        for(double& a:v) if(!(row>>a)) throw std::runtime_error("Incomplete CFG atom row");
        g.type.push_back(static_cast<int>(v[ti])); g.x.push_back({{v[xi],v[yi],v[zi]}});}
      break;
    }
  }
  if(!n || g.x.size()!=std::size_t(n)) throw std::runtime_error("Incomplete CFG"); return g;
}
Matrix inverse(const Matrix& a)
{
  Matrix b; double det=0;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    b[j][i]=a[(i+1)%3][(j+1)%3]*a[(i+2)%3][(j+2)%3]-a[(i+1)%3][(j+2)%3]*a[(i+2)%3][(j+1)%3];
  for(int i=0;i<3;++i) det+=a[0][i]*b[i][0];
  if(!(det>0)) throw std::runtime_error("Invalid CFG cell");
  for(auto& row:b) for(auto& x:row) x/=det; return b;
}
std::vector<Edge> edges(const Geometry& g,double cutoff)
{
  std::vector<Edge> out; Matrix inv=inverse(g.cell); V reach{};
  for(int q=0;q<3;++q) {for(int a=0;a<3;++a) reach[q]+=inv[a][q]*inv[a][q];reach[q]=cutoff*std::sqrt(reach[q]);}
  for(int i=0;i<int(g.x.size());++i) for(int j=0;j<int(g.x.size());++j) {
    V d{},f{}; for(int q=0;q<3;++q) d[q]=g.x[j][q]-g.x[i][q];
    for(int q=0;q<3;++q) for(int a=0;a<3;++a) f[q]+=d[a]*inv[a][q];
    std::array<int,3> lo,hi; long long count=1;
    for(int q=0;q<3;++q) {lo[q]=int(std::ceil(-f[q]-reach[q])); hi[q]=int(std::floor(-f[q]+reach[q]));count*=std::max(0,hi[q]-lo[q]+1);}
    if(count>1000000) throw std::runtime_error("Unreasonable periodic image count");
    for(int a=lo[0];a<=hi[0];++a) for(int b=lo[1];b<=hi[1];++b) for(int c=lo[2];c<=hi[2];++c) {
      if(i==j && a==0 && b==0 && c==0) continue;
      V v=d; double r2=0; for(int q=0;q<3;++q) {v[q]+=a*g.cell[0][q]+b*g.cell[1][q]+c*g.cell[2][q];r2+=v[q]*v[q];}
      if(r2<cutoff*cutoff) out.push_back({i,j,v});
    }
  }
  return out;
}
int main(int argc,char** argv)
{
 try {
  if(argc<3 || argc>6) throw std::runtime_error("Usage: embedded-efs MODEL CFG [domains] [direct|table] [tabstep]");
  Model model; model.load(argv[1]); Geometry g=read_cfg(argv[2]);
  int domains=argc>3?std::stoi(argv[3]):1; if(domains<1||domains>128) throw std::runtime_error("Invalid domains");
  std::string mode=argc>4?argv[4]:"direct"; if(mode!="direct"&&mode!="table") throw std::runtime_error("Invalid mode");
  double step=argc>5?std::stod(argv[5]):.001; auto ed=edges(g,model.cutoff());
  const int n=g.x.size(); std::vector<int> owner(n),owner_local(n);
  std::vector<std::vector<int>> map(domains),local(domains,std::vector<int>(n));
  std::vector<int> owned(domains); std::vector<std::unique_ptr<Kernel>> kernels;
  for(int p=0;p<domains;++p) {
    for(int i=0;i<n;++i) if(i%domains==p) {owner[i]=p;owner_local[i]=map[p].size();map[p].push_back(i);}
    owned[p]=map[p].size(); for(int i=0;i<n;++i) if(i%domains!=p) map[p].push_back(i);
    std::vector<int> types; for(int i=0;i<n;++i) {local[p][map[p][i]]=i; types.push_back(g.type[map[p][i]]);}
    std::vector<Edge> pe; for(auto e:ed) if(e.i%domains==p) {e.i=local[p][e.i];e.j=local[p][e.j];pe.push_back(e);}
    kernels.emplace_back(new Kernel(model)); kernels.back()->configure_radial(mode=="direct"?RadialTableMode::Direct:RadialTableMode::Table,step);
    kernels.back()->begin(owned[p],n,types,pe);
  }
  const int width=kernels[0]->communication_width(); std::vector<double> buffer(width);
  for(int p=0;p<domains;++p) for(int i=owned[p];i<n;++i) {
    const int global=map[p][i]; kernels[owner[global]]->pack_forward(owner_local[global],buffer.data()); kernels[p]->unpack_forward(i,buffer.data());
  }
  for(auto& k:kernels) k->outer();
  for(int p=0;p<domains;++p) for(int i=owned[p];i<n;++i) {
    const int global=map[p][i]; kernels[p]->pack_reverse(i,buffer.data()); kernels[owner[global]]->add_reverse(owner_local[global],buffer.data());
  }
  Result total;total.force.resize(n);total.site_energy.resize(n);
  for(int p=0;p<domains;++p) {auto r=kernels[p]->finish();total.energy+=r.energy;
    for(int i=0;i<n;++i) for(int a=0;a<3;++a) total.force[map[p][i]][a]+=r.force[i][a];
    for(int i=0;i<owned[p];++i) total.site_energy[map[p][i]]=r.site_energy[i];
    for(int a=0;a<6;++a) total.virial[a]+=r.virial[a];}
  auto vector=[](const V& x){std::cout<<'['<<x[0]<<','<<x[1]<<','<<x[2]<<']';};
  std::cout<<std::setprecision(17)<<"NATIVE_EFS_JSON {\"energy\":"<<total.energy<<",\"forces\":[";
  for(int i=0;i<n;++i) {if(i)std::cout<<',';vector(total.force[i]);}
  const auto& w=total.virial;std::cout<<"],\"virial\":[";vector({{w[0],w[3],w[4]}});std::cout<<',';
  vector({{w[3],w[1],w[5]}});std::cout<<',';vector({{w[4],w[5],w[2]}});std::cout<<"],\"site_energies\":[";
  for(int i=0;i<n;++i){if(i)std::cout<<',';std::cout<<total.site_energy[i];}std::cout<<"]}\n";
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
