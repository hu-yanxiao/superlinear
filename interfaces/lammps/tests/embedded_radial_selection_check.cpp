// Standalone radial selection/storage regression; external model stays read-only.
// See README.superlinear-embedded-cpu.md for the portable build command.
#include "sus2_sl_model.h"
#include "sus2_sl_radial.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS::SUS2SL;
namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
Radial* radial(const Model& m) {
  return new Radial(m.text("radial_basis_type"),m.min_dist,m.max_dist,m.radial_size,
                    m.scaling,m.text("radial_cutoff_type","legacy"),m.integer("radial_cutoff_p",5));
}
void equal(const std::vector<double>& a,const std::vector<double>& b) {
  require(a.size()==b.size(),"Frozen result size differs");
  for(std::size_t i=0;i<a.size();++i)
    require(a[i]==b[i],"Pruning changed frozen interpolation arithmetic or model indexing");
}
void selection(const Model& m) {
  require(m.species_count==8,"Selection fixture must contain eight native species");
  const std::vector<int> selected{0,1,2,3,6,7};
  std::unique_ptr<Radial> full(radial(m)),subset(radial(m));
  // A coarser spacing bounds regression runtime; arithmetic comparisons use
  // exactly the same spacing and require bit-for-bit agreement.
  full->configure(RadialTableMode::Table,.004);
  subset->configure(RadialTableMode::Table,.004);
  full->freeze(m);
  subset->freeze(m,{7,3,0,6,2,1,7,0});
  require(subset->stats().bytes==0&&subset->stats().tables==0,"Freeze eagerly allocated knots");
  const double r=std::max(0.,m.min_dist)+.371*(m.max_dist-std::max(0.,m.min_dist));
  std::vector<double> av,ad,bv,bd;
  for(int ti:selected)for(int tj:selected)for(int gate=0;gate<2;++gate) {
    const int pair=ti*m.species_count+tj;
    require(full->eval_frozen(r,pair,gate,av,ad),"Full request unexpectedly missing");
    require(subset->eval_frozen(r,pair,gate,bv,bd),"Selected original model pair missing");
    equal(av,bv);equal(ad,bd);
  }
  require(!subset->eval_frozen(r,4*8+0,false,bv,bd),"Absent species S retained a frozen request");
  require(!subset->eval_frozen(r,0*8+5,false,bv,bd),"Absent species Cl retained a frozen request");
  subset->freeze(m,{});
  require(!subset->frozen_available()&&subset->stats().bytes==0&&subset->stats().hits==0,
          "Empty selection retained old requests, knots or counters");
  subset->freeze(m,{5});
  require(!subset->eval_frozen(r,0,false,bv,bd),"Reconfiguration retained an old pair");
  require(subset->eval_frozen(r,5*8+5,false,bv,bd),"New selection failed to refill");
  subset->configure(RadialTableMode::Table,.004,0);
  subset->eval_frozen(r,5*8+5,false,bv,bd);
  require(subset->stats().bytes==0&&subset->stats().memory_fallbacks>0,"Explicit zero cap changed");
  subset->configure(RadialTableMode::Table,.004,1024);
  subset->eval_frozen(r,5*8+5,false,bv,bd);
  require(subset->stats().bytes<=1024&&subset->stats().memory_fallbacks>0,"Explicit small cap ignored");
  bool rejected=false;try{subset->freeze(m,{8});}catch(const std::out_of_range&){rejected=true;}
  require(rejected,"Out-of-range original model index accepted");
}
void default_storage() {
  Radial unlimited("RBChebyshev",.5,2.5,1),capped("RBChebyshev",.5,2.5,1);
  unlimited.configure(RadialTableMode::Table,2./4096.);
  capped.configure(RadialTableMode::Table,2./4096.,65552);
  std::vector<double> v,d;
  // Each distinct channel is a distinct complete generic request. Tiny basis
  // size keeps the >64 MiB admission check independent of large model fixtures.
  for(int channel=0;channel<1100;++channel)unlimited.eval(1.25,1.,0.,channel,v,d);
  const auto stats=unlimited.stats();
  require(stats.bytes>64u*1024u*1024u,"Default still imposes the old 64 MiB cap");
  require(stats.tables==1100&&stats.memory_fallbacks==0&&stats.invalid_fallbacks==0,
          "Safe generic requests did not all receive default tables");
  for(int channel=0;channel<2;++channel)capped.eval(1.25,1.,0.,channel,v,d);
  require(capped.stats().bytes<=65552&&capped.stats().memory_fallbacks>0,
          "Explicit generic cap ignored");
  std::cout<<"default knot bytes="<<stats.bytes<<" tables="<<stats.tables<<'\n';
}
}
int main(int argc,char**argv) {
  try {
    require(argc==2,"Usage: embedded-radial-selection-check EIGHT_SPECIES_MODEL");
    Model model;model.load(argv[1]);selection(model);default_storage();
    std::cout<<"PASS: original species indices, lazy selection/reset, explicit caps, default >64 MiB\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
