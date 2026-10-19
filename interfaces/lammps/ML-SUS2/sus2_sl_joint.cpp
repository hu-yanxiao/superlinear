#include "sus2_sl_joint.h"
#include <stdexcept>
#include <cmath>
namespace LAMMPS_NS { namespace SUS2SL {
JointReadout::JointReadout(const Model& model) {
 enabled_=model.boolean("joint_readout_enabled");
 if(!enabled_) return;
 species_=model.text("readout_mode","shared")=="species";
 lmax_=model.lmax;outer_rank_=model.integer("outer_k_rank",1);
 gate_rank_=model.integer("two_layer_gate_k_rank");
 outer_begin_=model.integer("outer_k_rank1_basic_begin");
 gate_begin_=model.integer("two_layer_gate_k_rank_basic_begin");
 moments_=model.integer("joint_readout_alpha_moments_count");
 scalar_count_=model.integer("joint_readout_scalar_count");
 coeffs_=model.numbers("joint_readout_coeffs");
 mapping_=model.integers("joint_readout_scalar_mapping");
 for(const auto& b:model.rows("joint_readout_alpha_index_basic",3)) {
  std::array<int,3> x;
  for(int a=0;a<3;++a) {if(b[a]!=std::floor(b[a])) throw std::runtime_error("Nonintegral joint primitive");x[a]=static_cast<int>(b[a]);}
  if((x[0]!=0 && x[0]!=1) || x[1]<1 || x[1]>lmax_ || std::abs(x[2])>x[1])
   throw std::runtime_error("Invalid joint primitive source or harmonic");
  basic_.push_back(x);
 }
 for(const auto& p:model.rows("joint_readout_sh_products",4)) {
  for(int a=0;a<3;++a) if(p[a]!=std::floor(p[a])) throw std::runtime_error("Nonintegral joint product index");
  Product x{static_cast<int>(p[0]),static_cast<int>(p[1]),static_cast<int>(p[2]),p[3]};
  if(x.a<0 || x.b<0 || x.out<static_cast<int>(basic_.size()) || x.a>=x.out || x.b>=x.out || x.out>=moments_ || !std::isfinite(x.c))
   throw std::runtime_error("Invalid joint product");
  products_.push_back(x);
 }
 if(outer_rank_<1 || gate_rank_<1 || moments_<static_cast<int>(basic_.size()) ||
    mapping_.size()!=static_cast<std::size_t>(scalar_count_) ||
    coeffs_.size()!=static_cast<std::size_t>(scalar_count_)*(species_?model.species_count:1))
  throw std::runtime_error("Invalid joint graph or coefficient dimensions");
 for(int node:mapping_) if(node<0 || node>=moments_) throw std::runtime_error("Invalid joint scalar mapping");
 for(const auto& b:basic_) {
  const int rank=b[0]?gate_rank_:outer_rank_;
  for(int a=0;a<rank;++a) {const int n=source_node(b[0],a,b[1],b[2]);if(n<0 || n>=model.moments)throw std::runtime_error("Invalid joint source node");}
 }
}
int JointReadout::source_node(int source,int a,int l,int m) const {
 const int rank=source?gate_rank_:outer_rank_,begin=source?gate_begin_:outer_begin_;
 const int H=(lmax_+1)*(lmax_+1);
 return begin+rank*(H-(l+1)*(l+1))+(rank-1-a)*(2*l+1)+m+l;
}
std::size_t JointReadout::memory_usage() const {
 return sizeof(*this)+sizeof(std::array<int,3>)*basic_.capacity()+sizeof(Product)*products_.capacity()+sizeof(int)*mapping_.capacity()+sizeof(double)*(coeffs_.capacity()+values_.capacity()+adj_.capacity());
}
double JointReadout::evaluate(int type,double center,const double* outer,const double* gate,
                             double* outer_adj,double* gate_adj) const {
 if(!enabled_) return 0.;
 if(!outer || !gate || type<0 || (species_ && static_cast<std::size_t>(type+1)*scalar_count_>coeffs_.size()))
  throw std::runtime_error("Invalid joint readout input");
 values_.assign(moments_,0.);adj_.assign(moments_,0.);
 auto& values=values_;auto& adj=adj_;
 for(std::size_t i=0;i<basic_.size();++i) {
  const auto& b=basic_[i];const int rank=b[0]?gate_rank_:outer_rank_;
  const double* source=b[0]?gate:outer;
  for(int a=0;a<rank;++a) values[i]+=source[source_node(b[0],a,b[1],b[2])];
  values[i]/=rank;
 }
 for(const auto& p:products_) values[p.out]+=p.c*values[p.a]*values[p.b];
 double energy=0.;
 for(int q=0;q<scalar_count_;++q) {
  const double weight=(species_?1.:center)*coeffs_[(species_?type*scalar_count_:0)+q];
  energy+=weight*values[mapping_[q]];adj[mapping_[q]]+=weight;
 }
 if(!outer_adj && !gate_adj) return energy;
 for(auto p=products_.rbegin();p!=products_.rend();++p) {
  const double a=p->c*adj[p->out];
  adj[p->a]+=a*values[p->b];adj[p->b]+=a*values[p->a];
 }
 for(std::size_t i=0;i<basic_.size();++i) {
  const auto& b=basic_[i];const int rank=b[0]?gate_rank_:outer_rank_;
  double* target=b[0]?gate_adj:outer_adj;
  if(target) for(int a=0;a<rank;++a) target[source_node(b[0],a,b[1],b[2])]+=adj[i]/rank;
 }
 return energy;
}
}}
