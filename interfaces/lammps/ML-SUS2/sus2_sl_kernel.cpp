#include "sus2_sl_kernel.h"
#include "sus2_sl_zbl.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <stdexcept>
namespace LAMMPS_NS { namespace SUS2SL {
namespace {
int h(int l,int m) { return l*l+l+m; }
// Normalized real solid-harmonic recurrence, including Cartesian derivatives.
// It remains regular on the polar axis and uses the native Condon-Shortley convention.
template<bool Derivatives>
void harmonics(const std::array<double,3>& x,int L,std::vector<double>& v,std::vector<double>& d) {
 const int H=(L+1)*(L+1); double r2=0; for(double a:x) r2+=a*a;
 const double r=std::sqrt(r2);
 if(!(r>0)) throw std::runtime_error("distributed EFS neighborhood contains an invalid source or distance");
 typedef std::complex<double> C; C s[49]{},ds[147]{}; s[0]=1;
 C u(x[0],x[1]),du[3]={C(1),C(0,1),C(0)};
 for(int m=1;m<=L;++m) {int i=h(m,m),p=h(m-1,m-1); double c=-(2*m-1);
  s[i]=c*u*s[p]; if(Derivatives)for(int a=0;a<3;++a) ds[3*i+a]=c*(du[a]*s[p]+u*ds[3*p+a]); }
 for(int m=0;m<=L;++m) {
  if(m<L) {int i=h(m+1,m),p=h(m,m); double c=2*m+1; s[i]=c*x[2]*s[p];
   if(Derivatives)for(int a=0;a<3;++a) ds[3*i+a]=c*((a==2?s[p]:C(0))+x[2]*ds[3*p+a]); }
  for(int l=m+2;l<=L;++l) {int i=h(l,m),p=h(l-1,m),q=h(l-2,m); double A=2*l-1,B=l+m-1,D=l-m;
   s[i]=(A*x[2]*s[p]-B*r2*s[q])/D;
   if(Derivatives)for(int a=0;a<3;++a) ds[3*i+a]=(A*((a==2?s[p]:C(0))+x[2]*ds[3*p+a])-B*(2*x[a]*s[q]+r2*ds[3*q+a]))/D; }
 }
 // The final (l,m) loop writes every positive/negative harmonic component.
 v.resize(H); if(Derivatives)d.resize(3*H);
 static const std::array<double,49> norm=[](){std::array<double,49>n{};const double pi=std::acos(-1.0);for(int l=0;l<=6;++l)for(int m=0;m<=l;++m){double ratio=1;for(int a=l-m+1;a<=l+m;++a)ratio/=a;n[h(l,m)]=std::sqrt((2*l+1)*ratio/(4*pi));}return n;}();
 double invpow=1;
 for(int l=0;l<=L;++l,invpow/=r) for(int m=0;m<=l;++m) {
  double n=norm[h(l,m)]*invpow;
  int i=h(l,m); C y=n*s[i]; double f=m?std::sqrt(2.0)*(m%2?-1:1):1;
  v[i]=f*y.real(); if(m) v[h(l,-m)]=f*y.imag();
  if(Derivatives)for(int a=0;a<3;++a) {C z=n*(ds[3*i+a]-double(l)*s[i]*x[a]/r2);
   d[3*i+a]=f*z.real(); if(m) d[3*h(l,-m)+a]=f*z.imag(); }
 }
}
double at(const std::vector<double>& v,int i,double fallback=0) { return i>=0&&i<int(v.size())?v[i]:fallback; }
}
Kernel::Kernel(const Model& m):model_(m),radial_(m.text("radial_basis_type"),m.min_dist,m.max_dist,m.radial_size,m.scaling,m.text("radial_cutoff_type","legacy"),m.integer("radial_cutoff_p",5),m.angular_max_dist) {
 m.validate(); gated_=m.boolean("two_layer_gate_enabled");
 gate_rank_=m.integer("two_layer_gate_k_rank"); outer_rank_=m.boolean("outer_k_rank1_enabled")?m.integer("outer_k_rank",1):0;
 gate_begin_=m.integer("two_layer_gate_k_rank_basic_begin"); outer_begin_=m.integer("outer_k_rank1_basic_begin");
 outer_product_=outer_rank_?m.integer("outer_k_rank1_product_begin"):0;
 weights_=m.numbers("two_layer_gate_weights"); gate_nodes_=m.integers("two_layer_gate_scalar_moment_indices"); body_=m.integers("two_layer_gate_scalar_body_orders");
 if(gate_nodes_.empty()) for(int q:m.integers("two_layer_gate_scalar_indices")) gate_nodes_.push_back(m.scalar_mapping.at(q));
 linear_=m.numbers("moment_coeffs"); gate_type_=m.numbers("two_layer_gate_type_coeffs"); gate_add_=m.numbers("two_layer_gate_additive_coeffs"); gate_radial_=m.numbers("two_layer_gate_radial_coeffs");
 scal_=m.numbers("scal_coeffs");gate_scal_=m.numbers("two_layer_gate_scaling_coeffs");gate_rank_weights_=m.numbers("two_layer_gate_k_rank_weights");outer_rank_weights_=m.numbers("outer_k_rank1_weights");species_=m.numbers("species_coeffs");bias_=m.numbers("shift_coeffs");body_mix_=m.numbers("two_layer_gate_body_mix_weights");
 if(gate_type_.empty()){gate_type_.resize(m.species_count*m.radial_funcs);for(int t=0;t<m.species_count;++t)for(int mu=0;mu<m.radial_funcs;++mu)gate_type_[t*m.radial_funcs+mu]=at(m.radial_coeffs,mu*(m.radial_size+m.species_count)+m.radial_size+t);}
 pure_=m.text("outer_radial_mode")=="pure-matrix";mixing_=m.text("two_layer_gate_radial_mixing","none")=="full";
 adaptive_mode_=m.text("outer_radial_mode")=="adaptive-coefficients";feature_width_=gated_?m.radial_funcs:0;
 const auto effect_option=[](const char* name){
  const char* value=std::getenv(name);
  if(value && std::string(value)!="0" && std::string(value)!="1")
   throw std::runtime_error(std::string(name)+" must be 0 or 1");
  return !value || std::string(value)=="1";
 };
 const bool effect=gated_ && !adaptive_mode_ && !pure_;
 effect_source_cache_=effect_option("SUS2_SL_EFFECT_SOURCE_CACHE") && effect;
 effect_forward_stream_=effect_option("SUS2_SL_EFFECT_FORWARD_STREAM") && effect;
 step_angular_cache_enabled_=effect_option("SUS2_SL_STEP_ANGULAR_CACHE") && effect;
 step_angular_budget_=64*1024*1024;
 if(const char* budget=std::getenv("SUS2_SL_STEP_ANGULAR_CACHE_MB")){
  char* end=nullptr;
  const unsigned long long mb=std::strtoull(budget,&end,10);
  if(!*budget || *end || *budget=='-' || mb>std::numeric_limits<std::size_t>::max()/1024/1024)
   throw std::runtime_error("Invalid SUS2_SL_STEP_ANGULAR_CACHE_MB");
  step_angular_budget_=std::size_t(mb)*1024*1024;
 }
 if(effect_source_cache_){
  const int C=m.species_count,F=m.radial_funcs,R=m.radial_size;
  effect_outer_type_.resize(std::size_t(C)*F);
  effect_gate_type_.resize(std::size_t(C)*F);
  effect_gate_add_.resize(std::size_t(C)*F);
  for(int t=0;t<C;++t)for(int mu=0;mu<F;++mu){
   const std::size_t index=std::size_t(t)*F+mu;
   effect_outer_type_[index]=at(m.radial_coeffs,mu*(R+C)+R+t);
   effect_gate_type_[index]=at(gate_type_,t*F+mu,at(gate_type_,t,1));
   effect_gate_add_[index]=at(gate_add_,t*F+mu,at(gate_add_,t,1));
  }
 }
 if(adaptive_mode_){adaptive_.reset(new Adaptive(m));feature_width_=adaptive_->signal_width();feature_nodes_.resize(feature_width_);for(int q=0;q<feature_width_;++q)feature_nodes_[q]=adaptive_->feature_moment_index(q);}
 const char* projected_comm=std::getenv("SUS2_SL_ADAPTIVE_PROJECTED_COMM");
 if(projected_comm && std::string(projected_comm)!="0" && std::string(projected_comm)!="1")
  throw std::runtime_error("SUS2_SL_ADAPTIVE_PROJECTED_COMM must be 0 or 1");
 // Owners perform the linear projection and its transpose once per atom.
 // Nonlinear/shared-n/tensor projections retain raw-signal communication.
 adaptive_projected_comm_=(!projected_comm || std::string(projected_comm)=="1") && adaptive_ && !mixing_ && adaptive_->projection_width()>0;

 mixer_=m.numbers("two_layer_gate_radial_mixing_coeffs");
 std::string sm=m.text("scaling_map","LK");scaling_mode_=sm=="pair"?0:(sm=="L"?1:(sm=="K"?2:3));
 species_readout_=m.text("readout_mode")=="species";body_combo_=m.text("two_layer_gate_mode")=="mu-body-linear-combo";
 gate_readout_=m.numbers("two_layer_gate_readout_coeffs");
 body_count_=m.integer("two_layer_gate_body_order_max")-1;gate_readout_enabled_=m.boolean("two_layer_gate_readout_enabled");
 gate_product_begin_=m.integer("two_layer_gate_k_rank_product_begin",0);gate_product_end_=gate_rank_?m.integer("two_layer_gate_k_rank_product_end",outer_product_):outer_product_;
 if(!outer_rank_&&!gate_rank_)gate_product_end_=m.products.size();
 if(m.boolean("joint_readout_enabled"))joint_.reset(new JointReadout(m));
 zbl_=m.boolean("zbl_enabled");if(zbl_){auto z=m.integers("zbl_atomic_numbers");int P=m.species_count*m.species_count;zbl_constants_.resize(P);zbl_inner_.resize(P);zbl_outer_.resize(P);std::vector<double>sq(P);sus2_sl_zbl::FillPairConstants(m.species_count,z.data(),zbl_constants_.data());sus2_sl_zbl::FillPairCutoffTables(m.species_count,z.data(),m.real("zbl_inner",.7),m.real("zbl_outer",1.4),m.boolean("zbl_typewise_cutoff_enabled"),m.real("zbl_typewise_cutoff_factor",.7),zbl_inner_.data(),zbl_outer_.data(),sq.data());}
 if(m.text("outer_radial_mode","legacy-plus-mixer")!="legacy-plus-mixer"&&!pure_&&!adaptive_mode_) throw std::runtime_error("SUS2 embedded: unknown outer radial mode");
 if(m.boolean("multi_layer_gate_enabled")||m.boolean("two_layer_gate_edge_l1_enabled")||m.boolean("two_layer_residual_enabled")) throw std::runtime_error("SUS2 embedded: multilayer, tensor-edge and residual modes are not implemented yet");
 if(gated_&&m.text("two_layer_gate_site_mode","neighbor")!="neighbor") throw std::runtime_error("SUS2 embedded: center-gate mode is not implemented yet");
 if(linear_.size()!=std::size_t(m.scalar_count)*(m.text("readout_mode")=="species"?m.species_count:1)) throw std::runtime_error("SUS2 embedded: invalid readout coefficient count");
 for(const auto&b:m.basic){basic_all_.push_back(int(basic_mu_.size()));if(b[1]==0)basic_l0_.push_back(int(basic_mu_.size()));basic_mu_.push_back(b[0]*(m.lmax+1)+b[1]);basic_sh_.push_back(h(b[1],b[2]));}
 for(int gate=0;gate<2;++gate){const int rank=gate?gate_rank_:outer_rank_,begin=gate?gate_begin_:outer_begin_,H=(m.lmax+1)*(m.lmax+1);const auto&w=gate?gate_rank_weights_:outer_rank_weights_;auto&out=gate?gate_contraction_:outer_contraction_;
  for(int a=0;a<rank;++a)for(std::size_t b=0;b<m.basic.size();++b){int k=m.basic[b][0],l=m.basic[b][1],mm=m.basic[b][2];out.push_back({int(b),begin+rank*(H-(l+1)*(l+1))+(rank-1-a)*(2*l+1)+mm+l,at(w,(l*rank+a)*m.kmax+k,1)});}}
}
void Kernel::configure_radial(RadialTableMode m,double s,std::size_t n){
 const char*reverse=std::getenv("SUS2_SL_REVERSE_STREAM");
 if(reverse&&std::string(reverse)!="0"&&std::string(reverse)!="1")
  throw std::runtime_error("SUS2_SL_REVERSE_STREAM must be 0 or 1");
 const char*setting=std::getenv("SUS2_SL_EDGE_METADATA");
 if(setting&&std::string(setting)!="0"&&std::string(setting)!="1")
  throw std::runtime_error("SUS2_SL_EDGE_METADATA must be 0 or 1");
 const char* step_cache=std::getenv("SUS2_SL_STEP_RADIAL_CACHE");
 if(step_cache && std::string(step_cache)!="0" && std::string(step_cache)!="1")
  throw std::runtime_error("SUS2_SL_STEP_RADIAL_CACHE must be 0 or 1");
 step_radial_cache_enabled_=!step_cache || std::string(step_cache)=="1";
 step_radial_cache_auto_=!step_cache;
 step_radial_budget_=128*1024*1024;
 if(const char* budget=std::getenv("SUS2_SL_STEP_RADIAL_CACHE_MB")){
  char* end=nullptr;
  const unsigned long long mb=std::strtoull(budget,&end,10);
  if(!*budget || *end || *budget=='-' || mb>std::numeric_limits<std::size_t>::max()/1024/1024)
   throw std::runtime_error("Invalid SUS2_SL_STEP_RADIAL_CACHE_MB");
  step_radial_budget_=std::size_t(mb)*1024*1024;
 }
 const bool enabled=setting&&std::string(setting)=="1";
 reverse_stream_=!reverse||std::string(reverse)=="1";
 std::vector<EdgeMetadata>().swap(edge_metadata_);
 radial_mode_=m;edge_metadata_requested_=enabled;
 radial_.configure(m,s,n);
 if(m!=RadialTableMode::Direct){
  if(species_configured_)radial_.freeze(model_,active_species_);
  else radial_.freeze(model_);
 }
 frozen_enabled_=radial_.frozen_available();
 // Static frozen tables already store contracted radial values. Caching them
 // again adds edge traffic; retain step caching for gated models and direct
 // fallback, while an explicit 0/1 setting always overrides this default.
 if(step_radial_cache_auto_)step_radial_cache_enabled_=gated_ || !frozen_enabled_;
 metadata_cache_enabled_=enabled&&frozen_enabled_&&(gated_||(!pure_&&!adaptive_mode_));
}
void Kernel::configure_species(std::vector<int> species){
 std::sort(species.begin(),species.end());
 species.erase(std::unique(species.begin(),species.end()),species.end());
 for(int t:species)if(t<0||t>=model_.species_count)
  throw std::out_of_range("Radial table selection has an invalid model species ID");
 if(species_configured_&&species==active_species_)return;
 active_species_=std::move(species);species_configured_=true;
 std::vector<EdgeMetadata>().swap(edge_metadata_);
 if(radial_mode_!=RadialTableMode::Direct)radial_.freeze(model_,active_species_);
 frozen_enabled_=radial_.frozen_available();
 if(step_radial_cache_auto_)step_radial_cache_enabled_=gated_ || !frozen_enabled_;
 metadata_cache_enabled_=edge_metadata_requested_&&frozen_enabled_&&(gated_||(!pure_&&!adaptive_mode_));
}
int Kernel::communication_width()const{return adaptive_projected_comm_?adaptive_->projection_width():feature_width_;}
Kernel::EdgeMetadata* Kernel::edge_metadata(const Edge&e){
 if(edge_metadata_.empty())return nullptr;
 // All callers traverse the owned edges_ copy, including begin's gate pass.
 const std::size_t index=static_cast<std::size_t>(&e-edges_.data());
 return index<edge_metadata_.size()?&edge_metadata_[index]:nullptr;
}
template<bool Frozen,bool Derivatives,bool Materialize>
double Kernel::edge_basis(const Edge& e,bool gate,std::vector<double>& v,std::vector<double>& d) {
 const Model&m=model_; const int C=m.species_count,R=m.radial_size,L=m.lmax+1;
 EdgeMetadata*metadata=Frozen?edge_metadata(e):nullptr;double r;
 if(metadata)r=metadata->distance;else {double r2=0;for(double x:e.d)r2+=x*x;r=std::sqrt(r2);}
 const int pair=metadata?metadata->pair:types_[e.i]*C+types_[e.j];
 auto&y=y_scratch_;auto&dy=dy_scratch_;auto&rb=rb_scratch_;auto&dr=dr_scratch_;
 if(Materialize){v.resize(m.basic.size());if(Derivatives)d.resize(3*m.basic.size());}
 active_basic_=r>=m.max_dist?&basic_none_:(r>=m.angular_max_dist?&basic_l0_:&basic_all_);
 if(r>=m.max_dist){
  // The ordinary path overwrites all outputs; only this early exit needs zeros.
  if(Materialize){std::fill(v.begin(),v.end(),0.);if(Derivatives)std::fill(d.begin(),d.end(),0.);}
  return r;
 }
 const int active_L=r>=m.angular_max_dist?1:L;
 const std::size_t edge_index=std::size_t(&e-edges_.data());
 const std::size_t H=std::size_t(L)*L;
 if(edge_index<step_angular_ready_.size()){
  double* slot=step_angular_.data()+edge_index*4*H;
  if(!step_angular_ready_[edge_index]){
   // The geometry and active L are identical in all four edge passes. Compute
   // derivatives once, using the original recurrence and evaluation order.
   harmonics<true>(e.d,active_L-1,y,dy);
   std::copy(y.begin(),y.end(),slot);
   std::copy(dy.begin(),dy.end(),slot+H);
   step_angular_ready_[edge_index]=1;
  }
  edge_y_=slot;edge_dy_=slot+H;
 }else{
  harmonics<Derivatives>(e.d,active_L-1,y,dy);
  edge_y_=y.data();edge_dy_=Derivatives?dy.data():nullptr;
 }
 if(Materialize&&active_L<L){std::fill(v.begin(),v.end(),0.);if(Derivatives)std::fill(d.begin(),d.end(),0.);}
 const auto&scale=(gate&&!gate_scal_.empty())?gate_scal_:scal_;
 const auto& coeff=(gate&&!gate_radial_.empty())?gate_radial_:m.radial_coeffs;
 auto&rv=rv_scratch_;auto&rd=rd_scratch_;rv.resize(m.radial_funcs);rd.resize(m.radial_funcs);
 const std::size_t cache_slot=2*edge_index+std::size_t(gate);
 const bool cache_admitted=cache_slot<step_fixed_ready_.size();
 const bool cached=cache_admitted && step_fixed_ready_[cache_slot];
 const bool radial_derivatives=Derivatives || cache_admitted;
 double* cached_values=cache_admitted?step_fixed_radial_.data()+cache_slot*2*std::size_t(m.radial_funcs):nullptr;
 if(cached){
  std::copy(cached_values,cached_values+m.radial_funcs,rv.begin());
  if(Derivatives)std::copy(cached_values+m.radial_funcs,cached_values+2*m.radial_funcs,rd.begin());
 }else{
 if(active_L<L){std::fill(rv.begin(),rv.end(),0.);std::fill(rd.begin(),rd.end(),0.);}
 if(!Frozen||!radial_.eval_frozen(r,pair,gate,rv,rd,metadata?&metadata->interpolation:nullptr,radial_derivatives))for(int k=0;k<m.kmax;++k)for(int l=0;l<active_L;++l) {
  const int mu=k*L+l;
  int sb=scaling_mode_==2?k:(scaling_mode_==1?l:(scaling_mode_==0?0:mu));
  int sc=2*sb*C*C+pair;
  double slope=at(scale,sc,1),shift=at(scale,sc+C*C,0);
  // K/pair maps share exactly the same warp and basis k across all l.
  if(l==0||(l==1&&m.angular_max_dist!=m.max_dist)||scaling_mode_==1||scaling_mode_==3)radial_.eval(r,slope,shift,k,rb,dr,radial_derivatives,l>0);
  int stride=(gate&&!gate_radial_.empty())?R:R+C;int off=mu*stride;
  // Native radial coefficients are shared across center species.
  double a=0,da=0;for(int q=0;q<R;++q){a+=at(coeff,off+q)*rb[q];if(radial_derivatives)da+=at(coeff,off+q)*dr[q];}
  rv[mu]=a;rd[mu]=da;
 }
 if(cache_admitted){
  std::copy(rv.begin(),rv.end(),cached_values);
  std::copy(rd.begin(),rd.end(),cached_values+m.radial_funcs);
  step_fixed_ready_[cache_slot]=1;
 }
 }
 if(Materialize)for(int b:*active_basic_){int mu=basic_mu_[b],sh=basic_sh_[b];v[b]=rv[mu]*edge_y_[sh];if(Derivatives)for(int q=0;q<3;++q)d[3*b+q]=rd[mu]*e.d[q]/r*edge_y_[sh]+rv[mu]*edge_dy_[3*sh+q];}
 return r;
}
void Kernel::contract(double* v,bool gate,bool reverse)const {
 if(!gate&&(pure_||adaptive_mode_))return;
 const auto& rows=gate?gate_contraction_:outer_contraction_;
 for(const auto&r:rows){if(reverse)v[r.src]+=r.coefficient*v[r.dst];else v[r.dst]+=r.coefficient*v[r.src];}
}
template<bool Frozen,bool Derivatives>
void Kernel::edge_mixer(const Edge&e,std::vector<double>&v,std::vector<double>&d,std::vector<double>&s){
 const Model&m=model_;int L=m.lmax+1,H=L*L,O=outer_rank_,F=m.radial_funcs,R=m.radial_size,C=m.species_count;
 v.assign(O*H,0);if(Derivatives){d.assign(3*O*H,0);s.assign(O*H*F,0);}auto&y=y_scratch_;auto&dy=dy_scratch_;auto&rb=rb_scratch_;auto&dr=dr_scratch_;
 EdgeMetadata*metadata=Frozen?edge_metadata(e):nullptr;double r;
 if(metadata)r=metadata->distance;else {double r2=0;for(double a:e.d)r2+=a*a;r=std::sqrt(r2);}
 if(r>=m.max_dist)return;
 const int active_L=r>=m.angular_max_dist?1:L;harmonics<Derivatives>(e.d,active_L-1,y,dy);
 for(int l=0;l<active_L;++l)for(int k=0;k<m.kmax;++k){int mu=k*L+l;
  int sb=scaling_mode_==2?k:(scaling_mode_==1?l:(scaling_mode_==0?0:mu));int sc=2*sb*C*C+types_[e.i]*C+types_[e.j];
  radial_.eval(r,at(scal_,sc,1),at(scal_,sc+C*C),k,rb,dr,Derivatives,l>0);
  double c=at(m.radial_coeffs,mu*(R+C)+R+types_[e.i]),z=at(m.radial_coeffs,mu*(R+C)+R+types_[e.j]);
  double a=at(gate_add_,types_[e.j]*F+mu,at(gate_add_,types_[e.j],1)),signal=c*(z+(gated_?a*gate_[e.j*F+mu]:0));
  for(int o=0;o<O;++o){double w=0,dw=0;int base=((l*O+o)*m.kmax+k)*R;for(int b=0;b<R;++b){double q=mixer_.at(base+b);w+=q*rb[b];if(Derivatives)dw+=q*dr[b];}
   for(int mm=-l;mm<=l;++mm){int sh=h(l,mm),dst=O*(H-(l+1)*(l+1))+(O-1-o)*(2*l+1)+mm+l;v[dst]+=signal*w*y[sh];if(Derivatives){s[dst*F+mu]+=c*a*w*y[sh];for(int q=0;q<3;++q)d[dst*3+q]+=signal*(dw*e.d[q]/r*y[sh]+w*dy[3*sh+q]);}}
  }
 }
}
void Kernel::graph(double* v,bool gate,bool reverse,const double* values)const {
 int lo=gate?gate_product_begin_:outer_product_;
 int hi=gate?gate_product_end_:int(model_.products.size());
 for(int q=0;q<hi-lo;++q){const auto&p=model_.products[reverse?hi-1-q:lo+q];
  if(reverse){double z=p.c*v[p.out];v[p.a]+=z*values[p.b];v[p.b]+=z*values[p.a];}else v[p.out]+=p.c*v[p.a]*v[p.b]; }
}
double Kernel::gate_weight(int mu,int q)const {
 if(!body_combo_)return weights_.at(mu*gate_nodes_.size()+q);
 return weights_.at(q)*body_mix_.at(mu*body_count_+body_.at(q)-2);
}
template<bool Gate>
void Kernel::prepare_edge_coefficients(const Edge&e){
 const Model&m=model_;const int F=m.radial_funcs,R=m.radial_size,C=m.species_count,ti=types_[e.i],tj=types_[e.j];
 edge_coeff_scratch_.resize(F);
 if(effect_source_cache_ && (Gate || effect_source_ready_[e.j])){
  const std::size_t center=std::size_t(ti)*F,source=std::size_t(tj)*F;
  for(int mu=0;mu<F;++mu){auto&x=edge_coeff_scratch_[mu];
   if(Gate){x[0]=effect_gate_type_[center+mu];x[1]=effect_gate_type_[source+mu];}
   else {x[0]=effect_outer_type_[center+mu];x[2]=effect_gate_add_[source+mu];
    // Outer consumers need x0/x2/x3; x1 is only used by the uncached formula.
    x[3]=effect_source_signal_[std::size_t(e.j)*F+mu];}
  }
  return;
 }
 for(int mu=0;mu<F;++mu){auto&x=edge_coeff_scratch_[mu];
  if(Gate){x[0]=at(gate_type_,ti*F+mu,at(gate_type_,ti,1));x[1]=at(gate_type_,tj*F+mu,at(gate_type_,tj,1));}
  else {x[0]=at(m.radial_coeffs,mu*(R+C)+R+ti);x[1]=at(m.radial_coeffs,mu*(R+C)+R+tj);x[2]=at(gate_add_,tj*F+mu,at(gate_add_,tj,1));
   // Only this already-parenthesized inner signal is hoisted. In particular,
   // adj*c*a*v and adj*c*signal are not reassociated or premultiplied.
   x[3]=x[1]+(gated_?x[2]*gate_[e.j*F+mu]:0);}}
}
template<bool Frozen,bool Gate,bool Stream>
void Kernel::forward_moments(std::vector<double>& moments){
 const int M=model_.moments;
 auto&v=v_scratch_;auto&d=d_scratch_;
 for(const auto&e:edges_){
  edge_basis<Frozen,false,!Stream>(e,Gate,v,d);
  prepare_edge_coefficients<Gate>(e);
  // The active set is empty outside the outer cutoff. Scratch values outside
  // that set may be stale and must never be read or accumulated.
  for(int b:*active_basic_){
   const int mu=basic_mu_[b];const auto&x=edge_coeff_scratch_[mu];
   const double basis=Stream?rv_scratch_[mu]*edge_y_[basic_sh_[b]]:v[b];
   moments[e.i*M+b]+=x[0]*x[Gate?1:3]*basis;
  }
 }
}
template<bool Frozen,bool Stream,bool Gate>
void Kernel::coordinate_reverse(const std::vector<double>&moment_adj,Result&result){
 const Model&m=model_;const int M=m.moments,F=m.radial_funcs;
 auto&v=v_scratch_;auto&d=d_scratch_;
 for(const auto&e:edges_){
  const double r=edge_basis<Frozen,true,!Stream>(e,Gate,v,d);
  prepare_edge_coefficients<Gate>(e);
  const bool within=r<m.max_dist;std::array<double,3>g{{0,0,0}};
  for(int b:*active_basic_){const int mu=basic_mu_[b],sh=basic_sh_[b];const auto&x=edge_coeff_scratch_[mu];
   double adj;
   if(Gate)adj=moment_adj[e.i*M+b]*x[0]*x[1];
   else {
    const double basis=Stream?(within?rv_scratch_[mu]*edge_y_[sh]:0.):v[b];
    if(gated_)gate_adj_[e.j*F+mu]+=moment_adj[e.i*M+b]*x[0]*x[2]*basis;
    adj=moment_adj[e.i*M+b]*x[0]*x[3];
   }
   for(int q=0;q<3;++q){
    // Keep each original gradient expression and the b/q accumulation order.
    // Outside cutoff, materialized edge_basis supplied positive-zero entries.
    const double gradient=Stream?(within?rd_scratch_[mu]*e.d[q]/r*edge_y_[sh]+rv_scratch_[mu]*edge_dy_[3*sh+q]:0.):d[3*b+q];
    g[q]+=adj*gradient;
   }
  }
  add_force(result,e,g);
 }
}
void Kernel::begin(int n,int all,const std::vector<int>&t,const std::vector<Edge>&e){
 if(frozen_enabled_)begin_impl<true>(n,all,t,e);else begin_impl<false>(n,all,t,e);
}
template<bool Frozen>
void Kernel::begin_impl(int n,int all,const std::vector<int>& t,const std::vector<Edge>& e){
 const Model&m=model_;int M=m.moments,F=m.radial_funcs,W=feature_width_;
 if(n<0||all<n||t.size()!=static_cast<std::size_t>(all))throw std::runtime_error("SUS2 embedded: invalid atom layout");
 // Graph and communication offsets below intentionally use LAMMPS int indices.
 // Check before multiplying or allocating; a huge layout must fail clearly,
 // not wrap into a small allocation. Empty ranks remain legal.
 const auto limit=static_cast<std::size_t>(std::numeric_limits<int>::max());
 if(static_cast<std::size_t>(n)*M>limit || static_cast<std::size_t>(all)*W>limit || (adaptive_projected_comm_ && static_cast<std::size_t>(all)*adaptive_->projection_width()>limit))
  throw std::runtime_error("SUS2 embedded: atom graph/communication layout exceeds int index capacity; use more MPI ranks");
 owned_=n;all_=all;types_=t;edges_=e;
 edge_y_=nullptr;edge_dy_=nullptr;
 step_angular_.clear();step_angular_ready_.clear();
 if(step_angular_cache_enabled_ && !edges_.empty()){
  const std::size_t width=4*std::size_t(m.lmax+1)*(m.lmax+1);
  const std::size_t cached_edges=std::min(edges_.size(),step_angular_budget_/(width*sizeof(double)+1));
  try{
   if(cached_edges*width>step_angular_.capacity())std::vector<double>(cached_edges*width).swap(step_angular_);
   else step_angular_.resize(cached_edges*width);
   // Replace rather than geometrically grow ready storage: the whole cache,
   // including this bitmap, must remain within its configured byte budget.
   if(cached_edges>step_angular_ready_.capacity())std::vector<unsigned char>(cached_edges,0).swap(step_angular_ready_);
   else step_angular_ready_.assign(cached_edges,0);
  }catch(const std::bad_alloc&){
   std::vector<double>().swap(step_angular_);
   std::vector<unsigned char>().swap(step_angular_ready_);
  }
 }
 if(effect_source_cache_){
  effect_source_signal_.resize(std::size_t(all)*F);
  effect_source_ready_.assign(all,0);
 }
 step_fixed_radial_.clear();
 step_adaptive_radial_.clear();
 step_fixed_ready_.clear();
 step_adaptive_width_=adaptive_?adaptive_->cached_radial_width():0;
 if(step_radial_cache_enabled_ && !edges_.empty()){
  // Retain complete slots for a bounded prefix; all other edges evaluate normally.
  const std::size_t fixed_width=4*std::size_t(F);
  const std::size_t bytes_per_edge=(fixed_width+step_adaptive_width_)*sizeof(double)+2;
  const std::size_t cached_edges=std::min(edges_.size(),step_radial_budget_/bytes_per_edge);
  const auto resize_exact=[](std::vector<double>& values,std::size_t size){
   if(size>values.capacity())std::vector<double>(size).swap(values);
   else values.resize(size);
  };
  try{
   resize_exact(step_fixed_radial_,cached_edges*fixed_width);
   resize_exact(step_adaptive_radial_,cached_edges*step_adaptive_width_);
   step_fixed_ready_.assign(2*cached_edges,0);
  }catch(const std::bad_alloc&){
   std::vector<double>().swap(step_fixed_radial_);
   std::vector<double>().swap(step_adaptive_radial_);
   std::vector<unsigned char>().swap(step_fixed_ready_);
  }
 }
 for(int a:t)if(a<0||a>=m.species_count)throw std::runtime_error("SUS2 embedded: atom species index out of range");
 // This separate 64 MiB cap is not the configured radial knot-storage budget.
 // Bound metadata independently of model shape and table capacity. Uncached
 // edges use precisely the existing evaluation path, in the same edge order.
 const std::size_t metadata_cap=64*1024*1024;
 const std::size_t metadata_count=Frozen&&metadata_cache_enabled_?std::min(edges_.size(),metadata_cap/sizeof(EdgeMetadata)):0;
 try {
  // Explicit exact-size replacement avoids vector's geometric growth taking
  // reserved storage above the metadata cap when a neighbor list grows.
  if(metadata_count>edge_metadata_.capacity())std::vector<EdgeMetadata>(metadata_count).swap(edge_metadata_);
  else edge_metadata_.resize(metadata_count);
 }
 catch(const std::bad_alloc&){edge_metadata_.clear();}
 for(std::size_t k=0;k<edges_.size();++k){const auto&x=edges_[k];double r2=0;for(double a:x.d)r2+=a*a;if(x.i<0||x.i>=n||x.j<0||x.j>=all||!(r2>0)||!std::isfinite(r2))throw std::runtime_error("distributed EFS neighborhood contains an invalid source or distance");
  if(Frozen&&k<edge_metadata_.size()){auto&meta=edge_metadata_[k];meta.distance=std::sqrt(r2);meta.pair=t[x.i]*m.species_count+t[x.j];meta.interpolation.shell[0].ready=false;meta.interpolation.shell[1].ready=false;}}
 // outer() initializes each complete adjoint row before seeding/backpropagation.
 if(adaptive_projected_comm_){
  const std::size_t width=adaptive_->projection_width();
  adaptive_projected_.assign(std::size_t(all)*width,0);
  adaptive_projected_adj_.assign(std::size_t(all)*width,0);
 }
 gate_.assign(all*W,0);gate_adj_.assign(all*W,0);outer_moments_.assign(n*M,0);outer_adj_.resize(n*M);
 local_gate_adj_.assign(joint_?n*M:0,0);extra_energy_.assign(n,0);
 // finish() assigns every owned site energy before gate-readout/ZBL additions.
 partial_.energy=0;partial_.virial.fill(0);partial_.force.assign(all,std::array<double,3>{{0,0,0}});partial_.site_energy.resize(n);
 if(!gated_)return;gate_moments_.assign(n*M,0);
 if(effect_forward_stream_)forward_moments<Frozen,true,true>(gate_moments_);
 else forward_moments<Frozen,true,false>(gate_moments_);
 for(int i=0;i<n;++i){double*a=gate_moments_.data()+i*M;contract(a,true);graph(a,true);
  if(adaptive_mode_)for(int q=0;q<W;++q)gate_[i*W+q]=a[feature_nodes_[q]];
  else for(int mu=0;mu<F;++mu)for(std::size_t q=0;q<gate_nodes_.size();++q)gate_[i*F+mu]+=gate_weight(mu,q)*a[gate_nodes_[q]];}
 if(adaptive_projected_comm_){
  const std::size_t width=adaptive_->projection_width();
  for(int i=0;i<n;++i){
   double* projected=adaptive_projected_.data()+std::size_t(i)*width;
   adaptive_->project(gate_.data()+i*W,projected);
   adaptive_->prepare_coefficients(projected,types_[i]);
  }
 }
}
void Kernel::pack_forward(int atom,double* output)const {
 const auto& values=adaptive_projected_comm_?adaptive_projected_:gate_;
 const std::size_t width=communication_width();
 std::copy(values.begin()+atom*width,values.begin()+(atom+1)*width,output);
}
void Kernel::unpack_forward(int atom,const double* input) {
 auto& values=adaptive_projected_comm_?adaptive_projected_:gate_;
 const std::size_t width=communication_width();
 std::copy(input,input+width,values.begin()+atom*width);
}
void Kernel::pack_reverse(int atom,double* output)const {
 const auto& values=adaptive_projected_comm_?adaptive_projected_adj_:gate_adj_;
 const std::size_t width=communication_width();
 std::copy(values.begin()+atom*width,values.begin()+(atom+1)*width,output);
}
void Kernel::add_reverse(int atom,const double* input) {
 auto& values=adaptive_projected_comm_?adaptive_projected_adj_:gate_adj_;
 const std::size_t width=communication_width();
 for(std::size_t q=0;q<width;++q)values[atom*width+q]+=input[q];
}
void Kernel::outer(){if(frozen_enabled_)outer_impl<true>();else outer_impl<false>();}
template<bool Frozen>
void Kernel::outer_impl(){const Model&m=model_;int M=m.moments,F=m.radial_funcs;
 typedef std::chrono::steady_clock Clock;Clock::time_point stamp;
 if(profile_enabled_){outer_profile_.fill(0);stamp=Clock::now();}
 const auto stage=[&](int index){if(profile_enabled_){const auto now=Clock::now();outer_profile_[index]=std::chrono::duration<double>(now-stamp).count();stamp=now;}};
 // Project only receiving slots referenced by this rank's edges. Retain scratch
 // capacity across MD steps, but rebuild values after forward communication.
 const int PW=adaptive_?adaptive_->projection_width():0;
 auto& projected=adaptive_projected_;
 auto& projected_adj=adaptive_projected_adj_;
 auto& active=adaptive_active_;
 if(PW && !adaptive_projected_comm_){
  projected.resize(types_.size()*PW);
  projected_adj.assign(types_.size()*PW,0);
  active.assign(types_.size(),0);
  for(const auto& edge:edges_)active[edge.j]=1;
  for(std::size_t i=0;i<types_.size();++i){
   if(active[i]){adaptive_->project(gate_.data()+i*feature_width_,projected.data()+i*PW);adaptive_->prepare_coefficients(projected.data()+i*PW,types_[i]);}
  }
 }
 if(effect_source_cache_){
  for(const auto&e:edges_)if(!effect_source_ready_[e.j]){
   const std::size_t row=std::size_t(e.j)*F,type=std::size_t(types_[e.j])*F;
   for(int mu=0;mu<F;++mu)
    effect_source_signal_[row+mu]=effect_outer_type_[type+mu]+effect_gate_add_[type+mu]*gate_[row+mu];
   effect_source_ready_[e.j]=1;
  }
 }
 if(!pure_&&!adaptive_mode_){
  if(effect_forward_stream_)forward_moments<Frozen,false,true>(outer_moments_);
  else forward_moments<Frozen,false,false>(outer_moments_);
 }
 auto&v=v_scratch_;auto&d=d_scratch_;auto&s=s_scratch_;
 const auto&species=species_;bool sp=species_readout_;
 for(int i=0;i<owned_;++i)contract(outer_moments_.data()+i*M,false);
 stage(0);
 if(mixing_)for(const auto&e:edges_){edge_mixer<Frozen,false>(e,v,d,s);for(std::size_t b=0;b<v.size();++b)outer_moments_[e.i*M+outer_begin_+b]+=v[b];}
 if(adaptive_)for(const auto&e:edges_){adaptive_->edge(radial_,e.d,types_[e.i],types_[e.j],gate_.data()+e.j*feature_width_,v,d,s,false,nullptr,nullptr,PW?projected.data()+e.j*PW:nullptr,nullptr,
      std::size_t(&e-edges_.data())<step_fixed_ready_.size()/2?step_adaptive_radial_.data()+std::size_t(&e-edges_.data())*step_adaptive_width_:nullptr,false);for(std::size_t b=0;b<v.size();++b)outer_moments_[e.i*M+outer_begin_+b]+=v[b];}
 stage(1);
 for(int i=0;i<owned_;++i){double*row=outer_moments_.data()+i*M;double*a=outer_adj_.data()+i*M;graph(row,false);
  std::fill(a,a+M,0);for(int q=0;q<m.scalar_count;++q)a[m.scalar_mapping[q]]+=(sp?1:at(species,types_[i]))*linear_[(sp?types_[i]*m.scalar_count:0)+q];if(joint_)extra_energy_[i]=joint_->evaluate(types_[i],at(species,types_[i]),row,gate_moments_.data()+i*M,a,local_gate_adj_.data()+i*M);graph(a,false,true,row);contract(a,false,true);}
 stage(2);
 // Outer coordinate and gate adjoints; coordinates are accumulated in finish.
 if(!pure_&&!adaptive_mode_){if(reverse_stream_)coordinate_reverse<Frozen,true,false>(outer_adj_,partial_);else coordinate_reverse<Frozen,false,false>(outer_adj_,partial_);}
 stage(3);
 if(mixing_)for(const auto&e:edges_){edge_mixer<Frozen,true>(e,v,d,s);std::array<double,3>g{{0,0,0}};for(std::size_t b=0;b<v.size();++b){double adj=outer_adj_[e.i*M+outer_begin_+b];if(gated_)for(int mu=0;mu<F;++mu)gate_adj_[e.j*F+mu]+=adj*s[b*F+mu];for(int q=0;q<3;++q)g[q]+=adj*d[3*b+q];}add_force(partial_,e,g);}
 if(adaptive_)for(const auto&e:edges_){adaptive_->edge(radial_,e.d,types_[e.i],types_[e.j],gate_.data()+e.j*feature_width_,v,d,s,true,outer_adj_.data()+e.i*M+outer_begin_,gate_adj_.data()+e.j*feature_width_,PW?projected.data()+e.j*PW:nullptr,PW?projected_adj.data()+e.j*PW:nullptr,
      std::size_t(&e-edges_.data())<step_fixed_ready_.size()/2?step_adaptive_radial_.data()+std::size_t(&e-edges_.data())*step_adaptive_width_:nullptr,true);std::array<double,3>g{{0,0,0}};for(std::size_t b=0;b<v.size();++b){double adj=outer_adj_[e.i*M+outer_begin_+b];for(int q=0;q<3;++q)g[q]+=adj*d[3*b+q];}add_force(partial_,e,g);}
 // Reduce to the original q-width signal before the existing reverse MPI exchange.
 if(PW && !adaptive_projected_comm_){
  for(std::size_t i=0;i<types_.size();++i){
   if(active[i])adaptive_->project_reverse(projected_adj.data()+i*PW,gate_adj_.data()+i*feature_width_);
  }
 }
 stage(4);
}
void Kernel::add_force(Result&r,const Edge&e,const std::array<double,3>&g)const {for(int a=0;a<3;++a){r.force[e.i][a]+=g[a];r.force[e.j][a]-=g[a];}
 r.virial[0]-=e.d[0]*g[0];r.virial[1]-=e.d[1]*g[1];r.virial[2]-=e.d[2]*g[2];r.virial[3]-=e.d[0]*g[1];r.virial[4]-=e.d[0]*g[2];r.virial[5]-=e.d[1]*g[2];}
const Result& Kernel::finish(){return frozen_enabled_?finish_impl<true>():finish_impl<false>();}
template<bool Frozen>
const Result& Kernel::finish_impl(){const Model&m=model_;int M=m.moments,F=m.radial_funcs;Result&out=partial_;
 if(adaptive_projected_comm_){
  const std::size_t width=adaptive_->projection_width();
  for(int i=0;i<owned_;++i)
   adaptive_->project_reverse(adaptive_projected_adj_.data()+i*width,gate_adj_.data()+i*feature_width_);
 }
 const auto&species=species_;const auto&bias=bias_;bool sp=species_readout_;
 for(int i=0;i<owned_;++i){double E=at(species,types_[i])+at(bias,types_[i])+extra_energy_[i];for(int q=0;q<m.scalar_count;++q)E+=(sp?1:at(species,types_[i]))*linear_[(sp?types_[i]*m.scalar_count:0)+q]*outer_moments_[i*M+m.scalar_mapping[q]];out.site_energy[i]=E;out.energy+=E;}
 const auto&gate_readout=gate_readout_;if(gate_readout_enabled_)for(int i=0;i<owned_;++i){double E=0;for(std::size_t q=0;q<gate_nodes_.size();++q)E+=(sp?1:at(species,types_[i]))*gate_readout.at(q)*gate_moments_[i*M+gate_nodes_[q]];out.site_energy[i]+=E;out.energy+=E;}
 if(gated_){gate_moment_adj_.assign(owned_*M,0);auto&adj=gate_moment_adj_;for(int i=0;i<owned_;++i){double*a=adj.data()+i*M;if(joint_)std::copy(local_gate_adj_.begin()+i*M,local_gate_adj_.begin()+(i+1)*M,a);if(adaptive_mode_)for(int q=0;q<feature_width_;++q)a[feature_nodes_[q]]+=gate_adj_[i*feature_width_+q];else for(int mu=0;mu<F;++mu)for(std::size_t q=0;q<gate_nodes_.size();++q)a[gate_nodes_[q]]+=gate_adj_[i*F+mu]*gate_weight(mu,q);if(gate_readout_enabled_)for(std::size_t q=0;q<gate_nodes_.size();++q)a[gate_nodes_[q]]+=(sp?1:at(species,types_[i]))*gate_readout.at(q);graph(a,true,true,gate_moments_.data()+i*M);contract(a,true,true);}
  if(reverse_stream_&&!pure_&&!adaptive_mode_)coordinate_reverse<Frozen,true,true>(adj,out);else coordinate_reverse<Frozen,false,true>(adj,out);}
 if(zbl_)for(const auto&e:edges_){double r2=0;for(double x:e.d)r2+=x*x;double r=std::sqrt(r2);int pair=types_[e.i]*m.species_count+types_[e.j];auto p=sus2_sl_zbl::ComputePairHostCached(zbl_constants_[pair],r,zbl_inner_[pair],zbl_outer_[pair]);double E=.5*p.energy;out.site_energy[e.i]+=E;out.energy+=E;std::array<double,3>g;for(int q=0;q<3;++q)g[q]=.5*p.dEdr*e.d[q]/r;add_force(out,e,g);}
 return out;
}
double Kernel::memory_usage() const {
 const std::vector<double>* arrays[]={&gate_moments_,&gate_,&gate_adj_,&outer_moments_,&outer_adj_,&local_gate_adj_,&extra_energy_,&weights_,&linear_,&mult_,&gate_type_,&gate_add_,&gate_radial_,&scal_,&gate_scal_,&gate_rank_weights_,&outer_rank_weights_,&species_,&bias_,&body_mix_,&mixer_,&y_scratch_,&dy_scratch_,&rb_scratch_,&dr_scratch_,&rv_scratch_,&rd_scratch_,&v_scratch_,&d_scratch_,&s_scratch_,&gate_moment_adj_,&adaptive_projected_,&adaptive_projected_adj_,&partial_.site_energy};
 std::size_t bytes=sizeof(*this)+sizeof(Edge)*edges_.capacity()+sizeof(int)*(types_.capacity()+gate_nodes_.capacity()+body_.capacity()+feature_nodes_.capacity()+basic_mu_.capacity()+basic_sh_.capacity()+basic_all_.capacity()+basic_l0_.capacity())+sizeof(Contraction)*(gate_contraction_.capacity()+outer_contraction_.capacity())+sizeof(std::array<double,3>)*partial_.force.capacity()+radial_.stats().bytes;
 if(joint_)bytes+=joint_->memory_usage();
 bytes+=sizeof(EdgeMetadata)*edge_metadata_.capacity();
 bytes+=sizeof(double)*(step_fixed_radial_.capacity()+step_adaptive_radial_.capacity());
 bytes+=sizeof(unsigned char)*step_fixed_ready_.capacity();
 bytes+=sizeof(double)*step_angular_.capacity()+sizeof(unsigned char)*step_angular_ready_.capacity();
 bytes+=sizeof(unsigned char)*adaptive_active_.capacity();
 bytes+=sizeof(double)*(effect_outer_type_.capacity()+effect_gate_type_.capacity()+effect_gate_add_.capacity()+effect_source_signal_.capacity());
 bytes+=sizeof(unsigned char)*effect_source_ready_.capacity();
 bytes+=sizeof(std::array<double,4>)*edge_coeff_scratch_.capacity();
 for(const auto*v:arrays)bytes+=sizeof(double)*v->capacity();
 // Include the immutable model's principal numeric storage; strings/maps are
 // allocator-dependent and are not part of the MD workspace estimate.
 bytes+=sizeof(double)*model_.radial_coeffs.capacity()+sizeof(Product)*model_.products.capacity()+sizeof(std::array<int,3>)*model_.basic.capacity()+sizeof(int)*model_.scalar_mapping.capacity();
 return double(bytes);
}
} }
