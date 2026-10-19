#include "sus2_sl_model.h"
#include "sus2_sl_zbl.h"
#include <cctype>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
namespace LAMMPS_NS { namespace SUS2SL {
namespace {
std::string trim(const std::string &s) {
  const auto a=s.find_first_not_of(" \t\r\n");
  return a==std::string::npos ? "" : s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
void require(bool ok,const std::string &why) {
  if (!ok) throw std::runtime_error("SUS2 superlinear model: "+why);
}
// All inference kernels use int for their local flattened offsets. Check the
// mathematical extent before evaluating an int multiplication or allocating
// default coefficient arrays. Division avoids overflowing even size_t/uint64.
int index_product(const std::string &name,std::initializer_list<long long> dimensions) {
  const long long limit=std::numeric_limits<int>::max();
  long long result=1;
  for(long long n:dimensions) {
    require(n>=0,"negative index dimension in "+name);
    require(n==0 || result<=limit/n,"int index capacity exceeded in "+name);
    result*=n;
  }
  return static_cast<int>(result);
}
void validate_index_capacity(const Model &m) {
  require(m.species_count>0 && m.radial_size>0 && m.radial_funcs>0 &&
          m.kmax>0 && m.lmax>=0 && m.lmax<=4,"invalid species or radial dimensions");
  const int C=m.species_count,R=m.radial_size,F=m.radial_funcs,L=m.lmax+1,H=L*L;
  const bool ranked=m.boolean("outer_k_rank1_enabled"),gated=m.boolean("two_layer_gate_enabled");
  const int O=ranked?m.integer("outer_k_rank",1):0,G=m.integer("two_layer_gate_k_rank");
  const int Q=m.integer("two_layer_gate_scalar_count",0),tl=m.integer("tensor_message_lmax");
  require(O>=0 && G>=0 && Q>=0 && tl>=0 && tl<=2 && tl<=m.lmax,"invalid rank or feature dimensions");
  const int body_max=m.integer("two_layer_gate_body_order_max");
  require(body_max>=0 && body_max<=6 && (!gated || body_max>=2),"invalid gate body order");
  const auto outer=m.text("outer_radial_mode","legacy-plus-mixer");
  if(outer=="adaptive-coefficients") {
    const bool shared=m.text("adaptive_radial_projection","full")=="shared-n";
    const int latent=shared?m.integer("adaptive_radial_shared_n_rank",1):1;
    require(O>0 && latent>0,"invalid adaptive rank");
    const int rows=index_product("Adaptive rows",{L,O,R});
    const int projection=shared?index_product("Adaptive projection rows",{L,O,latent}):rows;
    index_product("Adaptive projection weights",{projection,Q});
    index_product("Adaptive source amplitudes",{m.text("adaptive_radial_amplitude_mode","shared")=="species"?C:1,rows,latent});
    const int tensor_features=index_product("Adaptive tensor features",{G,(tl+1)*(tl+1)-1});
    const long long width=static_cast<long long>(Q)+tensor_features;
    require(width<=std::numeric_limits<int>::max(),"int index capacity exceeded in Adaptive signal width");
    index_product("Adaptive primitive signal Jacobian",{O,H,width});
    index_product("Adaptive tensor coefficients",{tl,O,G});
  }
  const long long stride=static_cast<long long>(R)+C;
  index_product("radial coefficient row stride",{stride});
  index_product("radial coefficients",{F,stride});
  const auto sm=m.text("scaling_map","LK");
  const int blocks=sm=="pair"?1:sm=="L"?L:sm=="K"?m.kmax:F;
  index_product("scaling coefficients",{2,C,C,blocks});
  const int heads=m.text("readout_mode","shared")=="species"?C:1;
  index_product("scalar readout",{heads,m.scalar_count});
  index_product("primitive coordinate Jacobian",{3,m.integer("alpha_index_basic_count")});
  if(ranked) {
    index_product("outer primitives",{O,H});
    index_product("outer primitive coordinate Jacobian",{3,O,H});
    if(m.integer("outer_k_rank1_weight_count")>0)
      index_product("outer contraction weights",{L,O,m.kmax});
  }
  if(G>0) {
    index_product("gate primitives",{G,H});
    index_product("gate contraction weights",{L,G,m.kmax});
  }
  if(gated) {
    index_product("gate radial coefficients",{F,R});
    index_product("gate type coefficients",{C,F});
    if(outer!="adaptive-coefficients") {
      if(m.text("two_layer_gate_mode")=="mu-body-linear-combo")
        index_product("gate body mix weights",{F,body_max-1});
      else index_product("gate scalar weights",{F,Q});
    }
  }
  if(m.text("two_layer_gate_radial_mixing","none")=="full") {
    index_product("radial mixer coefficients",{L,O,m.kmax,R});
    index_product("radial mixer signal Jacobian",{O,H,F});
  }
  if(m.boolean("joint_readout_enabled"))
    index_product("joint scalar readout",{heads,m.integer("joint_readout_scalar_count")});
}
int exact_int(double x,const std::string &key) {
  require(std::isfinite(x) && x>=std::numeric_limits<int>::min() &&
          x<=std::numeric_limits<int>::max() && std::floor(x)==x,"noninteger "+key);
  return static_cast<int>(x);
}
std::string list_string(const std::vector<double> &v) {
  std::ostringstream out; out<<std::setprecision(17)<<"{";
  for (size_t i=0;i<v.size();++i) out<<(i ? "," : "")<<v[i];
  return out.str()+"}";
}
// Recursive list parsing retains strict separators and balanced braces.
void list_values(const char *&p,std::vector<double> &out,const std::string &key) {
  while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
  require(*p=='{',"expected list in "+key); ++p;
  while (true) {
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (*p=='}') { ++p; return; }
    if (*p=='{') list_values(p,out,key);
    else {
      char *end=nullptr; double x=std::strtod(p,&end);
      require(end!=p && std::isfinite(x),"invalid finite number in "+key);
      out.push_back(x); p=end;
    }
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (*p=='}') { ++p; return; }
    require(*p==',',"missing list separator in "+key); ++p;
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    require(*p!='}',"trailing list separator in "+key);
  }
}
}
bool Model::has(const std::string &key) const { return fields.count(key)!=0; }
double Model::cutoff() const {
  if(!boolean("zbl_enabled")) return max_dist;
  const auto z=integers("zbl_atomic_numbers");
  return std::max(max_dist,sus2_sl_zbl::MaxPairOuterCutoff(species_count,z.data(),
      real("zbl_outer",1.4),boolean("zbl_typewise_cutoff_enabled"),real("zbl_typewise_cutoff_factor",.7)));
}
std::string Model::text(const std::string &key,const std::string &fallback) const {
  const auto it=fields.find(key); return it==fields.end() ? fallback : it->second;
}
double Model::real(const std::string &key,double fallback) const {
  if (!has(key)) return fallback;
  const std::string s=text(key); char *end=nullptr;
  const double v=std::strtod(s.c_str(),&end);
  require(end!=s.c_str() && trim(end).empty() && std::isfinite(v),"invalid number "+key);
  return v;
}
int Model::integer(const std::string &key,int fallback) const {
  return has(key) ? exact_int(real(key),key) : fallback;
}
bool Model::boolean(const std::string &key,bool fallback) const {
  if (!has(key)) return fallback;
  const auto s=text(key);
  require(s=="true" || s=="false" || s=="0" || s=="1","invalid boolean "+key);
  return s=="true" || s=="1";
}
std::vector<double> Model::numbers(const std::string &key) const {
  if (!has(key)) return {};
  const auto s=text(key); const char *p=s.c_str(); std::vector<double> result;
  list_values(p,result,key); require(trim(p).empty(),"trailing list text in "+key);
  return result;
}
std::vector<int> Model::integers(const std::string &key) const {
  std::vector<int> result;
  for (double x:numbers(key)) result.push_back(exact_int(x,key));
  return result;
}
std::vector<std::vector<double> > Model::rows(const std::string &key,int width) const {
  require(width>0,"invalid row width"); const auto v=numbers(key);
  require(v.size()%width==0,"invalid row width in "+key);
  std::vector<std::vector<double> > result;
  for(size_t i=0;i<v.size();i+=width) result.emplace_back(v.begin()+i,v.begin()+i+width);
  return result;
}
void Model::load(const std::string &path) {
  *this=Model(); std::ifstream in(path.c_str()); require(bool(in),"cannot open "+path);
  std::string line,radial_text; bool radial=false,header=false; int depth=0;
  while (std::getline(in,line)) {
    line=trim(line); if (line.empty() || line[0]=='#') continue;
    if (!header) { require(line=="MTP","missing MTP header"); header=true; continue; }
    const size_t eq=line.find('=');
    if (eq==std::string::npos) {
      if (line=="radial_coeffs") { require(!radial && radial_text.empty(),"duplicate radial_coeffs"); radial=true; continue; }
      require(radial,"unrecognized line: "+line);
      if (line=="0-0") continue;
      require(line[0]=='{',"invalid radial coefficient block");
      if (!radial_text.empty()) radial_text+=",";
      radial_text+=line; continue;
    }
    radial=false;
    const std::string key=trim(line.substr(0,eq)); std::string value=trim(line.substr(eq+1));
    require(!key.empty() && !has(key),"duplicate or empty field "+key);
    depth=0; for(char c:value) depth+=(c=='{')-(c=='}');
    while(depth>0 && std::getline(in,line)) {
      value+=" "+trim(line); for(char c:line) depth+=(c=='{')-(c=='}');
    }
    require(depth==0,"unbalanced braces in "+key); fields[key]=value;
  }
  require(header,"empty model");
  species_count=integer("species_count"); lmax=integer("sh_l_max"); kmax=integer("sh_k_max");
  radial_size=integer("radial_basis_size"); radial_funcs=integer("radial_funcs_count");
  moments=integer("alpha_moments_count"); scalar_count=integer("alpha_scalar_moments");
  min_dist=real("min_dist"); max_dist=real("max_dist"); scaling=real("scaling",1);
  validate_index_capacity(*this);
  const int radial_count=index_product("radial coefficients",{radial_funcs,static_cast<long long>(radial_size)+species_count});
  if(!radial_text.empty()) { fields["radial_coeffs"]="{"+radial_text+"}"; radial_coeffs=numbers("radial_coeffs"); }
  else {
    radial_coeffs.assign(static_cast<size_t>(radial_count),.01);
    for(int mu=0;mu<radial_funcs;++mu) for(int t=0;t<species_count;++t)
      radial_coeffs[static_cast<size_t>(mu)*(radial_size+species_count)+radial_size+t]=2.;
    fields["radial_coeffs"]=list_string(radial_coeffs);
  }
  if(!has("shift_coeffs")) fields["shift_coeffs"]=list_string(std::vector<double>(species_count,-1.));
  const std::string sm=text("scaling_map","LK");
  int blocks=sm=="pair" ? 1 : sm=="L" ? lmax+1 : sm=="K" ? kmax : radial_funcs;
  require(blocks>0,"invalid scaling block count");
  index_product("scaling coefficients",{2,species_count,species_count,blocks});
  if(!has("scal_coeffs")) {
    const int pairs=species_count*species_count; std::vector<double> v(2*pairs*blocks);
    const double start=text("radial_basis_type").find("RBLaguerre_log1p")==0 ? 3. : 4.;
    for(int b=0;b<blocks;++b) for(int t=0;t<species_count;++t) for(int u=0;u<species_count;++u) {
      const double x=species_count==1 ? .5 : (std::sqrt(t+1.)+std::sqrt(u+1.)-2.)/(2.*std::sqrt(double(species_count))-2.);
      v[2*b*pairs+t*species_count+u]=start+(2.-start)*x;
      v[(2*b+1)*pairs+t*species_count+u]=1.5+x;
    }
    fields["scal_coeffs"]=list_string(v);
  }
  index_product("scalar readout",{scalar_count,text("readout_mode","shared")=="species"?species_count:1});
  if(!has("species_coeffs")) {
    require(!has("moment_coeffs"),"moment_coeffs without species_coeffs");
    fields["species_coeffs"]=list_string(std::vector<double>(species_count,.001));
    fields["moment_coeffs"]=list_string(std::vector<double>(scalar_count*(text("readout_mode","shared")=="species" ? species_count : 1),text("readout_mode")=="species" ? 1e-6 : .001));
  }
  for(const auto &r:rows("alpha_index_basic",3)) basic.push_back({{exact_int(r[0],"basic k"),exact_int(r[1],"basic l"),exact_int(r[2],"basic m")}});
  for(const auto &r:rows("sh_products",4)) products.push_back({exact_int(r[0],"product a"),exact_int(r[1],"product b"),exact_int(r[2],"product out"),r[3]});
  scalar_mapping=integers("alpha_moment_mapping");
  if(boolean("two_layer_gate_enabled") && !has("two_layer_gate_scalar_count"))
    fields["two_layer_gate_scalar_count"]=std::to_string(numbers("two_layer_gate_scalar_indices").size());
  validate();
}
void Model::validate() const {
  validate_index_capacity(*this);
  const auto ver=text("version");
  require(ver=="1.1.0" || ver=="1.2.0" || ver=="1.3.0" || ver=="1.4.0" || ver=="1.5.0" || ver=="1.6.0" || ver=="1.7.0" || ver=="1.8.0","unsupported version "+ver);
  const std::set<std::string> adaptive_keys={"adaptive_radial_projection","adaptive_radial_amplitude_mode","adaptive_radial_shared_n_rank","adaptive_radial_activation","adaptive_radial_amplitude_count","adaptive_radial_amplitudes","adaptive_radial_weight_count","adaptive_radial_weights","adaptive_radial_amplitude_optimizer_scales","adaptive_radial_weight_optimizer_scales"};
  const std::set<std::string> tensor_keys={"tensor_message_lmax","tensor_message_coeff_count","tensor_message_coeffs","tensor_message_optimizer_scales"};
  const std::set<std::string> joint_keys={"joint_readout_enabled","joint_readout_l_max","joint_readout_body_order","joint_readout_body_l_max","joint_readout_alpha_moments_count","joint_readout_alpha_index_basic_count","joint_readout_alpha_index_basic","joint_readout_sh_product_count","joint_readout_sh_products","joint_readout_scalar_count","joint_readout_scalar_mapping","joint_readout_scalar_info_count","joint_readout_scalar_info","joint_readout_coeff_count","joint_readout_coeffs","joint_readout_optimizer_scales"};
  for(const auto &f:fields) {
    if(f.first.compare(0,16,"adaptive_radial_")==0) require(adaptive_keys.count(f.first),"unknown adaptive metadata "+f.first);
    if(f.first.compare(0,15,"tensor_message_")==0) require(tensor_keys.count(f.first),"unknown tensor metadata "+f.first);
    if(f.first.compare(0,14,"joint_readout_")==0) require(joint_keys.count(f.first),"unknown joint metadata "+f.first);
    if(ver=="1.8.0") require(!adaptive_keys.count(f.first) && !tensor_keys.count(f.first) && !joint_keys.count(f.first),"version 1.8.0 is reserved for Legacy scalar models");
  }
  if(ver=="1.5.0" || ver=="1.6.0" || ver=="1.7.0") {
    for(const char *key:{"readout_mode","tensor_message_lmax","adaptive_radial_projection","adaptive_radial_amplitude_mode","adaptive_radial_amplitude_count","adaptive_radial_amplitudes","adaptive_radial_weight_count","adaptive_radial_weights"}) require(has(key),"missing required field "+std::string(key));
  }
  if(ver=="1.7.0") for(const char *key:{"adaptive_radial_shared_n_rank","adaptive_radial_activation","joint_readout_enabled"}) require(has(key),"missing required v1.7 field "+std::string(key));
  require(text("potential_tag")=="SUS2-SH","requires a SUS2-SH spherical harmonic model");
  require(lmax>=0 && lmax<=4 && kmax>0 && integer("L")==lmax && static_cast<long long>(kmax)*(lmax+1)==radial_funcs,"inconsistent L/K dimensions");
  require(integer("sh_body_order")>=2 && integer("sh_body_order")<=6 && text("sh_parity")=="even","unsupported body order or parity");
  require(max_dist>min_dist && min_dist>=0 && scaling>0,"invalid distance or scaling");
  const auto sm=text("scaling_map","LK"); require(sm=="L" || sm=="K" || sm=="LK" || sm=="pair","unknown scaling_map");
  auto count=[&](const std::string &key,size_t n) { require(numbers(key).size()==n,"wrong coefficient count in "+key); };
  count("shift_coeffs",species_count); count("species_coeffs",species_count);
  count("scal_coeffs",static_cast<size_t>(2)*species_count*species_count*(sm=="pair" ? 1 : sm=="L" ? lmax+1 : sm=="K" ? kmax : radial_funcs));
  require(radial_coeffs.size()==static_cast<size_t>(radial_funcs)*(radial_size+species_count),"wrong radial coefficient count");
  require(moments>0 && basic.size()==static_cast<size_t>(integer("alpha_index_basic_count")) && moments>=static_cast<int>(basic.size()),"invalid basic graph dimensions");
  require(products.size()==static_cast<size_t>(integer("sh_product_count")),"wrong sh_product_count");
  require(integer("alpha_index_times_count")==0 && numbers("alpha_index_times").empty(),"Cartesian product graph is unsupported");
  for(const auto &b:basic) require(b[0]>=0 && b[0]<kmax && b[1]>=0 && b[1]<=lmax && b[2]>=-b[1] && b[2]<=b[1],"invalid basic (k,l,m)");
  for(const auto &p:products) require(p.a>=0 && p.b>=0 && p.out>=0 && p.a<p.out && p.b<p.out && p.out<moments,"invalid or cyclic product graph");
  require(scalar_mapping.size()==static_cast<size_t>(scalar_count),"wrong scalar mapping count");
  for(int i:scalar_mapping) require(i>=0 && i<moments,"scalar mapping out of range");
  for(int i:integers("sh_body_l_max")) require(i>=0 && i<=lmax,"body angular limit out of range");
  require(!boolean("two_layer_residual_enabled") && integer("multi_layer_gate_layers",1)<=1 && !boolean("multi_layer_gate_enabled"),"old residual/multilayer models are unsupported");
  const auto outer=text("outer_radial_mode","legacy-plus-mixer");
  require(outer=="legacy-plus-mixer" || outer=="pure-matrix" || outer=="adaptive-coefficients","unsupported outer_radial_mode");
  const auto radial_mode=text("two_layer_gate_radial_mode","base-radial");
  require(radial_mode=="shared-radial" || radial_mode=="base-radial" || radial_mode=="legacy","unknown two_layer_gate_radial_mode");
  // Native files omit the mixer section when inactive; an explicit section
  // only admits the current full/direct-k layout.
  if(has("two_layer_gate_radial_mixing")) {
    require(text("two_layer_gate_radial_mixing")=="full","unknown two_layer_gate_radial_mixing");
    require(text("two_layer_gate_radial_mixing_layout")=="direct-k","unsupported two_layer_gate_radial_mixing_layout");
  }
  const auto channel=text("outer_channel_mode","cross-channel"); require(channel=="cross-channel" || channel=="same-channel","invalid outer_channel_mode");
  const auto readout=text("readout_mode","shared"); require(readout=="shared" || readout=="species","invalid readout_mode");
  count("moment_coeffs",static_cast<size_t>(scalar_count)*(readout=="species" ? species_count : 1));
  if(has("sh_scalar_info")) {
    const auto info=rows("sh_scalar_info",7);
    require(info.size()==static_cast<size_t>(integer("sh_scalar_info_count")) && info.size()==static_cast<size_t>(scalar_count),"inconsistent scalar body metadata");
    for(const auto &r:info) for(double x:r) exact_int(x,"sh_scalar_info");
  }
  if(boolean("outer_k_rank1_enabled")) {
    const int ro=integer("outer_k_rank",1), begin=integer("outer_k_rank1_basic_begin"), n=integer("outer_k_rank1_basic_count"), p=integer("outer_k_rank1_product_begin");
    require(ro>0 && (outer!="legacy-plus-mixer" || ro<=kmax),"invalid outer rank");
    require(n==static_cast<long long>(ro)*(lmax+1)*(lmax+1) && begin>=static_cast<int>(basic.size()) && static_cast<long long>(begin)+n<=moments && p>=0 && p<=static_cast<int>(products.size()),"invalid outer rank graph segment");
    const size_t nw=numbers("outer_k_rank1_weights").size();
    require(nw==0 || nw==static_cast<size_t>(lmax+1)*ro*kmax,"invalid outer contraction weights");
  } else require(outer=="legacy-plus-mixer","outer radial mode requires ranked graph");
  if(integer("two_layer_gate_k_rank")>0) {
    const int rg=integer("two_layer_gate_k_rank"), begin=integer("two_layer_gate_k_rank_basic_begin"), n=integer("two_layer_gate_k_rank_basic_count"), p=integer("two_layer_gate_k_rank_product_begin"), end=integer("two_layer_gate_k_rank_product_end");
    require(rg<=kmax && n==static_cast<long long>(rg)*(lmax+1)*(lmax+1) && begin>=static_cast<int>(basic.size()) && static_cast<long long>(begin)+n<=moments && p>=0 && end>=p && end<=static_cast<int>(products.size()),"invalid gate rank graph segment");
    count("two_layer_gate_k_rank_weights",static_cast<size_t>(lmax+1)*rg*kmax);
  }
  // Validate every persisted count/list pair, including optional inference blocks.
  const char *pairs[][2]={{"outer_k_rank1_weight_count","outer_k_rank1_weights"},{"two_layer_gate_k_rank_weight_count","two_layer_gate_k_rank_weights"},{"two_layer_gate_radial_coeff_count","two_layer_gate_radial_coeffs"},{"two_layer_gate_type_coeff_count","two_layer_gate_type_coeffs"},{"two_layer_gate_weight_count","two_layer_gate_weights"},{"two_layer_gate_additive_coeff_count","two_layer_gate_additive_coeffs"},{"two_layer_gate_body_mix_weight_count","two_layer_gate_body_mix_weights"},{"two_layer_gate_scaling_coeff_count","two_layer_gate_scaling_coeffs"},{"two_layer_gate_radial_mixing_coeff_count","two_layer_gate_radial_mixing_coeffs"},{"two_layer_gate_readout_coeff_count","two_layer_gate_readout_coeffs"},{"adaptive_radial_amplitude_count","adaptive_radial_amplitudes"},{"adaptive_radial_weight_count","adaptive_radial_weights"},{"tensor_message_coeff_count","tensor_message_coeffs"},{"joint_readout_coeff_count","joint_readout_coeffs"}};
  for(const auto &p:pairs) if(has(p[0]) || has(p[1])) { const int n=integer(p[0],-1); require(n>=0,"missing/negative count "+std::string(p[0])); count(p[1],static_cast<size_t>(n)); }
  if(boolean("two_layer_gate_enabled")) {
    const int q=integer("two_layer_gate_scalar_count",static_cast<int>(numbers("two_layer_gate_scalar_indices").size())); require(q>0,"empty gate scalar basis");
    count("two_layer_gate_scalar_indices",q); count("two_layer_gate_scalar_body_orders",q);
    const auto ids=integers("two_layer_gate_scalar_moment_indices");
    if(has("two_layer_gate_scalar_moment_indices")) { require(ids.size()==static_cast<size_t>(q),"gate moment count mismatch"); for(int x:ids) require(x>=0 && x<moments,"gate moment out of range"); }
    const std::string mode=text("two_layer_gate_mode"), site=text("two_layer_gate_site_mode","neighbor"), prune=text("two_layer_gate_factor_pruning","q-total");
    require(mode=="mu-scalar-full" || mode=="mu-body-order" || mode=="mu-body-linear-combo","unsupported gate mode "+mode);
    require(site=="neighbor" && (prune=="q-total" || prune=="same-k"),"unsupported gate site/pruning mode");
    const auto gs=text("two_layer_gate_scaling_mode","shared-scaling"); require(gs=="shared-scaling" || gs=="legacy" || gs=="decoupled","invalid gate scaling mode");
    if(gs=="decoupled") count("two_layer_gate_scaling_coeffs",numbers("scal_coeffs").size());
    if(radial_mode=="shared-radial") count("two_layer_gate_radial_coeffs",static_cast<size_t>(radial_funcs)*radial_size);
    else if(has("two_layer_gate_radial_mode")) count("two_layer_gate_radial_coeffs",0);
  }
  if(outer=="adaptive-coefficients") {
    require(boolean("two_layer_gate_enabled") && boolean("outer_k_rank1_enabled"),"adaptive model requires gate and outer rank");
    const std::string proj=text("adaptive_radial_projection","full"), amp=text("adaptive_radial_amplitude_mode","shared"), act=text("adaptive_radial_activation","none");
    require(proj=="full" || proj=="shared-n","invalid adaptive projection"); require(amp=="shared" || amp=="species","invalid adaptive amplitude mode"); require(act=="none" || act=="tanh","invalid adaptive activation");
    const int rank=integer("adaptive_radial_shared_n_rank",1), ro=integer("outer_k_rank",1);
    require(rank>0 && (proj=="shared-n" || rank==1) && ro>0,"invalid adaptive rank");
    const size_t rows=static_cast<size_t>(lmax+1)*ro*radial_size, latent=proj=="shared-n" ? rank : 1;
    count("adaptive_radial_amplitudes",rows*latent*(amp=="species" ? species_count : 1));
    count("adaptive_radial_weights",(proj=="shared-n" ? rows/radial_size*rank : rows)*integer("two_layer_gate_scalar_count"));
    if(has("adaptive_radial_amplitude_optimizer_scales")) count("adaptive_radial_amplitude_optimizer_scales",numbers("adaptive_radial_amplitudes").size());
    if(has("adaptive_radial_weight_optimizer_scales")) count("adaptive_radial_weight_optimizer_scales",integer("two_layer_gate_scalar_count"));
  }
  const int tl=integer("tensor_message_lmax"); require(tl>=0 && tl<=2 && tl<=lmax,"invalid tensor message lmax");
  if(tl) { require(outer=="adaptive-coefficients","tensor requires adaptive coefficients"); count("tensor_message_coeffs",static_cast<size_t>(tl)*integer("outer_k_rank",1)*integer("two_layer_gate_k_rank")); }
  if(boolean("joint_readout_enabled")) {
    require(outer=="adaptive-coefficients","joint readout requires adaptive coefficients");
    const int jm=integer("joint_readout_alpha_moments_count"), jb=integer("joint_readout_alpha_index_basic_count"), jq=integer("joint_readout_scalar_count");
    require(jm>=jb && jb>=0 && jq>=0,"invalid joint graph dimensions");
    const auto b=rows("joint_readout_alpha_index_basic",3), p=rows("joint_readout_sh_products",4);
    require(b.size()==static_cast<size_t>(jb) && p.size()==static_cast<size_t>(integer("joint_readout_sh_product_count")),"joint graph count mismatch");
    for(const auto &r:b) { const int k=exact_int(r[0],"joint source"), l=exact_int(r[1],"joint l"), m=exact_int(r[2],"joint m"); require(k>=0 && k<=1 && l>=1 && l<=lmax && m>=-l && m<=l,"invalid joint basic"); }
    for(const auto &r:p) { const int a=exact_int(r[0],"joint a"), bb=exact_int(r[1],"joint b"), out=exact_int(r[2],"joint out"); require(a>=0 && bb>=0 && a<out && bb<out && out<jm,"invalid joint product graph"); }
    count("joint_readout_scalar_mapping",jq);
    for(int x:integers("joint_readout_scalar_mapping")) require(x>=0 && x<jm,"joint scalar out of range");
    count("joint_readout_coeffs",static_cast<size_t>(jq)*(readout=="species" ? species_count : 1));
  }
  if(boolean("zbl_enabled")) {
    const auto z=integers("zbl_atomic_numbers"); require(z.size()==static_cast<size_t>(species_count),"wrong ZBL atomic number count");
    for(int n:z) require(n>0 && n<=94,"invalid ZBL atomic number");
    require(real("zbl_inner")>=0 && real("zbl_outer")>real("zbl_inner") && real("zbl_typewise_cutoff_factor",.7)>0,"invalid ZBL cutoff");
  }
  // Parse all numeric lists once for finite-value and syntax validation.
  for(const auto &f:fields) if(!f.second.empty() && f.second[0]=='{') {
    const auto v=numbers(f.first);
    if(f.first.find("optimizer_scales")!=std::string::npos) for(double x:v) require(x>0,"nonpositive optimizer scale "+f.first);
  }
}
} }
