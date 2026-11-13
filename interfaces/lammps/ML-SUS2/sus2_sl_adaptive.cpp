#include "sus2_sl_adaptive.h"
#include <cmath>
#include <complex>
#include <stdexcept>
namespace LAMMPS_NS { namespace SUS2SL {
namespace {
int harmonic(int l,int m){return l*l+l+m;}
int node(int rank,int H,int l,int a,int m){return rank*(H-(l+1)*(l+1))+(rank-1-a)*(2*l+1)+m+l;}
void harmonics(const std::array<double,3>&x,int L,const std::array<double,25>&norm,std::vector<double>&v,std::vector<double>&d){
 const int H=(L+1)*(L+1);double r2=0;for(double a:x)r2+=a*a;
 if(!(r2>0))throw std::runtime_error("adaptive edge has zero/nonfinite distance");
 typedef std::complex<double>C;std::array<C,25>s{};std::array<C,75>ds{};s[0]=1;C u(x[0],x[1]),du[3]={C(1),C(0,1),C(0)};
 for(int m=1;m<=L;++m){int i=harmonic(m,m),p=harmonic(m-1,m-1);double c=-(2*m-1);s[i]=c*u*s[p];for(int a=0;a<3;++a)ds[3*i+a]=c*(du[a]*s[p]+u*ds[3*p+a]);}
 for(int m=0;m<=L;++m){
  if(m<L){int i=harmonic(m+1,m),p=harmonic(m,m);double c=2*m+1;s[i]=c*x[2]*s[p];for(int a=0;a<3;++a)ds[3*i+a]=c*((a==2?s[p]:C(0))+x[2]*ds[3*p+a]);}
  for(int l=m+2;l<=L;++l){int i=harmonic(l,m),p=harmonic(l-1,m),q=harmonic(l-2,m);double A=2*l-1,B=l+m-1,D=l-m;s[i]=(A*x[2]*s[p]-B*r2*s[q])/D;for(int a=0;a<3;++a)ds[3*i+a]=(A*((a==2?s[p]:C(0))+x[2]*ds[3*p+a])-B*(2*x[a]*s[q]+r2*ds[3*q+a]))/D;}}
 v.assign(H,0);d.assign(3*H,0);double inv_r=1./std::sqrt(r2),rpower=1;
 for(int l=0;l<=L;++l){for(int m=0;m<=l;++m){int i=harmonic(l,m);double n=norm[i]*rpower;C y=n*s[i];v[i]=y.real();if(m)v[harmonic(l,-m)]=y.imag();for(int a=0;a<3;++a){C z=n*(ds[3*i+a]-double(l)*s[i]*x[a]/r2);d[3*i+a]=z.real();if(m)d[3*harmonic(l,-m)+a]=z.imag();}}rpower*=inv_r;}
}
}
Adaptive::Adaptive(const Model&m):model_(m),q_(m.integer("two_layer_gate_scalar_count")),o_(m.integer("outer_k_rank",1)),g_(m.integer("two_layer_gate_k_rank")),tl_(m.integer("tensor_message_lmax")),latent_(1),rows_((m.lmax+1)*o_*m.radial_size),width_(q_),shared_n_(m.text("adaptive_radial_projection","full")=="shared-n"),species_(m.text("adaptive_radial_amplitude_mode","shared")=="species"),tanh_(m.text("adaptive_radial_activation","none")=="tanh") {
 m.validate();if(m.text("outer_radial_mode")!="adaptive-coefficients")throw std::runtime_error("Adaptive requires adaptive-coefficients model");
 latent_=shared_n_?m.integer("adaptive_radial_shared_n_rank",1):1;
 const std::string sm=m.text("scaling_map","LK");scaling_map_=sm=="pair"?0:sm=="L"?1:sm=="K"?2:3;
 width_+=g_*((tl_+1)*(tl_+1)-1);
 amplitude_=m.numbers("adaptive_radial_amplitudes");weight_=m.numbers("adaptive_radial_weights");tensor_=m.numbers("tensor_message_coeffs");scale_=m.numbers("scal_coeffs");
 harmonic_norm_.fill(0);const double pi=std::acos(-1.);
 for(int l=0;l<=m.lmax;++l)for(int mm=0;mm<=l;++mm){double ratio=1;for(int a=l-mm+1;a<=l+mm;++a)ratio/=a;harmonic_norm_[harmonic(l,mm)]=std::sqrt((2*l+1)*ratio/(4*pi))*(mm?std::sqrt(2.)*(mm%2?-1:1):1.);}
 y_.reserve((m.lmax+1)*(m.lmax+1));dy_.reserve(3*(m.lmax+1)*(m.lmax+1));rb_.reserve(m.radial_size);dr_.reserve(m.radial_size);rq_.reserve(q_);
 const int projection_rows=shared_n_?(m.lmax+1)*o_*latent_:rows_;phi_.resize(projection_rows);first_.resize(projection_rows);
 gate_nodes_=m.integers("two_layer_gate_scalar_moment_indices");
 if(gate_nodes_.empty())for(int q:m.integers("two_layer_gate_scalar_indices"))gate_nodes_.push_back(m.scalar_mapping.at(q));
}
void Adaptive::project(const double* s,double* p)const{for(int r=0;r<rows_;++r){double z=0;for(int q=0;q<q_;++q)z+=weight_[r*q_+q]*s[q];p[r]=z;}}
void Adaptive::project_reverse(const double* p,double* s)const{for(int r=0;r<rows_;++r)for(int q=0;q<q_;++q)s[q]+=p[r]*weight_[r*q_+q];}
int Adaptive::signal_width()const{return width_;}
int Adaptive::feature_moment_index(int f)const{
 if(f<0 || f>=width_)throw std::runtime_error("adaptive feature index out of range");
 if(f<q_)return gate_nodes_.at(f);
 f-=q_;for(int l=1;l<=tl_;++l){int n=g_*(2*l+1);if(f<n){int g=f/(2*l+1),m=f%(2*l+1)-l;return model_.integer("two_layer_gate_k_rank_basic_begin")+node(g_,(model_.lmax+1)*(model_.lmax+1),l,g,m);}f-=n;}
 throw std::runtime_error("adaptive tensor feature index out of range");
}
void Adaptive::edge(Radial&radial,const std::array<double,3>&d,int ti,int tj,const double*signal,std::vector<double>&value,std::vector<double>&coord,std::vector<double>&signal_der,bool derivatives,const double* adj,double* signal_adj,const double* projected,double* projected_adj)const{
 const Model&m=model_;const int L=m.lmax+1,H=L*L,R=m.radial_size,C=m.species_count,N=o_*H;
 if(ti<0||ti>=C||tj<0||tj>=C)throw std::runtime_error("adaptive species out of range");
 const bool factored=adj && !shared_n_ && !tanh_ && tl_==0;
 std::vector<double> projection_adj(factored&&!projected_adj?rows_:0,0);
 double* pa=projected_adj?projected_adj:projection_adj.data();
 value.assign(N,0);if(derivatives){coord.assign(3*N,0);if(!factored)signal_der.assign(static_cast<size_t>(N)*width_,0);}
 double r2=0;for(double x:d)r2+=x*x;double r=std::sqrt(r2);if(r>=m.max_dist)return;
 const int active_L=r>=m.angular_max_dist?1:L;
 auto&y=y_;auto&dy=dy_;auto&rb=rb_;auto&dr=dr_;harmonics(d,active_L-1,harmonic_norm_,y,dy);
 const int P=shared_n_?L*o_*latent_:rows_;
 auto&phi=phi_;auto&first=first_;
 for(int p=0;p<(active_L==1?P/L:P);++p){double z=0;if(projected)z=projected[p];else for(int q=0;q<q_;++q)z+=weight_[p*q_+q]*signal[q];phi[p]=tanh_?std::tanh(z):z;first[p]=tanh_?1.-phi[p]*phi[p]:1.;}
 // With K sharing, slope/shift and channel depend only on o and pair.
 // Refresh at l=1 when angular and scalar shells have different cutoffs;
 // later angular channels can then reuse that shell without reordering sums.
 auto& basis_cache=basis_cache_;
 auto& derivative_cache=derivative_cache_;
 if(scaling_map_==2){basis_cache.resize(o_);derivative_cache.resize(o_);}
 for(int l=0;l<active_L;++l)for(int o=0;o<o_;++o){
  const int mu=o*L+l,row=(l*o_+o)*R;
  const int sb=scaling_map_==0?0:scaling_map_==1?l:scaling_map_==2?o:mu,pair=ti*C+tj,sc=2*sb*C*C+pair;
  if(scaling_map_==2 && l>0 && (l>1 || m.angular_max_dist==m.max_dist)){
   rb=basis_cache[o];
   dr=derivative_cache[o];
  }else{
   radial.eval(r,scale_.at(sc),scale_.at(sc+C*C),o,rb,dr,true,l>0);
   if(scaling_map_==2){basis_cache[o]=rb;derivative_cache[o]=dr;}
  }
  double rv=0,rd=0;auto&rq=rq_;rq.assign(q_,0);
  for(int n=0;n<R;++n){double coeff=m.radial_coeffs[mu*(R+C)+n];for(int t=0;t<latent_;++t){int p=shared_n_?(l*o_+o)*latent_+t:row+n;double a=amplitude_[((species_?tj*rows_:0)+row+n)*latent_+t];coeff+=a*phi[p];if(derivatives&&!factored)for(int q=0;q<q_;++q)rq[q]+=rb[n]*a*first[p]*weight_[p*q_+q];}rv+=coeff*rb[n];rd+=coeff*dr[n];}
  double species=m.radial_coeffs[mu*(R+C)+R+ti]*m.radial_coeffs[mu*(R+C)+R+tj];
  double contraction=0;
  for(int mm=-l;mm<=l;++mm){const int at=node(o_,H,l,o,mm),h=harmonic(l,mm);double angular=y[h];if(factored)contraction+=adj[at]*angular;
   if(l>=1&&l<=tl_)for(int g=0;g<g_;++g){const int feature=q_+(l*l-1)*g_+g*(2*l+1)+mm+l;const double w=tensor_[((l-1)*o_+o)*g_+g];angular+=w*signal[feature];if(derivatives)signal_der[static_cast<size_t>(at)*width_+feature]+=species*rv*w;}
   value[at]=species*rv*angular;if(derivatives)for(int a=0;a<3;++a)coord[3*at+a]=species*(rd*d[a]/r*angular+rv*dy[3*h+a]);
   if(derivatives&&!factored)for(int q=0;q<q_;++q)signal_der[static_cast<size_t>(at)*width_+q]+=species*rq[q]*angular;
  }
  if(factored)for(int n=0;n<R;++n)pa[row+n]+=species*contraction*rb[n]*amplitude_[((species_?tj*rows_:0)+row+n)];
 }
 if(factored){if(!projected_adj)project_reverse(pa,signal_adj);}
 else if(adj)for(int b=0;b<N;++b)for(int q=0;q<width_;++q)signal_adj[q]+=adj[b]*signal_der[b*width_+q];
}
} }
