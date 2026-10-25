#ifndef LMP_SUS2_SL_KOKKOS_KERNEL_H
#define LMP_SUS2_SL_KOKKOS_KERNEL_H

#include "sus2_sl_kokkos_radial.h"
#include "sus2_sl_kokkos_math.h"
#include "sus2_sl_zbl.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace LAMMPS_NS { namespace SUS2SL {

// A device-only inference engine. Host work is model compilation/allocation;
// coordinates, neighbor traversals, graphs and the complete force chain remain
// on DeviceType. Each MPI rank supplies owned-center directed CSR neighbors.
template<class DeviceType> class KokkosKernel {
 public:
  using execution_space=typename DeviceType::execution_space;
  using I=Kokkos::View<int*,DeviceType>;
  using D=Kokkos::View<double*,DeviceType>;
  using PublicMatrix=Kokkos::View<double**,Kokkos::LayoutRight,DeviceType>;
  using Matrix=Kokkos::View<double**,Kokkos::LayoutLeft,DeviceType>;
  using Range=Kokkos::RangePolicy<execution_space,Kokkos::IndexType<std::size_t>>;
  using TeamPolicy=Kokkos::TeamPolicy<execution_space>;
  using TeamMember=typename TeamPolicy::member_type;
  using Scratch=Kokkos::View<double**,Kokkos::LayoutRight,
      typename TeamMember::scratch_memory_space,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  // Public because NVCC names this type in specialized launch signatures.
  enum class ForwardRole { OuterRaw,GateRaw,Message };
  PublicMatrix positions,signal,signal_adjoint,force,virial;
  I types,offsets,neighbors,error_flag;
  D site_energy;

 public:
  // NVCC's host stub names every type in the enclosing launch signature.
  struct Program {
    I output,offset,left,right; D coefficient;
    // Host launch metadata only. No graph operation runs on the host.
    std::vector<int> layers;
    std::size_t bytes()const{return (output.size()+offset.size()+left.size()+right.size())*sizeof(int)+coefficient.size()*sizeof(double);}
  };
 private:
  struct HostRow { int output;std::vector<int> left,right;std::vector<double> c; };
  struct Stage {
    std::vector<int> map;
    int width=0;
    Program forward,reverse,contract,transpose;
    Matrix values,adjoint;
  };
 public:
  // The captured type must also be nameable by NVCC's extended-lambda stub.
  // Only its type is public; the owning instance and graph storage are not.
  struct State {
    DeviceRadial<DeviceType> radial;
    int owned=0,all=0,edges=0,C=0,L=0,H=0,K=0,F=0,B=0,G=0,O=0,Q=0,W=0;
    int mode=0,tensor_lmax=0,tensor_components=0,latent=1,P=0,Drows=0,scaling_map=3;
    bool gated=false,mixing=false,species_readout=false,species_amplitude=false;
    bool shared_n=false,activation_tanh=false,gate_readout=false,zbl=false;
    double cutoff=0,zbl_inner=.7,zbl_outer=1.4,zbl_factor=0;
    PublicMatrix x,s,sa,f,v;
    I type,off,nbr,error,inner_basic,outer_basic,outer_rank,gate_feature,gate_scalar;
    I outer_scalar,joint_basic_source,joint_basic_l,joint_basic_m,joint_gate_source,joint_outer_source;
    I atomic_number;
    D norm,gate_species,outer_species,additive,gate_weights,gate_head,head,species,bias;
    D mixer,scale,base,amplitude,U,tensor,joint_head;
    Matrix gv,ga,ov,oa,jv,ja,projection,coeff,coeff_adj,tensor_value,tensor_adj;
    D energy;
    explicit State(const Model&m):radial(m){}
    KOKKOS_INLINE_FUNCTION bool valid_type(int i)const {
      if(type(i)>=0&&type(i)<C)return true;
      Kokkos::atomic_exchange(&error(0),3);return false;
    }
    KOKKOS_INLINE_FUNCTION bool edge_range(int i,int&lo,int&hi)const {
      lo=off(i);hi=off(i+1);
      if(lo<0||hi<lo||hi>edges){Kokkos::atomic_exchange(&error(0),4);return false;}
      return true;
    }
    KOKKOS_INLINE_FUNCTION bool geometry(int i,int j,double*d,double&r)const {
      if(j<0||j>=all){Kokkos::atomic_exchange(&error(0),1);return false;}
      if(type(i)<0||type(i)>=C||type(j)<0||type(j)>=C){Kokkos::atomic_exchange(&error(0),3);return false;}
      double r2=0.;for(int a=0;a<3;++a){d[a]=x(j,a)-x(i,a);r2+=d[a]*d[a];}
      if(!(r2>0.)||!(r2<1.7976931348623157e308)){Kokkos::atomic_exchange(&error(0),2);return false;}
      r=sqrt(r2);return true;
    }
    KOKKOS_INLINE_FUNCTION void add_edge(int i,int j,const double*d,const double*g)const {
      for(int a=0;a<3;++a){Kokkos::atomic_add(&f(i,a),g[a]);Kokkos::atomic_add(&f(j,a),-g[a]);}
      Kokkos::atomic_add(&v(i,0),-d[0]*g[0]);Kokkos::atomic_add(&v(i,1),-d[1]*g[1]);
      Kokkos::atomic_add(&v(i,2),-d[2]*g[2]);Kokkos::atomic_add(&v(i,3),-d[0]*g[1]);
      Kokkos::atomic_add(&v(i,4),-d[0]*g[2]);Kokkos::atomic_add(&v(i,5),-d[1]*g[2]);
    }
    KOKKOS_INLINE_FUNCTION void warp(int ti,int tj,int k,int l,double&slope,double&shift)const {
      const int slot=scaling_map==0?0:scaling_map==1?l:scaling_map==2?k:k*L+l;
      const int sc=2*slot*C*C+ti*C+tj;slope=scale(sc);shift=scale(sc+C*C);
    }
  };
 private:
  State d_;
  Model model_;
  Stage gate_,outer_,joint_;
  int owned_=0,all_=0,edges_=0;
  std::size_t model_bytes_=0;
  bool generic_reference_=false;
  bool forward_radial_tile_=false,reverse_center_reduce_=false;
  bool forward_role_specialize_=false;

  static bool environment_switch(const char* name) {
    const char* value=std::getenv(name);
    if(value&&std::string(value)!="0"&&std::string(value)!="1")
      throw std::runtime_error(std::string(name)+" must be 0 or 1");
    return value&&std::string(value)=="1";
  }

  template<class T> static Kokkos::View<T*,DeviceType> upload(const std::string&name,const std::vector<T>&a) {
    Kokkos::View<T*,DeviceType> result(name,a.size());
    auto h=Kokkos::create_mirror_view(result);for(std::size_t i=0;i<a.size();++i)h(i)=a[i];
    Kokkos::deep_copy(result,h);return result;
  }
  template<class T> Kokkos::View<T*,DeviceType> parameter(const std::string&name,const std::vector<T>&a) {
    model_bytes_+=a.size()*sizeof(T);return upload(name,a);
  }
  static double at(const std::vector<double>&a,int i,double fallback=0.) {
    return i>=0&&static_cast<std::size_t>(i)<a.size()?a[i]:fallback;
  }
  static int use(Stage&s,int logical) {
    if(logical<0||logical>=static_cast<int>(s.map.size()))throw std::runtime_error("SUS2 Kokkos graph node out of range");
    if(s.map[logical]<0)s.map[logical]=s.width++;
    return s.map[logical];
  }
  static Program program(const std::string&name,const std::vector<std::vector<HostRow>>&layers) {
    Program p;std::vector<int>out,offset(1,0),left,right;std::vector<double>c;p.layers.push_back(0);
    for(const auto&layer:layers){for(const auto&r:layer){out.push_back(r.output);left.insert(left.end(),r.left.begin(),r.left.end());right.insert(right.end(),r.right.begin(),r.right.end());c.insert(c.end(),r.c.begin(),r.c.end());offset.push_back(static_cast<int>(c.size()));}p.layers.push_back(static_cast<int>(out.size()));}
    p.output=upload(name+"_out",out);p.offset=upload(name+"_offset",offset);
    p.left=upload(name+"_left",left);p.right=upload(name+"_right",right);p.coefficient=upload(name+"_c",c);return p;
  }
  void graph_plan(Stage&s,const std::vector<Product>&products,int begin,int end,const std::string&name) {
    std::map<int,std::vector<int>> grouped;
    for(int p=begin;p<end;++p){use(s,products[p].a);use(s,products[p].b);use(s,products[p].out);grouped[products[p].out].push_back(p);}
    std::vector<int> depth(s.map.size(),0);int maximum=0;
    for(const auto&g:grouped){int z=0;for(int p:g.second)z=std::max(z,std::max(depth[products[p].a],depth[products[p].b]));depth[g.first]=z+1;maximum=std::max(maximum,z+1);}
    std::vector<std::vector<HostRow>>forward(maximum),reverse(maximum);
    for(const auto&g:grouped){HostRow r;r.output=s.map[g.first];for(int p:g.second){r.left.push_back(s.map[products[p].a]);r.right.push_back(s.map[products[p].b]);r.c.push_back(products[p].c);}forward[depth[g.first]-1].push_back(r);}
    // One writer per destination at each reverse depth, in reverse serialized
    // term order. This handles repeated products and a==b without graph atomics.
    for(int depth_index=maximum;depth_index>=1;--depth_index){std::map<int,HostRow>rows;
      for(int p=end;p-->begin;){const auto&t=products[p];if(depth[t.out]!=depth_index)continue;
        auto&a=rows[t.a];a.output=s.map[t.a];a.left.push_back(s.map[t.out]);a.right.push_back(s.map[t.b]);a.c.push_back(t.c);
        auto&b=rows[t.b];b.output=s.map[t.b];b.left.push_back(s.map[t.out]);b.right.push_back(s.map[t.a]);b.c.push_back(t.c);
      }for(const auto&r:rows)reverse[maximum-depth_index].push_back(r.second);
    }
    s.forward=program(name+"_forward",forward);s.reverse=program(name+"_reverse",reverse);
  }
  void rank_plan(Stage&s,bool gate) {
    const int rank=gate?d_.G:d_.O,begin=model_.integer(gate?"two_layer_gate_k_rank_basic_begin":"outer_k_rank1_basic_begin");
    if(!rank||(!gate&&(d_.mode==2||d_.mode==3)))return;
    const auto w=model_.numbers(gate?"two_layer_gate_k_rank_weights":"outer_k_rank1_weights");
    std::map<int,HostRow>forward,reverse;
    for(int a=0;a<rank;++a)for(std::size_t b=0;b<model_.basic.size();++b){const auto&t=model_.basic[b];
      const int dst=use(s,begin+kokkos_detail::ranked(rank,d_.H,t[1],a,t[2])),src=use(s,static_cast<int>(b));
      const double c=at(w,(t[1]*rank+a)*d_.K+t[0],1.);
      auto&f=forward[dst];f.output=dst;f.left.push_back(src);f.right.push_back(-1);f.c.push_back(c);
      auto&r=reverse[src];r.output=src;r.left.push_back(dst);r.right.push_back(-1);r.c.push_back(c);
    }
    std::vector<std::vector<HostRow>>f(1),r(1);for(const auto&x:forward)f[0].push_back(x.second);for(const auto&x:reverse)r[0].push_back(x.second);
    s.contract=program(gate?"gate_U":"outer_U",f);s.transpose=program(gate?"gate_Ut":"outer_Ut",r);
  }
 public:
  // NVCC extended lambdas require their enclosing host launch methods to be
  // accessible. Implementation state and graph storage remain private.
  void execute(const Program&p,const Matrix&values,const Matrix&adjoints,bool reverse,bool linear) {
    const auto out=p.output,off=p.offset,left=p.left,right=p.right;const auto c=p.coefficient;const int n=owned_;
    for(std::size_t layer=1;layer<p.layers.size();++layer){const int begin=p.layers[layer-1],count=p.layers[layer]-begin;if(!count||!n)continue;
      Kokkos::parallel_for(reverse?"sus2_graph_reverse":"sus2_graph_forward",Range(0,static_cast<std::size_t>(n)*count),KOKKOS_LAMBDA(std::size_t idx){
        const int atom=static_cast<int>(idx%n),row=begin+static_cast<int>(idx/n);double z=0.;
        for(int t=off(row);t<off(row+1);++t)z+=c(t)*(reverse?adjoints(atom,left(t)):values(atom,left(t)))*(linear?1.:values(atom,right(t)));
        if(reverse)adjoints(atom,out(row))+=z;else values(atom,out(row))+=z;
      });
    }
  }
  // Bounded per-team geometry/SH tile, reused by all dynamic radial channels.
  // No per-edge data survives a tile, and each raw/output row has one writer.
  void forward_team(bool gate,bool message) {
    if(forward_radial_tile_){forward_radial_tile(gate,message);return;}
    if(forward_role_specialize_){dispatch_forward_role(gate,message,false);return;}
    if(!owned_)return;const auto d=d_;const Matrix values=gate?d.gv:d.ov;
    const I basic=gate?d.inner_basic:d.outer_basic;const int rows=message?d.L*d.O:d.F,tile=16;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO).set_scratch_size(0,Kokkos::PerTeam(Scratch::shmem_size(tile,4+d.H)));
    Kokkos::parallel_for(message?"sus2_message_forward_team":gate?"sus2_gate_forward_team":"sus2_static_effect_forward_team",policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      Scratch shared(team.team_scratch(0),tile,4+d.H);
      for(int begin=lo;begin<hi;begin+=tile){const int count=hi-begin<tile?hi-begin:tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),[&](const int t){
          const int j=d.nbr(begin+t);double delta[3],r;
          if(!d.geometry(i,j,delta,r)||r>=d.cutoff){shared(t,3)=0.;return;}
          for(int q=0;q<3;++q)shared(t,q)=delta[q];shared(t,3)=r;
          double y[25],unused[75];kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
          for(int h=0;h<d.H;++h)shared(t,4+h)=y[h];
        });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,rows),[&](const int row){
          const int l=message?row/d.O:row%d.L,channel=message?row%d.O:row/d.L;double accum[9]={};
          for(int t=0;t<count;++t){const double r=shared(t,3);if(!(r>0.))continue;const int j=d.nbr(begin+t),tj=d.type(j);
            if(!message){double rv,rd;d.radial.contract(r,ti,tj,channel,l,gate,rv,rd);
              const double factor=gate?d.gate_species(ti*d.F+row)*d.gate_species(tj*d.F+row):d.outer_species(ti*d.F+row)*(d.outer_species(tj*d.F+row)+(d.gated?d.additive(tj*d.F+row)*d.s(j,row):0.));
              for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*shared(t,4+l*l+c);
            }else if(d.mode==3){const int mu=channel*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,channel,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,channel,n,b,db);rv+=d.coeff(j,cr+n)*b;}
              const double species=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
              for(int c=0;c<2*l+1;++c){double angular=shared(t,4+l*l+c);if(l&&l<=d.tensor_lmax)angular+=d.tensor_value(j,channel*d.tensor_components+l*l-1+c);accum[c]+=species*rv*angular;}
            }else for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);rv+=d.mixer(((l*d.O+channel)*d.K+k)*d.B+n)*b;}
              const double factor=d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));
              for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*shared(t,4+l*l+c);
            }
          }
          for(int c=0;c<2*l+1;++c){const int node=message?d.outer_rank(row*9+c):basic(row*9+c);if(node>=0)values(i,node)+=accum[c];}
        });
        team.team_barrier();
      }
    });
  }
  // Experimental row x neighbor producer. Fixed row chunks bound scratch for
  // every legal K/O; the existing generic and candidate10 paths remain intact.
  void forward_radial_tile(bool gate,bool message) {
    if(forward_role_specialize_){dispatch_forward_role(gate,message,true);return;}
    if(!owned_)return;const auto d=d_;const Matrix values=gate?d.gv:d.ov;
    const I basic=gate?d.inner_basic:d.outer_basic;
    const int rows=message?d.L*d.O:d.F,tile=16,row_tile=8;
    // Mixer factors depend on k. Keep each angular component's original k
    // contraction, then reduce neighbor contributions; do not freeze sources.
    const bool mixer=message&&d.mode!=3;
    const int stride=mixer?9:2,columns=4+d.H+row_tile*stride;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO,1).set_scratch_size(0,
        Kokkos::PerTeam(Scratch::shmem_size(tile,columns)));
    Kokkos::parallel_for(message?"sus2_message_forward_radial_tile":gate?"sus2_gate_forward_radial_tile":"sus2_static_effect_forward_radial_tile",policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      Scratch shared(team.team_scratch(0),tile,columns);
      for(int begin=lo;begin<hi;begin+=tile){const int count=hi-begin<tile?hi-begin:tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),[&](const int t){
          const int j=d.nbr(begin+t);double delta[3],r;
          if(!d.geometry(i,j,delta,r)||r>=d.cutoff){shared(t,3)=0.;return;}
          for(int q=0;q<3;++q)shared(t,q)=delta[q];shared(t,3)=r;
          double y[25],unused[75];kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
          for(int h=0;h<d.H;++h)shared(t,4+h)=y[h];
        });
        team.team_barrier();
        for(int first=0;first<rows;first+=row_tile){const int nr=rows-first<row_tile?rows-first:row_tile;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count*nr),[&](const int idx){
            const int local=idx/count,t=idx%count,row=first+local;
            const int l=message?row/d.O:row%d.L,channel=message?row%d.O:row/d.L;
            const int slot=4+d.H+local*stride;const double r=shared(t,3);
            if(!(r>0.))return;const int j=d.nbr(begin+t),tj=d.type(j);
            if(!message){double rv,rd;d.radial.contract(r,ti,tj,channel,l,gate,rv,rd);
              shared(t,slot)=rv;
              shared(t,slot+1)=gate?d.gate_species(ti*d.F+row)*d.gate_species(tj*d.F+row):d.outer_species(ti*d.F+row)*(d.outer_species(tj*d.F+row)+(d.gated?d.additive(tj*d.F+row)*d.s(j,row):0.));
            }else if(d.mode==3){const int mu=channel*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,channel,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,channel,n,b,db);rv+=d.coeff(j,cr+n)*b;}
              shared(t,slot)=rv;shared(t,slot+1)=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
            }else{double partial[9]={};
              for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.;
                for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);rv+=d.mixer(((l*d.O+channel)*d.K+k)*d.B+n)*b;}
                const double factor=d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));
                for(int c=0;c<2*l+1;++c)partial[c]+=factor*rv*shared(t,4+l*l+c);
              }
              for(int c=0;c<2*l+1;++c)shared(t,slot+c)=partial[c];
            }
          });
          team.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,nr),[&](const int local){
            const int row=first+local,l=message?row/d.O:row%d.L,channel=message?row%d.O:row/d.L;
            const int slot=4+d.H+local*stride;double accum[9]={};
            for(int t=0;t<count;++t){if(!(shared(t,3)>0.))continue;
              if(mixer){for(int c=0;c<2*l+1;++c)accum[c]+=shared(t,slot+c);}
              else{const double rv=shared(t,slot),factor=shared(t,slot+1);
                for(int c=0;c<2*l+1;++c){double angular=shared(t,4+l*l+c);
                  if(message&&l&&l<=d.tensor_lmax)angular+=d.tensor_value(d.nbr(begin+t),channel*d.tensor_components+l*l-1+c);
                  accum[c]+=factor*rv*angular;
                }
              }
            }
            for(int c=0;c<2*l+1;++c){const int node=message?d.outer_rank(row*9+c):basic(row*9+c);if(node>=0)values(i,node)+=accum[c];}
          });
          team.team_barrier();
        }
      }
    });
  }
  // Only these three roles are instantiated. Shape and radial family remain
  // runtime metadata; this switch changes compilation, not the model or layout.
  void dispatch_forward_role(bool gate,bool message,bool tiled) {
    if(message)launch_forward_role<ForwardRole::Message>(tiled);
    else if(gate)launch_forward_role<ForwardRole::GateRaw>(tiled);
    else launch_forward_role<ForwardRole::OuterRaw>(tiled);
  }
  template<ForwardRole Role> void launch_forward_role(bool tiled) {
    if(tiled)forward_radial_tile_role<Role>();else forward_team_role<Role>();
  }
  template<ForwardRole Role> void forward_team_role() {
    constexpr bool gate=Role==ForwardRole::GateRaw,message=Role==ForwardRole::Message;
    if(!owned_)return;const auto d=d_;const Matrix values=gate?d.gv:d.ov;
    const I basic=gate?d.inner_basic:d.outer_basic;const int rows=message?d.L*d.O:d.F,tile=16;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO).set_scratch_size(0,Kokkos::PerTeam(Scratch::shmem_size(tile,4+d.H)));
    Kokkos::parallel_for(message?"sus2_message_forward_role_team":gate?"sus2_gate_forward_role_team":"sus2_static_effect_forward_role_team",policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      Scratch shared(team.team_scratch(0),tile,4+d.H);
      for(int begin=lo;begin<hi;begin+=tile){const int count=hi-begin<tile?hi-begin:tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),[&](const int t){
          const int j=d.nbr(begin+t);double delta[3],r;
          if(!d.geometry(i,j,delta,r)||r>=d.cutoff){shared(t,3)=0.;return;}
          for(int q=0;q<3;++q)shared(t,q)=delta[q];shared(t,3)=r;
          double y[25],unused[75];kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
          for(int h=0;h<d.H;++h)shared(t,4+h)=y[h];
        });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,rows),[&](const int row){
          const int l=(Role==ForwardRole::Message)?row/d.O:row%d.L,channel=(Role==ForwardRole::Message)?row%d.O:row/d.L;double accum[9]={};
          for(int t=0;t<count;++t){const double r=shared(t,3);if(!(r>0.))continue;const int j=d.nbr(begin+t),tj=d.type(j);
            if(Role!=ForwardRole::Message){double rv,rd;d.radial.contract(r,ti,tj,channel,l,Role==ForwardRole::GateRaw,rv,rd);
              const double factor=(Role==ForwardRole::GateRaw)?d.gate_species(ti*d.F+row)*d.gate_species(tj*d.F+row):d.outer_species(ti*d.F+row)*(d.outer_species(tj*d.F+row)+(d.gated?d.additive(tj*d.F+row)*d.s(j,row):0.));
              for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*shared(t,4+l*l+c);
            }else if(d.mode==3){const int mu=channel*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,channel,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,channel,n,b,db);rv+=d.coeff(j,cr+n)*b;}
              const double species=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
              for(int c=0;c<2*l+1;++c){double angular=shared(t,4+l*l+c);if(l&&l<=d.tensor_lmax)angular+=d.tensor_value(j,channel*d.tensor_components+l*l-1+c);accum[c]+=species*rv*angular;}
            }else for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);rv+=d.mixer(((l*d.O+channel)*d.K+k)*d.B+n)*b;}
              const double factor=d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));
              for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*shared(t,4+l*l+c);
            }
          }
          for(int c=0;c<2*l+1;++c){const int node=(Role==ForwardRole::Message)?d.outer_rank(row*9+c):basic(row*9+c);if(node>=0)values(i,node)+=accum[c];}
        });
        team.team_barrier();
      }
    });
  }
  template<ForwardRole Role> void forward_radial_tile_role() {
    constexpr bool gate=Role==ForwardRole::GateRaw,message=Role==ForwardRole::Message;
    if(!owned_)return;const auto d=d_;const Matrix values=gate?d.gv:d.ov;
    const I basic=gate?d.inner_basic:d.outer_basic;
    const int rows=message?d.L*d.O:d.F,tile=16,row_tile=8;
    // Mixer factors depend on k. Keep each angular component's original k
    // contraction, then reduce neighbor contributions; do not freeze sources.
    const bool mixer=message&&d.mode!=3;
    const int stride=mixer?9:2,columns=4+d.H+row_tile*stride;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO,1).set_scratch_size(0,
        Kokkos::PerTeam(Scratch::shmem_size(tile,columns)));
    Kokkos::parallel_for(message?"sus2_message_forward_role_radial_tile":gate?"sus2_gate_forward_role_radial_tile":"sus2_static_effect_forward_role_radial_tile",policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      Scratch shared(team.team_scratch(0),tile,columns);
      for(int begin=lo;begin<hi;begin+=tile){const int count=hi-begin<tile?hi-begin:tile;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count),[&](const int t){
          const int j=d.nbr(begin+t);double delta[3],r;
          if(!d.geometry(i,j,delta,r)||r>=d.cutoff){shared(t,3)=0.;return;}
          for(int q=0;q<3;++q)shared(t,q)=delta[q];shared(t,3)=r;
          double y[25],unused[75];kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
          for(int h=0;h<d.H;++h)shared(t,4+h)=y[h];
        });
        team.team_barrier();
        for(int first=0;first<rows;first+=row_tile){const int nr=rows-first<row_tile?rows-first:row_tile;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,count*nr),[&](const int idx){
            const int local=idx/count,t=idx%count,row=first+local;
            const int l=(Role==ForwardRole::Message)?row/d.O:row%d.L,channel=(Role==ForwardRole::Message)?row%d.O:row/d.L;
            const int slot=4+d.H+local*((Role==ForwardRole::Message)?stride:2);const double r=shared(t,3);
            if(!(r>0.))return;const int j=d.nbr(begin+t),tj=d.type(j);
            if(Role!=ForwardRole::Message){double rv,rd;d.radial.contract(r,ti,tj,channel,l,Role==ForwardRole::GateRaw,rv,rd);
              shared(t,slot)=rv;
              shared(t,slot+1)=(Role==ForwardRole::GateRaw)?d.gate_species(ti*d.F+row)*d.gate_species(tj*d.F+row):d.outer_species(ti*d.F+row)*(d.outer_species(tj*d.F+row)+(d.gated?d.additive(tj*d.F+row)*d.s(j,row):0.));
            }else if(d.mode==3){const int mu=channel*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,channel,l,slope,shift);double rv=0.;
              for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,channel,n,b,db);rv+=d.coeff(j,cr+n)*b;}
              shared(t,slot)=rv;shared(t,slot+1)=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
            }else{double partial[9]={};
              for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.;
                for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);rv+=d.mixer(((l*d.O+channel)*d.K+k)*d.B+n)*b;}
                const double factor=d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));
                for(int c=0;c<2*l+1;++c)partial[c]+=factor*rv*shared(t,4+l*l+c);
              }
              for(int c=0;c<2*l+1;++c)shared(t,slot+c)=partial[c];
            }
          });
          team.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team,nr),[&](const int local){
            const int row=first+local,l=(Role==ForwardRole::Message)?row/d.O:row%d.L,channel=(Role==ForwardRole::Message)?row%d.O:row/d.L;
            const int slot=4+d.H+local*((Role==ForwardRole::Message)?stride:2);double accum[9]={};
            for(int t=0;t<count;++t){if(!(shared(t,3)>0.))continue;
              if(Role==ForwardRole::Message){
                if(d.mode!=3){for(int c=0;c<2*l+1;++c)accum[c]+=shared(t,slot+c);continue;}
              }
              const double rv=shared(t,slot),factor=shared(t,slot+1);
              for(int c=0;c<2*l+1;++c){double angular=shared(t,4+l*l+c);
                if(Role==ForwardRole::Message){if(l&&l<=d.tensor_lmax)angular+=d.tensor_value(d.nbr(begin+t),channel*d.tensor_components+l*l-1+c);}
                accum[c]+=factor*rv*angular;
              }
            }
            for(int c=0;c<2*l+1;++c){const int node=(Role==ForwardRole::Message)?d.outer_rank(row*9+c):basic(row*9+c);if(node>=0)values(i,node)+=accum[c];}
          });
          team.team_barrier();
        }
      }
    });
  }
  void raw_forward(bool gate){if(generic_reference_)raw_forward_generic(gate);else forward_team(gate,false);}
  void raw_forward_generic(bool gate) {
    const auto d=d_;const Matrix values=gate?d.gv:d.ov;const I basic=gate?d.inner_basic:d.outer_basic;
    Kokkos::parallel_for(gate?"sus2_gate_radial":"sus2_legacy_radial",Range(0,static_cast<std::size_t>(owned_)*d.F),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),mu=static_cast<int>(idx/d.owned),k=mu/d.L,l=mu%d.L;
      int lo,hi;if(!d.edge_range(i,lo,hi))return;double accum[9]={};const int ti=d.type(i);
      for(int e=lo;e<hi;++e){const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)continue;
        const int tj=d.type(j);double y[25],unused[75],rv,rd;
        kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
        d.radial.contract(r,ti,tj,k,l,gate,rv,rd);
        const double factor=gate?d.gate_species(ti*d.F+mu)*d.gate_species(tj*d.F+mu):d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));
        for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*y[l*l+c];
      }
      for(int c=0;c<2*l+1;++c){const int node=basic(mu*9+c);if(node>=0)values(i,node)=accum[c];}
    });
  }
  void raw_reverse_team(bool gate) {
    if(reverse_center_reduce_)raw_reverse_team_impl<true>(gate);else raw_reverse_team_impl<false>(gate);
  }
  template<bool Reduce> void raw_reverse_team_impl(bool gate) {
    if(!owned_)return;const auto d=d_;const Matrix adj=gate?d.ga:d.oa;const I basic=gate?d.inner_basic:d.outer_basic;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO,1);
    if(Reduce)policy.set_scratch_size(0,Kokkos::PerThread(Scratch::shmem_size(1,9)));
    Kokkos::parallel_for(gate?(Reduce?"sus2_gate_reverse_center_reduce":"sus2_gate_reverse_team"):(Reduce?"sus2_static_effect_reverse_center_reduce":"sus2_static_effect_reverse_team"),policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      double center[9]={};
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,lo,hi),[&](const int e){
        const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)return;const int tj=d.type(j);
        double y[25],dy[75],g[3]={};kokkos_detail::harmonics<true>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,dy);
        for(int mu=0;mu<d.F;++mu){const int k=mu/d.L,l=mu%d.L;double rv,rd;d.radial.contract(r,ti,tj,k,l,gate,rv,rd);
          const double center=gate?d.gate_species(ti*d.F+mu):d.outer_species(ti*d.F+mu);
          const double source=gate?d.gate_species(tj*d.F+mu):d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.);double sa=0.;
          for(int c=0;c<2*l+1;++c){const int node=basic(mu*9+c);if(node<0)continue;const double a=adj(i,node),aa=a*center*source;
            if(!gate&&d.gated)sa+=a*center*d.additive(tj*d.F+mu)*rv*y[l*l+c];
            for(int q=0;q<3;++q)g[q]+=aa*(rd*delta[q]/r*y[l*l+c]+rv*dy[3*(l*l+c)+q]);
          }if(!gate&&d.gated)Kokkos::atomic_add(&d.sa(j,mu),sa);
        }
        if(Reduce)accumulate_center(d,j,delta,g,center);else d.add_edge(i,j,delta,g);
      });
      if(Reduce)finish_center(team,d,i,center);
    });
  }
  // Vector length is explicitly one: each nested neighbor iteration updates
  // its calling team thread's private sum exactly once. Neighbor slots remain
  // atomic, including self-image ghosts and owned centers of other teams.
  KOKKOS_INLINE_FUNCTION static void accumulate_center(const State&d,int j,
      const double*delta,const double*g,double*center) {
    for(int q=0;q<3;++q){Kokkos::atomic_add(&d.f(j,q),-g[q]);center[q]+=g[q];}
    center[3]+=-delta[0]*g[0];center[4]+=-delta[1]*g[1];center[5]+=-delta[2]*g[2];
    center[6]+=-delta[0]*g[1];center[7]+=-delta[0]*g[2];center[8]+=-delta[1]*g[2];
  }
  KOKKOS_INLINE_FUNCTION static void finish_center(const TeamMember&team,
      const State&d,int i,const double*center) {
    Scratch partial(team.team_scratch(0),team.team_size(),9);
    for(int q=0;q<9;++q)partial(team.team_rank(),q)=center[q];
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,9),[&](const int q){
      double sum=0.;for(int rank=0;rank<team.team_size();++rank)sum+=partial(rank,q);
      // Other center teams can still scatter to f(i); only this team owns v(i).
      if(q<3)Kokkos::atomic_add(&d.f(i,q),sum);else d.v(i,q-3)+=sum;
    });
    // A backend may reuse this block and scratch for another league entry.
    team.team_barrier();
  }
  void raw_reverse(bool gate){if(generic_reference_)raw_reverse_generic(gate);else raw_reverse_team(gate);}
  void raw_reverse_generic(bool gate) {
    const auto d=d_;const Matrix adj=gate?d.ga:d.oa;const I basic=gate?d.inner_basic:d.outer_basic;
    Kokkos::parallel_for(gate?"sus2_gate_coordinate_reverse":"sus2_legacy_coordinate_reverse",Range(0,static_cast<std::size_t>(owned_)*d.F),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),mu=static_cast<int>(idx/d.owned),k=mu/d.L,l=mu%d.L;int lo,hi;if(!d.edge_range(i,lo,hi))return;
      const int ti=d.type(i);
      for(int e=lo;e<hi;++e){const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)continue;
        const int tj=d.type(j);double y[25],dy[75],rv,rd;kokkos_detail::harmonics<true>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,dy);
        d.radial.contract(r,ti,tj,k,l,gate,rv,rd);
        const double center=gate?d.gate_species(ti*d.F+mu):d.outer_species(ti*d.F+mu);
        const double source=gate?d.gate_species(tj*d.F+mu):d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.);
        double g[3]={},source_adj=0.;
        for(int c=0;c<2*l+1;++c){const int node=basic(mu*9+c);if(node<0)continue;const double a=adj(i,node),aa=a*center*source;
          if(!gate&&d.gated)source_adj+=a*center*d.additive(tj*d.F+mu)*rv*y[l*l+c];
          for(int q=0;q<3;++q)g[q]+=aa*(rd*delta[q]/r*y[l*l+c]+rv*dy[3*(l*l+c)+q]);
        }
        if(!gate&&d.gated)Kokkos::atomic_add(&d.sa(j,mu),source_adj);d.add_edge(i,j,delta,g);
      }
    });
  }
  void adaptive_prepare() {
    const auto d=d_;
    Kokkos::parallel_for("sus2_adaptive_source_U",Range(0,static_cast<std::size_t>(all_)*d.P),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),p=static_cast<int>(idx/d.all);double z=0.;for(int q=0;q<d.Q;++q)z+=d.U(p*d.Q+q)*d.s(j,q);d.projection(j,p)=d.activation_tanh?tanh(z):z;
    });
    Kokkos::parallel_for("sus2_adaptive_source_coefficients",Range(0,static_cast<std::size_t>(all_)*d.Drows),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),row=static_cast<int>(idx/d.all),p=d.shared_n?(row/d.B)*d.latent:row;
      if(!d.valid_type(j))return;
      double z=d.base(row);for(int t=0;t<d.latent;++t)z+=d.amplitude(((d.species_amplitude?d.type(j)*d.Drows:0)+row)*d.latent+t)*d.projection(j,p+t);d.coeff(j,row)=z;
    });
    const int tensor_rows=d.O*d.tensor_components;
    Kokkos::parallel_for("sus2_tensor_source_projection",Range(0,static_cast<std::size_t>(all_)*tensor_rows),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),row=static_cast<int>(idx/d.all),o=row/d.tensor_components,h=row%d.tensor_components,l=h<3?1:2,c=h-(l*l-1);double z=0.;
      for(int g=0;g<d.G;++g)z+=d.tensor(((l-1)*d.O+o)*d.G+g)*d.s(j,d.Q+(l*l-1)*d.G+g*(2*l+1)+c);d.tensor_value(j,row)=z;
    });
  }
  void message_forward(){if(generic_reference_)message_forward_generic();else forward_team(false,true);}
  void message_forward_generic() {
    const auto d=d_;
    Kokkos::parallel_for("sus2_outer_message_forward",Range(0,static_cast<std::size_t>(owned_)*d.L*d.O),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),row=static_cast<int>(idx/d.owned),l=row/d.O,o=row%d.O,ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;double accum[9]={};
      for(int e=lo;e<hi;++e){const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)continue;const int tj=d.type(j);double y[25],unused[75];kokkos_detail::harmonics<false>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,unused);
        if(d.mode==3){const int mu=o*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,o,l,slope,shift);double rv=0.;for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,o,n,b,db);rv+=d.coeff(j,cr+n)*b;}
          const double species=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
          for(int c=0;c<2*l+1;++c){double angular=y[l*l+c];if(l&&l<=d.tensor_lmax)angular+=d.tensor_value(j,o*d.tensor_components+l*l-1+c);accum[c]+=species*rv*angular;}
        }else for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.;for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);rv+=d.mixer(((l*d.O+o)*d.K+k)*d.B+n)*b;}
          const double factor=d.outer_species(ti*d.F+mu)*(d.outer_species(tj*d.F+mu)+(d.gated?d.additive(tj*d.F+mu)*d.s(j,mu):0.));for(int c=0;c<2*l+1;++c)accum[c]+=factor*rv*y[l*l+c];
        }
      }
      for(int c=0;c<2*l+1;++c)d.ov(i,d.outer_rank((l*d.O+o)*9+c))+=accum[c];
    });
  }
  void message_reverse_team() {
    if(reverse_center_reduce_)message_reverse_team_impl<true>();else message_reverse_team_impl<false>();
  }
  template<bool Reduce> void message_reverse_team_impl() {
    if(!owned_)return;const auto d=d_;
    auto policy=TeamPolicy(owned_,Kokkos::AUTO,1);
    if(Reduce)policy.set_scratch_size(0,Kokkos::PerThread(Scratch::shmem_size(1,9)));
    Kokkos::parallel_for(Reduce?"sus2_message_reverse_center_reduce":"sus2_message_reverse_team",policy,KOKKOS_LAMBDA(const TeamMember&team){
      const int i=team.league_rank(),ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      double center[9]={};
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,lo,hi),[&](const int e){
        const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)return;const int tj=d.type(j);double y[25],dy[75],g[3]={};
        kokkos_detail::harmonics<true>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,dy);
        for(int l=0;l<d.L;++l)for(int o=0;o<d.O;++o){const int row=l*d.O+o;
          if(d.mode==3){const int mu=o*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,o,l,slope,shift);double rv=0.,rd=0.,angular_adj=0.,dy_adj[3]={};
            for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,o,n,b,db);rv+=d.coeff(j,cr+n)*b;rd+=d.coeff(j,cr+n)*db;}
            const double species=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
            for(int c=0;c<2*l+1;++c){const double a=d.oa(i,d.outer_rank(row*9+c));double angular=y[l*l+c];if(l&&l<=d.tensor_lmax){const int tr=o*d.tensor_components+l*l-1+c;angular+=d.tensor_value(j,tr);Kokkos::atomic_add(&d.tensor_adj(j,tr),species*rv*a);}angular_adj+=a*angular;for(int q=0;q<3;++q)dy_adj[q]+=a*dy[3*(l*l+c)+q];}
            for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,o,n,b,db);Kokkos::atomic_add(&d.coeff_adj(j,cr+n),species*b*angular_adj);}
            for(int q=0;q<3;++q)g[q]+=species*(rd*delta[q]/r*angular_adj+rv*dy_adj[q]);
          }else for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.,rd=0.;
            for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);const double w=d.mixer((row*d.K+k)*d.B+n);rv+=w*b;rd+=w*db;}
            const double center=d.outer_species(ti*d.F+mu),alpha=d.additive(tj*d.F+mu),source=d.outer_species(tj*d.F+mu)+(d.gated?alpha*d.s(j,mu):0.);double sa=0.;
            for(int c=0;c<2*l+1;++c){const double a=d.oa(i,d.outer_rank(row*9+c));sa+=a*center*alpha*rv*y[l*l+c];for(int q=0;q<3;++q)g[q]+=a*center*source*(rd*delta[q]/r*y[l*l+c]+rv*dy[3*(l*l+c)+q]);}if(d.gated)Kokkos::atomic_add(&d.sa(j,mu),sa);
          }
        }
        if(Reduce)accumulate_center(d,j,delta,g,center);else d.add_edge(i,j,delta,g);
      });
      if(Reduce)finish_center(team,d,i,center);
    });
  }
  void message_reverse() {
    const auto d=d_;
    if(generic_reference_){
    Kokkos::parallel_for("sus2_outer_message_reverse",Range(0,static_cast<std::size_t>(owned_)*d.L*d.O),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),row=static_cast<int>(idx/d.owned),l=row/d.O,o=row%d.O,ti=d.type(i);int lo,hi;if(!d.edge_range(i,lo,hi))return;
      for(int e=lo;e<hi;++e){const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r)||r>=d.cutoff)continue;const int tj=d.type(j);double y[25],dy[75],g[3]={};kokkos_detail::harmonics<true>(delta[0],delta[1],delta[2],d.L-1,d.norm,y,dy);
        if(d.mode==3){const int mu=o*d.L+l,cr=row*d.B;double slope,shift;d.warp(ti,tj,o,l,slope,shift);double rv=0.,rd=0.,angular_adj=0.,dy_adj[3]={};
          for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,o,n,b,db);rv+=d.coeff(j,cr+n)*b;rd+=d.coeff(j,cr+n)*db;}
          const double species=d.outer_species(ti*d.F+mu)*d.outer_species(tj*d.F+mu);
          for(int c=0;c<2*l+1;++c){const double a=d.oa(i,d.outer_rank((l*d.O+o)*9+c));double angular=y[l*l+c];if(l&&l<=d.tensor_lmax){const int tr=o*d.tensor_components+l*l-1+c;angular+=d.tensor_value(j,tr);Kokkos::atomic_add(&d.tensor_adj(j,tr),species*rv*a);}angular_adj+=a*angular;for(int q=0;q<3;++q)dy_adj[q]+=a*dy[3*(l*l+c)+q];}
          for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,o,n,b,db);Kokkos::atomic_add(&d.coeff_adj(j,cr+n),species*b*angular_adj);}
          for(int q=0;q<3;++q)g[q]=species*(rd*delta[q]/r*angular_adj+rv*dy_adj[q]);
        }else for(int k=0;k<d.K;++k){const int mu=k*d.L+l;double slope,shift;d.warp(ti,tj,k,l,slope,shift);double rv=0.,rd=0.;for(int n=0;n<d.B;++n){double b,db;d.radial.basis(r,slope,shift,k,n,b,db);const double w=d.mixer(((l*d.O+o)*d.K+k)*d.B+n);rv+=w*b;rd+=w*db;}
          const double center=d.outer_species(ti*d.F+mu),alpha=d.additive(tj*d.F+mu),source=d.outer_species(tj*d.F+mu)+(d.gated?alpha*d.s(j,mu):0.);double sa=0.;
          for(int c=0;c<2*l+1;++c){const double a=d.oa(i,d.outer_rank((l*d.O+o)*9+c));sa+=a*center*alpha*rv*y[l*l+c];for(int q=0;q<3;++q)g[q]+=a*center*source*(rd*delta[q]/r*y[l*l+c]+rv*dy[3*(l*l+c)+q]);}if(d.gated)Kokkos::atomic_add(&d.sa(j,mu),sa);
        }
        d.add_edge(i,j,delta,g);
      }
    });
    }else message_reverse_team();
    if(d.mode!=3)return;
    Kokkos::parallel_for("sus2_adaptive_source_pullback",Range(0,static_cast<std::size_t>(all_)*d.P),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),p=static_cast<int>(idx/d.all),first=d.shared_n?(p/d.latent)*d.B:p,t=d.shared_n?p%d.latent:0,count=d.shared_n?d.B:1;double z=0.;
      if(!d.valid_type(j))return;
      for(int n=0;n<count;++n)z+=d.amplitude(((d.species_amplitude?d.type(j)*d.Drows:0)+first+n)*d.latent+t)*d.coeff_adj(j,first+n);
      if(d.activation_tanh){const double phi=d.projection(j,p);z*=1.-phi*phi;}d.projection(j,p)=z;
    });
    Kokkos::parallel_for("sus2_adaptive_source_Utranspose",Range(0,static_cast<std::size_t>(all_)*d.Q),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),q=static_cast<int>(idx/d.all);double z=0.;for(int p=0;p<d.P;++p)z+=d.U(p*d.Q+q)*d.projection(j,p);d.sa(j,q)+=z;
    });
    const int count=d.G*d.tensor_components;
    Kokkos::parallel_for("sus2_tensor_source_transpose",Range(0,static_cast<std::size_t>(all_)*count),KOKKOS_LAMBDA(std::size_t idx){
      const int j=static_cast<int>(idx%d.all),row=static_cast<int>(idx/d.all),l=row<3*d.G?1:2,within=row-(l*l-1)*d.G,g=within/(2*l+1),c=within%(2*l+1);double z=0.;
      for(int o=0;o<d.O;++o)z+=d.tensor(((l-1)*d.O+o)*d.G+g)*d.tensor_adj(j,o*d.tensor_components+l*l-1+c);d.sa(j,d.Q+row)+=z;
    });
  }
  void joint_forward_reverse() {
    if(!model_.boolean("joint_readout_enabled"))return;const auto d=d_;const int basics=static_cast<int>(d.joint_basic_source.size());
    Kokkos::parallel_for("sus2_joint_primitives",Range(0,static_cast<std::size_t>(owned_)*basics),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),b=static_cast<int>(idx/d.owned),source=d.joint_basic_source(b),rank=source?d.G:d.O;double z=0.;
      for(int a=0;a<rank;++a){const int pos=b*(source?d.G:d.O)+a;z+=source?d.gv(i,d.joint_gate_source(pos)):d.ov(i,d.joint_outer_source(pos));}d.jv(i,b)=z/rank;
    });
    execute(joint_.forward,d.jv,d.ja,false,false);
    const int scalars=model_.integer("joint_readout_scalar_count");const auto mapping=joint_scalar_;
    Kokkos::parallel_for("sus2_joint_readout",Range(0,owned_),KOKKOS_LAMBDA(std::size_t ii){const int i=static_cast<int>(ii),type=d.type(i);if(!d.valid_type(i))return;double energy=0.;const double center=d.species_readout?1.:d.species(type);for(int q=0;q<scalars;++q){const double c=center*d.joint_head((d.species_readout?type*scalars:0)+q);energy+=c*d.jv(i,mapping(q));d.ja(i,mapping(q))+=c;}d.energy(i)+=energy;});
    execute(joint_.reverse,d.jv,d.ja,true,false);
    Kokkos::parallel_for("sus2_joint_primitive_adjoint",Range(0,static_cast<std::size_t>(owned_)*basics),KOKKOS_LAMBDA(std::size_t idx){
      const int i=static_cast<int>(idx%d.owned),b=static_cast<int>(idx/d.owned),source=d.joint_basic_source(b),rank=source?d.G:d.O;const double a=d.ja(i,b)/rank;
      for(int channel=0;channel<rank;++channel){const int pos=b*rank+channel;if(source)Kokkos::atomic_add(&d.ga(i,d.joint_gate_source(pos)),a);else Kokkos::atomic_add(&d.oa(i,d.joint_outer_source(pos)),a);}
    });
  }
 private:
  I joint_scalar_;

 public:
  explicit KokkosKernel(const Model&m):d_(m),model_(m) {
    m.validate();d_.C=m.species_count;d_.L=m.lmax+1;d_.H=d_.L*d_.L;d_.K=m.kmax;d_.F=m.radial_funcs;d_.B=m.radial_size;
    generic_reference_=environment_switch("SUS2_SL_KOKKOS_GENERIC");
    const bool forward_tile=environment_switch("SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE");
    const bool reverse_reduce=environment_switch("SUS2_SL_KOKKOS_REVERSE_CENTER_REDUCE");
    const bool forward_role=environment_switch("SUS2_SL_KOKKOS_FORWARD_ROLE_SPECIALIZE");
    forward_radial_tile_=!generic_reference_&&forward_tile;
    reverse_center_reduce_=!generic_reference_&&reverse_reduce;
    forward_role_specialize_=!generic_reference_&&forward_role;
    d_.G=m.integer("two_layer_gate_k_rank");d_.O=m.boolean("outer_k_rank1_enabled")?m.integer("outer_k_rank",1):0;
    d_.gated=m.boolean("two_layer_gate_enabled");const auto mode=m.text("outer_radial_mode","legacy-plus-mixer");d_.mode=mode=="adaptive-coefficients"?3:mode=="pure-matrix"?2:d_.gated?1:0;
    d_.mixing=m.text("two_layer_gate_radial_mixing")=="full";d_.cutoff=m.max_dist;d_.species_readout=m.text("readout_mode")=="species";
    d_.Q=m.integer("two_layer_gate_scalar_count");d_.tensor_lmax=m.integer("tensor_message_lmax");d_.tensor_components=(d_.tensor_lmax+1)*(d_.tensor_lmax+1)-1;
    d_.shared_n=m.text("adaptive_radial_projection","full")=="shared-n";d_.latent=d_.shared_n?m.integer("adaptive_radial_shared_n_rank",1):1;
    d_.species_amplitude=m.text("adaptive_radial_amplitude_mode","shared")=="species";d_.activation_tanh=m.text("adaptive_radial_activation","none")=="tanh";
    d_.Drows=d_.mode==3?d_.L*d_.O*d_.B:0;d_.P=d_.mode==3?(d_.shared_n?d_.L*d_.O*d_.latent:d_.Drows):0;
    d_.W=d_.gated?(d_.mode==3?d_.Q+d_.G*d_.tensor_components:d_.F):0;
    const auto sm=m.text("scaling_map","LK");d_.scaling_map=sm=="pair"?0:sm=="L"?1:sm=="K"?2:3;
    d_.gate_readout=m.boolean("two_layer_gate_readout_enabled");d_.zbl=m.boolean("zbl_enabled");
    if(m.boolean("multi_layer_gate_enabled")||m.boolean("two_layer_gate_edge_l1_enabled")||m.boolean("two_layer_residual_enabled")|| (d_.gated&&m.text("two_layer_gate_site_mode","neighbor")!="neighbor"))throw std::runtime_error("SUS2 Kokkos requires the current native neighbor-gate model");
    if(d_.mode==2&&!d_.mixing)throw std::runtime_error("SUS2 Kokkos Pure-W requires its serialized full mixer");
    gate_.map.assign(m.moments,-1);outer_.map.assign(m.moments,-1);
    std::vector<int>gscalar=m.integers("two_layer_gate_scalar_moment_indices");if(gscalar.empty())for(int q:m.integers("two_layer_gate_scalar_indices"))gscalar.push_back(m.scalar_mapping.at(q));
    std::vector<int>gfeatures=gscalar;
    const int gb=m.integer("two_layer_gate_k_rank_basic_begin"),ob=m.integer("outer_k_rank1_basic_begin");
    if(d_.gated){for(std::size_t b=0;b<m.basic.size();++b)use(gate_,static_cast<int>(b));for(int q:gscalar)use(gate_,q);for(int a=0;a<d_.G;++a)for(int l=0;l<d_.L;++l)for(int mm=-l;mm<=l;++mm)use(gate_,gb+kokkos_detail::ranked(d_.G,d_.H,l,a,mm));}
    if(d_.mode==3)for(int l=1;l<=d_.tensor_lmax;++l)for(int g=0;g<d_.G;++g)for(int mm=-l;mm<=l;++mm)gfeatures.push_back(gb+kokkos_detail::ranked(d_.G,d_.H,l,g,mm));
    if(d_.mode<2)for(std::size_t b=0;b<m.basic.size();++b)use(outer_,static_cast<int>(b));
    for(int q:m.scalar_mapping)use(outer_,q);for(int o=0;o<d_.O;++o)for(int l=0;l<d_.L;++l)for(int mm=-l;mm<=l;++mm)use(outer_,ob+kokkos_detail::ranked(d_.O,d_.H,l,o,mm));
    const int outer_begin=d_.O?m.integer("outer_k_rank1_product_begin"):0;
    if(d_.gated){const int begin=m.integer("two_layer_gate_k_rank_product_begin",0),end=d_.G?m.integer("two_layer_gate_k_rank_product_end",outer_begin):outer_begin;
      graph_plan(gate_,m.products,begin,(!d_.O&&!d_.G)?static_cast<int>(m.products.size()):end,"sus2_gate");rank_plan(gate_,true);}
    graph_plan(outer_,m.products,outer_begin,static_cast<int>(m.products.size()),"sus2_outer");rank_plan(outer_,false);
    std::vector<int>ib(d_.F*9,-1),obasic(d_.F*9,-1),rankmap(d_.L*d_.O*9,-1),features,gs;
    for(std::size_t b=0;b<m.basic.size();++b){const auto&t=m.basic[b];const int pos=(t[0]*d_.L+t[1])*9+t[2]+t[1];if(d_.gated)ib[pos]=gate_.map[b];if(d_.mode<2)obasic[pos]=outer_.map[b];}
    for(int l=0;l<d_.L;++l)for(int o=0;o<d_.O;++o)for(int mm=-l;mm<=l;++mm)rankmap[(l*d_.O+o)*9+mm+l]=outer_.map[ob+kokkos_detail::ranked(d_.O,d_.H,l,o,mm)];
    for(int q:gfeatures)features.push_back(d_.gated?gate_.map[q]:-1);for(int q:gscalar)gs.push_back(d_.gated?gate_.map[q]:-1);
    std::vector<int>os;for(int q:m.scalar_mapping)os.push_back(outer_.map[q]);
    d_.inner_basic=parameter("sus2_inner_basic",ib);d_.outer_basic=parameter("sus2_outer_basic",obasic);d_.outer_rank=parameter("sus2_outer_rank",rankmap);d_.gate_feature=parameter("sus2_gate_features",features);d_.gate_scalar=parameter("sus2_gate_scalars",gs);d_.outer_scalar=parameter("sus2_outer_scalars",os);
    std::vector<double>norm(25,0.);const double pi=acos(-1.);for(int l=0;l<d_.L;++l)for(int mm=0;mm<=l;++mm){double ratio=1.;for(int a=l-mm+1;a<=l+mm;++a)ratio/=a;norm[kokkos_detail::harmonic(l,mm)]=sqrt((2*l+1)*ratio/(4*pi))*(mm?sqrt(2.)*(mm%2?-1:1):1.);}
    d_.norm=parameter("sus2_SH_norm",norm);
    const auto gt=m.numbers("two_layer_gate_type_coeffs"),add=m.numbers("two_layer_gate_additive_coeffs");std::vector<double>gate_species(d_.C*d_.F),outer_species(d_.C*d_.F),additive(d_.C*d_.F);
    for(int t=0;t<d_.C;++t)for(int mu=0;mu<d_.F;++mu){const double tau=at(m.radial_coeffs,mu*(d_.B+d_.C)+d_.B+t);outer_species[t*d_.F+mu]=tau;gate_species[t*d_.F+mu]=gt.empty()?tau:at(gt,t*d_.F+mu,at(gt,t,1.));additive[t*d_.F+mu]=at(add,t*d_.F+mu,at(add,t,1.));}
    d_.gate_species=parameter("sus2_gate_species",gate_species);d_.outer_species=parameter("sus2_outer_species",outer_species);d_.additive=parameter("sus2_additive",additive);
    const auto w=m.numbers("two_layer_gate_weights"),body=m.numbers("two_layer_gate_body_mix_weights");const auto order=m.integers("two_layer_gate_scalar_body_orders");const int count=static_cast<int>(gscalar.size()),nb=m.integer("two_layer_gate_body_order_max")-1;
    std::vector<double>gw(d_.mode==3?0:static_cast<std::size_t>(d_.F)*count);for(int mu=0;mu<d_.F&&d_.mode!=3;++mu)for(int q=0;q<count;++q)gw[mu*count+q]=m.text("two_layer_gate_mode")=="mu-body-linear-combo"?w.at(q)*body.at(mu*nb+order.at(q)-2):at(w,mu*count+q);
    d_.gate_weights=parameter("sus2_gate_projection",gw);d_.head=parameter("sus2_readout",m.numbers("moment_coeffs"));d_.species=parameter("sus2_species",m.numbers("species_coeffs"));d_.bias=parameter("sus2_bias",m.numbers("shift_coeffs"));d_.gate_head=parameter("sus2_gate_head",m.numbers("two_layer_gate_readout_coeffs"));
    d_.mixer=parameter("sus2_direct_k_mixer",m.numbers("two_layer_gate_radial_mixing_coeffs"));d_.scale=parameter("sus2_scaling",m.numbers("scal_coeffs"));
    std::vector<double>base(d_.Drows);for(int l=0;l<d_.L&&d_.mode==3;++l)for(int o=0;o<d_.O;++o)for(int n=0;n<d_.B;++n)base[(l*d_.O+o)*d_.B+n]=m.radial_coeffs.at((o*d_.L+l)*(d_.B+d_.C)+n);
    d_.base=parameter("sus2_adaptive_base",base);d_.amplitude=parameter("sus2_adaptive_amplitude",m.numbers("adaptive_radial_amplitudes"));d_.U=parameter("sus2_adaptive_U",m.numbers("adaptive_radial_weights"));d_.tensor=parameter("sus2_tensor_B",m.numbers("tensor_message_coeffs"));
    if(m.boolean("joint_readout_enabled")){
      const int jm=m.integer("joint_readout_alpha_moments_count");joint_.map.assign(jm,-1);const auto basics=m.rows("joint_readout_alpha_index_basic",3);std::vector<int>source,lvalues,mvalues,gate_source(basics.size()*d_.G,-1),outer_source(basics.size()*d_.O,-1);
      for(std::size_t b=0;b<basics.size();++b){use(joint_,static_cast<int>(b));const int src=static_cast<int>(basics[b][0]),l=static_cast<int>(basics[b][1]),mm=static_cast<int>(basics[b][2]);source.push_back(src);lvalues.push_back(l);mvalues.push_back(mm);for(int a=0;a<(src?d_.G:d_.O);++a){const int logical=(src?gb:ob)+kokkos_detail::ranked(src?d_.G:d_.O,d_.H,l,a,mm);if(src)gate_source[b*d_.G+a]=use(gate_,logical);else outer_source[b*d_.O+a]=use(outer_,logical);}}
      std::vector<Product>p;for(const auto&t:m.rows("joint_readout_sh_products",4))p.push_back({static_cast<int>(t[0]),static_cast<int>(t[1]),static_cast<int>(t[2]),t[3]});std::vector<int>mapping;for(int q:m.integers("joint_readout_scalar_mapping"))mapping.push_back(use(joint_,q));graph_plan(joint_,p,0,static_cast<int>(p.size()),"sus2_joint");
      d_.joint_basic_source=parameter("sus2_joint_source",source);d_.joint_basic_l=parameter("sus2_joint_l",lvalues);d_.joint_basic_m=parameter("sus2_joint_m",mvalues);d_.joint_gate_source=parameter("sus2_joint_gate_indices",gate_source);d_.joint_outer_source=parameter("sus2_joint_outer_indices",outer_source);joint_scalar_=parameter("sus2_joint_scalar",mapping);d_.joint_head=parameter("sus2_joint_head",m.numbers("joint_readout_coeffs"));
    }
    d_.atomic_number=parameter("sus2_ZBL_z",m.integers("zbl_atomic_numbers"));d_.zbl_inner=m.real("zbl_inner",.7);d_.zbl_outer=m.real("zbl_outer",1.4);d_.zbl_factor=m.boolean("zbl_typewise_cutoff_enabled")?m.real("zbl_typewise_cutoff_factor",.7):0.;
    error_flag=I("sus2_error",1);d_.error=error_flag;
  }
  int communication_width()const{return d_.W;}
  bool uses_generic_reference()const{return generic_reference_;}
  bool uses_forward_radial_tile()const{return forward_radial_tile_;}
  bool uses_reverse_center_reduce()const{return reverse_center_reduce_;}
  bool uses_forward_role_specialize()const{return forward_role_specialize_;}
  void configure_radial_counting(bool enabled){d_.radial.set_counting(enabled);}
  void configure_radial(RadialTableMode mode,double spacing,std::size_t budget,const std::vector<int>&species){d_.radial.configure(mode,spacing,budget,species);}
  auto radial_stats()const -> decltype(d_.radial.stats()){return d_.radial.stats();}
  int error_code()const{auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),error_flag);return h(0);}
  void prepare(int nowned,int nall,int nedges) {
    if(nowned<0||nall<nowned||nedges<0)throw std::runtime_error("SUS2 Kokkos invalid atom/edge layout");
    owned_=nowned;all_=nall;edges_=nedges;d_.owned=nowned;d_.all=nall;d_.edges=nedges;
    if(positions.extent(0)!=static_cast<std::size_t>(nall)||positions.extent(1)!=3){positions=PublicMatrix("sus2_positions",nall,3);types=I("sus2_types",nall);signal=PublicMatrix("sus2_signal",nall,d_.W);signal_adjoint=PublicMatrix("sus2_signal_adj",nall,d_.W);force=PublicMatrix("sus2_force",nall,3);}
    if(offsets.extent(0)!=static_cast<std::size_t>(nowned)+1)offsets=I("sus2_CSR_offsets",static_cast<std::size_t>(nowned)+1);
    if(neighbors.extent(0)!=static_cast<std::size_t>(nedges))neighbors=I("sus2_CSR_neighbors",nedges);
    if(site_energy.extent(0)!=static_cast<std::size_t>(nowned)||virial.extent(1)!=6){site_energy=D("sus2_site_energy",nowned);virial=PublicMatrix("sus2_virial",nowned,6);}
    auto matrix=[](Matrix&v,const char*name,int atoms,int width){if(v.extent(0)!=static_cast<std::size_t>(atoms)||v.extent(1)!=static_cast<std::size_t>(width))v=Matrix(std::string(name),atoms,width);};
    matrix(gate_.values,"sus2_gate_values",nowned,d_.gated?gate_.width:0);matrix(gate_.adjoint,"sus2_gate_adjoints",nowned,d_.gated?gate_.width:0);matrix(outer_.values,"sus2_outer_values",nowned,outer_.width);matrix(outer_.adjoint,"sus2_outer_adjoints",nowned,outer_.width);
    matrix(joint_.values,"sus2_joint_values",nowned,joint_.width);matrix(joint_.adjoint,"sus2_joint_adjoints",nowned,joint_.width);
    matrix(d_.projection,"sus2_projection",nall,d_.P);matrix(d_.coeff,"sus2_coefficients",nall,d_.Drows);matrix(d_.coeff_adj,"sus2_coefficient_adjoints",nall,d_.Drows);matrix(d_.tensor_value,"sus2_tensor_values",nall,d_.mode==3?d_.O*d_.tensor_components:0);matrix(d_.tensor_adj,"sus2_tensor_adjoints",nall,d_.mode==3?d_.O*d_.tensor_components:0);
    d_.x=positions;d_.type=types;d_.off=offsets;d_.nbr=neighbors;d_.s=signal;d_.sa=signal_adjoint;d_.f=force;d_.v=virial;d_.energy=site_energy;
    d_.gv=gate_.values;d_.ga=gate_.adjoint;d_.ov=outer_.values;d_.oa=outer_.adjoint;d_.jv=joint_.values;d_.ja=joint_.adjoint;
  }
  void begin() {
    Kokkos::deep_copy(error_flag,0);Kokkos::deep_copy(force,0.);Kokkos::deep_copy(virial,0.);Kokkos::deep_copy(site_energy,0.);Kokkos::deep_copy(signal,0.);Kokkos::deep_copy(signal_adjoint,0.);
    Kokkos::deep_copy(d_.ov,0.);Kokkos::deep_copy(d_.oa,0.);Kokkos::deep_copy(d_.gv,0.);Kokkos::deep_copy(d_.ga,0.);Kokkos::deep_copy(d_.jv,0.);Kokkos::deep_copy(d_.ja,0.);Kokkos::deep_copy(d_.coeff_adj,0.);Kokkos::deep_copy(d_.tensor_adj,0.);
    const auto layout=d_;
    Kokkos::parallel_for("sus2_validate_atom_types",Range(0,all_),KOKKOS_LAMBDA(std::size_t i){layout.valid_type(static_cast<int>(i));});
    if(!d_.gated||!owned_)return;raw_forward(true);execute(gate_.contract,d_.gv,d_.ga,false,true);execute(gate_.forward,d_.gv,d_.ga,false,false);const auto d=d_;const int qcount=static_cast<int>(d.gate_scalar.size());
    Kokkos::parallel_for("sus2_inner_signal",Range(0,static_cast<std::size_t>(owned_)*d.W),KOKKOS_LAMBDA(std::size_t idx){const int i=static_cast<int>(idx%d.owned),q=static_cast<int>(idx/d.owned);double z=0.;if(d.mode==3)z=d.gv(i,d.gate_feature(q));else for(int s=0;s<qcount;++s)z+=d.gate_weights(q*qcount+s)*d.gv(i,d.gate_scalar(s));d.s(i,q)=z;});
  }
  void outer() {
    if(d_.mode==3)adaptive_prepare();if(d_.mode<2){raw_forward(false);execute(outer_.contract,d_.ov,d_.oa,false,true);}if(d_.mixing||d_.mode==3)message_forward();execute(outer_.forward,d_.ov,d_.oa,false,false);
    const auto d=d_;const int scalars=static_cast<int>(d.outer_scalar.size());
    Kokkos::parallel_for("sus2_outer_readout",Range(0,owned_),KOKKOS_LAMBDA(std::size_t ii){const int i=static_cast<int>(ii),type=d.type(i);if(!d.valid_type(i))return;double energy=d.species(type)+d.bias(type),center=d.species_readout?1.:d.species(type);for(int q=0;q<scalars;++q){const double c=center*d.head((d.species_readout?type*scalars:0)+q);energy+=c*d.ov(i,d.outer_scalar(q));d.oa(i,d.outer_scalar(q))+=c;}d.energy(i)=energy;});
    joint_forward_reverse();execute(outer_.reverse,d_.ov,d_.oa,true,false);execute(outer_.transpose,d_.ov,d_.oa,true,true);
    if(d_.mode<2)raw_reverse(false);if(d_.mixing||d_.mode==3)message_reverse();
  }
  void finish() {
    const auto d=d_;const int qcount=static_cast<int>(d.gate_scalar.size());
    if(d_.gated){
      // These are owned-node seeds AFTER the caller has completed reverse
      // signal communication. Joint local primitive seeds already live in ga.
      Kokkos::parallel_for("sus2_inner_adjoint_seeds",Range(0,owned_),KOKKOS_LAMBDA(std::size_t ii){const int i=static_cast<int>(ii);if(!d.valid_type(i))return;if(d.mode==3){for(int q=0;q<d.W;++q)d.ga(i,d.gate_feature(q))+=d.sa(i,q);}else for(int q=0;q<qcount;++q){double z=0.;for(int mu=0;mu<d.F;++mu)z+=d.sa(i,mu)*d.gate_weights(mu*qcount+q);d.ga(i,d.gate_scalar(q))+=z;}
        if(d.gate_readout){const double center=d.species_readout?1.:d.species(d.type(i));double energy=0.;for(int q=0;q<qcount;++q){const double c=center*d.gate_head(q);d.ga(i,d.gate_scalar(q))+=c;energy+=c*d.gv(i,d.gate_scalar(q));}d.energy(i)+=energy;}
      });
      execute(gate_.reverse,d_.gv,d_.ga,true,false);execute(gate_.transpose,d_.gv,d_.ga,true,true);raw_reverse(true);
    }
    if(d_.zbl)Kokkos::parallel_for("sus2_ZBL",Range(0,owned_),KOKKOS_LAMBDA(std::size_t ii){const int i=static_cast<int>(ii);int lo,hi;if(!d.edge_range(i,lo,hi))return;double energy=0.;for(int e=lo;e<hi;++e){const int j=d.nbr(e);double delta[3],r;if(!d.geometry(i,j,delta,r))continue;const auto p=sus2_sl_zbl::ComputePairKokkos<double>(d.atomic_number(d.type(i)),d.atomic_number(d.type(j)),r,d.zbl_inner,d.zbl_outer,d.zbl_factor);energy+=.5*p.energy;double g[3];for(int q=0;q<3;++q)g[q]=.5*p.dEdr*delta[q]/r;d.add_edge(i,j,delta,g);}d.energy(i)+=energy;});
  }
  double memory_usage()const {
    std::size_t bytes=model_bytes_+d_.radial.memory_usage();
    const Matrix*a[]={&d_.gv,&d_.ga,&d_.ov,&d_.oa,&d_.jv,&d_.ja,&d_.projection,&d_.coeff,&d_.coeff_adj,&d_.tensor_value,&d_.tensor_adj};for(const auto*v:a)bytes+=v->size()*sizeof(double);
    bytes+=(positions.size()+signal.size()+signal_adjoint.size()+force.size()+virial.size()+site_energy.size())*sizeof(double)+(types.size()+offsets.size()+neighbors.size()+error_flag.size())*sizeof(int);
    const Stage*s[]={&gate_,&outer_,&joint_};for(const auto*stage:s)bytes+=stage->forward.bytes()+stage->reverse.bytes()+stage->contract.bytes()+stage->transpose.bytes();return static_cast<double>(bytes);
  }
};
}}
#endif
