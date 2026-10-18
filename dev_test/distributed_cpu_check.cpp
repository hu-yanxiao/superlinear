// Deterministic in-process MPI simulation. Uses the actual staged inference API,
// including a rank with no owned atoms; no production formula is copied here.
#include "../dev_src/mtpr.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Result { double energy=0; std::vector<Vector3> forces; Matrix3 stress; };
struct Domain {
  std::unique_ptr<MLMTPR> model;
  Configuration cfg;
  Neighborhoods nbhs;
  std::vector<int> global, local;
  int owned=0;
};
void check(bool condition,const char* what) { if(!condition) throw std::runtime_error(what); }
void near(double a,double b,double atol,double rtol,const char* what,double& maximum) {
  check(std::isfinite(a)&&std::isfinite(b),"non-finite numerical result");
  maximum=std::max(maximum,std::abs(a-b));
  if(std::abs(a-b)>atol+rtol*std::max(std::abs(a),std::abs(b))) {
    std::cerr<<what<<": "<<std::setprecision(17)<<a<<" versus "<<b<<"\n";
    throw std::runtime_error(what);
  }
}
Result evaluate(const std::string& path,const Configuration& input,int ranks,bool table) {
  MLMTPR geometry_model(path);
  Neighborhoods global(input,geometry_model.CutOff());
  std::vector<std::unique_ptr<Domain>> domains;
  const int n=input.size();
  for(int rank=0;rank<ranks;++rank) {
    domains.emplace_back(new Domain);
    Domain& d=*domains.back();
    d.model.reset(new MLMTPR(path));
    d.model->p_RadialBasis->ConfigureInferenceRadialTable(
        table ? RadialTableMode::Table : RadialTableMode::Direct,0.001,64u*1024u*1024u);
    for(int i=0;i<n;++i) if(i%ranks==rank) d.global.push_back(i);
    d.owned=d.global.size();
    for(int i=0;i<n;++i) if(i%ranks!=rank) d.global.push_back(i);
    d.local.resize(n);
    for(int i=0;i<n;++i) d.local[d.global[i]]=i;
    d.cfg=input;
    d.nbhs.resize(n); d.nbhs.cutoff=global.cutoff;
    for(int i=0;i<n;++i) {
      const int g=d.global[i];
      d.cfg.pos(i)=input.pos(g); d.cfg.type(i)=input.type(g);
      Neighborhood& hood=d.nbhs[i]; hood.my_type=input.type(g); hood.count=0;
      if(i<d.owned) {
        hood=global[g];
        for(int& j:hood.inds) j=d.local[j];
      }
    }
    d.model->BeginDistributedEFS(d.cfg,d.nbhs,d.owned);
  }
  for(auto& target:domains) {
    const int width=target->model->DistributedForwardWidth();
    std::vector<double> buffer(std::max(1,width));
    for(int i=target->owned;i<n;++i) {
      const int g=target->global[i]; Domain& owner=*domains[g%ranks];
      check(width==owner.model->DistributedForwardWidth(),"forward width mismatch");
      owner.model->PackDistributedForward(owner.local[g],buffer.data());
      target->model->UnpackDistributedForward(i,buffer.data());
    }
  }
  for(auto& d:domains) d->model->EvaluateDistributedOuterEFS(d->cfg,d->nbhs);
  for(auto& source:domains) {
    const int width=source->model->DistributedReverseWidth();
    std::vector<double> buffer(std::max(1,width));
    for(int i=source->owned;i<n;++i) {
      const int g=source->global[i]; Domain& owner=*domains[g%ranks];
      check(width==owner.model->DistributedReverseWidth(),"reverse width mismatch");
      source->model->PackDistributedReverse(i,buffer.data());
      owner.model->AddDistributedReverse(owner.local[g],buffer.data());
    }
  }
  Result result; result.forces.resize(n);
  for(int a=0;a<3;++a) for(int b=0;b<3;++b) result.stress[a][b]=0;
  for(Vector3& f:result.forces) for(int a=0;a<3;++a) f[a]=0;
  size_t hits=0;
  for(auto& d:domains) {
    d->model->FinishDistributedEFS(d->cfg,d->nbhs);
    result.energy+=d->cfg.energy;
    for(int a=0;a<3;++a) for(int b=0;b<3;++b) result.stress[a][b]+=d->cfg.stresses[a][b];
    for(int i=0;i<n;++i) result.forces[d->global[i]]+=d->cfg.force(i);
    hits+=d->model->p_RadialBasis->GetInferenceRadialTableStats().hits;
  }
  if(table) check(hits>0,"table requested but no table evaluation occurred");
  return result;
}
}

int main(int argc,char** argv) {
  try {
    check(argc==3,"usage: distributed_cpu_check MODEL CFG");
    Configuration input; std::ifstream stream(argv[2]); check(input.Load(stream),"cannot read cfg");
    MLMTPR model(argv[1]); Configuration reference=input; model.CalcEFS(reference);
    double direct_error=0,table_error=0,fd_error=0;
    for(bool table:{false,true}) for(int ranks:{1,2,4}) {
      Result r=evaluate(argv[1],input,ranks,table);
      double& error=table?table_error:direct_error;
      const double tol=table?2e-7:2e-10;
      near(r.energy,reference.energy,tol,tol,"energy",error);
      for(int i=0;i<input.size();++i) for(int a=0;a<3;++a)
        near(r.forces[i][a],reference.force(i)[a],tol,tol,"force",error);
      for(int a=0;a<3;++a) for(int b=0;b<3;++b)
        near(r.stress[a][b],reference.stresses[a][b],tol,tol,"virial",error);
    }
    // Differentiate the actual table energy, not the analytic reference.
    const double h=2e-6;
    Result center=evaluate(argv[1],input,2,true);
    Configuration plus=input,minus=input;
    plus.pos(0)[0]+=h; minus.pos(0)[0]-=h;
    near(center.forces[0][0],-(evaluate(argv[1],plus,2,true).energy-
         evaluate(argv[1],minus,2,true).energy)/(2*h),2e-6,2e-5,"table energy-force FD",fd_error);
    // x-strain changes both cell vectors and coordinates. Native stress is
    // minus the energy derivative w.r.t. strain (not pressure / volume).
    plus=input; minus=input;
    for(int i=0;i<input.size();++i) {plus.pos(i)[0]*=(1+h);minus.pos(i)[0]*=(1-h);}
    for(int i=0;i<3;++i) {plus.lattice[i][0]*=(1+h);minus.lattice[i][0]*=(1-h);}
    near(center.stress[0][0],-(evaluate(argv[1],plus,2,true).energy-
         evaluate(argv[1],minus,2,true).energy)/(2*h),3e-6,2e-5,"table energy-strain FD",fd_error);
    std::cout<<std::setprecision(17)<<"DISTRIBUTED_CPU_PASS {\"direct_max_abs\":"<<direct_error
      <<",\"table_max_abs\":"<<table_error<<",\"table_fd_max_abs\":"<<fd_error<<"}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
