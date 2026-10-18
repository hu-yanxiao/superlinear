// ZBL oracle independent of native CalcEFS's historical adaptive ZBL path.
// Run against static, Legacy/Effect, pure-W, and Adaptive fixture models.
#include "../dev_src/mtpr.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
void Near(double a,double b,double tolerance,const char* message,double& error) {
    Require(std::isfinite(a)&&std::isfinite(b),"non-finite ZBL oracle result");
    error=std::max(error,std::abs(a-b));
    if (std::abs(a-b)>tolerance*(1.0+std::max(std::abs(a),std::abs(b)))) {
        std::cerr<<message<<": "<<a<<" != "<<b<<"\n";
        throw std::runtime_error(message);
    }
}
void Configure(MLMTPR& model,bool typewise) {
    std::vector<int> zs(model.species_count,1);
    if (model.species_count>1) zs[1]=8;
    const double cutoff=model.CutOff();
    // H-H gets 1.2*ML cutoff; H-O/O-O reach the global 1.4*ML cap.
    const double factor=1.2*cutoff/(2.0*ZBLCovalentRadius(1));
    model.ConfigureZBL(zs,0.2*cutoff,1.4*cutoff,typewise,factor);
}
Configuration Geometry(MLMTPR& model) {
    Configuration cfg;
    cfg.resize(4);
    const double cutoff=model.CutOff();
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
        cfg.lattice[a][b]=a==b ? 8.0*cutoff : 0.0;
    const double coordinates[4][3]={{0.30,0.40,0.20},{0.52,0.43,0.22},
        {1.42,0.40,0.20},{0.77,0.64,0.43}};
    for (int i=0;i<4;++i) {
        cfg.type(i)=i%model.species_count;
        cfg.unique_elems.insert(cfg.type(i));
        for (int a=0;a<3;++a) cfg.pos(i)[a]=coordinates[i][a]*cutoff;
    }
    return cfg;
}
struct Domain {
    std::unique_ptr<MLMTPR> model;
    Configuration cfg;
    Neighborhoods neighborhoods;
    std::vector<int> global,local;
    int owned=0;
};
Configuration Staged(const std::string& path,const Configuration& input,
                     int ranks,bool typewise,bool table) {
    MLMTPR template_model(path); Configure(template_model,typewise);
    Neighborhoods global(input,template_model.ZBL().MaxOuterCutoff());
    const int n=input.size();
    std::vector<std::unique_ptr<Domain>> domains;
    for (int rank=0;rank<ranks;++rank) {
        domains.emplace_back(new Domain);
        Domain& d=*domains.back(); d.model.reset(new MLMTPR(path));
        Configure(*d.model,typewise);
        d.model->ConfigureDistributedRadialTable(table ? RadialTableMode::Table : RadialTableMode::Direct,0.0002);
        for (int i=0;i<n;++i) if (i%ranks==rank) d.global.push_back(i);
        d.owned=d.global.size();
        for (int i=0;i<n;++i) if (i%ranks!=rank) d.global.push_back(i);
        d.local.resize(n);
        for (int i=0;i<n;++i) d.local[d.global[i]]=i;
        d.cfg=input; d.neighborhoods.resize(n); d.neighborhoods.cutoff=global.cutoff;
        for (int i=0;i<n;++i) {
            const int g=d.global[i];
            d.cfg.type(i)=input.type(g); d.cfg.pos(i)=input.pos(g);
            Neighborhood& nbh=d.neighborhoods[i]; nbh.count=0; nbh.my_type=d.cfg.type(i);
            if (i<d.owned) {
                nbh=global[g];
                for (int& j:nbh.inds) j=d.local[j];
            }
        }
        d.model->BeginDistributedEFS(d.cfg,d.neighborhoods,d.owned);
    }
    for (auto& target:domains) {
        std::vector<double> buffer(std::max(1,target->model->DistributedForwardWidth()));
        for (int i=target->owned;i<n;++i) {
            const int g=target->global[i]; Domain& owner=*domains[g%ranks];
            owner.model->PackDistributedForward(owner.local[g],buffer.data());
            target->model->UnpackDistributedForward(i,buffer.data());
        }
    }
    for (auto& d:domains) d->model->EvaluateDistributedOuterEFS(d->cfg,d->neighborhoods);
    for (auto& source:domains) {
        std::vector<double> buffer(std::max(1,source->model->DistributedReverseWidth()));
        for (int i=source->owned;i<n;++i) {
            const int g=source->global[i]; Domain& owner=*domains[g%ranks];
            source->model->PackDistributedReverse(i,buffer.data());
            owner.model->AddDistributedReverse(owner.local[g],buffer.data());
        }
    }
    Configuration result=input;
    result.has_forces(true); result.has_site_energies(true); result.has_stresses(true);
    result.has_energy(true); result.energy=0;
    result.cal_se.assign(n,0.0);
    for (int i=0;i<n;++i) result.force(i)=Vector3(0,0,0);
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) result.stresses[a][b]=0;
    for (auto& d:domains) {
        d->model->FinishDistributedEFS(d->cfg,d->neighborhoods);
        result.energy+=d->cfg.energy;
        for (int i=0;i<n;++i) result.force(d->global[i])+=d->cfg.force(i);
        for (int i=0;i<d->owned;++i) result.cal_se[d->global[i]]+=d->cfg.cal_se[i];
        for (int a=0;a<3;++a) for (int b=0;b<3;++b)
            result.stresses[a][b]+=d->cfg.stresses[a][b];
    }
    return result;
}

// LAMMPS gives distinct ghost slots to different periodic images of the same
// physical atom. Test those slots, not just repeated owner indices in rows.
Configuration ImageStaged(const std::string& path,const Configuration& input,bool typewise) {
    MLMTPR model(path); Configure(model,typewise);
    model.ConfigureDistributedRadialTable(RadialTableMode::Table,0.0002);
    Neighborhoods global(input,model.ZBL().MaxOuterCutoff());
    Require(input.size()==1 && global[0].count>1,"periodic image fixture must contain repeated self images");
    const int count=global[0].count;
    Configuration cfg=input; cfg.resize(1+count);
    Neighborhoods neighborhoods; neighborhoods.resize(1+count); neighborhoods.cutoff=global.cutoff;
    neighborhoods[0]=global[0];
    for (int j=0;j<count;++j) {
        cfg.type(1+j)=cfg.type(0);
        cfg.pos(1+j)=cfg.pos(0)+global[0].vecs[j];
        neighborhoods[0].inds[j]=1+j;
        neighborhoods[1+j].count=0; neighborhoods[1+j].my_type=cfg.type(0);
    }
    model.BeginDistributedEFS(cfg,neighborhoods,1);
    std::vector<double> forward(std::max(1,model.DistributedForwardWidth()));
    model.PackDistributedForward(0,forward.data());
    for (int j=0;j<count;++j) model.UnpackDistributedForward(1+j,forward.data());
    model.EvaluateDistributedOuterEFS(cfg,neighborhoods);
    std::vector<double> reverse(std::max(1,model.DistributedReverseWidth()));
    for (int j=0;j<count;++j) {
        model.PackDistributedReverse(1+j,reverse.data());
        model.AddDistributedReverse(0,reverse.data());
    }
    model.FinishDistributedEFS(cfg,neighborhoods);
    for (int j=0;j<count;++j) cfg.force(0)+=cfg.force(1+j);
    return cfg;
}
}

int main(int argc,char** argv) {
    try {
        Require(argc==2,"usage: distributed_zbl_oracle_check MODEL");
        MLMTPR model(argv[1]);
        const Configuration geometry=Geometry(model);
        double oracle_error=0,fd_error=0;
        for (bool typewise:{false,true}) {
            Configure(model,typewise);
            Configuration reference=geometry;
            model.SetZBLEvaluationEnabled(false);
            model.CalcEFS(reference);
            // This standalone native ZBL oracle never enters CalcEFS.
            const ZBLEFS zbl=model.ZBL().Compute(geometry);
            Require(zbl.energy>0,"vacuous ZBL oracle");
            for (bool table:{false,true}) for (int ranks:{1,2,5}) {
                const Configuration actual=Staged(argv[1],geometry,ranks,typewise,table);
                const double tolerance=table ? 2e-7 : 2e-10;
                Near(actual.energy,reference.energy+zbl.energy,tolerance,"ZBL total energy",oracle_error);
                double site_sum=0;
                for (double value:actual.cal_se) site_sum+=value;
                Near(site_sum,actual.energy,2e-10,"ZBL site energy sum",oracle_error);
                for (int i=0;i<geometry.size();++i) for (int a=0;a<3;++a)
                    Near(actual.force(i)[a],reference.force(i)[a]+zbl.forces[i][a],
                         tolerance,"ZBL total force",oracle_error);
                for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                    Near(actual.stresses[a][b],reference.stresses[a][b]+zbl.stresses[a][b],
                         tolerance,"ZBL total virial",oracle_error);
            }
            // Check derivatives of the composed staged energy, including the
            // cutoff-separated source gate chain and all six box components.
            const double step=2e-6;
            const Configuration center=Staged(argv[1],geometry,2,typewise,true);
            Configuration plus=geometry,minus=geometry;
            plus.pos(0)[0]+=step; minus.pos(0)[0]-=step;
            Near(center.force(0)[0],-(Staged(argv[1],plus,2,typewise,true).energy
                 -Staged(argv[1],minus,2,typewise,true).energy)/(2*step),
                 2e-5,"ZBL composed force FD",fd_error);
            for (int a=0;a<3;++a) for (int b=a;b<3;++b) {
                plus=geometry; minus=geometry;
                for (int i=0;i<geometry.size();++i) {
                    plus.pos(i)[a]+=step*geometry.pos(i)[b];
                    minus.pos(i)[a]-=step*geometry.pos(i)[b];
                }
                for (int k=0;k<3;++k) {
                    plus.lattice[k][a]+=step*geometry.lattice[k][b];
                    minus.lattice[k][a]-=step*geometry.lattice[k][b];
                }
                Near(center.stresses[a][b],-(Staged(argv[1],plus,2,typewise,true).energy
                     -Staged(argv[1],minus,2,typewise,true).energy)/(2*step),
                     2e-5,"ZBL composed strain FD",fd_error);
            }

            Configuration periodic;
            periodic.resize(1); periodic.type(0)=0; periodic.unique_elems.insert(0);
            const double sides[3]={0.72,0.83,0.94};
            for (int a=0;a<3;++a) {
                periodic.pos(0)[a]=0.2*model.CutOff();
                for (int b=0;b<3;++b)
                    periodic.lattice[a][b]=a==b ? sides[a]*model.CutOff() : 0.0;
            }
            Configuration periodic_reference=periodic;
            model.SetZBLEvaluationEnabled(false); model.CalcEFS(periodic_reference);
            const ZBLEFS periodic_zbl=model.ZBL().Compute(periodic);
            const Configuration images=ImageStaged(argv[1],periodic,typewise);
            Near(images.energy,periodic_reference.energy+periodic_zbl.energy,
                 2e-7,"duplicate periodic image energy",oracle_error);
            for (int a=0;a<3;++a) {
                Near(images.force(0)[a],periodic_reference.force(0)[a]+periodic_zbl.forces[0][a],
                     2e-7,"duplicate periodic image force",oracle_error);
                for (int b=0;b<3;++b)
                    Near(images.stresses[a][b],periodic_reference.stresses[a][b]+periodic_zbl.stresses[a][b],
                         2e-7,"duplicate periodic image virial",oracle_error);
            }
            Require(std::abs(images.stresses[0][0])>1e-12,"periodic self-image virial is vacuous");
            plus=periodic; minus=periodic;
            plus.lattice[0][0]*=(1+step); minus.lattice[0][0]*=(1-step);
            plus.pos(0)[0]*=(1+step); minus.pos(0)[0]*=(1-step);
            Near(images.stresses[0][0],-(ImageStaged(argv[1],plus,typewise).energy
                 -ImageStaged(argv[1],minus,typewise).energy)/(2*step),
                 2e-5,"periodic self-image strain FD",fd_error);
        }
        std::cout<<"DISTRIBUTED_ZBL_ORACLE_PASS max_abs="<<oracle_error
                 <<" fd_max_abs="<<fd_error<<"\n";
        return 0;
    } catch (const MlipException& error) {
        std::cerr<<error.What()<<"\n"; return 1;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<"\n"; return 1;
    }
}
