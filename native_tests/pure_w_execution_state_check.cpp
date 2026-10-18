#include "dev_src/mtpr.h"
#include <cmath>
#include <fstream>
#include <iostream>
#ifdef MLIP_MPI
#include <mpi.h>
#endif

static int Check(const char* model_path, const char* data_path)
{
    MLMTPR model(model_path);
    Configuration seed;
    std::ifstream input(data_path);
    if (!seed.Load(input) || seed.size() == 0) return 2;
    std::vector<double> original(model.Coeff(),model.Coeff()+model.CoeffCount());
    // Nonuniform species factors catch omitted head multipliers in barQ.
    for (int t=0;t<model.species_count;++t)
        original[model.LinearCoeffOffset()+t]=0.65+0.9*t;
    double max_error=0.0, max_net_force=0.0;
    for (int frame=0;frame<12;++frame) {
        for (int p=0;p<model.CoeffCount();++p)
            model.Coeff()[p]=original[p]*(1.0+0.003*std::sin(0.7*frame+0.3*p));
        Configuration geometry=seed;
        for (int i=0;i<geometry.size();++i)
            for (int a=0;a<3;++a)
                geometry.pos(i,a)+=0.007*std::sin(frame*0.37+i*0.71+a*0.43);
        for (int a=0;a<3;++a) geometry.lattice[a][a]*=1.0+0.0001*frame;
        // Change neighborhood topology and buffer lengths, including an
        // isolated frame, then return to dense geometry on the same object.
        const double dilation=frame%4==1 ? 8.0 : (frame%4==2 ? 0.9 : 1.0);
        for (int i=0;i<geometry.size();++i)
            for (int a=0;a<3;++a) geometry.pos(i,a)*=dilation;
        for (int a=0;a<3;++a) for (int b=0;b<3;++b) geometry.lattice[a][b]*=dilation;
        Configuration fast=geometry, reference=geometry, again=geometry, energy=geometry;
        model.CalcEFS(fast);
        model.RequestTwoLayerFullEdgeCacheForNextCalcEFS();
        model.CalcEFS(reference); // original EFS path, complete derivative cache
        model.CalcEnergyAndSiteEnergies(energy);
        model.CalcEFS(again); // must not reuse stale graph/W/geometry state
        auto compare=[&](double x,double y) {
            if (!std::isfinite(x)||!std::isfinite(y)) max_error=1.0;
            else max_error=std::max(max_error,std::abs(x-y)/std::max(1.0,std::max(std::abs(x),std::abs(y))));
        };
        compare(fast.energy,reference.energy);
        compare(fast.energy,again.energy);
        compare(fast.energy,energy.energy);
        double net[3]={0,0,0};
        for (int i=0;i<geometry.size();++i) {
            compare(fast.cal_se[i],reference.cal_se[i]);
            for (int a=0;a<3;++a) {
                compare(fast.force(i,a),reference.force(i,a));
                compare(fast.force(i,a),again.force(i,a));
                net[a]+=fast.force(i,a);
            }
        }
        for (int a=0;a<3;++a) {
            max_net_force=std::max(max_net_force,std::abs(net[a]));
            for (int b=0;b<3;++b) compare(fast.stresses[a][b],reference.stresses[a][b]);
        }
    }
    std::cout << "mutable_geometry_coefficients_full_cache_transition max_scaled_error="
              << max_error << " net_force=" << max_net_force << std::endl;
    return max_error<2e-10 && max_net_force<2e-10 ? 0 : 1;
}
int main(int argc,char** argv)
{
#ifdef MLIP_MPI
    MPI_Init(&argc,&argv);
#endif
    const int rc=argc==3 ? Check(argv[1],argv[2]) : 2;
#ifdef MLIP_MPI
    MPI_Finalize();
#endif
    return rc;
}
