// Empty spatial domains are valid MPI participants even with no ghost atoms.
#include "../dev_src/mtpr.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("usage: distributed_empty_domain_check MODEL");
        MLMTPR model(argv[1]);
        Configuration cfg;
        Neighborhoods neighborhoods;
        neighborhoods.cutoff=model.CutOff();
        if (model.HasZBL())
            neighborhoods.cutoff=std::max(neighborhoods.cutoff,model.ZBL().MaxOuterCutoff());
        for (RadialTableMode mode:{RadialTableMode::Direct,RadialTableMode::Table,RadialTableMode::Auto}) {
            model.ConfigureDistributedRadialTable(mode,0.001);
            model.BeginDistributedEFS(cfg,neighborhoods,0);
            if (model.p_RadialBasis->IsInferenceRadialTableActive())
                throw std::runtime_error("radial inference scope escaped Begin");
            model.EvaluateDistributedOuterEFS(cfg,neighborhoods);
            model.FinishDistributedEFS(cfg,neighborhoods);
            if (cfg.energy!=0.0 || !cfg.cal_se.empty())
                throw std::runtime_error("empty domain produced energy");
            for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                if (cfg.stresses[a][b]!=0.0)
                    throw std::runtime_error("empty domain produced stress");
            if (model.p_RadialBasis->IsInferenceRadialTableActive())
                throw std::runtime_error("radial inference scope escaped Finish");
            // Abort restores the stage after host communication cancellation.
            model.BeginDistributedEFS(cfg,neighborhoods,0);
            model.AbortDistributedEFS();
        }
        std::cout<<"DISTRIBUTED_EMPTY_DOMAIN_PASS\n";
        return 0;
    } catch (const MlipException& e) {
        std::cerr<<e.What()<<"\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<"\n";
        return 1;
    }
}
