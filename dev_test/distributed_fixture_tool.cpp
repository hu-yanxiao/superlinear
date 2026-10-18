// Small native fixture driver, deliberately sharing production initialization
// and model I/O without linking the optimizer/trainer into interface tests.
#include "../dev_src/mtpr.h"
#include "../dev_src/sh_model_init.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
  try {
    if(argc<3) throw std::runtime_error("expected init-sh, roundtrip-model-dev, or eval-efs");
    const std::string command=argv[1];
    if(command=="init-sh") {
      std::map<std::string,std::string> opts;
      for(int i=3;i<argc;++i) {
        std::string arg=argv[i];
        if(arg.substr(0,2)!="--") throw std::runtime_error("expected option");
        arg=arg.substr(2);const auto eq=arg.find('=');
        opts[arg.substr(0,eq)]=eq==std::string::npos?"true":arg.substr(eq+1);
      }
      WriteSphericalHarmonicModel(argv[2],opts);
    } else if(command=="roundtrip-model-dev" && argc==4) {
      MLMTPR model(argv[2]);model.Save(argv[3]);
    } else if(command=="eval-efs" && argc==4) {
      MLMTPR model(argv[2]); Configuration cfg;
      std::ifstream input(argv[3]);
      if(!cfg.Load(input)) throw std::runtime_error("cannot read configuration");
      // Compose independent native ML and exactly one standalone ZBL term;
      // this avoids the historical Adaptive CalcEFS double-ZBL path.
      const bool zbl_enabled=model.HasZBL();
      model.SetZBLEvaluationEnabled(false);
      model.CalcEFS(cfg);
      if(zbl_enabled) {
        const auto& zbl=model.ZBL();
        const Neighborhoods neighborhoods(cfg,zbl.MaxOuterCutoff());
        const ZBLEFS repulsion=zbl.Compute(cfg,neighborhoods);
        cfg.energy+=repulsion.energy;
        for(int i=0;i<cfg.size();++i) cfg.force(i)+=repulsion.forces[i];
        cfg.stresses+=repulsion.stresses;
        const auto& zs=zbl.AtomicNumbers();
        for(int i=0;i<cfg.size();++i) {
          const auto& hood=neighborhoods[i];
          for(int edge=0;edge<hood.count;++edge)
            cfg.cal_se[i]+=0.5*ComputeZBLPair(zs.at(cfg.type(i)),zs.at(hood.types[edge]),
              hood.dists[edge],zbl.InnerCutoff(),zbl.OuterCutoff(),
              zbl.TypewiseCutoffEnabled()?zbl.TypewiseCutoffFactor():0.0).energy;
        }
      }
      auto number=[](double value) {
        if(!std::isfinite(value)) throw std::runtime_error("non-finite native EFS");
        std::cout<<value;
      };
      if(cfg.cal_se.size()!=static_cast<std::size_t>(cfg.size()))
        throw std::runtime_error("native site energies missing");
      std::cout<<std::setprecision(17)<<"NATIVE_EFS_JSON {\"energy\":";
      number(cfg.energy); std::cout<<",\"forces\":[";
      for(int i=0;i<cfg.size();++i) {
        if(i) std::cout<<','; std::cout<<'[';
        for(int a=0;a<3;++a) {if(a) std::cout<<',';number(cfg.force(i)[a]);}
        std::cout<<']';
      }
      std::cout<<"],\"virial\":[";
      for(int a=0;a<3;++a) {
        if(a) std::cout<<','; std::cout<<'[';
        for(int b=0;b<3;++b) {if(b) std::cout<<',';number(cfg.stresses[a][b]);}
        std::cout<<']';
      }
      std::cout<<"],\"site_energies\":[";
      for(int i=0;i<cfg.size();++i) {if(i) std::cout<<',';number(cfg.cal_se[i]);}
      std::cout<<"]}\n";
    } else throw std::runtime_error("invalid fixture command");
    return 0;
  } catch(const MlipException& e) {std::cerr<<e.What()<<"\n";return 1;}
    catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
