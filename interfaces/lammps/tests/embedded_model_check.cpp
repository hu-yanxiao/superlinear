// Standalone parser regression check. Build with sus2_sl_model.cpp only.
// Usage: embedded_model_check MODEL... | --defaults INPUT NATIVE_SAVED
//        embedded_model_check --self-test VALID_MODEL
#include "sus2_sl_model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>
using LAMMPS_NS::SUS2SL::Model;
namespace {
void check(bool b,const std::string &why) {if(!b)throw std::runtime_error(why);}
struct Temporary {
  std::string path;
  Temporary() {char name[]="/tmp/sus2-embedded-model-XXXXXX";int fd=mkstemp(name);check(fd>=0,"mkstemp failed");close(fd);path=name;}
  ~Temporary(){std::remove(path.c_str());}
};
std::string replace(const std::string &s,const std::string &key,const std::string &value) {
  const auto start=s.find(key+" = ");check(start!=std::string::npos,"missing mutation field "+key);
  const auto end=s.find('\n',start);return s.substr(0,start)+key+" = "+value+(end==std::string::npos ? "\n" : s.substr(end));
}
void reject(const std::string &s,const std::string &name) {
  Temporary tmp;{std::ofstream out(tmp.path.c_str());out<<s;}
  bool failed=false;try{Model m;m.load(tmp.path);}catch(const std::exception&){failed=true;}
  check(failed,"parser accepted malformed model: "+name);
}
void self_test(const std::string &path) {
  Model original;original.load(path);
  std::ifstream in(path.c_str());std::string s((std::istreambuf_iterator<char>(in)),{});
  reject(s+"\nspecies_count = 2\n","duplicate fields");
  reject(replace(s,"species_count","0"),"zero species");
  reject(replace(s,"alpha_moment_mapping","{999999999}"),"invalid scalar graph");
  reject(replace(s,"sh_k_max","1.5"),"fractional dimension");
  reject(replace(s,"sh_l_max","7"),"angular range");
  reject(replace(s,"max_dist","nan"),"nonfinite radius");
  reject(replace(s,"moment_coeffs","{}"),"missing readout coefficients");
  reject(replace(s,"moment_coeffs","{1,}"),"trailing comma");
  reject(replace(s,"moment_coeffs","{1} garbage"),"trailing list text");
  reject(replace(s,"version","9.9.0"),"unknown version");
  reject(s+"\ntwo_layer_residual_enabled = true\n","old residual");
  reject(s+"\nmulti_layer_gate_enabled = true\n","old multilayer");
  if(original.has("adaptive_radial_weights")) {
    reject(replace(s,"adaptive_radial_weights","{nan}"),"nonfinite adaptive weights");
    reject(replace(s,"adaptive_radial_projection","unknown"),"unknown adaptive projection");
    reject(s+"\nadaptive_radial_unknown = 1\n","unknown adaptive field");
    reject(replace(s,"adaptive_radial_amplitude_count","0"),"inconsistent adaptive count");
  }
}
void defaults(const std::string &a,const std::string &b) {
  Model source,saved;source.load(a);saved.load(b);
  const char* keys[]={"shift_coeffs","scal_coeffs","radial_coeffs","species_coeffs","moment_coeffs"};
  for(const char *key:keys){const auto x=source.numbers(key),y=saved.numbers(key);check(x.size()==y.size(),std::string(key)+" size differs");for(size_t i=0;i<x.size();++i)check(std::abs(x[i]-y[i])<=2e-14*std::max(1.,std::max(std::abs(x[i]),std::abs(y[i]))),std::string(key)+" differs from native saved defaults");}
}
}
int main(int argc,char**argv) {
  try {
    check(argc>1,"expected models, --self-test VALID_MODEL, or --defaults INPUT NATIVE_SAVED");
    if(std::string(argv[1])=="--self-test") {check(argc==3,"self-test needs one valid model");self_test(argv[2]);std::cout<<"malformed-model rejection checks passed\n";}
    else if(std::string(argv[1])=="--defaults") {check(argc==4,"defaults needs input and native saved output");defaults(argv[2],argv[3]);std::cout<<"native default coefficients match\n";}
    else {for(int i=1;i<argc;++i){Model m;m.load(argv[i]);}std::cout<<"validated "<<argc-1<<" native model files\n";}
    return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
