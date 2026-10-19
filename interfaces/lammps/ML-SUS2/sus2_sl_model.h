// Standalone native SUS2-SH model reader; no training-library dependency.
#ifndef LMP_SUS2_SL_MODEL_H
#define LMP_SUS2_SL_MODEL_H
#include <array>
#include <map>
#include <string>
#include <vector>
namespace LAMMPS_NS { namespace SUS2SL {
struct Product { int a, b, out; double c; };
struct Model {
  std::map<std::string,std::string> fields;
  int species_count=0, lmax=0, kmax=0, radial_size=0, radial_funcs=0;
  int moments=0, scalar_count=0;
  double min_dist=0, max_dist=0, scaling=1;
  std::vector<double> radial_coeffs;
  std::vector<std::array<int,3> > basic;
  std::vector<Product> products;
  std::vector<int> scalar_mapping;
  // Neighbor cutoff includes the optional ZBL contribution.
  double cutoff() const;
  void load(const std::string &path);
  void validate() const;
  bool has(const std::string &key) const;
  std::string text(const std::string &key,const std::string &fallback="") const;
  int integer(const std::string &key,int fallback=0) const;
  double real(const std::string &key,double fallback=0) const;
  bool boolean(const std::string &key,bool fallback=false) const;
  std::vector<double> numbers(const std::string &key) const;
  std::vector<int> integers(const std::string &key) const;
  std::vector<std::vector<double> > rows(const std::string &key,int width) const;
};
} }
#endif
