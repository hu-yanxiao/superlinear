#ifndef SUS2_SH_MODEL_INIT_H
#define SUS2_SH_MODEL_INIT_H

#include <map>
#include <array>
#include <iosfwd>
#include <string>
#include <vector>

// Independent K=2, l>=1 invariant graph. Input 0 is the outer-channel mean;
// input 1 is the post-rank gate-channel mean. All indices are graph-local.
struct JointReadoutProduct { int left, right, target; double coeff; };
struct JointReadoutTopology {
    int l_max = 0, body_order = 0, moments_count = 0;
    std::vector<int> body_l_max;
    std::vector<std::array<int, 3>> basic;
    std::vector<JointReadoutProduct> products;
    std::vector<int> scalar_mapping;
    std::vector<std::array<int, 7>> scalar_info;
};
JointReadoutTopology BuildJointReadoutTopology(int l_max, int body_order,
                                              const std::vector<int>& body_l_max);
void WriteJointReadoutSection(std::ostream&, const JointReadoutTopology&,
                             const std::vector<double>& coefficients,
                             const std::vector<double>& optimizer_scales);
// The caller has consumed the joint_readout_enabled key, but not its value.
void ReadJointReadoutSection(std::istream&, JointReadoutTopology&,
                            std::vector<double>& coefficients,
                            std::vector<double>& optimizer_scales,
                            int species_count, bool species_readout,
                            int l_max, int body_order, const std::vector<int>& body_l_max);

void WriteSphericalHarmonicModel(const std::string& filename,
                                 const std::map<std::string, std::string>& opts);

#endif
