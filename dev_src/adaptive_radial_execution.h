#ifndef SUS2_ADAPTIVE_RADIAL_EXECUTION_H
#define SUS2_ADAPTIVE_RADIAL_EXECUTION_H

#include <vector>

// Scratch only: no trainable state. A requested full EFS cache can be consumed
// once by the next gradient, after exact geometry/parameter equality checks.
// An EFS-only radial cache is never consumed by mixed derivatives.
struct AdaptiveRadialExecution {
    bool derivatives = false;
    bool parameters = false;
    int rows = 0, scalars = 0, radial_stride = 0, harmonics = 0;
    std::vector<int> outer_nodes;
    std::vector<double> signals, coefficients, coefficient_tangents;
    std::vector<double> coefficient_adjoints, coefficient_tangent_adjoints;
    std::vector<double> scalar_tangents, scalar_adjoints, scalar_tangent_adjoints;
    std::vector<double> gate_moment_tangents;
    std::vector<double> tensor_messages, tensor_message_tangents;
    std::vector<double> tensor_message_adjoints, tensor_message_tangent_adjoints;
    std::vector<double> tensor_gate_moment_adjoints;
    std::vector<double> tensor_gate_moment_tangent_adjoints;
    std::vector<double> component_tensor_message_adjoints;
    std::vector<double> values, tangents, adjoints, tangent_adjoints;
    std::vector<double> radial_values, radial_derivatives, radial_tangents;
    std::vector<double> site_values;
    std::vector<double> component_coefficient_adjoints;
    bool reusable_full_cache = false;
    double cached_scaling = 0.0;
    std::vector<double> cached_coefficients, cached_edge_geometry;
    std::vector<int> cached_edge_metadata;
};

#endif
