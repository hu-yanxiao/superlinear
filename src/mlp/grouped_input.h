#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

// User-facing configuration only. Translation never changes evaluator equations,
// coefficient ordering, initialization algorithms, or the optimizer profile.
namespace superlinear_input {
std::vector<std::string> NormalizeArguments(int argc, char* argv[]);
// Returns true for --check-config (no model/data loading or output creation).
bool Resolve(const std::string& command, std::vector<std::string>& args,
             std::map<std::string, std::string>& opts, std::ostream* output,
             const std::string& run_token = "");
// MPI callers generate once on rank zero and broadcast before Resolve.
std::string GenerateRunToken();
const char* Help();
}
