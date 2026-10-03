#include "../src/mlp/grouped_input.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
typedef std::map<std::string, std::string> Options;
int checks = 0;
std::string root;
void Require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
std::string Fixture(const std::string& text, const std::string& name = "test.yaml") {
    const std::string path = root + "/" + name;
    std::ofstream file(path.c_str()); file << text;
    if (!file) throw std::runtime_error("cannot write fixture");
    return path;
}
struct Result { bool check; std::vector<std::string> args; Options opts; };
Result Run(std::vector<std::string> tokens) {
    std::vector<char*> pointers;
    for (auto& token : tokens) pointers.push_back(&token[0]);
    tokens = superlinear_input::NormalizeArguments(pointers.size(), pointers.data());
    Result result; bool end = false;
    // Same token protocol as ParseOptions; executable tests also exercise the
    // production parser. This standalone test needs no MKL, MPI or trainer.
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        if (!end && token == "--") { end = true; continue; }
        if (!end && token.compare(0, 2, "--") == 0) {
            const auto eq = token.find('=');
            result.opts[token.substr(2, eq == std::string::npos ? eq : eq-2)] =
                eq == std::string::npos ? "true" : token.substr(eq+1);
        } else result.args.push_back(token);
    }
    result.check = superlinear_input::Resolve(tokens[0], result.args, result.opts, nullptr);
    return result;
}
void Bad(const std::vector<std::string>& tokens, const std::string& expected) {
    try { Run(tokens); } catch (const std::exception& e) {
        Require(std::string(e.what()).find(expected) != std::string::npos,
                "wrong error: " + std::string(e.what()) + " expected " + expected);
        return;
    }
    throw std::runtime_error("expected rejection: " + expected);
}
const std::string model =
    "schema_version: 1\nmodel:\n  kind: coordination-adaptive\n  elements: [H, O]\n"
    "  l_max: 3\n  radial_channels: 8\n  neighbor_cutoff: 6.5\n"
    "  coordination:\n    channels: 4\n    distance_warp: shared\n"
    "  interaction:\n    channels: 4\n    body_order: 4\n";
const std::string training =
    "training:\n  engine: native\n  device: cpu\n  dtype: float64\n"
    "  optimizer: bfgs\n  iterations: 2000\n"
    "loss:\n  energy_weight: 20\n  force_weight: 20\n  virial_weight: 5\n"
    "data:\n  train: 'train with spaces.cfg'\n  valid: val.cfg\n"
    "output:\n  directory: run\n";
}
int main() {
    try {
        char temp[] = "/tmp/superlinear-config-XXXXXX";
        char* created = mkdtemp(temp); if (!created) return 2; root = created;
        const std::string config = Fixture(model + training);
        const Result a = Run({"train", config, "--check-config"});
        Require(a.check, "check-config should short-circuit");
        Require(a.args.size() == 1 && a.args[0] == root + "/train with spaces.cfg", "relative input path");
        Require(a.opts.at("k-max") == "8" && a.opts.at("two-layer-gate-k-rank") == "4" && a.opts.at("outer-k-rank") == "4", "K/G/O mapping");
        Require(a.opts.at("max-iter") == "2000" && a.opts.at("stress-weight") == "5", "training mapping");
        Require(a.opts.at("species-count") == "2", "element ordering count");
        Require(!a.opts.count("two-layer-gate-decoupled-scaling") && !a.opts.count("zbl-elements"), "presence flags must be absent");
        struct stat info; Require(stat((root + "/run").c_str(), &info) != 0, "check must create nothing");
        const auto alias = Run({"train", "data.cfg", "--radial-channels", "8", "--coord-channels=4",
            "--interaction-channels=4", "--iterations=2000", "--output.directory", root + "/cli", "--check-config"});
        Require(alias.opts.at("two-layer-gate-k-rank") == "4" && alias.opts.at("init-params") == "same", "short alias and defaults");
        Run({"train", config, "--energy-weight=20.0", "--check-config"}); ++checks;
        Bad({"train", config, "--energy-weight=19", "--check-config"}, "conflicting");
        Bad({"train", config, "--epochs=2000", "--check-config"}, "not supported");
        Bad({"train", config, "--optimizer=adam", "--check-config"}, "conflicting");
        Bad({"train", config, "--init-params=random", "--check-config"}, "conflicting");
        Bad({"train", "--coord-channels=4", "--model.coordination.channels=8"}, "duplicate/conflicting");
        Bad({"train", config, "--max-iter=2", "--max-iter=2000", "--check-config"}, "duplicate/conflicting");
        Bad({"train", "--config="}, "requires a YAML path");
        Bad({"train", config, "--coord-channel=4", "--check-config"}, "not supported");
        Bad({"train", Fixture(model + training + "training:\n  iterations: 3\n"), "--check-config"}, "duplicate mapping");
        Bad({"train", Fixture(model + training + "unknown: yes\n"), "--check-config"}, "unknown setting");
        Bad({"train", Fixture(model + "training:\n  epochs: 2000\n"), "--check-config"}, "unknown setting");
        Bad({"train", Fixture(model + training), "--model.coordination.channels=9", "--check-config"}, "conflict");
        Bad({"train", Fixture(model + training), "--model.response.layout=per-basis", "--model.response.latent_channels=2", "--check-config"}, "latent channels");
        Bad({"train", Fixture(model + training), "--model.angular_limit_by_body_order.7=1", "--check-config"}, "unknown setting");
        const auto angular = Run({"init-sh", Fixture(model +
            "  angular_limit_by_body_order: {2: 3, 3: 2, 4: 1, 5: 1}\noutput:\n  model: seed.mtp\n"), "--check-config"});
        Require(angular.opts.at("body-l-max") == "3,2,1,1", "angular limit flow map");
        const auto block = Run({"init-sh", Fixture(
            "model:\n  elements:\n    - H\n    - O\n  radial_channels: 2\n  interaction:\n    channels: 1\noutput:\n  model: seed.mtp\n"), "--check-config"});
        Require(block.opts.at("species-count") == "2" && block.opts.at("outer-k-rank") == "1", "block list");
        Bad({"init-sh", Fixture(model + "output:\n  model: seed.mtp\n"), "--max-iter=0", "--check-config"}, "does not use");
        Bad({"init-sh", Fixture(model + "elements: [H, H]\noutput:\n  model: seed.mtp\n"), "--check-config"}, "duplicate element");
        const auto fixed = Run({"train", Fixture(model + training +
            "regularization:\n  radial_smoothness: 1e-6\n  radial_smoothness_grid: 128\n  energy_readout_l2: 1e-5\n  coordination_response_l2: 1e-4\n"), "--check-config"});
        Require(!fixed.opts.count("radial-smooth") && !fixed.opts.count("gate-full-l2"), "fixed controls must not become forbidden legacy overrides");
        Bad({"train", Fixture(model + training), "--regularization.energy_readout_l2=0", "--check-config"}, "fixed to");
        const auto resumed = Run({"train", Fixture("data:\n  model: checkpoint.mtp\n  train: train.cfg\noutput:\n  directory: resumed\ntraining:\n  iterations: 5\n"), "--check-config"});
        Require(resumed.args.size() == 2 && resumed.opts.at("init-params") == "same" && !resumed.opts.count("init-sh"), "continuation");
        Bad({"train", Fixture(model + "data:\n  model: checkpoint.mtp\n  train: train.cfg\noutput:\n  directory: resumed\n"), "--check-config"}, "inherits");
        const auto legacy = Run({"train", "old.mtp", "old.cfg", "--max-iter=7", "--unknown=true"});
        Require(!legacy.check && legacy.args.size() == 2 && legacy.opts.size() == 2, "legacy options untouched");
        const auto terminator = Run({"init-sh", "--", "--model.output.mtp"});
        Require(terminator.args.size() == 1 && terminator.opts.empty(), "end-of-options preserved");
        Fixture(model + training);
        const auto fresh = Run({"train", config});
        Require(!fresh.check && fresh.opts.at("grouped-input") == "fresh", "trainer marker");
        Require(stat((root + "/run").c_str(), &info) == 0 && S_ISDIR(info.st_mode), "create output directory only for execution");
        Fixture("keep", "run/initial.mtp");
        Bad({"train", config}, "refusing to reuse/overwrite");
        std::cout << "GROUPED_INPUT_PASS checks=" << checks << " fixtures=" << root << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
