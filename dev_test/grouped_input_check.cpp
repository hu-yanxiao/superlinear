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
const std::string static_model =
    "model:\n  elements: [H, O]\n  l_max: 3\n  radial_channels: 8\n"
    "  coordination:\n    enabled: false\n  interaction:\n    channels: 4\n    body_order: 4\n";
const std::string legacy_model =
    "model:\n  kind: coordination-legacy\n  elements: [H, O]\n  l_max: 3\n  radial_channels: 8\n"
    "  coordination:\n    channels: 4\n  interaction:\n    channels: 4\n    body_order: 4\n"
    "  readout:\n    element_mode: per-element\n";
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
        const std::pair<const char*, const char*> warp_choices[] = {
            {"L", "L"}, {"K", "K"}, {"LK", "LK"},
            {"angular", "L"}, {"channel", "K"}, {"channel-angular", "LK"}
        };
        for (const auto& choice : warp_choices) {
            const auto warp = Run({"train", Fixture(model + "  radial:\n    warp_sharing: " + choice.first + "\n" + training), "--check-config"});
            Require(warp.opts.at("scaling-map") == choice.second, "YAML warp sharing: " + std::string(choice.first));
        }
        for (const char* sharing : {"L", "K", "LK"}) {
            const auto warp = Run({"train", Fixture(model + training), "--distance-warp-sharing", sharing,
                "--scaling-map=" + std::string(sharing), "--check-config"});
            Require(warp.opts.at("scaling-map") == sharing, "CLI warp sharing: " + std::string(sharing));
        }
        Bad({"train", Fixture(model + training), "--distance-warp-sharing=lk", "--check-config"}, "unsupported value");
        Bad({"train", Fixture(model + training), "--distance-warp-sharing=L", "--scaling-map=K", "--check-config"}, "conflicting");
        const std::pair<const char*, const char*> response_choices[] = {
            {"full", "full"}, {"per-basis", "full"},
            {"shared-n", "shared-n"}, {"shared-latent", "shared-n"}
        };
        for (const auto& choice : response_choices) {
            const auto response = Run({"train", Fixture(model + "  response:\n    layout: " + choice.first + "\n" + training), "--check-config"});
            Require(response.opts.at("adaptive-radial-projection") == choice.second, "YAML response layout: " + std::string(choice.first));
        }
        for (const char* rank : {"1", "2", "10"}) {
            const auto response = Run({"train", Fixture(model + "  response:\n    layout: shared-n\n    latent_channels: " + rank + "\n" + training),
                "--response-layout=shared-latent", "--adaptive-radial-projection=shared-n", "--check-config"});
            Require(response.opts.at("adaptive-radial-shared-n-rank") == rank, "shared-n projection rank: " + std::string(rank));
        }
        const auto full_response = Run({"train", Fixture(model + training), "--response-layout=full", "--check-config"});
        Require(full_response.opts.at("adaptive-radial-projection") == "full", "full response CLI");
        Bad({"train", Fixture(model + training), "--response-layout=full", "--model.response.latent_channels=2", "--check-config"}, "latent channels");
        Bad({"train", Fixture(model + training), "--response-layout=shared-n", "--model.response.latent_channels=11", "--check-config"}, "latent channels");
        Bad({"train", Fixture(model + training), "--response-layout=shared-n", "--model.response.latent_channels=0", "--check-config"}, "out of range");
        Bad({"train", Fixture(model + training), "--response-layout=full", "--adaptive-radial-projection=shared-n", "--check-config"}, "conflicting");
        Fixture(model + training);
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
            "  angular_limit_by_body_order: {2: 3, 3: 2, 4: 1, 5: 1}\noutput:\n  model: seed.sus2\n"), "--check-config"});
        Require(angular.opts.at("body-l-max") == "3,2,1,1", "angular limit flow map");
        const auto block = Run({"init-sh", Fixture(
            "model:\n  elements:\n    - H\n    - O\n  radial_channels: 2\n  interaction:\n    channels: 1\noutput:\n  model: seed.sus2\n"), "--check-config"});
        Require(block.opts.at("species-count") == "2" && block.opts.at("outer-k-rank") == "1", "block list");
        Bad({"init-sh", Fixture(model + "output:\n  model: seed.sus2\n"), "--max-iter=0", "--check-config"}, "does not use");
        Bad({"init-sh", Fixture(model + "elements: [H, H]\noutput:\n  model: seed.sus2\n"), "--check-config"}, "duplicate element");
        const auto fixed = Run({"train", Fixture(model + training +
            "regularization:\n  radial_smoothness: 1e-6\n  radial_smoothness_grid: 128\n  energy_readout_l2: 1e-5\n  coordination_response_l2: 1e-4\n"), "--check-config"});
        Require(!fixed.opts.count("radial-smooth") && !fixed.opts.count("gate-full-l2"), "fixed controls must not become forbidden legacy overrides");
        Bad({"train", Fixture(model + training), "--regularization.energy_readout_l2=0", "--check-config"}, "fixed to");
        const auto resumed = Run({"train", Fixture("data:\n  model: checkpoint.mtp\n  train: train.cfg\noutput:\n  directory: resumed\ntraining:\n  iterations: 5\n"), "--check-config"});
        Require(resumed.args.size() == 2 && resumed.opts.at("init-params") == "same" && !resumed.opts.count("init-sh"), "continuation");
        const auto static_run = Run({"train", Fixture(static_model + training), "--check-config"});
        Require(static_run.opts.at("superlinear-mode") == "legacy" && static_run.opts.at("sh-factor-pruning") == "q-total"
            && static_run.opts.at("outer-k-rank") == "4", "static topology without gate");
        Require(!static_run.opts.count("two-layer-gate-k-rank") && !static_run.opts.count("adaptive-radial-projection")
            && static_run.opts.at("skip-preinit") == "true", "omit unused gate and hidden pretraining");
        for (const char* sharing : {"L", "K", "LK"}) {
            const auto s = Run({"train", Fixture(static_model + training), "--distance-warp-sharing", sharing, "--check-config"});
            Require(s.opts.at("scaling-map") == sharing, "static warp sharing");
        }
        const auto static_cli = Run({"train", "data.cfg", "--coord-enabled", "false", "--interaction-channels=2",
            "--output.directory=" + root + "/static-cli", "--check-config"});
        Require(static_cli.opts.at("superlinear-mode") == "legacy", "static short CLI");
        const auto static_kind = Run({"init-sh", Fixture("model:\n  kind: static-radial\n  interaction:\n    channels: 2\noutput:\n  model: static.sus2\n"), "--check-config"});
        Require(static_kind.opts.at("superlinear-mode") == "legacy", "static kind implies coordination disabled");
        const auto static_quoted = Run({"train", Fixture("model:\n  kind: 'static-radial'\n  interaction:\n    channels: 2\n" + training
            + "regularization:\n  energy_readout_l2: 0\n"), "--training.profile='native-static-v1'", "--check-config"});
        Require(static_quoted.opts.at("superlinear-mode") == "legacy" && static_quoted.opts.at("scalar-head-l2") == "0", "quoted static mode/profile");
        const auto static_family = Run({"train", Fixture(static_model + training), "--model.kind=coordination-adaptive", "--check-config"});
        Require(static_family.opts.at("superlinear-mode") == "legacy", "explicit switch overrides adaptive family default");
        for (const char* key : {"model.coordination.channels", "model.coordination.body_order",
             "model.coordination.channel_coupling", "model.coordination.distance_warp", "model.coordination.tensor_l_max",
             "model.response.layout", "model.response.latent_channels", "model.response.activation", "model.response.element_mode"})
            Bad({"train", Fixture(static_model + training), std::string("--") + key + "=1", "--check-config"}, "unavailable");
        Bad({"train", Fixture(static_model + training), "--two-layer-gate-k-rank=2", "--check-config"}, "unavailable");
        Bad({"train", Fixture(static_model + training), "--adaptive-radial-projection=full", "--check-config"}, "unavailable");
        const auto static_species = Run({"train", Fixture(static_model + training), "--model.readout.element_mode=per-element", "--check-config"});
        Require(static_species.opts.at("readout-mode") == "species", "static Legacy species readout");
        Bad({"train", Fixture(static_model + training), "--model.readout.joint=true", "--check-config"}, "joint: false");
        Bad({"train", Fixture(static_model + training), "--interaction-coupling=diagonal", "--check-config"}, "conflicting");
        Bad({"train", Fixture(static_model + training), "--regularization.coordination_response_l2=0", "--check-config"}, "unavailable");
        Bad({"train", Fixture(static_model + training), "--training.profile=native-adaptive-v1", "--check-config"}, "must match");
        Bad({"train", Fixture(model + training), "--training.profile=native-static-v1", "--check-config"}, "must match");
        Bad({"train", Fixture(static_model + training), "--initialization.seed=41", "--check-config"}, "requires a fresh model");
        const auto random_static = Run({"train", Fixture(static_model + training), "--initialization.parameters=random",
            "--initialization.seed=41", "--check-config"});
        Require(random_static.opts.at("init-params") == "random" && random_static.opts.at("init-random-seed") == "41", "static random initialization");
        const auto static_regularizers = Run({"train", Fixture(static_model + training), "--regularization.radial_smoothness=0",
            "--regularization.radial_smoothness_grid=32", "--regularization.energy_readout_l2=1e-5", "--check-config"});
        Require(static_regularizers.opts.at("radial-smooth") == "0" && static_regularizers.opts.at("radial-smooth-grid") == "32"
            && std::stod(static_regularizers.opts.at("scalar-head-l2")) == 1e-5, "static selectable regularizers");
        const std::string static_continue = "data:\n  model: static.mtp\n  train: train.cfg\ntraining:\n  profile: native-static-v1\noutput:\n  directory: resumed-static\n";
        const auto sresumed = Run({"train", Fixture(static_continue), "--check-config"});
        Require(sresumed.args.size() == 2 && sresumed.opts.at("skip-preinit") == "true", "static continuation profile");
        Bad({"train", Fixture(static_continue), "--training.trainable_groups=radial", "--check-config"}, "requires an adaptive checkpoint");
        const auto legacy_species = Run({"train", Fixture(legacy_model + training), "--check-config"});
        Require(legacy_species.opts.at("superlinear-mode") == "legacy" && legacy_species.opts.at("readout-mode") == "species", "Legacy family and element readout");
        Require(legacy_species.opts.at("two-layer-gate-mode") == "mu-scalar-full" && legacy_species.opts.at("two-layer-gate-radial-mode") == "shared-radial", "ranked Legacy coordination");
        Require(legacy_species.opts.at("sh-factor-pruning") == "same-k" && legacy_species.opts.at("two-layer-gate-factor-pruning") == "q-total", "Legacy matches Adaptive graph defaults");
        Require(legacy_species.opts.at("k-max") == "8" && legacy_species.opts.at("two-layer-gate-k-rank") == "4" && legacy_species.opts.at("outer-k-rank") == "4", "Legacy K/R/O mapping");
        Require(!legacy_species.opts.count("adaptive-radial-projection") && legacy_species.opts.at("skip-preinit") == "true", "Legacy no adaptive U or hidden pretraining");
        for (const char* gate : {"cross", "diagonal"}) for (const char* outer : {"cross", "diagonal"}) {
            const auto combination = Run({"train", Fixture(legacy_model + training), std::string("--coord-coupling=") + gate,
                std::string("--interaction-coupling=") + outer, "--check-config"});
            Require(combination.opts.at("two-layer-gate-factor-pruning") == (std::string(gate) == "cross" ? "q-total" : "same-k"), "independent Legacy inner coupling");
            Require(combination.opts.at("sh-factor-pruning") == (std::string(outer) == "cross" ? "q-total" : "same-k"), "independent Legacy outer coupling");
        }
        const auto compact = Run({"init-sh", "--model.kind=coordination-legacy", "--L=3", "--K=8", "--R=4", "--O=4",
            "--output.model=" + root + "/compact.sus2", "--check-config"});
        Require(compact.opts.at("l-max") == "3" && compact.opts.at("k-max") == "8" && compact.opts.at("two-layer-gate-k-rank") == "4" && compact.opts.at("outer-k-rank") == "4", "compact L/K/R/O aliases");
        Bad({"train", Fixture(legacy_model + training), "--model.response.layout=full", "--check-config"}, "unavailable");
        Bad({"train", Fixture(legacy_model + training), "--model.coordination.tensor_l_max=1", "--check-config"}, "tensor messages");
        Bad({"train", Fixture(legacy_model + training), "--model.readout.joint=true", "--check-config"}, "joint readout");
        Bad({"train", Fixture(legacy_model + training), "--training.profile=native-adaptive-v1", "--check-config"}, "must match");
        const auto legacy_regularizers = Run({"train", Fixture(legacy_model + training), "--regularization.coordination_response_l2=0", "--regularization.energy_readout_l2=0", "--check-config"});
        Require(legacy_regularizers.opts.at("gate-full-l2") == "0" && legacy_regularizers.opts.at("scalar-head-l2") == "0", "Legacy explicit selectable regularizers");
        Bad({"train", Fixture(model + "data:\n  model: checkpoint.mtp\n  train: train.cfg\noutput:\n  directory: resumed\n"), "--check-config"}, "inherits");
        const auto legacy = Run({"train", "old.mtp", "old.cfg", "--max-iter=7", "--unknown=true"});
        Require(!legacy.check && legacy.args.size() == 2 && legacy.opts.size() == 2, "legacy options untouched");
        const auto terminator = Run({"init-sh", "--", "--model.output.mtp"});
        Require(terminator.args.size() == 1 && terminator.opts.empty(), "end-of-options preserved");
        Fixture(model + training);
        const auto fresh = Run({"train", config});
        Require(!fresh.check && fresh.opts.at("grouped-input") == "fresh", "trainer marker");
        Require(stat((root + "/run").c_str(), &info) == 0 && S_ISDIR(info.st_mode), "create output directory only for execution");
        Fixture("keep", "run/initial.sus2");
        Bad({"train", config}, "refusing to reuse/overwrite");
        std::cout << "GROUPED_INPUT_PASS checks=" << checks << " fixtures=" << root << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
