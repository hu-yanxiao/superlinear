// Reuse the standalone parser fixture helpers, without running its main suite.
#define main grouped_base_suite_main
#include "grouped_input_check.cpp"
#undef main
#include <set>

static std::string OutputParent(const std::string& path) {
    return path.substr(0, path.find_last_of('/'));
}

int main() {
    try {
        char temp[] = "/tmp/superlinear-kind-output-XXXXXX";
        char* created = mkdtemp(temp);
        if (!created) return 2;
        root = created;
        for (const char* kind : {"coordination-effect", "effect", "coordination-legacy", "legacy"}) {
            std::string input = legacy_model;
            input.replace(input.find("coordination-legacy"), std::string("coordination-legacy").size(), kind);
            for (const char* profile : {"native-effect-v1", "native-legacy-v1"}) {
                const std::string directory = root + "/" + kind + "-" + profile;
                const auto result = Run({"train", Fixture(input + "data:\n  train: train.cfg\n"
                    "training:\n  profile: " + profile + "\noutput:\n  directory: " + directory + "\n")});
                Require(result.opts.at("grouped-profile") == "native-effect-v1", "canonical effect profile");
                Require(result.opts.at("superlinear-mode") == "legacy", "saved mathematics unchanged");
                Require(result.opts.at("inline-sh-model") == directory + "/initial.sus2", "fresh .sus2 output");
                Require(result.opts.at("trained-pot-name") == directory + "/final.sus2", "final .sus2 output");
                Require(result.opts.at("optimizer-trace-file") == directory + "/optimizer.csv", "CSV unchanged");
            }
            const auto defaults = Run({"train", Fixture(input + training), "--check-config"});
            Require(defaults.opts.at("curr-pot-name") == root + "/run/current.sus2", "default current suffix");
            Require(defaults.opts.at("best-pot-name") == root + "/run/best.sus2", "default best suffix");
        }
        const auto topology = Run({"init-sh", "--model.kind=effect", "--O=2",
            "--output.directory=" + root + "/topology", "--check-config"});
        Require(topology.args.at(0) == root + "/topology/initial.sus2", "topology default suffix");
        for (const char* suffix : {".mtp", ".SUS2", ".sus2.tmp", ""}) {
            Bad({"init-sh", "--model.kind=effect", "--O=2", "--output.model=" + root + "/seed" + suffix,
                "--check-config"}, "must end in .sus2");
            for (const char* option : {"output.initial_model", "output.current", "output.best", "output.final"})
                Bad({"train", Fixture(model + training), std::string("--") + option + "=" + root + "/bad" + suffix,
                    "--check-config"}, "must end in .sus2");
        }
        for (const char* input_suffix : {".mtp", ".sus2"}) {
            const auto resumed = Run({"train", Fixture("data:\n  model: checkpoint" + std::string(input_suffix)
                + "\n  train: train.cfg\ntraining:\n  profile: native-effect-v1\noutput:\n  directory: resumed\n"),
                "--check-config"});
            Require(resumed.args.at(0) == root + "/checkpoint" + input_suffix, "input compatibility");
            Require(!resumed.opts.count("inline-sh-model"), "continuation has no initial output");
        }
        for (const std::string& path : {std::string(".sus2"), root + "/dir/.sus2",
                root + "/dir/\\.sus2", root + "/bad" + std::string(1, '\0') + ".sus2"}) {
            // Direct Resolve preserves embedded NUL; argv cannot carry it.
            std::vector<std::string> args;
            Options options{{"model.kind", "effect"}, {"model.interaction.channels", "2"},
                {"output.model", path}, {"check-config", "true"}};
            bool rejected = false;
            try { superlinear_input::Resolve("init-sh", args, options, nullptr); }
            catch (const std::exception& e) {
                rejected = std::string(e.what()).find("must end in .sus2") != std::string::npos;
            }
            Require(rejected, "reject empty basename or NUL model output");
        }
        const auto static_result = Run({"train", Fixture(static_model + training), "--check-config"});
        Require(!static_result.opts.count("two-layer-gate-k-rank") && static_result.opts.at("outer-k-rank") == "4",
            "static has no R, retains O");
        Bad({"train", Fixture(static_model + training), "--interaction-coupling=diagonal", "--check-config"}, "conflicting");
        const std::string auto_model = "model:\n  kind: coordination-effect\n  l_max: 3\n  radial_channels: 4\n"
            "  coordination:\n    channels: 2\n  interaction:\n    channels: 2\n"
            "data:\n  train: train.cfg\n";
        const std::string auto_config = Fixture(auto_model, "automatic.yaml");
        const auto first = Run({"train", auto_config, "--check-config"});
        const auto second = Run({"train", auto_config, "--check-config"});
        Require(first.opts.at("trained-pot-name") != second.opts.at("trained-pot-name"), "repeated auto checks get unique paths");
        Require(first.opts.at("trained-pot-name").find("runs/coordination-effect-l3-k4-r2-o2-") != std::string::npos,
            "fresh name reflects architecture");
        struct stat info;
        Require(stat(OutputParent(first.opts.at("trained-pot-name")).c_str(), &info) != 0, "auto check creates no directory");
        Options rank_outputs[2];
        for (auto& options : rank_outputs) {
            options["check-config"] = "true";
            std::vector<std::string> args{auto_config};
            superlinear_input::Resolve("train", args, options, nullptr, "shared-mpi-token");
        }
        Require(rank_outputs[0] == rank_outputs[1], "broadcast token yields identical rank options");
        std::set<std::string> tokens;
        for (int i = 0; i < 100; ++i) tokens.insert(superlinear_input::GenerateRunToken());
        Require(tokens.size() == 100, "run token uniqueness");
        Bad({"train", auto_config, "--output.final=" + root + "/partial.sus2", "--check-config"}, "partial explicit");
        const auto auto_static = Run({"train", Fixture("model:\n  kind: static-radial\n  radial_channels: 4\n"
            "  interaction:\n    channels: 2\ndata:\n  train: train.cfg\n"), "--check-config"});
        Require(auto_static.opts.at("trained-pot-name").find("static-radial-l3-k4-o2-") != std::string::npos, "static auto name omits R");
        const auto auto_resume = Run({"train", Fixture("data:\n  model: old-checkpoint.mtp\n  train: train.cfg\n"
            "training:\n  profile: native-effect-v1\n"), "--check-config"});
        Require(auto_resume.opts.at("trained-pot-name").find("resume-old-checkpoint-") != std::string::npos, "resume auto label");
        const auto auto_execution = Run({"train", auto_config});
        Require(stat(OutputParent(auto_execution.opts.at("trained-pot-name")).c_str(), &info) == 0 && S_ISDIR(info.st_mode),
            "real auto execution creates run directory");
        const auto repeated_execution = Run({"train", auto_config});
        Require(auto_execution.opts.at("trained-pot-name") != repeated_execution.opts.at("trained-pot-name"),
            "repeated execution does not reuse output directory");
        const auto all_explicit = Run({"train", auto_config, "--output.initial_model=" + root + "/i.sus2",
            "--output.current=" + root + "/c.sus2", "--output.best=" + root + "/b.sus2",
            "--output.final=" + root + "/f.sus2", "--output.optimizer_trace=" + root + "/trace.csv", "--check-config"});
        Require(all_explicit.opts.at("trained-pot-name") == root + "/f.sus2", "fully explicit paths retained");
        std::cout << "GROUPED_KIND_OUTPUT_PASS checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
