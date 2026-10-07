#include "grouped_input.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace superlinear_input {
namespace {
typedef std::map<std::string, std::string> Values;
struct Rule { const char* key; const char* option; const char* type; };
const Rule rules[] = {
    {"model.kind", "superlinear-mode", "coordination-adaptive=radial-coefficient-gate|coordination-legacy=legacy|legacy=legacy|static-radial=legacy"},
    {"model.name", "potential-name", "text"},
    {"model.species_count", "species-count", "positive-int"},
    {"model.l_max", "l-max", "int"},
    {"model.radial_channels", "k-max", "positive-int"},
    {"model.neighbor_cutoff", "cutoff", "positive-real"},
    {"model.radial.basis", "radial-basis-type", "chebyshev-sss=RBChebyshev_sss|laguerre-log1p=RBLaguerre_log1p"},
    {"model.radial.basis_size", "radial-basis-size", "positive-int"},
    {"model.radial.basis_scale", "scaling", "positive-real"},
    {"model.radial.reference_min_distance", "min-dist", "nonnegative-real"},
    {"model.radial.warp_sharing", "scaling-map", "L=L|K=K|LK=LK|angular=L|channel=K|channel-angular=LK"},
    {"model.coordination.enabled", "@coordination-enabled", "bool"},
    {"model.coordination.channels", "two-layer-gate-k-rank", "positive-int"},
    {"model.coordination.body_order", "two-layer-gate-body-order", "positive-int"},
    {"model.coordination.channel_coupling", "two-layer-gate-factor-pruning", "cross=q-total|diagonal=same-k"},
    {"model.coordination.distance_warp", "two-layer-gate-decoupled-scaling", "shared=false|independent=true"},
    {"model.coordination.tensor_l_max", "tensor-message-lmax", "int"},
    {"model.interaction.channels", "outer-k-rank", "positive-int"},
    {"model.interaction.body_order", "body-order", "positive-int"},
    {"model.interaction.channel_coupling", "sh-factor-pruning", "diagonal=same-k|cross=q-total"},
    {"model.response.layout", "adaptive-radial-projection", "full=full|shared-n=shared-n|per-basis=full|shared-latent=shared-n"},
    {"model.response.latent_channels", "adaptive-radial-shared-n-rank", "positive-int"},
    {"model.response.activation", "adaptive-radial-activation", "linear=none|tanh=tanh"},
    {"model.response.element_mode", "adaptive-radial-amplitude-mode", "shared=shared|source-element=species"},
    {"model.readout.element_mode", "readout-mode", "shared=shared|per-element=species"},
    {"model.readout.joint", "joint-readout", "bool"},
    {"initialization.parameters", "init-params", "same=same|random=random"},
    {"initialization.seed", "init-random-seed", "seed"},
    {"initialization.warp_slope_endpoints", "scal-range", "pair"},
    {"initialization.warp_shift_endpoints", "s-range", "pair"},
    {"training.iterations", "max-iter", "int"},
    {"training.optimizer", "optimizer", "bfgs=bfgs"},
    {"training.convergence_tolerance", "optimizer-conv-tol", "nonnegative-real"},
    {"training.trainable_groups", "trainable-groups", "groups"},
    {"loss.energy_weight", "energy-weight", "nonnegative-real"},
    {"loss.force_weight", "force-weight", "nonnegative-real"},
    {"loss.virial_weight", "stress-weight", "nonnegative-real"},
    {"data.train", "@train", "path"},
    {"data.valid", "valid-cfgs", "path"},
    {"data.model", "@model", "path"},
    {"output.directory", "@directory", "path"},
    {"output.model", "@output", "path"},
    {"output.initial_model", "inline-sh-model", "path"},
    {"output.current", "curr-pot-name", "path"},
    {"output.best", "best-pot-name", "path"},
    {"output.final", "trained-pot-name", "path"},
    {"output.optimizer_trace", "optimizer-trace-file", "path"},
    {"training.engine", "@fixed", "native=native"},
    {"training.device", "@fixed", "cpu=cpu"},
    {"training.dtype", "@fixed", "float64=float64"},
    {"training.profile", "@profile", "native-adaptive-v1=native-adaptive-v1|native-legacy-v1=native-legacy-v1|native-static-v1=native-static-v1"},
    {"loss.profile", "@fixed", "sus2-legacy=sus2-legacy"},
    {"regularization.radial_smoothness", "@fixed:0.000001", "nonnegative-real"},
    {"regularization.radial_smoothness_grid", "@fixed:128", "positive-int"},
    {"regularization.energy_readout_l2", "@fixed:0.00001", "nonnegative-real"},
    {"regularization.coordination_response_l2", "@fixed:0.0001", "nonnegative-real"},
};
const Values& Aliases() {
    static const Values values = {
        {"L", "model.l_max"},
        {"K", "model.radial_channels"},
        {"R", "model.coordination.channels"},
        {"O", "model.interaction.channels"},
        {"radial-channels", "model.radial_channels"},
        {"coord-enabled", "model.coordination.enabled"},
        {"coord-channels", "model.coordination.channels"},
        {"interaction-channels", "model.interaction.channels"},
        {"neighbor-cutoff", "model.neighbor_cutoff"},
        {"coord-body-order", "model.coordination.body_order"},
        {"interaction-body-order", "model.interaction.body_order"},
        {"coord-coupling", "model.coordination.channel_coupling"},
        {"interaction-coupling", "model.interaction.channel_coupling"},
        {"distance-warp-sharing", "model.radial.warp_sharing"},
        {"response-layout", "model.response.layout"},
        {"response-activation", "model.response.activation"},
        {"iterations", "training.iterations"},
        {"virial-weight", "loss.virial_weight"},
    };
    return values;
}
void Fail(const std::string& message) { throw std::runtime_error("Grouped input: " + message); }
std::string Trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \r\n");
    return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(" \r\n") - a + 1);
}
bool Starts(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool KnownKey(const std::string& key) {
    if (key == "schema_version" || key == "elements" || key == "model.elements") return true;
    for (const Rule& r : rules) if (key == r.key) return true;
    const std::string prefix = "model.angular_limit_by_body_order.";
    return Starts(key, prefix) && key.size() == prefix.size() + 1 &&
           key.back() >= '2' && key.back() <= '6';
}
bool KnownGroup(const std::string& key) {
    if (key == "model.angular_limit_by_body_order") return true;
    for (const Rule& r : rules) if (Starts(r.key, key + ".")) return true;
    return key == "model";
}
std::string Scalar(const std::string& text) {
    const std::string s = Trim(text);
    if (s.empty()) Fail("empty scalar");
    if (s.front() == '\'' || s.front() == '"') {
        const char quote = s.front();
        if (s.size() < 2 || s.back() != quote) Fail("unterminated quoted scalar");
        std::string result;
        for (std::size_t i = 1; i + 1 < s.size(); ++i) {
            if (quote == '\'' && s[i] == '\'' && i + 2 < s.size() && s[i+1] == '\'') { result += '\''; ++i; }
            else if (quote == '"' && s[i] == '\\') {
                if (++i + 1 >= s.size() || (s[i] != '\\' && s[i] != '"'))
                    Fail("only escaped quote/backslash are supported in double-quoted scalars");
                result += s[i];
            } else if (s[i] == quote) Fail("unexpected quote inside scalar");
            else result += s[i];
        }
        if (result.empty()) Fail("empty quoted scalar");
        return result;
    }
    if (s.find_first_of("\t\r\n") != std::string::npos || s == "null" || s == "~" ||
        s.find_first_of("&*!{}[]") != std::string::npos || s == "|" || s == ">")
        Fail("unsupported YAML scalar: " + s);
    return s;
}
std::vector<std::string> SplitFlow(const std::string& s) {
    std::vector<std::string> result; std::string part; char quote = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote) {
            part += c;
            if (c == '\\' && quote == '"' && i + 1 < s.size()) part += s[++i];
            else if (c == quote) {
                if (quote == '\'' && i + 1 < s.size() && s[i+1] == '\'') part += s[++i];
                else quote = 0;
            }
        } else if (c == '\'' || c == '"') { quote = c; part += c; }
        else if (c == ',') { if (Trim(part).empty()) Fail("empty flow-list item"); result.push_back(Trim(part)); part.clear(); }
        else part += c;
    }
    if (quote || Trim(part).empty()) Fail("unterminated or empty flow collection");
    result.push_back(Trim(part)); return result;
}
std::string List(const std::string& value) {
    std::string s = Trim(value), result;
    if (!s.empty() && s.front() == '[') {
        if (s.back() != ']') Fail("unterminated flow list");
        s = s.substr(1, s.size()-2);
    }
    for (const std::string& item : SplitFlow(s)) {
        if (!result.empty()) result += ',';
        const std::string part = Scalar(item);
        if (part.find(',') != std::string::npos) Fail("commas inside list items are unsupported");
        result += part;
    }
    return result;
}
std::string WithoutComment(const std::string& line) {
    char quote = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote) {
            if (c == '\\' && quote == '"') ++i;
            else if (c == quote) {
                if (quote == '\'' && i + 1 < line.size() && line[i+1] == '\'') ++i;
                else quote = 0;
            }
        } else if (c == '\'' || c == '"') quote = c;
        else if (c == '#' && (i == 0 || line[i-1] == ' ')) return line.substr(0, i);
    }
    return line;
}
void Add(Values& values, const std::string& key, const std::string& value) {
    if (!KnownKey(key)) Fail("unknown setting '" + key + "' (PT-only settings cannot be used by the native trainer)");
    if (!values.emplace(key, value).second) Fail("duplicate setting '" + key + "'");
}
Values ReadConfig(const std::string& filename) {
    std::ifstream file(filename.c_str()); if (!file) Fail("cannot open " + filename);
    Values values; std::set<std::string> nodes;
    std::vector<std::pair<int, std::string> > stack;
    std::string line; std::size_t bytes = 0; int number = 0;
    while (std::getline(file, line)) {
        ++number; bytes += line.size(); if (bytes > 1048576) Fail("configuration exceeds 1 MiB");
        if (line.find('\0') != std::string::npos) Fail("NUL bytes are unsupported");
        if (line.find('\t') != std::string::npos) Fail(filename + ":" + std::to_string(number) + ": tabs are unsupported; use spaces");
        line = WithoutComment(line); const std::string text = Trim(line);
        if (text.empty()) continue;
        if (text == "---" || text == "...") Fail("document markers/multiple YAML documents are unsupported");
        const int indent = static_cast<int>(line.find_first_not_of(' '));
        while (!stack.empty() && stack.back().first >= indent) stack.pop_back();
        if (stack.empty() && indent != 0) Fail("unexpected indentation at line " + std::to_string(number));
        if (Starts(text, "- ")) {
            if (stack.empty()) Fail("a block list requires a setting");
            const std::string key = stack.back().second;
            if (key != "elements" && key != "model.elements" &&
                key != "training.trainable_groups" && key != "initialization.warp_slope_endpoints" &&
                key != "initialization.warp_shift_endpoints") Fail("block lists are unsupported for " + key);
            std::string item = Scalar(text.substr(2));
            if (item.find(',') != std::string::npos) Fail("invalid block-list item");
            if (!values[key].empty()) values[key] += ',';
            values[key] += item; continue;
        }
        if (!stack.empty() && KnownKey(stack.back().second)) Fail("a scalar/list cannot contain mapping entries");
        const std::size_t colon = text.find(':');
        if (colon == std::string::npos) Fail("expected key: value at line " + std::to_string(number));
        const std::string local = Trim(text.substr(0, colon));
        if (local.empty() || local.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
            Fail("invalid mapping key at line " + std::to_string(number));
        const std::string key = (stack.empty() ? "" : stack.back().second + ".") + local;
        if (!nodes.insert(key).second) Fail("duplicate mapping key '" + key + "'");
        const std::string value = Trim(text.substr(colon + 1));
        if (value.empty()) {
            if (!KnownGroup(key) && key != "elements" && key != "model.elements" &&
                key != "training.trainable_groups" && key != "initialization.warp_slope_endpoints" &&
                key != "initialization.warp_shift_endpoints") Fail("unknown/empty group '" + key + "'");
            stack.push_back(std::make_pair(indent, key)); continue;
        }
        if (key == "model.angular_limit_by_body_order" && value.front() == '{') {
            if (value.back() != '}') Fail("unterminated angular-limit mapping");
            for (const std::string& part : SplitFlow(value.substr(1, value.size()-2))) {
                const std::size_t sep = part.find(':'); if (sep == std::string::npos) Fail("expected body order: angular limit");
                Add(values, key + "." + Trim(part.substr(0, sep)), Trim(part.substr(sep + 1)));
            }
        } else Add(values, key, value);
    }
    if (file.bad()) Fail("failed while reading " + filename);
    for (const std::string& node : nodes) {
        if (values.count(node)) continue;
        bool child = false; for (const auto& v : values) if (Starts(v.first, node + ".")) child = true;
        if (!child) Fail("empty group/list '" + node + "'");
    }
    return values;
}
std::string Integer(const std::string& text, bool positive) {
    const std::string s = Scalar(text);
    if (s.find_first_not_of("0123456789") != std::string::npos) Fail("expected nonnegative integer, got " + s);
    errno = 0; char* end = nullptr; const unsigned long long n = std::strtoull(s.c_str(), &end, 10);
    if (errno || *end || n > static_cast<unsigned long long>(std::numeric_limits<int>::max()) || (positive && n == 0)) Fail("integer out of range: " + s);
    return std::to_string(n);
}
std::string Real(const std::string& text, bool positive, bool nonnegative) {
    const std::string s = Scalar(text); char* end = nullptr; errno = 0;
    const double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end || errno || !std::isfinite(value) ||
        (positive && value <= 0) || (nonnegative && value < 0)) Fail("invalid finite number: " + s);
    std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
std::string Convert(const Rule& r, const std::string& value) {
    const std::string type = r.type;
    if (type == "int" || type == "positive-int") return Integer(value, type == "positive-int");
    if (type == "positive-real" || type == "nonnegative-real") return Real(value, type == "positive-real", true);
    if (type == "path" || type == "text") return Scalar(value);
    if (type == "bool") {
        const std::string s = Scalar(value); if (s != "true" && s != "false") Fail(std::string(r.key) + " requires true or false"); return s;
    }
    if (type == "seed") {
        const std::string s = Scalar(value); errno = 0; char* end = nullptr;
        if (s.find_first_not_of("0123456789") != std::string::npos) Fail("seed must be an unsigned integer");
        const unsigned long long n = std::strtoull(s.c_str(), &end, 10);
        if (errno || *end) Fail("seed is outside uint64 range");
        return std::to_string(n);
    }
    if (type == "pair" || type == "groups") {
        const std::string s = List(value); const auto items = SplitFlow(s);
        if (type == "pair") {
            if (items.size() != 2) Fail(std::string(r.key) + " requires two endpoints");
            return Real(items[0], false, false) + "," + Real(items[1], false, false);
        }
        const std::set<std::string> allowed = {"shift", "readout", "radial", "adaptive_amplitude", "adaptive_weight"};
        std::set<std::string> seen;
        for (const std::string& item : items) if (!allowed.count(item) || !seen.insert(item).second) Fail("unknown/duplicate trainable group: " + item);
        return s;
    }
    const std::string s = Scalar(value); std::istringstream choices(type); std::string choice;
    while (std::getline(choices, choice, '|')) {
        const std::size_t eq = choice.find('=');
        if (s == choice.substr(0, eq)) return choice.substr(eq + 1);
    }
    Fail(std::string(r.key) + " has unsupported value '" + s + "'; choices: " + type);
    return "";
}
std::string Absolute(const std::string& value, const std::string& base) {
    const std::string path = value.front() == '/' ? value : base + "/" + value;
    std::vector<std::string> parts; std::istringstream in(path); std::string part;
    while (std::getline(in, part, '/')) {
        if (part.empty() || part == ".") continue;
        if (part == "..") { if (!parts.empty()) parts.pop_back(); }
        else parts.push_back(part);
    }
    std::string result; for (const std::string& p : parts) result += "/" + p;
    return result.empty() ? "/" : result;
}
std::string Parent(const std::string& s) { const auto p = s.rfind('/'); return p == 0 ? "/" : s.substr(0, p); }
bool Exists(const std::string& path) { struct stat info; return ::lstat(path.c_str(), &info) == 0; }
void MakeDirectory(const std::string& path) {
    if (path.empty() || path == "/") return;
    struct stat info;
    if (::stat(path.c_str(), &info) == 0) { if (!S_ISDIR(info.st_mode)) Fail(path + " is not a directory"); return; }
    if (errno != ENOENT) Fail("cannot inspect output directory: " + path);
    MakeDirectory(Parent(path));
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) Fail("cannot create output directory: " + path);
}
bool Equivalent(const std::string& a, const std::string& b) {
    if (a == b) return true;
    char *ea = nullptr, *eb = nullptr; const double x = std::strtod(a.c_str(), &ea), y = std::strtod(b.c_str(), &eb);
    return ea != a.c_str() && eb != b.c_str() && !*ea && !*eb && std::isfinite(x) && x == y;
}
void Put(Values& opts, const std::string& option, const std::string& value) {
    const auto old = opts.find(option);
    if (old != opts.end() && !Equivalent(old->second, value)) Fail("conflicting values for --" + option);
    opts[option] = value;
}
int GetInt(const Values& opts, const std::string& key, int fallback) {
    const auto it = opts.find(key); return it == opts.end() ? fallback : std::stoi(Integer(it->second, false));
}
std::string Quoted(const std::string& s) {
    std::string result = "\"";
    for (char c : s) { if (c == '\\' || c == '"') result += '\\'; result += c; }
    return result + "\"";
}
void ValidateLegacyOptions(const Values& opts, bool static_profile, bool legacy_profile) {
    // Grouped input is deliberately narrower than the historical CLI. Do not
    // let a typo, an Adam flag or a legacy-only control be silently ignored.
    std::set<std::string> allowed = {"init-sh", "max-dist", "bfgs-conv-tol", "bfgs-trace-file",
        "body-l-max", "std-weight", "stdd-weight"};
    for (const Rule& r : rules) if (r.option[0] != '@') allowed.insert(r.option);
    if (static_profile || legacy_profile)
        for (const char* key : {"radial-smooth", "radial-smooth-grid", "scalar-head-l2"}) allowed.insert(key);
    if (legacy_profile) allowed.insert("gate-full-l2");
    for (const auto& option : opts)
        if (!allowed.count(option.first)) Fail("--" + option.first +
            " is not supported with grouped native input; use a documented grouped key or the legacy CLI separately");
    for (const Rule& r : rules) if (r.option[0] != '@' && opts.count(r.option)) {
        const std::string type = r.type;
        if (type == "int" || type == "positive-int" || type == "positive-real" ||
            type == "nonnegative-real" || type == "seed" || type == "bool" ||
            type == "groups" || type == "pair") Convert(r, opts.at(r.option));
        else if (type != "path" && type != "text") {
            bool valid = false; std::istringstream in(type); std::string item;
            while (std::getline(in, item, '|')) if (item.substr(item.find('=') + 1) == opts.at(r.option)) valid = true;
            if (!valid) Fail("unsupported native value for --" + std::string(r.option));
        }
    }
    if (opts.count("two-layer-gate-decoupled-scaling") && opts.at("two-layer-gate-decoupled-scaling") != "true")
        Fail("a legacy decoupled-scaling flag enables the feature by presence; use model.coordination.distance_warp: shared instead");
    for (const char* key : {"std-weight", "stdd-weight"}) if (opts.count(key)) Real(opts.at(key), false, true);
    if (static_profile || legacy_profile) {
        for (const char* key : {"radial-smooth", "scalar-head-l2"}) if (opts.count(key)) Real(opts.at(key), false, true);
        if (opts.count("radial-smooth-grid")) Integer(opts.at("radial-smooth-grid"), true);
    }
    if (legacy_profile && opts.count("gate-full-l2")) Real(opts.at("gate-full-l2"), false, true);
}
}

std::vector<std::string> NormalizeArguments(int argc, char* argv[]) {
    std::vector<std::string> result;
    const std::string command = argc ? argv[0] : "";
    const bool active = command == "train" || command == "init-sh";
    bool grouped = false;
    for (int i = 1; active && i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--") break;
        if (Starts(token, "--")) {
            const std::string key = token.substr(2, token.find('=') == std::string::npos ? std::string::npos : token.find('=')-2);
            if (key == "config" || KnownKey(key) || Aliases().count(key) || key.find('.') != std::string::npos) grouped = true;
        } else if ((token.size() >= 5 && token.substr(token.size()-5) == ".yaml") ||
                   (token.size() >= 4 && token.substr(token.size()-4) == ".yml")) grouped = true;
    }
    bool end_options = false;
    Values seen;
    for (int i = 0; i < argc; ++i) {
        std::string token = argv[i];
        if (token == "--") end_options = true;
        if (active && !end_options && Starts(token, "--")) {
            const std::size_t eq = token.find('='); std::string key = token.substr(2, eq == std::string::npos ? std::string::npos : eq-2);
            const auto alias = Aliases().find(key); if (alias != Aliases().end()) key = alias->second;
            if (key == "config" || KnownKey(key) || key.find('.') != std::string::npos) {
                if (eq == std::string::npos) {
                    if (i + 1 >= argc || Starts(argv[i+1], "--")) Fail("--" + key + " requires a value");
                    token = "--" + key + "=" + argv[++i];
                } else token = "--" + key + token.substr(eq);
            }
            const auto split = token.find('=');
            const std::string value = split == std::string::npos ? "true" : token.substr(split + 1);
            if (grouped && seen.count(key) && !Equivalent(seen[key], value)) Fail("duplicate/conflicting CLI setting --" + key);
            seen[key] = value;
        }
        result.push_back(token);
    }
    return result;
}

bool Resolve(const std::string& command, std::vector<std::string>& args, Values& opts, std::ostream* output) {
    if (command != "train" && command != "init-sh") return false;
    std::string config;
    const auto file = opts.find("config");
    if (file != opts.end()) {
        config = file->second;
        if (config.empty() || config == "true") Fail("--config requires a YAML path");
        opts.erase(file);
    }
    if (args.size() == 1 &&
        ((args[0].size() >= 5 && args[0].substr(args[0].size()-5) == ".yaml") ||
         (args[0].size() >= 4 && args[0].substr(args[0].size()-4) == ".yml"))) {
        if (!config.empty()) Fail("use a positional YAML or --config, not both");
        config = args[0]; args.clear();
    }
    Values settings = config.empty() ? Values() : ReadConfig(config);
    std::set<std::string> cli_paths;
    bool active = !config.empty();
    for (auto it = opts.begin(); it != opts.end();) {
        if (KnownKey(it->first) || it->first.find('.') != std::string::npos) {
            active = true; const auto old = settings.find(it->first);
            if (old != settings.end() && old->second != it->second) {
                // Compare translated, typed values rather than e.g. 1 vs 1.0.
                bool same = false;
                for (const Rule& r : rules) if (it->first == r.key) same = Equivalent(Convert(r, old->second), Convert(r, it->second));
                if (!same) Fail("configuration/CLI conflict for " + it->first);
            }
            if (old == settings.end()) cli_paths.insert(it->first);
            settings[it->first] = it->second; it = opts.erase(it);
        } else ++it;
    }
    const bool check = opts.count("check-config") != 0;
    if (!active) { if (check) Fail("--check-config requires grouped settings or a YAML configuration"); return false; }
    if (check && opts["check-config"] != "true") Fail("--check-config is a boolean flag");
    opts.erase("check-config");
    char cwd[4096]; if (!::getcwd(cwd, sizeof(cwd))) Fail("cannot resolve working directory");
    const std::string base = config.empty() ? cwd : Parent(Absolute(config, cwd));
    const bool continuation = settings.count("data.model") || (command == "train" && args.size() == 2);
    const std::string kind = settings.count("model.kind") ? Scalar(settings.at("model.kind")) : "coordination-adaptive";
    const bool coordination_enabled = settings.count("model.coordination.enabled")
        ? Convert(Rule{"model.coordination.enabled", "@coordination-enabled", "bool"}, settings.at("model.coordination.enabled")) == "true"
        : kind != "static-radial";
    const bool legacy_kind = kind == "coordination-legacy" || kind == "legacy";
    const std::string profile = settings.count("training.profile") ? Scalar(settings.at("training.profile"))
        : ((!continuation && !coordination_enabled) ? "native-static-v1"
            : ((!continuation && legacy_kind) ? "native-legacy-v1" : "native-adaptive-v1"));
    const bool static_profile = profile == "native-static-v1";
    const bool legacy_profile = profile == "native-legacy-v1";
    if (!continuation && (static_profile == coordination_enabled ||
        (coordination_enabled && legacy_profile != legacy_kind)))
        Fail("training.profile must match model.kind and coordination.enabled (native-static-v1 for false, native-legacy-v1 for coordination-legacy, native-adaptive-v1 for coordination-adaptive)");
    if (kind == "static-radial" && coordination_enabled)
        Fail("static-radial requires coordination.enabled: false");
    if (!continuation && !coordination_enabled) {
        for (const auto& value : settings)
            if ((Starts(value.first, "model.coordination.") && value.first != "model.coordination.enabled") ||
                Starts(value.first, "model.response."))
                Fail(value.first + " is unavailable when coordination.enabled is false; omit the inactive setting");
    }
    if (static_profile && settings.count("regularization.coordination_response_l2"))
        Fail("regularization.coordination_response_l2 is unavailable without coordination response");
    if (!continuation && legacy_kind)
        for (const auto& value : settings)
            if (Starts(value.first, "model.response."))
                Fail(value.first + " is unavailable for coordination-legacy; Legacy has no adaptive U projection");
    Values paths; std::string elements;
    for (const auto& value : settings) {
        if (!KnownKey(value.first)) Fail("unknown setting '" + value.first + "'");
        if (value.first == "schema_version") { if (Integer(value.second, true) != "1") Fail("only schema_version: 1 is supported"); continue; }
        if (value.first == "elements" || value.first == "model.elements") {
            const std::string list = List(value.second);
            if (!elements.empty() && elements != list) Fail("conflicting element order");
            elements = list;
            static const std::string symbols = " H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og ";
            std::set<std::string> seen; const auto items = SplitFlow(list);
            for (const std::string& item : items) if (symbols.find(" " + item + " ") == std::string::npos || !seen.insert(item).second) Fail("unknown/duplicate element symbol: " + item);
            Put(opts, "species-count", std::to_string(items.size())); continue;
        }
        if (Starts(value.first, "model.angular_limit_by_body_order.")) continue;
        for (const Rule& r : rules) if (value.first == r.key) {
            std::string converted = Convert(r, value.second);
            const std::string option = r.option;
            if (option == "@coordination-enabled" || option == "@profile") {
                // Adapter metadata, never forwarded as a native presence flag.
            } else if (Starts(option, "@fixed")) {
                if ((static_profile || legacy_profile) && Starts(value.first, "regularization.")) {
                    const Values static_regularizers = {{"regularization.radial_smoothness", "radial-smooth"},
                        {"regularization.radial_smoothness_grid", "radial-smooth-grid"},
                        {"regularization.energy_readout_l2", "scalar-head-l2"},
                        {"regularization.coordination_response_l2", "gate-full-l2"}};
                    Put(opts, static_regularizers.at(value.first), converted);
                } else if (Starts(option, "@fixed:") && !Equivalent(converted, option.substr(7)))
                    Fail(value.first + " is fixed to " + option.substr(7) + " by native-adaptive-v1");
            } else if (std::string(r.type) == "path") {
                converted = Absolute(converted, cli_paths.count(value.first) ? cwd : base);
                if (option.front() == '@') paths[option] = converted;
                else Put(opts, option, converted);
            } else if (option == "two-layer-gate-decoupled-scaling" && converted == "false") {
                if (opts.count(option) || opts.count("two-layer-gate-full-decoupled")) Fail("shared distance warp conflicts with legacy decoupled-scaling flag");
            } else if (value.first != "model.kind" || coordination_enabled || continuation)
                Put(opts, option, converted);
        }
    }
    // Legacy aliases must not bypass the conflict checks.
    for (const auto& alias : Values{{"max-dist", "cutoff"}, {"bfgs-conv-tol", "optimizer-conv-tol"}, {"bfgs-trace-file", "optimizer-trace-file"}})
        if (opts.count(alias.first) && opts.count(alias.second) && !Equivalent(opts[alias.first], opts[alias.second])) Fail("conflicting --" + alias.first + " and --" + alias.second);
    ValidateLegacyOptions(opts, static_profile, legacy_profile);
    if ((paths.count("@train") || paths.count("@model") || paths.count("@output")) && !args.empty())
        Fail("do not mix positional model/data/output paths with grouped data/output.model paths");
    if (continuation) {
        for (const auto& value : settings) if (Starts(value.first, "model.") || value.first == "elements")
            Fail("continuation inherits the saved architecture; omit " + value.first + " and use data.model");
        for (const Rule& r : rules) if (Starts(r.key, "model.") && opts.count(r.option))
            Fail("continuation inherits the saved architecture; omit --" + std::string(r.option));
        if (opts.count("init-sh")) Fail("continuation cannot use --init-sh");
        Put(opts, "init-params", "same");
    } else {
        Put(opts, "superlinear-mode", coordination_enabled && !legacy_kind ? "radial-coefficient-gate" : "legacy");
        if (legacy_kind && coordination_enabled) {
            // The same K/R/O interface selects ranked Legacy graphs. Do not
            // obtain an Adaptive model by merely relabeling its family.
            for (const auto& option : Values{{"two-layer-gate", "true"},
                 {"two-layer-gate-mode", "mu-scalar-full"},
                 {"two-layer-gate-site-mode", "neighbor"},
                 {"two-layer-gate-radial-mode", "shared-radial"}})
                Put(opts, option.first, option.second);
            if (opts.count("tensor-message-lmax") && GetInt(opts, "tensor-message-lmax", 0) != 0)
                Fail("coordination-legacy does not support tensor messages");
            // Explicit zero is metadata, not an unsupported architecture flag.
            opts.erase("tensor-message-lmax");
            if (opts.count("joint-readout") && opts.at("joint-readout") != "false")
                Fail("coordination-legacy does not support joint readout");
            // New grouped models share the Adaptive graph defaults; old
            // positional native CLI/model defaults remain untouched.
            if (!opts.count("sh-factor-pruning")) Put(opts, "sh-factor-pruning", "same-k");
            if (!opts.count("two-layer-gate-factor-pruning")) Put(opts, "two-layer-gate-factor-pruning", "q-total");
        }
        if (!coordination_enabled) {
            for (const Rule& r : rules)
                if ((Starts(r.key, "model.coordination.") || Starts(r.key, "model.response.")) &&
                    r.option[0] != '@' && opts.count(r.option))
                    Fail("--" + std::string(r.option) + " is unavailable when coordination.enabled is false");
            if (opts.count("joint-readout") && opts.at("joint-readout") != "false")
                Fail("static-radial supports only joint: false");
            Put(opts, "sh-factor-pruning", "q-total");
        }
        const int l = GetInt(opts, "l-max", 3), k = GetInt(opts, "k-max", 3);
        const int o = GetInt(opts, "outer-k-rank", 0), g = GetInt(opts, "two-layer-gate-k-rank", k);
        const int body = GetInt(opts, "body-order", 5), gate_body = GetInt(opts, "two-layer-gate-body-order", body);
        const int n = GetInt(opts, "radial-basis-size", 10), tensor = GetInt(opts, "tensor-message-lmax", 0);
        if (coordination_enabled && legacy_kind && !opts.count("two-layer-gate-k-rank"))
            Put(opts, "two-layer-gate-k-rank", std::to_string(g));
        if (l > 6 || !k || !o || o > k || !g || g > k || body < 2 || body > 6 || gate_body < 2 || gate_body > 6 || !n || tensor > 2 || tensor > l)
            Fail("invalid architecture: require L=0..6, 1<=R,O<=K, body orders=2..6, tensor L<=min(2,L)");
        if (opts.count("joint-readout") && opts["joint-readout"] == "true" && l < 1)
            Fail("joint readout requires l_max >= 1");
        const int d = GetInt(opts, "adaptive-radial-shared-n-rank", 1);
        const std::string projection = opts.count("adaptive-radial-projection") ? opts["adaptive-radial-projection"] : "full";
        if (d < 1 || d > n || (projection == "full" && d != 1)) Fail("latent channels require shared-n (shared-latent) layout and 1<=d<=basis_size");
        std::string limits; bool explicit_limits = false;
        for (const auto& value : settings) if (Starts(value.first, "model.angular_limit_by_body_order.")) explicit_limits = true;
        if (explicit_limits) {
            for (int b = 2; b <= (body >= 6 ? 6 : 5); ++b) {
                const std::string key = "model.angular_limit_by_body_order." + std::to_string(b);
                if (!settings.count(key)) Fail("angular limits must name each body order 2.." + std::to_string(body >= 6 ? 6 : 5));
                const std::string v = Integer(settings[key], false); if (std::stoi(v) > l) Fail("angular limit exceeds l_max");
                if (!limits.empty()) limits += ',';
                limits += v;
            }
            if (body < 6 && settings.count("model.angular_limit_by_body_order.6")) Fail("body order 6 angular limit requires interaction.body_order: 6");
            Put(opts, "body-l-max", limits);
        }
        if (command == "train") {
            if (opts.count("trainable-groups")) Fail("trainable_groups is for continuation from a complete checkpoint");
            if ((coordination_enabled && !legacy_kind) || !opts.count("init-params")) Put(opts, "init-params", "same");
            Put(opts, "init-sh", "true");
        }
    }
    if (command == "train" && (static_profile || legacy_profile)) {
        if (static_profile && opts.count("trainable-groups")) Fail("trainable_groups requires an adaptive checkpoint, not native-static-v1");
        if (opts.count("init-random-seed") && (continuation || opts.at("init-params") != "random"))
            Fail("Legacy/static initialization.seed requires a fresh model with initialization.parameters: random");
        // Keep native static rescale/linear warmup, but no hidden 75-step
        // pretraining or unscoped ini.mtp output outside the declared run paths.
        opts["skip-preinit"] = "true";
    }
    const std::string dir = paths.count("@directory") ? paths["@directory"] : "";
    if (!dir.empty()) {
        if (command == "init-sh") {
            if (!paths.count("@output")) paths["@output"] = dir + "/initial.mtp";
        } else {
            for (const auto& p : Values{{"inline-sh-model", "initial.mtp"}, {"curr-pot-name", "current.mtp"}, {"best-pot-name", "best.mtp"}, {"trained-pot-name", "final.mtp"}, {"optimizer-trace-file", "optimizer.csv"}})
                if ((!continuation || p.first != "inline-sh-model") && !opts.count(p.first)) opts[p.first] = dir + "/" + p.second;
        }
    }
    if (command == "train") {
        if (paths.count("@output")) Fail("use output.final for train; output.model is for init-sh");
        if (paths.count("@train")) { if (continuation) args.push_back(paths.at("@model")); args.push_back(paths["@train"]); }
        else if (paths.count("@model")) Fail("data.model requires data.train");
        if (args.size() != (continuation ? 2U : 1U)) Fail("training requires data.train (and data.model for continuation)");
        if (!opts.count("trained-pot-name") || !opts.count("curr-pot-name") ||
            !opts.count("best-pot-name") || !opts.count("optimizer-trace-file") ||
            (!continuation && !opts.count("inline-sh-model")))
            Fail("grouped training requires output.directory or all explicit output paths");
    } else {
        if (paths.count("@train") || paths.count("@model") || settings.count("data.valid")) Fail("init-sh does not consume training/validation/model inputs");
        for (const auto& value : settings) if (Starts(value.first, "training.") || Starts(value.first, "loss.") || Starts(value.first, "regularization.") || Starts(value.first, "initialization."))
            Fail("init-sh only writes topology; use train for " + value.first);
        std::set<std::string> model_options = {"body-l-max", "max-dist"};
        for (const Rule& r : rules) if (Starts(r.key, "model.")) model_options.insert(r.option);
        if (legacy_kind && coordination_enabled)
            for (const char* key : {"two-layer-gate", "two-layer-gate-mode", "two-layer-gate-site-mode", "two-layer-gate-radial-mode"})
                model_options.insert(key);
        for (const auto& option : opts) if (!model_options.count(option.first)) Fail("init-sh does not use --" + option.first);
        if (paths.count("@output")) args.push_back(paths["@output"]);
        if (args.size() != 1) Fail("init-sh requires output.model or one positional output path");
    }
    if (output) {
        *output << "Grouped input schema=1 command=" << command << " mode=" << (continuation ? "continue" : "fresh") << '\n';
        if (!elements.empty()) *output << "elements_in_cfg_type_order=" << elements << " (does not enable ZBL)\n";
        *output << "resolved_arguments:"; for (const std::string& arg : args) *output << " " << Quoted(arg); *output << '\n';
        for (const auto& option : opts) *output << "resolved_option --" << option.first << "=" << Quoted(option.second) << '\n';
        if (command == "train") {
            if (legacy_profile) *output << "native-legacy-v1: ranked Legacy coordination and interaction; K/R/O share Adaptive meanings; physical radial coordinates; native vibrations weighting; no adaptive U or 75-step pretraining; iterations are BFGS steps, not epochs\n";
            else if (static_profile) *output << "native-static-v1: no coordination/response; physical radial coordinates; native vibrations weighting; radial_smoothness defaults to 1e-6; energy_readout_l2 defaults to 0; no 75-step pretraining; iterations are BFGS steps, not epochs\n";
            else *output << "native-adaptive-v1: fixed RMS; force-RMS output scale; radial_smoothness=1e-6; energy_readout_l2=1e-5; coordination_response_l2=1e-4; iterations are BFGS steps, not epochs\n";
        }
        if (check) *output << "CONFIG_CHECK_PASS (syntax/translation only; model/data not loaded, no outputs created)\n";
    }
    if (check) return true;
    // New grouped runs are fail-closed about accidental output reuse. Legacy
    // invocations remain unchanged and complete source checkpoints are untouched.
    if (command == "init-sh") {
        if (Exists(args[0])) Fail("refusing to overwrite existing topology: " + args[0]);
    } else {
        std::set<std::string> outputs;
        for (const std::string key : {"inline-sh-model", "curr-pot-name", "best-pot-name", "trained-pot-name", "optimizer-trace-file"})
            if (opts.count(key)) {
                const std::string target = Absolute(opts[key], cwd);
                if (!outputs.insert(target).second) Fail("output paths must be distinct: " + target);
                for (const std::string& input : args) if (target == Absolute(input, cwd)) Fail("output path equals an input: " + target);
                if (opts.count("valid-cfgs") && target == Absolute(opts["valid-cfgs"], cwd)) Fail("output path equals validation input: " + target);
                if (Exists(target)) Fail("refusing to reuse/overwrite output: " + target + "; choose a new output directory");
            }
    }
    if (!dir.empty()) MakeDirectory(dir);
    if (command == "train") {
        opts["grouped-input"] = continuation ? "continue" : "fresh";
        opts["grouped-profile"] = profile;
    }
    return false;
}

const char* Help() {
    return "Grouped physical input (native adaptive, Legacy or static-radial models):\n"
           "  mlp-sus2 train run.yaml [--check-config]\n"
           "  mlp-sus2 train --config run.yaml [--check-config]\n"
           "  mlp-sus2 init-sh model.yaml [--check-config]\n"
           "  New CLI names: --model.kind coordination-adaptive --radial-channels K\n"
           "    --coord-channels R --interaction-channels O --neighbor-cutoff ANGSTROM\n"
           "    Compact aliases: --L L --K K --R R --O O (identical meanings in both families)\n"
           "    --model.kind coordination-legacy: ranked Legacy, no adaptive response U\n"
           "    --coord-coupling cross|diagonal --interaction-coupling cross|diagonal\n"
           "    --coord-enabled false: static radial weights, no coordination/response keys\n"
           "    --distance-warp-sharing L|K|LK (default LK; semantic aliases remain valid)\n"
           "    --response-layout full|shared-n (default full); --model.response.latent_channels d\n"
           "  All grouped scalar keys also accept --group.key=value. Legacy flags remain valid.\n"
           "  Conflicting values, unknown keys and unsupported native/PT-only options are errors.\n"
           "  Configuration paths are relative to the YAML directory; no output is overwritten.\n"
           "  Continue with data.model; omit model.* and inherit the saved architecture.\n"
           "  See docs/grouped-native-input.md and examples/grouped-native-*.yaml.\n";
}
}
