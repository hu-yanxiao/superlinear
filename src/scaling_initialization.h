#ifndef SUS2_SCALING_INITIALIZATION_H
#define SUS2_SCALING_INITIALIZATION_H

#include "common/utils.h"
#include "element_radii.h"
#include "zbl.h"
#include <algorithm>
#include <map>
#include <cmath>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ScalingInitialization {
inline std::string Trim(const std::string& text)
{
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

inline std::vector<std::string> Elements(std::string text, bool required = false)
{
    text = Trim(text);
    if (text.empty() && !required) return {};
    if (!text.empty() && (text.front() == '{' || text.front() == '[')) {
        const char close = text.front() == '{' ? '}' : ']';
        if (text.back() != close) ERROR("Malformed scaling element list brackets");
        text = Trim(text.substr(1, text.size() - 2));
    }
    if (text.empty()) ERROR("Scaling elements must contain at least one symbol");
    if (text.find_first_of("{}[]") != std::string::npos)
        ERROR("Malformed scaling element list brackets");
    std::vector<std::string> tokens;
    if (text.find(',') == std::string::npos) {
        std::istringstream input(text);
        std::string symbol;
        while (input >> symbol) tokens.push_back(symbol);
    } else {
        size_t start = 0;
        while (true) {
            const size_t comma = text.find(',', start);
            const std::string symbol = Trim(text.substr(start, comma - start));
            if (symbol.empty() || symbol.find_first_of(" \t\r\n") != std::string::npos)
                ERROR("Scaling element lists cannot contain empty or compound entries");
            tokens.push_back(symbol);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    std::set<std::string> seen;
    for (const auto& symbol : tokens) {
        double covalent, vdw;
        if (!ScalingElementRadii::Lookup(symbol, covalent, vdw) || !seen.insert(symbol).second)
            ERROR("Unknown or duplicate scaling element symbol: " + symbol);
    }
    return tokens;
}

inline void Validate(const std::vector<std::string>& elements, const std::string& mode)
{
    if (mode != "legacy" && mode != "element-radii")
        ERROR("scaling-init must be auto, legacy, or element-radii");
    if (mode != "element-radii") return;
    if (elements.empty())
        ERROR("element-radii scaling initialization requires --elements/model.elements");
    for (const auto& symbol : elements) {
        double covalent, vdw;
        if (!ScalingElementRadii::Lookup(symbol, covalent, vdw))
            ERROR("No reliable covalent/van der Waals radius for scaling element " + symbol);
    }
}

struct Options {
    std::string mode;
    std::vector<std::string> elements;
};

inline Options Resolve(const std::map<std::string, std::string>& options)
{
    const auto get = [&](const char* key) {
        const auto it = options.find(key);
        return it == options.end() ? std::string() : it->second;
    };
    Options out;
    out.elements = Elements(get("elements"), options.count("elements") != 0);
    out.mode = get("scaling-init");
    const bool ranges = !get("scal-range").empty() || !get("s-range").empty();
    // Explicit endpoint ranges preserve the legacy initialization contract.
    if (out.mode.empty() || out.mode == "auto")
        out.mode = !out.elements.empty() && !ranges ? "element-radii" : "legacy";
    if (out.mode == "element-radii" && ranges)
        ERROR("element-radii conflicts with --scal-range/--s-range; use --scaling-init=legacy for manual ranges");
    Validate(out.elements, out.mode);
    return out;
}

inline std::pair<double, double> Range(const std::map<std::string, std::string>& options,
                                     const char* name, double first, double last)
{
    const auto it = options.find(name);
    if (it == options.end() || it->second.empty()) return {first, last};
    std::string text = it->second;
    if (std::count(text.begin(), text.end(), ',') != 1)
        ERROR(std::string(name) + " requires two comma-separated finite endpoints");
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    std::string extra;
    if (!(input >> first >> last) || (input >> extra) || !std::isfinite(first) || !std::isfinite(last))
        ERROR(std::string(name) + " requires two finite endpoints");
    return {first, last};
}

inline double Fraction(const std::string& sharing, int block, int raw_k, int angular_channels)
{
    if (raw_k < 2 || (sharing != "K" && sharing != "LK")) return 0.0;
    const int k = sharing == "K" ? block : block / angular_channels;
    return static_cast<double>(k) / (raw_k - 1);
}

inline std::vector<std::pair<double,double>> Radii(const std::vector<std::string>& elements)
{
    std::vector<std::pair<double,double>> result;
    for (const auto& symbol : elements) {
        double covalent, vdw;
        if (!ScalingElementRadii::Lookup(symbol, covalent, vdw))
            ERROR("Missing scaling element radii for " + symbol);
        result.emplace_back(covalent, vdw);
    }
    return result;
}

inline void CheckZBL(const std::vector<std::string>& elements, const std::vector<int>& atomic_numbers)
{
    if (elements.empty() || atomic_numbers.empty()) return;
    if (elements.size() != atomic_numbers.size())
        ERROR("elements and zbl-elements must have identical type order");
    for (size_t i = 0; i < elements.size(); ++i)
        if (ZBLAtomicNumberFromSymbol(elements[i]) != atomic_numbers[i])
            ERROR("elements and zbl-elements must have identical type order");
}

// Trailer fields are ignored by older ordered readers; physical warp coefficients
// remain in the existing scal_coeffs block, including in untrained templates.
inline void Write(std::ostream& out, const std::vector<std::string>& elements, const std::string& mode)
{
    if (elements.empty()) return;
    out << "\nscaling_elements = {";
    for (size_t i = 0; i < elements.size(); ++i) {
        if (i) out << ", ";
        out << elements[i];
    }
    out << "}\nscaling_init = " << mode << '\n';
}
}
#endif
