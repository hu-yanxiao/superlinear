#ifndef SUS2_MODEL_PATH_H
#define SUS2_MODEL_PATH_H

#include <stdexcept>
#include <string>

namespace sus2 {

// Reading historical model names is unrestricted. Every new model write uses
// this exact, case-sensitive suffix, including atomic temporary snapshots.
inline bool HasModelOutputSuffix(const std::string& path)
{
	const std::string suffix = ".sus2";
	return path.find('\0') == std::string::npos
		&& path.size() > suffix.size()
		&& path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0
		&& path[path.size() - suffix.size() - 1] != '/'
		&& path[path.size() - suffix.size() - 1] != '\\';
}

inline void ValidateModelOutputPath(const std::string& path)
{
	if (!HasModelOutputSuffix(path))
		throw std::invalid_argument(
			"model output path must end in .sus2: " + path
			+ " (legacy .mtp files are supported for reading only)");
}

inline std::string ModelOutputStem(const std::string& path)
{
	ValidateModelOutputPath(path);
	return path.substr(0, path.size() - 5);
}

} // namespace sus2

#endif
