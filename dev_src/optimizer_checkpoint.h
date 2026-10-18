#ifndef MLIP_OPTIMIZER_CHECKPOINT_H
#define MLIP_OPTIMIZER_CHECKPOINT_H

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/model_path.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sus2 {

// Track the best objective among points that the trainer has both evaluated
// and classified as accepted.  In particular, a reverse-communication trial
// must never reach this object before its objective has been evaluated.
class BestAcceptedCheckpoint {
public:
	BestAcceptedCheckpoint()
		: valid_(false), loss_(std::numeric_limits<double>::infinity()), step_(-1) {}

	bool Observe(double loss, int step)
	{
		if (!std::isfinite(loss))
			return false;
		if (valid_ && loss >= loss_)
			return false;
		valid_ = true;
		loss_ = loss;
		step_ = step;
		return true;
	}

	bool Valid() const { return valid_; }
	double Loss() const { return loss_; }
	int Step() const { return step_; }

private:
	bool valid_;
	double loss_;
	int step_;
};

inline std::string DefaultBestCheckpointPath(
	const std::string& current_checkpoint_path)
{
	if (current_checkpoint_path.empty())
		return std::string();
	ValidateModelOutputPath(current_checkpoint_path);
	const std::string extension = ".sus2";
	if (current_checkpoint_path.size() >= extension.size()
	    && current_checkpoint_path.compare(
	        current_checkpoint_path.size() - extension.size(),
	        extension.size(),
	        extension) == 0) {
		return current_checkpoint_path.substr(
		           0, current_checkpoint_path.size() - extension.size())
			+ ".best" + extension;
	}
	return current_checkpoint_path + ".best.sus2";
}

inline bool CheckpointFileExists(const std::string& path)
{
#ifndef _WIN32
	struct stat status;
	return ::lstat(path.c_str(), &status) == 0;
#else
	std::ifstream input(path.c_str(), std::ios::binary);
	return input.good();
#endif
}

inline std::string CheckpointParentDirectory(const std::string& path)
{
	const std::string::size_type slash = path.find_last_of('/');
	if (slash == std::string::npos)
		return ".";
	if (slash == 0)
		return "/";
	return path.substr(0, slash);
}

inline std::string AbsoluteLexicalCheckpointPath(const std::string& path)
{
	if (path.empty())
		return std::string();
	std::string absolute = path;
#ifndef _WIN32
	if (absolute[0] != '/') {
		std::vector<char> cwd(256, '\0');
		while (::getcwd(cwd.data(), cwd.size()) == nullptr) {
			if (errno != ERANGE)
				throw std::runtime_error(
					"cannot resolve checkpoint working directory: "
					+ std::string(std::strerror(errno)));
			cwd.resize(cwd.size() * 2, '\0');
		}
		absolute = std::string(cwd.data()) + "/" + absolute;
	}
#endif

	std::vector<std::string> components;
	std::string component;
	for (std::string::size_type index = 0; index <= absolute.size(); ++index) {
		const char character =
			index < absolute.size() ? absolute[index] : '/';
		if (character != '/') {
			component.push_back(character);
			continue;
		}
		if (component.empty() || component == ".") {
			component.clear();
			continue;
		}
		if (component == "..") {
			if (!components.empty())
				components.pop_back();
		} else {
			components.push_back(component);
		}
		component.clear();
	}

#ifndef _WIN32
	std::string normalized = "/";
#else
	std::string normalized;
#endif
	for (std::size_t index = 0; index < components.size(); ++index) {
		if (index != 0)
			normalized += '/';
		normalized += components[index];
	}
	return normalized;
}

inline std::string CanonicalCheckpointDestination(const std::string& path)
{
	const std::string absolute = AbsoluteLexicalCheckpointPath(path);
#ifndef _WIN32
	char* resolved = ::realpath(absolute.c_str(), nullptr);
	if (resolved != nullptr) {
		const std::string result(resolved);
		std::free(resolved);
		return result;
	}

	const std::string parent = CheckpointParentDirectory(absolute);
	resolved = ::realpath(parent.c_str(), nullptr);
	if (resolved != nullptr) {
		const std::string::size_type slash = absolute.find_last_of('/');
		const std::string basename = absolute.substr(slash + 1);
		std::string result(resolved);
		std::free(resolved);
		if (result != "/")
			result += '/';
		result += basename;
		return result;
	}
#endif
	return absolute;
}

inline bool CheckpointDestinationsAlias(
	const std::string& first,
	const std::string& second)
{
	if (first.empty() || second.empty())
		return false;
#ifndef _WIN32
	struct stat first_status;
	struct stat second_status;
	if (::stat(first.c_str(), &first_status) == 0
	    && ::stat(second.c_str(), &second_status) == 0
	    && first_status.st_dev == second_status.st_dev
	    && first_status.st_ino == second_status.st_ino)
		return true;
#endif
	return CanonicalCheckpointDestination(first)
		== CanonicalCheckpointDestination(second);
}

inline void SyncCheckpointFile(const std::string& path)
{
#ifndef _WIN32
	const int descriptor = ::open(path.c_str(), O_RDONLY);
	if (descriptor < 0)
		throw std::runtime_error(
			"cannot open checkpoint for fsync " + path + ": "
			+ std::strerror(errno));
	if (::fsync(descriptor) != 0) {
		const int saved_errno = errno;
		::close(descriptor);
		throw std::runtime_error(
			"cannot fsync checkpoint " + path + ": "
			+ std::strerror(saved_errno));
	}
	if (::close(descriptor) != 0)
		throw std::runtime_error(
			"cannot close checkpoint descriptor " + path + ": "
			+ std::strerror(errno));
#else
	(void)path;
#endif
}

inline void SyncCheckpointDirectory(const std::string& path)
{
#ifndef _WIN32
	int flags = O_RDONLY;
#ifdef O_DIRECTORY
	flags |= O_DIRECTORY;
#endif
	const std::string directory = CheckpointParentDirectory(path);
	const int descriptor = ::open(directory.c_str(), flags);
	if (descriptor < 0)
		throw std::runtime_error(
			"cannot open checkpoint directory for fsync " + directory + ": "
			+ std::strerror(errno));
	if (::fsync(descriptor) != 0) {
		const int saved_errno = errno;
		::close(descriptor);
		throw std::runtime_error(
			"cannot fsync checkpoint directory " + directory + ": "
			+ std::strerror(saved_errno));
	}
	if (::close(descriptor) != 0)
		throw std::runtime_error(
			"cannot close checkpoint directory descriptor " + directory + ": "
			+ std::strerror(errno));
#else
	(void)path;
#endif
}

// Write beside the destination and rename only after the writer has produced
// a non-empty file.  A failed or interrupted write therefore cannot truncate
// the last usable checkpoint.
template <typename Writer>
void AtomicWriteCheckpoint(const std::string& destination, Writer writer)
{
	ValidateModelOutputPath(destination);
	static unsigned long long write_counter = 0;
	std::string temporary = destination + ".tmp";
#ifndef _WIN32
	temporary += "." + std::to_string(
		static_cast<unsigned long long>(::getpid()));
#endif
	temporary += "." + std::to_string(++write_counter) + ".sus2";
	std::remove(temporary.c_str());
	try {
		writer(temporary);
	} catch (...) {
		std::remove(temporary.c_str());
		throw;
	}

	std::ifstream check(temporary.c_str(), std::ios::binary | std::ios::ate);
	if (!check.is_open() || check.tellg() <= 0) {
		check.close();
		std::remove(temporary.c_str());
		throw std::runtime_error(
			"checkpoint writer did not produce a non-empty temporary file: "
			+ temporary);
	}
	check.close();
	try {
		SyncCheckpointFile(temporary);
	} catch (...) {
		std::remove(temporary.c_str());
		throw;
	}

	if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
		const int saved_errno = errno;
		std::remove(temporary.c_str());
		throw std::runtime_error(
			"cannot atomically replace checkpoint " + destination + ": "
			+ std::strerror(saved_errno));
	}
	SyncCheckpointDirectory(destination);
}

// Preserve a best checkpoint left by an earlier invocation before starting a
// new best-of-run stream.  Copy it atomically instead of moving it, so a crash
// between archive creation and the first new best write still leaves the
// original best path usable.  Add a numeric disambiguator rather than ever
// overwriting an older archive.
inline std::string ArchiveExistingCheckpoint(
	const std::string& path,
	const std::string& archive_suffix)
{
	if (path.empty() || !CheckpointFileExists(path))
		return std::string();
	const std::string stem = ModelOutputStem(path);
	std::string archive = stem + archive_suffix + ".sus2";
	int disambiguator = 1;
	while (CheckpointFileExists(archive)) {
		archive = stem + archive_suffix + "."
			+ std::to_string(disambiguator++) + ".sus2";
	}
	AtomicWriteCheckpoint(archive, [&](const std::string& temporary) {
		std::ifstream input(path.c_str(), std::ios::binary);
		if (!input.is_open())
			throw std::runtime_error(
				"cannot open existing checkpoint for archive: " + path);
		std::ofstream output(
			temporary.c_str(), std::ios::binary | std::ios::trunc);
		if (!output.is_open())
			throw std::runtime_error(
				"cannot open checkpoint archive temporary file: " + temporary);
		output << input.rdbuf();
		output.flush();
		if (!input.good() && !input.eof())
			throw std::runtime_error(
				"failed while reading checkpoint for archive: " + path);
		if (!output.good())
			throw std::runtime_error(
				"failed while writing checkpoint archive: " + temporary);
		output.close();
		if (output.fail())
			throw std::runtime_error(
				"failed while closing checkpoint archive: " + temporary);
	});
	return archive;
}

} // namespace sus2

#endif // MLIP_OPTIMIZER_CHECKPOINT_H
