#include "../src/model_path.h"
#include "../dev_src/optimizer_checkpoint.h"

#include <cassert>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

static std::string Read(const std::string& path)
{
	std::ifstream input(path.c_str(), std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(input),
	                   std::istreambuf_iterator<char>());
}

int main(int argc, char** argv)
{
	assert(argc == 2); // Caller supplies an isolated temporary directory.
	for (const std::string path : {"model.sus2", "dir/model.sus2", "old.mtp.sus2"}) {
		assert(sus2::HasModelOutputSuffix(path));
		sus2::ValidateModelOutputPath(path);
	}
	for (const std::string path : {"", ".sus2", "dir/.sus2", "model.mtp",
	                              "model.sus2.tmp", "model.SUS2", "model.sus2/"}) {
		assert(!sus2::HasModelOutputSuffix(path));
		bool rejected = false;
		try { sus2::ValidateModelOutputPath(path); }
		catch (const std::invalid_argument&) { rejected = true; }
		assert(rejected);
		bool called = false;
		try {
			sus2::AtomicWriteCheckpoint(path, [&](const std::string&) { called = true; });
		} catch (const std::invalid_argument&) {}
		assert(!called);
	}
	assert(!sus2::HasModelOutputSuffix(std::string("legacy.mtp\0.sus2", 16)));
	const std::string checkpoint = std::string(argv[1]) + "/current.sus2";
	assert(sus2::DefaultBestCheckpointPath(checkpoint)
	       == std::string(argv[1]) + "/current.best.sus2");
	assert(sus2::DefaultBestCheckpointPath("").empty());
	std::string temporary;
	sus2::AtomicWriteCheckpoint(checkpoint, [&](const std::string& path) {
		temporary = path;
		assert(path != checkpoint);
		sus2::ValidateModelOutputPath(path);
		std::ofstream output(path.c_str());
		output << "MTP\noriginal\n";
	});
	assert(!sus2::CheckpointFileExists(temporary));
	assert(Read(checkpoint) == "MTP\noriginal\n");
	try {
		sus2::AtomicWriteCheckpoint(checkpoint, [&](const std::string& path) {
			temporary = path;
			std::ofstream output(path.c_str());
			output << "partial";
			throw std::runtime_error("simulated failed model write");
		});
		assert(false);
	} catch (const std::runtime_error&) {}
	assert(!sus2::CheckpointFileExists(temporary));
	assert(Read(checkpoint) == "MTP\noriginal\n");
	const std::string archive = sus2::ArchiveExistingCheckpoint(checkpoint, ".before-run");
	assert(archive == std::string(argv[1]) + "/current.before-run.sus2");
	assert(Read(archive) == Read(checkpoint));
	const std::string second = sus2::ArchiveExistingCheckpoint(checkpoint, ".before-run");
	assert(second == std::string(argv[1]) + "/current.before-run.1.sus2");
	assert(Read(second) == Read(checkpoint));
	return 0;
}
