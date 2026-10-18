#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "dev_src/optimizer_checkpoint.h"

namespace {

void Require(bool condition, const char* message)
{
	if (!condition) {
		std::cerr << message << '\n';
		std::exit(1);
	}
}

std::string ReadAll(const std::string& path)
{
	std::ifstream input(path.c_str(), std::ios::binary);
	return std::string(
		(std::istreambuf_iterator<char>(input)),
		std::istreambuf_iterator<char>());
}

void WriteText(const std::string& path, const std::string& text)
{
	std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
	output << text;
	output.close();
}

} // namespace

int main(int argc, char** argv)
{
	Require(argc == 2, "expected a temporary checkpoint path");
	const std::string checkpoint = argv[1];
	const std::string temporary = checkpoint + ".tmp";
	const std::string archive = sus2::ModelOutputStem(checkpoint) + ".previous.sus2";
	const std::string archive_1 = sus2::ModelOutputStem(checkpoint) + ".previous.1.sus2";
	const std::string hardlink_path = checkpoint + ".hardlink";
	const std::string symlink_path = checkpoint + ".symlink";
	std::remove(checkpoint.c_str());
	std::remove(temporary.c_str());
	std::remove(archive.c_str());
	std::remove(archive_1.c_str());
	std::remove(hardlink_path.c_str());
	std::remove(symlink_path.c_str());

	Require(
		sus2::DefaultBestCheckpointPath("fecn6_current.sus2")
			== "fecn6_current.best.sus2",
		"default .sus2 best-checkpoint path mismatch");
	bool extensionless_rejected = false;
	try { sus2::DefaultBestCheckpointPath("current"); }
	catch (const std::invalid_argument&) { extensionless_rejected = true; }
	Require(extensionless_rejected, "extensionless model output was accepted");
	Require(
		sus2::DefaultBestCheckpointPath("").empty(),
		"empty current checkpoint unexpectedly produced a best path");
	const std::string::size_type slash = checkpoint.find_last_of('/');
	const std::string lexical_alias = slash == std::string::npos
		? "./" + checkpoint
		: checkpoint.substr(0, slash + 1) + "./"
			+ checkpoint.substr(slash + 1);
	Require(
		sus2::CheckpointDestinationsAlias(checkpoint, lexical_alias),
		"lexically equivalent checkpoint paths were not detected");

	sus2::BestAcceptedCheckpoint best;
	Require(best.Observe(3.798, 1304), "first accepted loss was not saved");
	Require(!best.Observe(9004.469, 1305), "worse accepted loss replaced best");
	Require(!best.Observe(
	            std::numeric_limits<double>::quiet_NaN(), 1306),
	        "non-finite loss replaced best");
	Require(best.Step() == 1304, "best step changed after worse observations");
	Require(std::abs(best.Loss() - 3.798) < 1.0e-15,
	        "best loss changed after worse observations");
	Require(best.Observe(3.700, 1307), "improved accepted loss was not saved");

	sus2::AtomicWriteCheckpoint(checkpoint, [](const std::string& path) {
		WriteText(path, "accepted-step-1304\n");
	});
	Require(ReadAll(checkpoint) == "accepted-step-1304\n",
	        "initial atomic checkpoint content mismatch");

#ifndef _WIN32
	Require(::link(checkpoint.c_str(), hardlink_path.c_str()) == 0,
	        "could not create checkpoint hardlink test fixture");
	Require(
		sus2::CheckpointDestinationsAlias(checkpoint, hardlink_path),
		"hardlinked checkpoint destinations were not detected");
	const std::string canonical_checkpoint =
		sus2::CanonicalCheckpointDestination(checkpoint);
	Require(::symlink(canonical_checkpoint.c_str(), symlink_path.c_str()) == 0,
	        "could not create checkpoint symlink test fixture");
	Require(
		sus2::CheckpointDestinationsAlias(checkpoint, symlink_path),
		"symlinked checkpoint destinations were not detected");
	std::remove(hardlink_path.c_str());
	std::remove(symlink_path.c_str());
#endif

	bool failed_write_threw = false;
	try {
		sus2::AtomicWriteCheckpoint(checkpoint, [](const std::string& path) {
			WriteText(path, "partial-bad-trial\n");
			throw std::runtime_error("simulated interrupted save");
		});
	} catch (const std::runtime_error&) {
		failed_write_threw = true;
	}
	Require(failed_write_threw, "failed checkpoint writer did not propagate");
	Require(ReadAll(checkpoint) == "accepted-step-1304\n",
	        "failed checkpoint write damaged previous destination");
	Require(!sus2::CheckpointFileExists(temporary),
	        "failed checkpoint write left a temporary file");

	bool empty_write_threw = false;
	try {
		sus2::AtomicWriteCheckpoint(checkpoint, [](const std::string&) {});
	} catch (const std::runtime_error&) {
		empty_write_threw = true;
	}
	Require(empty_write_threw, "empty checkpoint writer did not fail");
	Require(ReadAll(checkpoint) == "accepted-step-1304\n",
	        "empty checkpoint write damaged previous destination");

	const std::string first_archive =
		sus2::ArchiveExistingCheckpoint(checkpoint, ".previous");
	Require(first_archive == archive, "first archive path mismatch");
	Require(ReadAll(archive) == "accepted-step-1304\n",
	        "archived checkpoint content mismatch");
	Require(ReadAll(checkpoint) == "accepted-step-1304\n",
	        "archive operation removed the live best checkpoint");

	WriteText(checkpoint, "next-run-best\n");
	const std::string second_archive =
		sus2::ArchiveExistingCheckpoint(checkpoint, ".previous");
	Require(second_archive == archive_1, "archive disambiguation mismatch");
	Require(ReadAll(archive_1) == "next-run-best\n",
	        "second archived checkpoint content mismatch");

	std::remove(checkpoint.c_str());
	std::remove(temporary.c_str());
	std::remove(archive.c_str());
	std::remove(archive_1.c_str());
	std::remove(hardlink_path.c_str());
	std::remove(symlink_path.c_str());
	std::cout << "optimizer_checkpoint_check passed\n";
	return 0;
}
