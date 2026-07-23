#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/sus2-sh-release3-version.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

cat > "$tmp_dir/banner_check.cpp" <<'CPP'
#include <iostream>
#include "src/mlp/mlp.h"

int main()
{
	PrintSUS2Banner(std::cout, true, 4);
	return 0;
}
CPP

"${CXX:-c++}" -std=c++11 -I"$repo_dir" \
	"$tmp_dir/banner_check.cpp" -o "$tmp_dir/banner_check"
"$tmp_dir/banner_check" > "$tmp_dir/banner.out"

grep -Fxq "SUS2-SH Release 3" "$tmp_dir/banner.out"
grep -Fxq "Branch: release3" "$tmp_dir/banner.out"
grep -Fxq "Release Date: 2026-07-23" "$tmp_dir/banner.out"
grep -Fxq "Mode: MPI (4 ranks)" "$tmp_dir/banner.out"

grep -Fq \
	'const char version_str[] = "SUS2-SH Release 3 (2026-07-23)";' \
	"$repo_dir/src/mlp/mlp_commands.cpp"
grep -Fq \
	'std::cout << "SUS2-SH Release 3 (2026-07-23)"' \
	"$repo_dir/dev_src/mlp/mtpr_train.cpp"

if grep -En \
	'SUS2-MLIP Developer Version|SUS2-MLIP developer version|codex/developer|2026-04-17 developer version' \
	"$repo_dir/src/mlp/mlp.h" \
	"$repo_dir/src/mlp/mlp_commands.cpp" \
	"$repo_dir/dev_src/mlp/mtpr_train.cpp"; then
	echo "stale developer-version text remains" >&2
	exit 1
fi

echo "release3 version banner check passed"
