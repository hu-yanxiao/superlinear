#!/usr/bin/env python3
"""Execute extracted native metadata guards; optionally check the full binary.

The default test builds a small host-only harness from the actual reader scan,
nonrank parser and value checks. It does not compile or qualify an MTP runtime.
--binary additionally tests init-sh/roundtrip-model-dev in a temporary folder.
No supplied model or frozen qualification fixture is edited.
"""
import argparse
import itertools
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('readout_mode', 'outer_channel_mode', 'outer_radial_mode')
NATIVE = KEYS
PT = ('outer_channel_mode', 'outer_radial_mode', 'readout_mode')
VALUES = {'readout_mode': 'species', 'outer_channel_mode': 'cross-channel',
          'outer_radial_mode': 'legacy-plus-mixer'}


def block(text, start):
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    # These extracted metadata blocks contain no string/comment braces.
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def harness(source):
    start = source.index('const std::streampos metadata_start = ifs.tellg();')
    end = source.index('ifs.seekg(metadata_start);', start) + len('ifs.seekg(metadata_start);')
    scan = source[start:end]
    start = source.index('if (legacy_v18 && !OuterKRankEnabled()')
    parser = block(source, start)
    start = source.index('void MLMTPR::ValidateOuterModeConfiguration() const')
    values_start = source.index('if (outer_channel_mode_', start)
    values_end = source.index('const bool adaptive = outer_radial_mode_ == "adaptive-coefficients";', values_start)
    values_end += len('const bool adaptive = outer_radial_mode_ == "adaptive-coefficients";')
    values = source[values_start:values_end]
    unranked = re.search(r'if \(!pure && !adaptive && outer_channel_mode_ == "same-channel" && !OuterKRankEnabled\(\)\)\s*ERROR\([^;]+;', source[start:])
    assert unranked
    legacy = re.search(r'if \(legacy_v18 && \(OuterUsesAdaptiveRadialCoefficients\(\) \|\| OuterUsesPureRadialMatrix\(\) \|\| !readout_mode_seen\)\)\s*ERROR\([^;]+;', source)
    assert legacy
    return r'''#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <stdexcept>
#define ERROR(message) throw std::runtime_error(message)
std::string outer_channel_mode_="cross-channel", outer_radial_mode_="legacy-plus-mixer", readout_mode_="shared";
bool OuterKRankEnabled() { return false; }
bool OuterUsesAdaptiveRadialCoefficients() { return outer_radial_mode_=="adaptive-coefficients"; }
bool OuterUsesPureRadialMatrix() { return outer_radial_mode_=="pure-matrix"; }
std::string TrimCopy(const std::string& value) {
  const auto first=value.find_first_not_of(" \t\r\n");
  return first==std::string::npos ? "" : value.substr(first,value.find_last_not_of(" \t\r\n")-first+1);
}
int main() {
  try {
    std::ostringstream input; input << std::cin.rdbuf(); std::stringstream ifs(input.str());
    const bool legacy_v18=true, adaptive_v15=false, adaptive_v17=false, joint_v16=false;
    const std::string model_file_version="1.8.0"; bool readout_mode_seen=false;
    std::string tmpstr;
''' + scan + '\nifs >> tmpstr;\n' + parser + '\n' + values + '\n' + unranked.group() + '\n' + legacy.group() + r'''
    if (tmpstr!="species_coeffs") ERROR("unexpected metadata after required fields");
    std::cout << readout_mode_ << " " << outer_channel_mode_ << " " << outer_radial_mode_ << "\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
'''


def text(order, values=None):
    values = VALUES if values is None else values
    return ''.join(key + ' = ' + values[key] + '\n' for key in order)


def cases():
    rows = []
    for order in (NATIVE, PT):
        for head in ('species', 'shared'):
            rows.append(('accept-' + '-'.join(order) + '-' + head, text(order, {**VALUES, 'readout_mode': head}), True))
        for key in KEYS:
            rows.append(('duplicate-' + key, text(order) + key + ' = ' + VALUES[key] + '\n', False))
            rows.append(('missing-' + key, text(tuple(value for value in order if value != key)), False))
        for key, value in [('readout_mode', 'bogus'), ('outer_channel_mode', 'bogus'),
                           ('outer_radial_mode', 'bogus'), ('outer_radial_mode', 'pure-matrix'),
                           ('outer_radial_mode', 'adaptive-coefficients'), ('outer_channel_mode', 'same-channel')]:
            rows.append(('invalid-' + key + '-' + value, text(order, {**VALUES, key: value}), False))
    for order in itertools.permutations(KEYS):
        if order not in (NATIVE, PT): rows.append(('wrong-order-' + '-'.join(order), text(order), False))
    return rows


def run_static(folder):
    compiler = shutil.which('c++')
    assert compiler, 'A host C++ compiler is required for the extracted-reader test'
    cpp = folder / 'metadata.cpp'; cpp.write_text(harness((ROOT / 'dev_src/mtpr.cpp').read_text()))
    binary = folder / 'metadata-check'
    subprocess.run([compiler, '-std=c++11', '-O0', str(cpp), '-o', str(binary)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for label, metadata, expected in cases():
        result = subprocess.run([str(binary)], input=metadata + 'species_coeffs\n', text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert (result.returncode == 0) == expected, (label, result.stdout, result.stderr)
    return len(cases())


def run_binary(binary, folder):
    bases = {}
    for head in ('species', 'shared'):
        initial = folder / ('initial-' + head + '.mtp')
        subprocess.run([binary, 'init-sh', str(initial), '--species-count=2', '--l-max=2',
                        '--k-max=3', '--body-order=3', '--readout-mode=' + head, '--write-sh-scalar-info'],
                       check=True, stdout=subprocess.DEVNULL)
        original = initial.read_text()
        assert 'outer_k_rank1_enabled' not in original
        if head == 'species':
            assert re.search(r'^version = 1\.8\.0$', original, re.M)
            for key in KEYS:
                assert len(re.findall(r'^' + key + r'\s*=', original, re.M)) == 1, key
        # Independent synthetic shared-head input keeps its own coefficient
        # count; only the metadata regression format is made explicitly v1.8.
        stripped = re.sub(r'^(?:' + '|'.join(KEYS) + r')\s*=.*\n', '', original, flags=re.M)
        bases[head] = re.sub(r'^version = .*$', 'version = 1.8.0', stripped, flags=re.M)
        assert bases[head].count('species_coeffs =') == 1
    for index, (label, metadata, expected) in enumerate(cases()):
        stripped = bases['shared' if re.search(r'^readout_mode = shared$', metadata, re.M) else 'species']
        path = folder / ('case' + str(index) + '.mtp')
        path.write_text(stripped.replace('species_coeffs =', metadata + 'species_coeffs =', 1))
        result = subprocess.run([binary, 'roundtrip-model-dev', str(path), str(folder / ('saved' + str(index) + '.mtp'))],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert (result.returncode == 0) == expected, (label, result.stdout)
    return len(cases())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='legacy-nonrank-metadata-') as temporary:
        folder = Path(temporary)
        count = run_static(folder)
        if args.binary: run_binary(str(args.binary.resolve()), folder)
    print('PASS: ' + str(count) + ' extracted native-reader cases' + (' + full-binary roundtrips' if args.binary else ''))


if __name__ == '__main__': main()
