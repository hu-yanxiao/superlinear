#!/usr/bin/env python3
"""Pure-W E/F/stress and mixed parameter-derivative regression, no training data needed.

Uses the production CLI and a fixed asymmetric, two-species fixture. All gate,
matrix and readout blocks are nonzero. Logs/fixtures are retained for diagnosis.
"""
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'bin/mlp-sus2'
(ROOT / '.local_tmp').mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix='pure-w-physics-', dir=ROOT / '.local_tmp'))
CFG = ROOT / 'native_tests/pure_w_regularization.cfg'
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
DEFAULT_OUTER_RANK = int(os.environ.get('PURE_W_TEST_OUTER_RANK', '3'))


def run(label, *args, env=None):
    result = subprocess.run([str(BIN), *map(str, args)], cwd=ROOT,
                            env=dict(ENV, **(env or {})), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (TMP / (label + '.log')).write_text(result.stdout)
    if result.returncode:
        raise AssertionError(f'{label}: exit {result.returncode}\n{result.stdout[-6000:]}')
    return result.stdout


def count(src, key):
    return int(re.search(rf'(?m)^\s*{key}\s*=\s*(\d+)\s*$', src)[1])


def values(src, key):
    return list(map(float, re.findall(NUM, re.search(
        rf'(?m)^\s*{key}\s*=\s*\{{([^}}]*)\}}', src)[1])))


def vector(src, key, data):
    replacement = key + ' = {' + ', '.join(f'{v:.17e}' for v in data) + '}'
    out, n = re.subn(rf'(?m)^\s*{key}\s*=\s*\{{[^}}]*\}}', replacement, src)
    assert n == 1, (key, n)
    return out


def model(mapping, width=DEFAULT_OUTER_RANK, k_width=2, radial_size=3, decoupled=False):
    tag = f'{mapping}-k{k_width}-o{width}-r{radial_size}-d{int(decoupled)}'
    path = TMP / f'{tag}.sus2'
    run(tag+'-init', 'init-sh', path, '--species-count=2',
        '--l-max=2', f'--k-max={k_width}', '--body-order=4', '--body-l-max=2,2,2,2',
        f'--radial-basis-size={radial_size}', '--cutoff=5', f'--scaling-map={mapping}',
        '--radial-basis-type=' + os.environ.get('PURE_W_RADIAL_BASIS', 'RBChebyshev_sss'),
        '--superlinear-mode=pure-w-channelwise', f'--outer-k-rank={width}',
        '--two-layer-gate-mode=mu-scalar-full', '--two-layer-gate-body-order=4',
        f'--two-layer-gate-k-rank={min(2,k_width)}', '--write-sh-scalar-info',
        *(['--two-layer-gate-decoupled-scaling'] if decoupled else []))
    seed = TMP / f'{tag}.seed.sus2'
    run(tag+'-complete', 'roundtrip-model-dev', path, seed)
    path.write_text(seed.read_text())
    src = path.read_text()
    src = re.sub(r'(?m)^scaling = .*$', 'scaling = 1.0', src)
    # Nonparallel W rows and nonzero gate features prevent vacuous FD passes.
    for key, amplitude, bias in [
        ('two_layer_gate_radial_mixing_coeffs', 0.7, 0.1),
        ('two_layer_gate_radial_coeffs', 0.3, 0.05),
        ('two_layer_gate_weights', 0.16, 0.0),
        ('two_layer_gate_k_rank_weights', 0.4, 0.1),
        ('two_layer_gate_additive_coeffs', 0.13, 0.7),
        ('two_layer_gate_type_coeffs', 0.1, 1.0),
    ]:
        src = vector(src, key, [bias + amplitude * math.cos(0.71 * (i + 1))
                              for i in range(len(values(src, key)))])
    scales = values(src, 'scal_coeffs')
    src = vector(src, 'scal_coeffs', [
        (2.6 if i % 8 < 4 else 1.85) + 0.025 * i for i in range(len(scales))])
    if decoupled:
        gate_scales = values(src, 'two_layer_gate_scaling_coeffs')
        src = vector(src, 'two_layer_gate_scaling_coeffs', [
            (2.25 if i % 8 < 4 else 1.55) + 0.017*i for i in range(len(gate_scales))])
    src = vector(src, 'species_coeffs', [0.12, -0.07])
    src = vector(src, 'moment_coeffs', [0.001 * math.cos(0.51 * (i + 1))
                                       for i in range(count(src, 'alpha_scalar_moments'))])
    path.write_text(src)
    return path


def predict(label, path, env=None):
    out = TMP / (label + '.cfg')
    run(label, 'calc-efs', path, CFG, out, env=env)
    src = out.read_text()
    energy = float(re.search(r'(?m)^\s*Energy\s*\n\s*(' + NUM + ')', src)[1])
    stress = list(map(float, re.search(r'(?m)^\s*PlusStress:[^\n]*\n([^\n]*)', src)[1].split()))
    atoms = re.search(r'(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy', src)[1].strip().splitlines()
    force = [float(x) for row in atoms for x in row.split()[-3:]]
    assert len(force) == 12 and len(stress) == 6
    return [energy, *force, *stress]


def equal(a, b, label, atol=2e-10):
    error = max(abs(x - y) for x, y in zip(a, b))
    assert error <= atol, (label, error)
    print(f'PASS {label}: max_abs={error:.3e}', flush=True)


GRAD = ['--max-configs=1', '--radial-smooth=0', '--gate-x-l2=0',
        '--gate-scalar-l2=0', '--gate-mix-l2=0', '--gate-full-l2=0', '--scalar-head-l2=0',
        '--displacement=1e-5', '--abs-tolerance=3e-8', '--rel-tolerance=3e-5']
CACHE_OFF = {'SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE': '1', 'SUS2_SH_SITE_DER_CACHE': '0',
             'SUS2_SH_PRODUCT_ROWS': '0', 'SUS2_SH_PRODUCT_HVT_REVERSE': '0'}
CACHE_ON = {'SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE': '0', 'SUS2_SH_SITE_DER_CACHE': '1',
            'SUS2_SH_PRODUCT_ROWS': '0', 'SUS2_SH_PRODUCT_HVT_REVERSE': '1'}

print(f'Evidence: {TMP}', flush=True)
for mapping in ('K', 'L', 'LK'):
    path = model(mapping)
    src = path.read_text()
    reference = predict(mapping + '-cache', path, CACHE_ON)
    equal(reference, predict(mapping + '-recompute', path, CACHE_OFF), mapping + ' cache/recompute')
    assert max(map(abs, reference[1:13])) > 1e-5, 'fixture force is too small'
    run(mapping + '-efs-fd', 'check-efs-fd-dev', path, CFG, '--max-configs=1',
        '--displacement=2e-5', '--abs-tolerance=3e-7', '--rel-tolerance=3e-4')
    for loss, weights, env in [
        ('energy', (1, 0, 0), CACHE_OFF), ('force', (0, 1, 0), CACHE_ON),
        ('stress', (0, 0, 1), CACHE_OFF), ('efs-rows', (1, 0.3, 0.1),
                                           dict(CACHE_ON, SUS2_SH_PRODUCT_ROWS='1')),
    ]:
        output = run(mapping + '-' + loss, 'check-loss-gradient-dev', path, CFG, *GRAD,
                     f'--energy-weight={weights[0]}', f'--force-weight={weights[1]}',
                     f'--stress-weight={weights[2]}', env=env)
        print(f'PASS {mapping} {loss}: {output.strip().splitlines()[-1]}', flush=True)
    # Unused compatibility R is mathematically absent, not an extra residual.
    match = re.search(r'(?ms)(^\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count)', src)
    def unused_row(m):
        row = list(map(float, re.findall(NUM, m[1])))
        return '{' + ', '.join(map(str, [3.7, -2.9, 4.1, *row[3:]])) + '}'
    unused = src[:match.start(2)] + re.sub(r'\{([^}]*)\}', unused_row, match[2]) + src[match.end(2):]
    unused_path = TMP / (mapping + '-unused-r.sus2')
    unused_path.write_text(unused)
    equal(reference, predict(mapping + '-unused-r', unused_path), mapping + ' unused R invariance')
    zero_path = TMP / (mapping + '-zero-w.sus2')
    zero_path.write_text(vector(src, 'two_layer_gate_radial_mixing_coeffs',
                                [0.0] * len(values(src, 'two_layer_gate_radial_mixing_coeffs'))))
    zero = predict(mapping + '-zero-w', zero_path, CACHE_ON)
    equal(zero[1:], [0.0] * 18, mapping + ' zero W force/stress', atol=1e-13)
    equal(zero, predict(mapping + '-zero-w-recompute', zero_path, CACHE_OFF), mapping + ' zero W cache')
    run(mapping + '-zero-w-gradient', 'check-loss-gradient-dev', zero_path, CFG, *GRAD,
        '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1', env=CACHE_ON)
    roundtrip = TMP / (mapping + '-roundtrip.sus2')
    run(mapping + '-roundtrip', 'roundtrip-model-dev', path, roundtrip)
    equal(reference, predict(mapping + '-roundtrip-efs', roundtrip), mapping + ' save/load')
    if mapping != 'LK':
        # Same input scale shared over l or k is exactly the corresponding LK expansion.
        compact = values(src, 'scal_coeffs')
        expanded = [x for k in range(2) for l in range(3)
                    for x in compact[(k if mapping == 'K' else l)*8:][:8]]
        expanded_src = src.replace(f'scaling_map = {mapping}', 'scaling_map = LK')
        expanded_src = vector(expanded_src, 'scal_coeffs', expanded)
        expanded_path = TMP / (mapping + '-expanded-lk.sus2')
        expanded_path.write_text(expanded_src)
        equal(reference, predict(mapping + '-expanded-lk', expanded_path), mapping + ' shared-scaling LK expansion')
# For O=1 the channelwise and cross-channel angular graphs coincide. Set
# legacy R exactly zero: the old residual implementation then computes the
# same physical W, providing an independent check of removing the baseline.
one = model('LK', width=1)
src = one.read_text()
legacy = re.sub(r'(?m)^outer_(?:channel|radial)_mode = .*\n', '', src)
legacy = legacy.replace('outer_k_rank1_weight_count = 0', 'outer_k_rank1_weight_count = 6')
legacy = vector(legacy, 'outer_k_rank1_weights', [1.0] * 6)
match = re.search(r'(?ms)(^\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count)', legacy)
def zero_legacy_row(m):
    row = list(map(float, re.findall(NUM, m[1])))
    return '{' + ', '.join(map(str, [0.0, 0.0, 0.0, *row[3:]])) + '}'
legacy = legacy[:match.start(2)] + re.sub(r'\{([^}]*)\}', zero_legacy_row, match[2]) + legacy[match.end(2):]
legacy_path = TMP / 'o1-legacy-zero-base.sus2'
legacy_path.write_text(legacy)
one_ref = predict('o1-pure', one)
equal(one_ref, predict('o1-legacy', legacy_path), 'O1 pure vs legacy zero-baseline')
run('o1-efs-gradient', 'check-loss-gradient-dev', one, CFG, *GRAD,
    '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1')
# R > K*O exercises the pure contraction scratch-size boundary. O=3 also
# separates gate scaling from the W input scaling, while keeping gate, W,
# type factors and the head nonzero so mixed-force derivatives cannot pass
# merely because their physical chain is absent.
for mapping in ('K', 'L', 'LK'):
    for outer in (1, 3):
        tag = f'{mapping}-k1-o{outer}-r4'
        path = model(mapping, width=outer, k_width=1, radial_size=4,
                     decoupled=(outer == 3))
        reference = predict(tag+'-cache', path, CACHE_ON)
        equal(reference, predict(tag+'-off', path, CACHE_OFF), tag+' cache/recompute')
        assert max(map(abs, reference[1:13])) > 1e-6, 'K1 fixture force is too small'
        src = path.read_text()
        ablated = TMP/(tag+'-zero-gate.sus2')
        ablated.write_text(vector(src, 'two_layer_gate_weights',
                                  [0.0]*len(values(src, 'two_layer_gate_weights'))))
        zero_gate = predict(tag+'-zero-gate', ablated, CACHE_ON)
        gate_effect = max(abs(a-b) for a,b in zip(reference[1:13], zero_gate[1:13]))
        assert gate_effect > 1e-7, ('K1 gate force signal too small', tag, gate_effect)
        for mode, env in [('cache', CACHE_ON), ('recompute', CACHE_OFF)]:
            out = run(tag+'-gradient-'+mode, 'check-loss-gradient-dev', path, CFG, *GRAD,
                      '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1', env=env)
            print(f'PASS {tag} {mode}: {out.strip().splitlines()[-1]}', flush=True)
        run(tag+'-efs-fd', 'check-efs-fd-dev', path, CFG, '--max-configs=1',
            '--displacement=2e-5', '--abs-tolerance=3e-7', '--rel-tolerance=3e-4')
if os.environ.get('OLD_BIN'):
    BIN = Path(os.environ['OLD_BIN']).resolve()
    equal(one_ref, predict('o1-old-binary', legacy_path), 'O1 pure vs original binary')
print('PASS pure-W physics regression', flush=True)
