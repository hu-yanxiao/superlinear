#!/usr/bin/env python3
"""Pure-W TrainLinear columns must reproduce physical E/F/stress, including gate chains."""
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve()
(ROOT / '.codex_tmp').mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix='pure-w-linear-', dir=ROOT / '.codex_tmp'))
CFG = ROOT / 'codex_tests/pure_w_regularization.cfg'
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           SUS2_SH_PRODUCT_ROWS='0', SUS2_SH_PRODUCT_HVT_REVERSE='0')


def run(label, *args, cache=True):
    env = dict(ENV, SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE='0' if cache else '1')
    result = subprocess.run([str(BIN), *map(str, args)], cwd=ROOT, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (TMP / (label + '.log')).write_text(result.stdout)
    if result.returncode:
        raise AssertionError(f'{label}: exit {result.returncode}\n{result.stdout[-5000:]}')
    return result.stdout


def replace_vector(src, key, values):
    result, count = re.subn(rf'(?m)^\s*{key}\s*=\s*\{{[^}}]*\}}',
                           key + ' = {' + ', '.join(f'{x:.17e}' for x in values) + '}', src)
    assert count == 1, (key, count)
    return result


def vector_count(src, key):
    match = re.search(rf'(?m)^\s*{key}\s*=\s*\{{([^}}]*)\}}', src)
    assert match is not None, key
    return len(match[1].split(','))


def field(text, key):
    return re.findall(rf'\b{key}=([^\s]+)', text)[-1]


def predict(label, model):
    output = TMP / f'{label}.cfg'
    run(label, 'calc-efs', model, CFG, output)
    text = output.read_text()
    atoms = re.search(r'(?ms)^\s*AtomData:[^\n]*\n(.*?)^\s*Energy', text)[1].strip().splitlines()
    forces = [float(value) for row in atoms for value in row.split()[-3:]]
    stress = list(map(float, re.search(r'(?m)^\s*PlusStress:[^\n]*\n([^\n]*)', text)[1].split()))
    assert len(forces) == 12 and len(stress) == 6
    return forces, stress


print(f'Evidence: {TMP}', flush=True)
for mapping in ('K', 'L', 'LK'):
    initial = TMP / f'{mapping}.initial.mtp'
    model = TMP / f'{mapping}.mtp'
    run(mapping + '-init', 'init-sh', initial, '--species-count=2', '--l-max=2',
        '--k-max=2', '--body-order=4', '--body-l-max=2,2,2,2', '--radial-basis-size=3',
        '--cutoff=5', '--scaling=0.3', '--superlinear-mode=pure-w-channelwise',
        '--outer-k-rank=3', '--two-layer-gate-k-rank=2',
        '--two-layer-gate-mode=mu-scalar-full', '--two-layer-gate-body-order=4',
        '--write-sh-scalar-info', f'--scaling-map={mapping}')
    run(mapping + '-complete', 'roundtrip-model-dev', initial, model)
    src = model.read_text()
    for key, amplitude, bias in [
        ('two_layer_gate_radial_mixing_coeffs', 0.3, 0.05),
        ('two_layer_gate_radial_coeffs', 0.25, 0.03),
        ('two_layer_gate_k_rank_weights', 0.4, 0.1),
        ('two_layer_gate_weights', 0.16, 0.0),
        ('two_layer_gate_additive_coeffs', 0.13, 0.7),
        ('moment_coeffs', 0.12, 0.0),
    ]:
        src = replace_vector(src, key, [bias + amplitude * math.cos(.71 * (i + 1))
                                      for i in range(vector_count(src, key))])
    # The established full linear solve uses a shared scalar head and resets
    # species prefactors to one. Its component-readout checker assumes that
    # representation, not arbitrary nonlinear species-prefactor coordinates.
    src = replace_vector(src, 'species_coeffs', [1.0, 1.0])
    src = replace_vector(src, 'scal_coeffs', [
        (2.6 if i % 8 < 4 else 1.85) + 0.02 * i
        for i in range(vector_count(src, 'scal_coeffs'))])
    model.write_text(src)
    force, stress = predict(mapping + '-physical', model)
    assert max(map(abs, force)) > 1e-5 and max(map(abs, stress)) > 1e-5
    zero_gate = TMP / f'{mapping}.zero-gate.mtp'
    zero_gate.write_text(replace_vector(src, 'two_layer_gate_weights',
                                       [0.0] * vector_count(src, 'two_layer_gate_weights')))
    no_gate_force, _ = predict(mapping + '-zero-gate-physical', zero_gate)
    gate_force_change = max(abs(x-y) for x, y in zip(force, no_gate_force))
    assert gate_force_change > 1e-7, (mapping, 'vacuous gate-chain fixture', gate_force_change)
    print(f'PASS {mapping} active force/stress; gate force change={gate_force_change:.3e}', flush=True)
    for cache in (False, True):
        label = mapping + ('-cache' if cache else '-recompute')
        readout = run(label + '-readout', 'check-linear-readout-dev', model, CFG,
                      '--max-atoms=0', '--check-stress', '--abs-tolerance=2e-10',
                      '--rel-tolerance=2e-10', cache=cache)
        assert int(field(readout, 'stress_components')) == 9
        run(label + '-columns-fd', 'check-linear-components-fd-dev', model, CFG,
            '--max-atoms=0', '--displacement=2e-5', '--abs-tolerance=2e-7',
            '--rel-tolerance=2e-5', cache=cache)
        layout = run(label + '-layout', 'check-loss-gradient-dev', model, CFG,
                     '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1',
                     '--radial-smooth=0', '--gate-x-l2=0', '--coeff-start=0', '--coeff-end=1',
                     '--abs-tolerance=2e-7', '--rel-tolerance=2e-5', cache=cache)
        head_begin = int(field(layout, 'linear_begin')) + 2
        run(label + '-head-fd', 'check-loss-gradient-dev', model, CFG,
            '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1',
            '--radial-smooth=0', '--gate-x-l2=0', f'--coeff-start={head_begin}',
            '--displacement=1e-6', '--abs-tolerance=2e-7', '--rel-tolerance=2e-5', cache=cache)
        print(f'PASS {label}: E/F/stress readout, all column spatial FD, head coefficient FD', flush=True)

    zero = TMP / f'{mapping}.zero-w.mtp'
    zero.write_text(replace_vector(src, 'two_layer_gate_radial_mixing_coeffs',
                                  [0.0] * vector_count(src, 'two_layer_gate_radial_mixing_coeffs')))
    run(mapping + '-zero-readout', 'check-linear-readout-dev', zero, CFG,
        '--max-atoms=0', '--check-stress', '--abs-tolerance=1e-13', '--rel-tolerance=1e-13')
    trained = TMP / f'{mapping}.trained.mtp'
    run(mapping + '-train', 'train', model, CFG, '--max-iter=2', '--skip-preinit',
        '--init-params=same', '--radial-parameterization=weight-norm',
        '--do-lin', '--do-lin-freq=1', '--do-lin-steps=2', '--scalar-head-l2=1e-5',
        '--energy-weight=1', '--force-weight=0.3', '--stress-weight=0.1',
        '--radial-smooth=1e-6', '--gate-x-l2=0', f'--trained-pot-name={trained}',
        f'--curr-pot-name={TMP / (mapping + ".current.mtp")}')
    # A nonlinear step may update the species prefactors after the last linear
    # solve. Preserve the real checkpoint; test its learned W/gate/head blocks
    # separately in the unit-prefactor representation used by the linear solve.
    trained_linear = TMP / f'{mapping}.trained-linear-representation.mtp'
    trained_linear.write_text(replace_vector(trained.read_text(), 'species_coeffs', [1.0, 1.0]))
    run(mapping + '-trained-readout', 'check-linear-readout-dev', trained_linear, CFG,
        '--max-atoms=0', '--check-stress', '--abs-tolerance=2e-9', '--rel-tolerance=2e-9')
print('PASS pure-W linear component regression', flush=True)
