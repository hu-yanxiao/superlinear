#!/usr/bin/env python3
"""Native effect projection schema, backward compatibility, and fail-closed I/O."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', default='bin/mlp-sus2')
    ap.add_argument('--output', default='.codex_tmp/effect-tanh-schema')
    a = ap.parse_args()
    binary = str(Path(a.binary).resolve())
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    checked = []
    def run(label, *args, fail=False):
        result = subprocess.run([binary, *map(str, args)], text=True, capture_output=True)
        (out / (label + '.log')).write_text(result.stdout + result.stderr)
        assert bool(result.returncode) == fail, (label, result.stdout[-2000:], result.stderr[-2000:])
        checked.append(label)
    options = ['--species-count=2', '--l-max=1', '--k-max=3', '--body-order=3',
        '--radial-basis-size=3', '--cutoff=4', '--outer-k-rank=2', '--two-layer-gate',
        '--two-layer-gate-k-rank=2', '--two-layer-gate-mode=mu-scalar-full',
        '--two-layer-gate-radial-mode=shared-radial']
    old = out / 'linear.sus2'; explicit = out / 'linear-explicit.sus2'
    run('linear-default', 'init-sh', old, *options)
    run('linear-explicit', 'init-sh', explicit, *options, '--effect-gate-projection=linear')
    assert old.read_bytes() == explicit.read_bytes()
    assert 'effect_gate_projection' not in old.read_text()
    for label, extra, version in (
        ('legacy-grouped', ['--outer-channel-block-size=1'], '1.9.0'),
        ('adaptive-shared-o', ['--superlinear-mode=radial-coefficient-gate', '--adaptive-radial-projection=shared-o'], '1.10.0'),
        ('native-tanh-dual', ['--effect-gate-projection=tanh-gate', '--angular-cutoff=3'], '1.11.0'),
    ):
        seed = out / (label + '.sus2'); saved = out / (label + '-saved.sus2')
        run(label + '-init', 'init-sh', seed, *options, *extra)
        run(label + '-save', 'roundtrip-model-dev', seed, saved)
        assert ('version = ' + version) in saved.read_text()
        if label == 'native-tanh-dual':
            repeated = out / (label + '-repeated.sus2')
            run(label + '-repeat', 'init-sh', repeated, *options, *extra)
            assert seed.read_bytes() == repeated.read_bytes()

    for coupling in ('diagonal', 'cross', 'grouped-cross'):
        for readout in ('shared', 'per-element'):
            label = coupling + '-' + readout
            config = out / (label + '.yaml'); seed = out / (label + '.sus2'); saved = out / (label + '-saved.sus2')
            group = coupling == 'grouped-cross'
            group_yaml = '    channel_block_size: 2\n' if group else ''
            config.write_text(f'''schema_version: 1
model:
  kind: coordination-effect
  elements: [H, O]
  l_max: 1
  radial_channels: 4
  neighbor_cutoff: 4
  angular_cutoff: 3
  radial:
    basis_size: 3
  coordination:
    gate_projection: tanh-gate
    channels: 2
    body_order: 3
    channel_coupling: cross
  interaction:
{group_yaml}    channels: {4 if group else 2}
    body_order: 3
    channel_coupling: {'cross' if group else coupling}
  readout:
    element_mode: {readout}
output:
  model: {seed}
''')
            run(label + '-init', 'init-sh', config)
            run(label + '-roundtrip', 'roundtrip-model-dev', seed, saved)
            tx = saved.read_text()
            assert 'version = 1.11.0' in tx and 'effect_gate_projection = tanh-gate' in tx
            integer = lambda key: int(re.search(r'(?m)^\s*' + key + r'\s*=\s*(\d+)', tx)[1])
            K = integer('sh_k_max'); Q = integer('two_layer_gate_scalar_count'); D = integer('radial_funcs_count')
            assert float(re.search(r'(?m)^\s*angular_max_dist\s*=\s*(\S+)', tx)[1]) == 3.0
            if group:
                assert 'outer_channel_mode = grouped-cross' in tx
                assert integer('outer_channel_block_size') == 2
            count = integer('two_layer_gate_weight_count')
            assert count == K * Q + D * K, (K, Q, D, count)
            vals = [float(x) for x in re.search(r'(?m)^two_layer_gate_weights\s*=\s*\{([^}]*)\}', tx)[1].split(',')]
            assert len(vals) == count and all(math.isfinite(x) and x != 0 for x in vals)
            def rank(flat, rows, cols):
                matrix = [flat[r*cols:(r+1)*cols] for r in range(rows)]
                pivots = 0
                for col in range(cols):
                    pivot = max(range(pivots, rows), key=lambda r: abs(matrix[r][col]), default=None)
                    if pivot is None or abs(matrix[pivot][col]) < 1e-11:
                        continue
                    matrix[pivots], matrix[pivot] = matrix[pivot], matrix[pivots]
                    divisor = matrix[pivots][col]
                    matrix[pivots] = [v/divisor for v in matrix[pivots]]
                    for row in range(pivots+1, rows):
                        factor = matrix[row][col]
                        matrix[row] = [x-factor*y for x,y in zip(matrix[row], matrix[pivots])]
                    pivots += 1
                    if pivots == rows: break
                return pivots
            assert K >= 4 and rank(vals[:K*Q], K, Q) == min(K, Q)
            assert rank(vals[K*Q:], D, K) == min(D, K)

            for bad_label, malformed in (
                ('old-version', tx.replace('version = 1.11.0', 'version = 1.8.0')),
                ('missing-projection', re.sub(r'(?m)^effect_gate_projection.*\n', '', tx)),
                ('duplicate-projection', tx + '\neffect_gate_projection = tanh-gate\n'),
                ('unknown-projection', tx.replace('effect_gate_projection = tanh-gate', 'effect_gate_projection = sigmoid')),
                ('missing-Q', re.sub(r'(?m)^two_layer_gate_scalar_count.*\n', '', tx)),
                ('wrong-count', re.sub(r'(?m)^(two_layer_gate_weight_count\s*=\s*)\d+', lambda m: m[1]+str(count+1), tx)),
                ('extra-weight', re.sub(r'(?m)^(two_layer_gate_weights\s*=\s*\{)', r'\g<1>0.1, ', tx)),
            ):
                path = out / (label + '-' + bad_label + '.sus2'); path.write_text(malformed)
                run(label + '-' + bad_label, 'roundtrip-model-dev', path, out / 'invalid-out.sus2', fail=True)
    (out / 'report.json').write_text(json.dumps({'checks': checked, 'status': 'PASS'}, indent=2) + '\n')
    print(f'PASS {len(checked)} native schema checks; artifacts: {out}')


if __name__ == '__main__':
    main()
