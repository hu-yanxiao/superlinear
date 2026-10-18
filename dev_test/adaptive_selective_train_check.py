#!/usr/bin/env python3
"""Actual BFGS updates with an adaptive coefficient allowlist (serial/MPI).

Usage: python adaptive_selective_train_check.py BIN [--mpi-command 'mpiexec -n 2']
Reuses the existing asymmetric E/F/stress fixture, not a production checkpoint.
"""
import json
import re
import numpy as np
import adaptive_pytorch_profile_check as p

GROUP_FIELDS = {
    'shift': 'shift_coeffs',
    'radial': 'radial_coeffs',
    'readout': 'moment_coeffs',
    'adaptive_amplitude': 'adaptive_radial_amplitudes',
    'adaptive_weight': 'adaptive_radial_weights',
}
FROZEN_FIELDS = ('scal_coeffs', 'species_coeffs', 'two_layer_gate_radial_coeffs',
                 'two_layer_gate_scal_coeffs', 'two_layer_gate_type_coeffs',
                 'two_layer_gate_k_rank_weights')


def signature(path):
    text = path.read_text()
    result = {'radial_coeffs': np.asarray(p.radial_rows(text)).ravel()}
    for field in (*GROUP_FIELDS.values(), *FROZEN_FIELDS):
        if field == 'radial_coeffs':
            continue
        values = p.optional_numbers(text, field)
        if values is not None:
            result[field] = np.asarray(values)
    assert set(GROUP_FIELDS.values()) <= set(result), result.keys()
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def compare(before, after, groups):
    a, b = signature(before), signature(after)
    assert a.keys() == b.keys()
    active = {GROUP_FIELDS[g] for g in groups}
    changes = {}
    for field in a:
        assert a[field].shape == b[field].shape
        delta = float(np.max(np.abs(a[field] - b[field]))) if a[field].size else 0.0
        changes[field] = delta
        if field in active:
            assert delta > 1e-13, ('selected group did not update', field, delta)
        else:
            # The runtime checks in-memory frozen values exactly before every
            # objective. Allow decimal export roundoff when comparing files.
            assert np.allclose(a[field], b[field], rtol=2e-14, atol=2e-14), (field, delta)
    return changes


def main():
    print('Selective regression directory:', p.TMP, flush=True)
    p.write_cfg()
    topology = p.make_topology()
    model = p.materialized_asymmetric_model(topology)
    results = {}
    groups_all = list(GROUP_FIELDS)
    for i, groups in enumerate([[g] for g in groups_all] + [groups_all]):
        name = 'selective-' + '-'.join(groups)
        out = p.TMP / (name + '.sus2')
        trace = p.TMP / (name + '.csv')
        log = p.run(name, 'train', model, p.CFG, *p.PROFILE_TRAIN_ARGS,
                    '--init-params=same', '--optimizer=bfgs', '--max-iter=3',
                    '--optimizer-conv-tol=0', '--trainable-groups=' + ','.join(groups),
                    '--trained-pot-name=' + str(out), '--bfgs-trace-file=' + str(trace),
                    '--curr-pot-name=' + str(p.TMP / (name + '.current.sus2')))
        assert 'active_coordinates=' in log
        assert len(trace.read_text().splitlines()) >= 2
        results[name] = compare(model, out, groups)
    if p.MPI_COMMAND:
        out = p.TMP / 'mpi-selected.sus2'
        # Two frames ensure both ranks participate in distributed dense BFGS.
        doubled = p.TMP / 'two.cfg'
        doubled.write_text(p.CFG.read_text() * 2)
        log = p.run('mpi-selected', 'train', model, doubled, *p.PROFILE_TRAIN_ARGS,
                    '--init-params=same', '--max-iter=3', '--optimizer-conv-tol=0',
                    '--trainable-groups=' + ','.join(groups_all),
                    '--curr-pot-name=' + str(p.TMP / 'mpi.current.sus2'),
                    '--trained-pot-name=' + str(out), mpi=True)
        assert 'mode=distributed-dense-bfgs' in log
        results['mpi'] = compare(model, out, groups_all)
        serial = p.TMP / ('selective-' + '-'.join(groups_all) + '.sus2')
        a, b = p.calc_efs('selected-serial-efs', serial), p.calc_efs('selected-mpi-efs', out)
        assert np.allclose(a, b, rtol=2e-7, atol=2e-8), np.max(np.abs(a-b))
        results['mpi_serial_efs_max_abs'] = float(np.max(np.abs(a-b)))
        # More ranks than frames: idle rank excluded without deadlock or drift.
        idle = p.TMP / 'idle-rank.sus2'
        p.run('mpi-idle-rank', 'train', model, p.CFG, *p.PROFILE_TRAIN_ARGS,
              '--init-params=same', '--max-iter=2', '--optimizer-conv-tol=0',
              '--trainable-groups=shift', '--trained-pot-name=' + str(idle), mpi=True)
        results['idle_rank'] = compare(model, idle, ['shift'])
    for i, extra in enumerate([
        ['--trainable-groups='], ['--trainable-groups=bogus'], ['--trainable-groups=shift,shift'],
        ['--trainable-groups=shift,'], ['--trainable-groups=,shift'],
        ['--trainable-groups=shift', '--init-params=random'],
        ['--trainable-groups=shift', '--fine-tune'],
        ['--trainable-groups=shift', '--do-lin'],
    ]):
        log = p.run('rejected-' + str(i), 'train', model, p.CFG,
                    '--max-iter=0', *extra, expect_success=False)
        assert 'trainable-groups' in log
    p.run('incomplete-rejected', 'train', topology, p.CFG, '--max-iter=0',
          '--trainable-groups=shift', expect_success=False)
    # Verify max-iter=0 retains every physical coefficient, including selected.
    same = p.TMP / 'zero.sus2'
    p.run('selected-zero', 'train', model, p.CFG, *p.PROFILE_TRAIN_ARGS,
          '--max-iter=0', '--trainable-groups=' + ','.join(groups_all),
          '--trained-pot-name=' + str(same))
    results['zero_iterations'] = compare(model, same, [])
    report = p.TMP / 'SELECTIVE_TRAIN_PASS.json'
    report.write_text(json.dumps(results, indent=2) + '\n')
    print('SELECTIVE_TRAIN_PASS', report, flush=True)


if __name__ == '__main__':
    main()
