import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from linear_components_make_cases import topology_options, reconstruction_chart

SPEC = importlib.util.spec_from_file_location('linear_check', Path(__file__).with_name('linear_components_benchmark_check.py'))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def dump(path, energy, force, virial, targets, labels=(1, 1, 1)):
    atoms, columns, _ = force.shape
    with path.open('wb') as handle:
        handle.write(b'LCOMP001' + struct.pack('=Q', 1))
        handle.write(struct.pack('=5Q', atoms, columns, *labels))
        for values in (energy, force, virial, targets):
            flat = np.asarray(values, dtype='=f8').reshape(-1)
            handle.write(struct.pack('=Q', flat.size))
            handle.write(flat.tobytes())


class LinearComponentsCheckTests(unittest.TestCase):
    def test_legacy_reconstruction_chart_is_separate_and_changes_only_centers(self):
        text = 'species_coeffs = {0.3, -0.7}\nmoment_coeffs = {0.2, 0.8}\n'
        chart = reconstruction_chart(text, 'shared')
        self.assertIn('species_coeffs = {0.3, -0.7}', text)
        self.assertIn('moment_coeffs = {0.2, 0.8}', chart)
        self.assertEqual(reconstruction_chart(text, 'species'), text)
        self.assertEqual(chart.splitlines()[0], 'species_coeffs = {1.00000000000000000e+00, 1.00000000000000000e+00}')

    def test_fixture_matrix_preserves_model_family_and_production_layout(self):
        for lmax in range(1, 5):
            for kind in ('static', 'effect'):
                for readout in ('shared', 'species'):
                    options = topology_options(lmax, kind, readout)
                    self.assertIn('--superlinear-mode=legacy', options)
                    self.assertIn('--readout-mode=' + readout, options)
                    self.assertEqual('--two-layer-gate' in options, kind == 'effect')
                    self.assertIn('--sh-factor-pruning=' + ('q-total' if kind == 'static' else 'same-k'), options)
                    if lmax == 4:
                        for option in ('--k-max=4', '--outer-k-rank=4', '--radial-basis-size=8',
                                       '--body-l-max=4,4,3,2'):
                            self.assertIn(option, options)
                        if kind == 'effect':
                            self.assertIn('--two-layer-gate-k-rank=3', options)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.e = np.array([2., 3.])
        self.f = np.arange(12., dtype=float).reshape(2, 2, 3) / 10
        self.v = np.arange(18., dtype=float).reshape(2, 9) / 20
        self.y = np.arange(16., dtype=float) / 30

    def test_normal_equations_match_explicit_native_weighted_rows(self):
        path = self.root/'features.dump'
        dump(path, self.e, self.f, self.v, self.y)
        frames = CHECK.read_dump(path)
        matrix, rhs = CHECK.normal_equations(frames)
        expected_m = .5 * np.outer(self.e, self.e)
        expected_r = .5 * self.e * self.y[0]
        for atom in range(2):
            for axis in range(3):
                row = self.f[atom, :, axis]
                expected_m += .1 * np.outer(row, row)
                expected_r += .1 * row * self.y[1 + atom * 3 + axis]
        for tensor in range(9):
            row = self.v[:, tensor]
            expected_m += .005 * np.outer(row, row)
            expected_r += .005 * row * self.y[7 + tensor]
        np.testing.assert_allclose(matrix, expected_m, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(rhs, expected_r, rtol=1e-15, atol=1e-15)

    def test_each_column_checked_despite_checksum_cancellation(self):
        a, b = self.root/'a.dump', self.root/'b.dump'
        dump(a, self.e, self.f, self.v, self.y)
        altered = self.f.copy()
        altered[0, 0, 0] += .02
        altered[0, 1, 0] -= .02
        self.assertEqual(float(altered.sum()), float(self.f.sum()))
        dump(b, self.e, altered, self.v, self.y)
        report = CHECK.compare(a, b)
        self.assertFalse(report['passed'])
        self.assertFalse(report['frames'][0]['force']['passed'])
        self.assertFalse(report['normal_matrix']['passed'])
        self.assertTrue(CHECK.compare(a, a)['passed'])

    def test_missing_labels_skip_objective_rows_without_fabricating_zero_targets(self):
        path = self.root/'energy-only.dump'
        dump(path, self.e, self.f, self.v, self.y, labels=(1, 0, 0))
        matrix, rhs = CHECK.normal_equations(CHECK.read_dump(path))
        np.testing.assert_allclose(matrix, .5 * np.outer(self.e, self.e))
        np.testing.assert_allclose(rhs, .5 * self.e * self.y[0])

    def test_nonfinite_and_truncated_dumps_rejected(self):
        path = self.root/'bad.dump'
        dump(path, [np.nan, 3], self.f, self.v, self.y)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            CHECK.read_dump(path)
        path.write_bytes(b'LCOMP001')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            CHECK.read_dump(path)


if __name__ == '__main__':
    unittest.main()
