#!/usr/bin/env python3
"""Contract tests for scientific matching, parser failures and backend proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import kokkos_benchmark as bench
import kokkos_qualification as qual
import kokkos_complete_qualification as complete
import kokkos_timing_matrix as timing
import kokkos_gpu_identity as identity
import kokkos_oracle_binding as oracle


class HarnessTest(unittest.TestCase):
    def test_unset_cvd_requires_lsf_single_gpu_nvml_and_runtime_match(self):
        scheduler = ('EXTERNAL MESSAGES:\n node:gpus=6;\nRESOURCE REQUIREMENT DETAILS:\n'
                     'GPU REQUIREMENT DETAILS:\n Effective: num=1/host:mode=exclusive_process\n'
                     'GPU_ALLOCATION:\n HOST TASK GPU_ID MODEL\n node 0 6 A100\n 1 6 A100\n')
        def probe(cmd):
            return dict(returncode=0, stdout='6, GPU-a, 00000000:85:00.0, A100, 81920\n' if cmd[0]=='nvidia-smi' else scheduler)
        with patch.dict('os.environ', {'LSB_JOBID':'7'}, clear=True), patch.object(identity,'probe',side_effect=probe):
            gpu=identity.assigned_gpu(dict(node='node'))
        self.assertEqual(gpu['index'],6)
        self.assertEqual(gpu['binding_method'],'LSF_SINGLE_NAMESPACE_UUID')
        marker='SUS2_KOKKOS_DEVICE device=0 visible_devices=1 uuid=GPU-a name=A100 memory_bytes=85000000000'
        self.assertEqual(identity.runtime_device(marker,gpu)[0]['device'],0)
        with self.assertRaisesRegex(RuntimeError,'UUID/name'):
            identity.runtime_device(marker.replace('GPU-a','GPU-b'),gpu)
        with self.assertRaisesRegex(RuntimeError,'exactly one'):
            identity.runtime_device(marker.replace('visible_devices=1','visible_devices=2'),gpu)
        with patch.dict('os.environ', {'LSB_JOBID':'7'}, clear=True), patch.object(identity,'probe',return_value=dict(returncode=0,stdout='0, GPU-a, 00000000:85:00.0, A100, 81920\n6, GPU-b, 00000000:86:00.0, A100, 81920\n')):
            with self.assertRaisesRegex(RuntimeError,'exactly one NVML-visible'):
                identity.assigned_gpu(dict(node='node'))
        def remapped_index(cmd):
            return dict(returncode=0, stdout='0, GPU-a, 00000000:85:00.0, A100, 81920\n' if cmd[0]=='nvidia-smi' else scheduler)
        for cvd in ('','6','0','GPU-a'):
            with patch.dict('os.environ', {'LSB_JOBID':'7','CUDA_VISIBLE_DEVICES':cvd}, clear=True), patch.object(identity,'probe',side_effect=remapped_index):
                remapped=identity.assigned_gpu(dict(node='node'))
                self.assertEqual((remapped['lsf_host_gpu_id'],remapped['nvml_visible_index']),(6,0))
                self.assertTrue(identity.runtime_device(marker,remapped)[0]['identity_verified'])
        with patch.dict('os.environ', {'LSB_JOBID':'7','CUDA_VISIBLE_DEVICES':'GPU-b'}, clear=True), patch.object(identity,'probe',side_effect=remapped_index):
            with self.assertRaisesRegex(RuntimeError,'CUDA UUID disagrees'):
                identity.assigned_gpu(dict(node='node'))
        bad_uuid=scheduler+' GPU-b\n'
        def wrong_host_uuid(cmd):
            return dict(returncode=0,stdout='0, GPU-a, 00000000:85:00.0, A100, 81920\n' if cmd[0]=='nvidia-smi' else bad_uuid)
        with patch.dict('os.environ', {'LSB_JOBID':'7'}, clear=True), patch.object(identity,'probe',side_effect=wrong_host_uuid):
            with self.assertRaisesRegex(RuntimeError,'LSF UUID disagrees'):
                identity.assigned_gpu(dict(node='node'))

    def test_native_hash_guard_is_not_removed_for_portability(self):
        with patch.object(oracle,'sha',return_value='0'*64):
            with self.assertRaisesRegex(RuntimeError,'Unrecognized native'):
                oracle.validate_reference(Path('/not-an-accepted-reference'))
        with patch.object(oracle,'sha',return_value=oracle.PORTABLE):
            with self.assertRaisesRegex(RuntimeError,'exact raw-audit'):
                oracle.validate_reference(Path('/portable-without-proof'))

    def test_portable_reference_rejects_audit_label_or_regenerated57(self):
        with patch.object(oracle,'sha',side_effect=[oracle.PORTABLE,'0'*64]):
            with self.assertRaisesRegex(RuntimeError,'independent raw-evidence audit changed'):
                oracle.validate_reference(Path('/portable'),Path('/audit'),Path('/matrix'))
        with patch.object(oracle,'sha',side_effect=[oracle.PORTABLE,oracle.AUDIT,oracle.BRIDGE,oracle.BUILD,oracle.AUDITOR,'0'*64]):
            with self.assertRaisesRegex(RuntimeError,'do not regenerate'):
                oracle.validate_reference(Path('/portable'),Path('/audit'),Path('/matrix'))

    def test_core_extension_without_radial_is_waiting_not_gpu_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            with patch.object(complete, 'verify_core', return_value=dict(binary_sha256='a'*64, native_sha256='b'*64)), \
                 patch.object(complete, 'verify_extension', return_value=dict(status='PASS')), \
                 patch('sys.argv', ['finalize', '--core-report', 'core.json', '--extension-report', 'extra.json', '--out', str(path)]):
                self.assertEqual(complete.main(), 0)
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'WAITING_RADIAL')
            self.assertFalse(result['gpu_qualified'])
            self.assertEqual(result['radial_gate']['status'], 'NOT_RUN')

    def test_radial_host_pass_cannot_satisfy_final_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'radial.json'
            path.write_text(json.dumps(dict(schema='sus2-radial-device-qualification-v1', status='PASS',
                backend=dict(kokkos_version=40701, cuda=False, execution_space='Serial', kernel_cuda_arch=0))))
            with self.assertRaisesRegex(RuntimeError, 'actually execute CUDA'):
                complete.verify_radial(path, None, None, 'a'*64)

    def test_gpu_timing_preserves_all66_windows(self):
        windows = timing.planned_windows(bench.manifest())
        self.assertEqual(len(windows), 66)
        self.assertEqual(len({(w['case'], w['radial_mode'], w['repeat']) for w in windows}), 66)
        self.assertTrue(all(w['status'] == 'NOT_RUN' for w in windows))

    def test_gpu_identity_and_cross_gpu_work_isolation(self):
        result = dict(returncode=0, stdout='1, GPU-b, 00000000:85:00.0, A100, 81920\n')
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-b'}), patch.object(identity, 'probe', return_value=result):
            gpu = timing.assigned_gpu()
        self.assertEqual(gpu['uuid'], 'GPU-b')
        self.assertEqual(gpu['memory_total_MiB'], 81920)
        with patch.object(bench, 'probe', return_value=dict(returncode=0, stdout='GPU-a, 100, lmp, 500\n')):
            self.assertEqual(timing.require_idle_gpu(gpu)['competing_processes'], [])
        with patch.object(bench, 'probe', return_value=dict(returncode=0, stdout='GPU-b, 100, lmp, 500\n')), patch.object(timing.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'already has compute processes'):
                timing.require_idle_gpu(gpu)

    def test_full_gate_rejects_missing_or_duplicate_execution_slots(self):
        expected = [(7, 'direct', 1, 'run0'), (7, 'table', 1, 'run0')]
        record = dict(case=7, mode='direct', ranks=1, operation='run0')
        with self.assertRaisesRegex(RuntimeError, 'Missing or extra'):
            complete.exact_slots([record], expected)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            complete.exact_slots([record, record], expected)

    def test_full_gate_does_not_trust_cuda_pass_label(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'case').mkdir()
            (directory / 'case' / 'screen.log').write_text('SUS2_KOKKOS_BACKEND Serial\nLoop time of 1\n')
            record = dict(name='case', status='PASS', device_backend='CUDA', native_comparison='PASS', operation='run0')
            with self.assertRaisesRegex(RuntimeError, 'actual CUDA backend'):
                complete.verify_run(record, directory, 1e-9)

    def test_complete_timing_matrix_and_matched_inputs(self):
        plan = bench.manifest()
        self.assertEqual(len(plan['cases']), 11)
        self.assertEqual(sum(c['family'] == 'fecn6' for c in plan['cases']), 3)
        for case in plan['cases']:
            cpu = bench.input_text(plan, case, 'direct', 'cpu120')
            gpu = bench.input_text(plan, case, 'direct', 'cuda1')
            self.assertEqual(cpu.replace('sus2/embedded ', 'sus2/embedded/kk '), gpu)
            self.assertIn('fix integrate all nve', cpu)
            self.assertNotIn(' npt ', cpu)
        self.assertEqual([c['atoms'] for c in plan['cases'][:4]], [1000, 9600, 101568, 1000000])

    def test_no_pending_or_missing_result_becomes_zero_or_pass(self):
        result = bench.summarize(bench.manifest(), [])
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertTrue(all(r['speedup'] is None and r['status'] == 'NOT_RUN' for r in result['rows']))

    def test_loop_dimensions_and_cuda_required(self):
        case = bench.manifest()['cases'][0]
        log = ('Loop time of 1 on 1 procs for 200 steps with 1000 atoms\n'
               'Loop time of 2 on 1 procs for 2000 steps with 1000 atoms\nBENCHMARK_COMPLETE\n')
        self.assertEqual(bench.parse_log(log, case, 1)['atom_steps_per_s'], 1000000)
        with self.assertRaisesRegex(RuntimeError, 'CUDA backend'):
            bench.parse_log(log + 'SUS2_KOKKOS_BACKEND Serial\n', case, 1, True)
        with self.assertRaisesRegex(RuntimeError, 'dimensions'):
            bench.parse_log(log.replace('2000 steps', '1999 steps'), case, 1)
        self.assertEqual(bench.parse_log(log + 'SUS2_KOKKOS_BACKEND CUDA\n', case, 1, True)['actual_atoms'], 1000)
        memory_log = log + ('SUS2_KOKKOS_BACKEND CUDA\n'
            'SUS2 Kokkos radial mode=table budget_bytes=536870912\n'
            'SUS2_KOKKOS_RADIAL tables=2 bytes=128 hits=4 memory_bytes=8512\n')
        self.assertEqual(bench.parse_log(memory_log, case, 1, True)['table_memory_observations'],
                         [dict(kind='bytes',bytes=128)])

    def test_reverse_signal_and_force_are_independent_requirements(self):
        with self.assertRaisesRegex(RuntimeError, 'reverse signal'):
            qual.backend_evidence('SUS2_KOKKOS_BACKEND CUDA\nSUS2_KOKKOS_COMM signal_reverse=0 force_exports=2', 'cuda', True)
        with self.assertRaisesRegex(RuntimeError, 'ghost force exports'):
            qual.backend_evidence('SUS2_KOKKOS_BACKEND CUDA\nSUS2_KOKKOS_COMM signal_reverse=2 force_exports=0', 'cuda', True)
        evidence = qual.backend_evidence('SUS2_KOKKOS_BACKEND CUDA\nSUS2_KOKKOS_COMM signal_reverse=2 force_exports=2', 'cuda', True)
        self.assertTrue(evidence['gpu_execution_observed'])
        self.assertTrue(evidence['force_exports_are_not_mpi_counts'])
        self.assertFalse(qual.backend_evidence('SUS2_KOKKOS_BACKEND Serial', 'serial')['gpu_execution_observed'])

    def test_speedup_matches_protocol_and_never_uses_npt(self):
        plan = bench.manifest()
        case = plan['cases'][0]
        reports = []
        for backend, seconds in (('cpu120', 2.0), ('cuda1', 0.5)):
            reports.append(dict(kind='throughput', case=case, radial_mode='direct', backend=backend,
                status='PASS', protocol=plan['protocol'], model_sha256=plan['models']['static'][1],
                binary_sha256=plan['cpu_sha256'] if backend == 'cpu120' else 'a'*64,
                hardware=dict(gpu=dict(returncode=0)), manifest_sha256='f'*64,
                median_loop_seconds=seconds, median_atom_steps_per_s=case['atoms']*case['steps']/seconds,
                runs=[dict(status='PASS', returncode=0, actual_atoms=case['atoms'], iterations=case['steps'],
                           ranks=120 if backend == 'cpu120' else 1, device_backend='CUDA' if backend == 'cuda1' else None)]*3))
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f'{i}.json' for i in range(2)]
            def emit(values):
                for path, value in zip(paths, values):
                    path.write_text(json.dumps(value))
            emit(reports)
            self.assertEqual(bench.summarize(plan, paths)['rows'][0]['speedup'], 4.0)
            altered = copy.deepcopy(reports)
            altered[1]['protocol'] = dict(plan['protocol'], velocity_seed=999)
            emit(altered)
            with self.assertRaisesRegex(RuntimeError, 'protocol'):
                bench.summarize(plan, paths)
            altered = copy.deepcopy(reports)
            altered[1]['kind'] = 'npt-validation'
            emit(altered)
            self.assertIsNone(bench.summarize(plan, paths)['rows'][0]['speedup'])
            altered = copy.deepcopy(reports)
            for row in altered[1]['runs']:
                row['device_backend'] = 'Serial'
            emit(altered)
            with self.assertRaisesRegex(RuntimeError, 'CUDA/device'):
                bench.summarize(plan, paths)


if __name__ == '__main__':
    unittest.main()
