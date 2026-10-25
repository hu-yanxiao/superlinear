#!/usr/bin/env python3
"""CPU-only contract tests: fake query output, no GPU/process launch required."""
import copy
import json
import subprocess
import threading
import unittest
from unittest.mock import patch

import kokkos_gpu_memory as memory

UUID='GPU-275a7a6e-3e02-c2e9-93ae-c0efa3f6e161'
OTHER='GPU-7216f1a0-2a8a-7dd2-5431-429f8efbe771'


def fixture():
    scheduler=('EXTERNAL MESSAGES:\n node:gpus=5;\nRESOURCE REQUIREMENT DETAILS:\n'
               'GPU REQUIREMENT DETAILS:\n Effective: num=1/host:mode=exclusive_process:mps=no:j_exclusive=yes\n'
               'GPU_ALLOCATION:\n HOST TASK GPU_ID MODEL\n node 0 5 A100\n 1 5 A100\n')
    gpu=dict(uuid=UUID,pci_bus_id='00000000:B1:00.0',model='A100',memory_total_MiB=81920,
             binding_method='EXPLICIT_CVD_ORDINAL_NAMESPACE',cuda_visible_devices='0',
             cuda_mps_pipe_directory=None,lsf_host_gpu_id=5,nvml_visible_index=0,
             scheduler_evidence=dict(returncode=0,stdout=scheduler))
    runtime=[dict(identity_verified=True,device=0,visible_devices=1,uuid=UUID,name='A100',memory_bytes=81920*1048576)]
    return gpu,runtime


def result(stdout='',code=0,**kwargs):
    return dict(stdout=stdout,stderr='',returncode=code,timed_out=False,**kwargs)


def device(used=123,total=81920,uuid=UUID,pci='0000:B1:00.0'):
    return result(f'{uuid}, {pci}, {used}, {total}\n')


class MemoryTest(unittest.TestCase):
    def test_remapped_namespace_is_legal_not_host_index_equality(self):
        gpu,runtime=fixture()
        proof=memory.validate_identity(gpu,runtime)
        self.assertEqual((proof['lsf_host_gpu_id'],proof['nvml_visible_index'],proof['cuda_runtime_ordinal']),(5,0,0))

    def test_ordinal_only_or_unverified_runtime_rejected(self):
        for gpu,runtime in [(dict(index=0),[]),(fixture()[0],[])]:
            with self.assertRaises(ValueError): memory.GpuMemorySampler(gpu,runtime)
        gpu,runtime=fixture();runtime[0]['identity_verified']=False
        with self.assertRaises(ValueError): memory.validate_identity(gpu,runtime)

    def test_wrong_runtime_uuid_or_count_rejected(self):
        for key,value in [('uuid',OTHER),('visible_devices',2),('device',1)]:
            gpu,runtime=fixture();runtime[0][key]=value
            with self.assertRaises(ValueError): memory.validate_identity(gpu,runtime)

    def test_mps_shared_or_multiple_assignment_rejected(self):
        for old,new in [('mps=no','mps=yes'),('num=1/host','num=2/host'),
                        ('j_exclusive=yes','j_exclusive=no'),('mode=exclusive_process','mode=shared')]:
            gpu,runtime=fixture();gpu['scheduler_evidence']['stdout']=gpu['scheduler_evidence']['stdout'].replace(old,new)
            with self.assertRaises(ValueError): memory.validate_identity(gpu,runtime)
        gpu,runtime=fixture();gpu['cuda_mps_pipe_directory']='/tmp/mps'
        with self.assertRaises(ValueError): memory.validate_identity(gpu,runtime)

    def test_empty_permission_timeout_na_are_unknown_not_zero(self):
        gpu,_=fixture()
        for query in [result(),result('denied',1),dict(returncode=None,timed_out=True,stdout=''),device('N/A'),device('')]:
            parsed=memory.parse_device_query(query,gpu)
            self.assertEqual(parsed['status'],'UNKNOWN')
            self.assertIsNone(parsed['used_MiB'])
        self.assertEqual(memory.parse_apps_query(result(),gpu)['status'],'EMPTY')
        self.assertIsNone(memory.parse_apps_query(result(),gpu)['sum_used_MiB'])

    def test_explicit_numeric_zero_is_observed_not_missing(self):
        gpu,_=fixture();parsed=memory.parse_device_query(device(0),gpu)
        self.assertEqual((parsed['status'],parsed['used_MiB']),('OK',0))

    def test_device_uuid_pci_or_multiple_rows_rejected(self):
        gpu,_=fixture()
        self.assertEqual(memory.parse_device_query(device(uuid=OTHER),gpu)['status'],'IDENTITY_ERROR')
        self.assertEqual(memory.parse_device_query(device(pci='0000:E1:00.0'),gpu)['status'],'IDENTITY_ERROR')
        self.assertEqual(memory.parse_device_query(result(device()['stdout']*2),gpu)['status'],'UNKNOWN')
        self.assertEqual(memory.parse_device_query(device(90000),gpu)['status'],'UNKNOWN')

    def test_processes_outside_launcher_pgid_are_not_filtered(self):
        gpu,_=fixture()
        query=result(f'{UUID}, 99999, "/path/has,comma/lmp", 1234\n{UUID}, 88888, lmp, 22\n')
        parsed=memory.parse_apps_query(query,gpu)
        self.assertEqual(parsed['sum_used_MiB'],1256)
        self.assertEqual([p['pid'] for p in parsed['processes']],[99999,88888])
        self.assertEqual(parsed['processes'][0]['process_name'],'/path/has,comma/lmp')

    def test_process_na_duplicates_or_other_uuid_not_summed(self):
        gpu,_=fixture()
        query=result(f'{UUID}, 9, lmp, N/A\n')
        parsed=memory.parse_apps_query(query,gpu)
        self.assertEqual(parsed['status'],'PARTIAL_UNKNOWN');self.assertIsNone(parsed['sum_used_MiB'])
        for query in [result(f'{UUID}, 9, lmp, 2\n'*2),result(f'{UUID}, 0, lmp, 2\n'),result('bad'),result('"unterminated')]:
            self.assertEqual(memory.parse_apps_query(query,gpu)['status'],'UNKNOWN')
        self.assertIsNone(memory.parse_apps_query(result(f'{UUID}, 9, lmp, 18446744073709551615\n'),gpu)['sum_used_MiB'])
        self.assertEqual(memory.parse_apps_query(result(f'{OTHER}, 9, lmp, 22\n'),gpu)['status'],'IDENTITY_ERROR')

    def test_baseline_peak_raw_events_and_idempotent_finish(self):
        gpu,runtime=fixture();values=iter([100,1400,900])
        def runner(command,timeout):
            self.assertIn(UUID,command);self.assertNotIn('-n',command)
            if any('--query-gpu=' in x for x in command): return device(next(values))
            return result(f'{UUID}, 99999, lmp, 1200\n')
        sampler=memory.GpuMemorySampler(gpu,runtime,runner=runner)
        sampler.sample_once('baseline');sampler.sample_once('running');report=sampler.finish()
        self.assertEqual((report['baseline_device_used_MiB'],report['sampled_device_peak_used_MiB']),(100,1400))
        self.assertEqual(report['sampled_process_peak_sum_MiB'],1200)
        self.assertEqual(len(report['events']),3)
        self.assertFalse(report['exact_peak']);self.assertFalse(report['pgid_filter_applied'])
        self.assertTrue(report['attribution_verified']);json.dumps(report)
        self.assertEqual(sampler.finish(),report)
        with self.assertRaises(ValueError): sampler.sample_once()

    def test_failed_queries_preserve_raw_stderr_and_unknown_peak(self):
        gpu,runtime=fixture()
        def runner(command,timeout): return dict(returncode=13,stdout='',stderr='permission denied',timed_out=False)
        sampler=memory.GpuMemorySampler(gpu,runtime,runner=runner)
        sampler.sample_once('baseline');report=sampler.finish()
        self.assertEqual(report['status'],'UNKNOWN');self.assertIsNone(report['sampled_device_peak_used_MiB'])
        self.assertIsNone(report['sampled_process_peak_sum_MiB']);self.assertFalse(report['attribution_verified'])
        self.assertEqual(report['events'][0]['device_query']['stderr'],'permission denied')

    def test_identity_error_survives_prior_valid_sample(self):
        gpu,runtime=fixture();queries=iter([device(123),device(uuid=OTHER)])
        def runner(command,timeout):
            return next(queries) if any('--query-gpu=' in x for x in command) else result()
        sampler=memory.GpuMemorySampler(gpu,runtime,runner=runner);sampler.sample_once('running');report=sampler.finish()
        self.assertEqual(report['status'],'IDENTITY_ERROR');self.assertFalse(report['attribution_verified'])
        self.assertEqual(report['sampled_device_peak_used_MiB'],123)

    def test_runner_exception_is_a_raw_unknown_event(self):
        gpu,runtime=fixture()
        def runner(command,timeout): raise RuntimeError('probe exploded')
        sampler=memory.GpuMemorySampler(gpu,runtime,runner=runner);report=sampler.finish()
        self.assertEqual(report['status'],'UNKNOWN')
        self.assertIn('probe exploded',report['events'][0]['device_query']['stderr'])

    def test_background_lifecycle_and_no_candidate_process_control(self):
        gpu,runtime=fixture();observed=threading.Event()
        def runner(command,timeout):
            self.assertEqual(command[:3],['nvidia-smi','-i',UUID]);observed.set()
            return device(222) if any('--query-gpu=' in x for x in command) else result()
        sampler=memory.GpuMemorySampler(gpu,runtime,runner=runner,interval_s=.005).start()
        self.assertTrue(observed.wait(1));report=sampler.finish()
        self.assertFalse(report['sampler_thread_still_running'])
        self.assertEqual(report['status'],'OBSERVED');self.assertEqual(report['process_observation_status'],'UNKNOWN')
        with self.assertRaises(ValueError): sampler.start()

    def test_default_runner_preserves_owned_probe_timeout(self):
        error=subprocess.TimeoutExpired(['nvidia-smi'],3,output=b'partial',stderr=b'timed out')
        with patch.object(memory.subprocess,'run',side_effect=error):
            result=memory.default_runner(['nvidia-smi'],3)
        self.assertTrue(result['timed_out']);self.assertEqual(result['stdout'],'partial')
        self.assertEqual(result['stderr'],'timed out')


if __name__=='__main__':
    unittest.main()
