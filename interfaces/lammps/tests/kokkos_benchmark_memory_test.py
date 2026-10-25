"""Offline matrix-memory/preflight contracts. No MPI, CUDA or NVML execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import kokkos_benchmark as bench
import kokkos_gpu_memory as memory
import kokkos_timing_matrix as matrix
import kokkos_gpu_identity as gpu_identity
import kokkos_scheduler_readiness as readiness

UUID='GPU-275a7a6e-3e02-c2e9-93ae-c0efa3f6e161'


def identity():
    text=('EXTERNAL MESSAGES:\n node:gpus=5;\nRESOURCE REQUIREMENT DETAILS:\n'
          'GPU REQUIREMENT DETAILS:\n Effective: num=1/host:mode=exclusive_process:mps=no:j_exclusive=yes\n'
          'GPU_ALLOCATION:\n HOST TASK GPU_ID MODEL\n node 0 5 A100\n')
    gpu=dict(uuid=UUID,pci_bus_id='00000000:B1:00.0',model='A100',memory_total_MiB=81920,
        binding_method='EXPLICIT_CVD_ORDINAL_NAMESPACE',cuda_mps_pipe_directory=None,
        lsf_host_gpu_id=5,nvml_visible_index=0,scheduler_evidence=dict(returncode=0,stdout=text))
    runtime=[dict(identity_verified=True,device=0,visible_devices=1,uuid=UUID,name='A100',memory_bytes=81920*1048576)]
    return gpu,runtime


def stdout(case):
    return ('SUS2_KOKKOS_BACKEND CUDA\n'+
        f'SUS2_KOKKOS_DEVICE device=0 visible_devices=1 uuid={UUID} name=A100 memory_bytes={81920*1048576}\n'+
        f'Loop time of 0.1 on 1 procs for {case["warmup"]} steps with {case["atoms"]} atoms\n'+
        f'Loop time of 2 on 1 procs for {case["steps"]} steps with {case["atoms"]} atoms\nBENCHMARK_COMPLETE\n')


class NoRss:
    def __init__(self,*args):self.thread=self
    def start(self):pass
    def finish(self):return dict(sampled_group_peak_rss_KiB=99,sampled_group_peak_gpu_MiB=None,exact_peak=False)


class Run(unittest.TestCase):
    def exercise(self,directory,*,binding=True,runtime=True,code=0,timeout=False,query='numeric',backend='cuda1'):
        root=Path(directory);binary=root/'lmp';model=root/'model';binary.write_text('fake executable');model.write_text('frozen model')
        plan=bench.manifest();plan['models']['static']=(str(model),bench.sha(model));plan['protocol']['repeats']=3
        if backend=='cpu120':plan.update(cpu_binary=str(binary),cpu_sha256=bench.sha(binary))
        manifest=root/'manifest.json';manifest.write_text(json.dumps(plan))
        gpu,proof=identity();events=[];samplers=[]
        args=argparse.Namespace(manifest=manifest,case=plan['cases'][0]['id'],radial='direct',
            backend=backend,candidate=str(binary),candidate_sha256=bench.sha(binary),validation_npt=False,
            out=root/'attempt',mpiexec='mpirun',timeout=3)
        if binding:args.expected_gpu=gpu
        if runtime:args.gpu_runtime_evidence=proof
        original=memory.GpuMemorySampler
        class Instrument(original):
            def start(self):
                events.append('memory_start');self.sample_once('baseline');self.sample_once('running');return self
        def factory(g,r):
            def runner(argv,limit):
                if any('--query-gpu=' in s for s in argv):
                    selected='GPU-7216f1a0-2a8a-7dd2-5431-429f8efbe771' if query=='identity_error' else UUID
                    return dict(returncode=0,timed_out=False,stdout=f'{selected}, 0000:B1:00.0, '+
                        ('N/A' if query=='unknown' else '1234')+', 81920\n',stderr='')
                # Deliberately not the launcher/group PID; card observation must retain it.
                return dict(returncode=0,timed_out=False,stdout=f'{UUID}, 88888, /candidate/lmp, N/A\n',stderr='')
            sampler=Instrument(g,r,runner=runner);samplers.append(sampler);return sampler
        case=plan['cases'][0];calls=[]
        class Process:
            pid=777
            def __init__(self,argv,**kwargs):
                self.returncode=None;self.waits=0;calls.append(argv);events.append('popen')
                text=stdout(case)
                if backend=='cpu120':text=text.replace('on 1 procs','on 120 procs')
                kwargs['stdout'].write(text);kwargs['stdout'].flush()
            def wait(self,timeout=None):
                self.waits+=1
                if timeout is not None and self.waits==1 and timeout_flag:raise subprocess.TimeoutExpired('fake MPI',timeout)
                self.returncode=-15 if timeout_flag else code;return self.returncode
        timeout_flag=timeout
        with patch.object(bench,'allocation',return_value={'offline':True}),patch.object(bench,'MemorySampler',NoRss),\
             patch.object(bench.subprocess,'Popen',Process),patch.object(bench.os,'killpg'),\
             patch.object(memory,'GpuMemorySampler',side_effect=factory):
            report=bench.run(args)
        return report,events,calls,samplers

    def test_bound_uuid_starts_before_every_formal_launch_and_ignores_pgid(self):
        with tempfile.TemporaryDirectory() as directory:r,events,calls,samplers=self.exercise(directory)
        self.assertEqual(r['status'],'PASS');self.assertEqual(len(calls),3)
        self.assertEqual(events,['memory_start','popen']*3)
        for row in r['runs']:
            self.assertEqual(row['returncode'],0);self.assertFalse(row['timed_out'])
            m=row['memory'];self.assertIsNone(m['sampled_group_peak_gpu_MiB'])
            self.assertEqual(m['sampled_device_peak_gpu_MiB'],1234)
            self.assertFalse(m['gpu_device_memory']['pgid_filter_applied']);self.assertFalse(m['exact_peak'])
            self.assertIsNone(m['gpu_device_memory']['sampled_process_peak_sum_MiB'])
            self.assertEqual(m['gpu_device_memory']['events'][1]['processes']['processes'][0]['pid'],88888)

    def test_unknown_queries_stay_null(self):
        with tempfile.TemporaryDirectory() as directory:r,_,_,_=self.exercise(directory,query='unknown')
        self.assertEqual(r['status'],'PASS')
        self.assertTrue(all(x['memory']['sampled_device_peak_gpu_MiB'] is None and
            x['memory']['gpu_observation_status']=='UNKNOWN' for x in r['runs']))

    def test_memory_uuid_drift_fails_round_not_a_successful_peak(self):
        with tempfile.TemporaryDirectory() as directory:r,_,calls,_=self.exercise(directory,query='identity_error')
        self.assertEqual(r['status'],'FAIL');self.assertEqual(len(calls),1)
        self.assertEqual(r['runs'][0]['returncode'],0)
        self.assertIsNone(r['runs'][0]['memory']['sampled_device_peak_gpu_MiB'])
        self.assertEqual(r['runs'][0]['memory']['gpu_observation_status'],'IDENTITY_ERROR')

    def test_no_preflight_is_unknown_never_posthoc_runtime_backfill(self):
        for binding,runtime in ((True,False),(False,True),(False,False)):
            with tempfile.TemporaryDirectory() as directory:r,events,_,samplers=self.exercise(directory,binding=binding,runtime=runtime)
            self.assertEqual(r['status'],'PASS');self.assertFalse(samplers)
            self.assertTrue(all(x['memory']['gpu_observation_status']=='UNKNOWN_NO_VERIFIED_PREFLIGHT' for x in r['runs']))
            self.assertNotIn('memory_start',events)

    def test_nonzero_and_timeout_returncodes_survive(self):
        for code,timeout,want in ((7,False,7),(0,True,-15)):
            with tempfile.TemporaryDirectory() as directory:r,_,calls,_=self.exercise(directory,code=code,timeout=timeout)
            self.assertEqual(r['status'],'FAIL');self.assertEqual(len(calls),1)
            self.assertEqual((r['runs'][0]['returncode'],r['runs'][0]['timed_out']),(want,timeout))
            self.assertIsNotNone(r['runs'][0]['memory']['gpu_device_memory'])

    def test_cpu_has_no_new_gpu_sampler_or_preflight(self):
        with tempfile.TemporaryDirectory() as directory:r,events,calls,samplers=self.exercise(directory,backend='cpu120')
        self.assertEqual(r['status'],'PASS');self.assertEqual(len(calls),3);self.assertFalse(samplers)
        self.assertNotIn('memory_start',events)
        self.assertTrue(all(x['memory']['gpu_observation_status']=='NOT_REQUESTED' for x in r['runs']))

    def test_rss_stays_original_group_scope_no_gpu_pid_query(self):
        sampler=bench.MemorySampler(7,True)
        def ps(argv,**kwargs):
            self.assertEqual(argv,['ps','-eo','pid=,pgid=,rss=']);sampler.stop_event.set()
            return '10 7 100\n88888 8 900\n'
        with patch.object(bench.subprocess,'check_output',side_effect=ps):sampler.sample()
        self.assertEqual(sampler.rss_kib,100);self.assertIsNone(sampler.gpu_mib)


class Preflight(unittest.TestCase):
    def test_matrix_readiness_delayed_publication_then_strict_binding_same_raw(self):
        gpu,_=identity()
        published='Job <123>, Status <RUN>\n'+gpu['scheduler_evidence']['stdout']
        bare=published.replace('EXTERNAL MESSAGES:\n node:gpus=5;\nRESOURCE REQUIREMENT DETAILS:\n','')
        queries=[]
        def query(command,timeout):
            queries.append(command)
            return dict(command=command,returncode=0,timed_out=False,stderr='raw stderr',
                stdout=bare if len(queries)==1 else published)
        def ready(hardware,parser,save):
            return readiness.scheduler_readiness(hardware,parser,save,query=query,
                clock=lambda:0.,sleep=lambda _:None)
        def nvml(command):
            self.assertEqual(command[0],'nvidia-smi')  # No third bjobs read allowed.
            self.assertEqual(len(queries),2)
            return dict(returncode=0,stdout=f'0, {UUID}, 00000000:B1:00.0, A100, 81920\n')
        hardware=dict(job_id='123',node='node')
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',
                {'LSB_JOBID':'123','CUDA_VISIBLE_DEVICES':'0'},clear=True),\
             patch.object(matrix,'scheduler_readiness',side_effect=ready),\
             patch.object(gpu_identity,'probe',side_effect=nvml):
            bound=matrix.ready_gpu(hardware,Path(directory))
            raw=json.loads((Path(directory)/'scheduler-readiness.json').read_text())
        self.assertEqual(raw['successful_attempt'],2)
        self.assertEqual(raw['attempts'][0]['raw']['stdout'],bare)
        self.assertEqual(raw['attempts'][1]['raw']['stdout'],published)
        self.assertEqual(bound['scheduler_evidence'],hardware['gpu_scheduler_evidence'])
        self.assertEqual(hardware['scheduler_job'],hardware['gpu_scheduler_evidence'])
        self.assertEqual(bound['scheduler_evidence']['stdout'],published)
        self.assertEqual((bound['lsf_host_gpu_id'],bound['nvml_visible_index']),(5,0))

    def exercise(self,directory,*,bad_mps=False,code=0,timed_out=False):
        root=Path(directory);binary=root/'lmp';model=root/'model';binary.write_text('fake');model.write_text('model')
        plan=bench.manifest();plan['models']['static']=(str(model),bench.sha(model));gpu,_=identity()
        if bad_mps:gpu['scheduler_evidence']['stdout']=gpu['scheduler_evidence']['stdout'].replace('mps=no','mps=yes')
        def process(owner,argv,cwd,log,timeout):
            self.assertEqual(owner.env['OMP_NUM_THREADS'],'1')
            Path(log).write_text(stdout(dict(warmup=0,steps=1,atoms=8)))
            return dict(returncode=code,timed_out=timed_out,elapsed_seconds=.01,process_group_killed_on_timeout=timed_out)
        with patch.object(matrix.cpu.Qualification,'process',side_effect=process),\
             patch.dict('os.environ',{'LSB_JOBID':'123','SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE':'1'}):
            return matrix.runtime_preflight(plan,binary,bench.sha(binary),gpu,root/'preflight','mpirun',3)

    def test_untimed_preflight_complete_receipt_and_validated_exclusive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt=self.exercise(directory);on_disk=json.loads((Path(directory)/'preflight/process.json').read_text())
        self.assertEqual(receipt,on_disk);self.assertFalse(receipt['formal_timing_window'])
        self.assertEqual((receipt['returncode'],receipt['timed_out']),(0,False))
        self.assertEqual(receipt['job_id'],'123');self.assertEqual(receipt['runtime_device'][0]['uuid'],UUID)
        self.assertEqual(receipt['gpu_identity']['uuid'],UUID);self.assertTrue(receipt['log_sha256'])
        self.assertEqual(receipt['binary_sha256_before'],receipt['binary_sha256_after'])
        self.assertEqual(receipt['environment_subset']['SUS2_SL_KOKKOS_FORWARD_RADIAL_TILE'],'1')

    def test_shared_allocation_nonzero_timeout_fail_closed_with_receipt(self):
        for kwargs in ({'bad_mps':True},{'code':7},{'code':-15,'timed_out':True}):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((ValueError,RuntimeError)):self.exercise(directory,**kwargs)
                receipt=json.loads((Path(directory)/'preflight/process.json').read_text())
                self.assertEqual(receipt['status'],'FAIL');self.assertTrue(receipt['log_sha256'])
                self.assertEqual(receipt['returncode'],kwargs.get('code',0))

    def test_formal_scientific_manifest_inputs_and66_windows_unchanged(self):
        # Hashes independently computed from the immutable r11 timing-r3 archive.
        plan=bench.manifest()
        self.assertEqual(hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
            'efbfdc57aee7900eb4da01bcd142d7ac225803638695c66dad39648ca4b3ebce')
        inputs=[bench.input_text(plan,c,m,b,npt) for c in plan['cases']for m in ('direct','table')
                for b in ('cpu120','cuda1')for npt in (False,True)]
        self.assertEqual(hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
            'd275181949a7831abda5610d47d83795935872b66cc5d8a4c4ca032acd4b7bb0')
        self.assertEqual(len(matrix.planned_windows(plan)),66)


if __name__=='__main__':unittest.main()
