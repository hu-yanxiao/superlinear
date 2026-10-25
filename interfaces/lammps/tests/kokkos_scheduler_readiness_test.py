"""Offline current-job publication-race and UUID idle-proof controls."""
import copy
import unittest
from kokkos_gpu_identity import lsf_gpu_id
import kokkos_scheduler_readiness as ready


class Readiness(unittest.TestCase):
    # Field layout/assignment from actual successful profile job4350254.
    bare='Job <4350254>, Status <RUN>\nGPU REQUIREMENT DETAILS:\n Effective: num=1/host:mode=exclusive_process:mps=no:j_exclusive=yes\nGPU_ALLOCATION:\n a05u22g 0 5 - - NVIDIAA100_S 80G\n 1 5 - - NVIDIAA100_S 80G\n'
    published=('Job <4350254>, Status <RUN>\nEXTERNAL MESSAGES:\n'
        ' 0 phy-weigw Oct24 a05u22g:gpus=5; N\nRESOURCE REQUIREMENT DETAILS:\n'+bare.split('\n',1)[1])

    def run_sequence(self,texts,rc=0,delay=0,fail=False):
        self.calls=[];self.saved=[];self.now=0.
        hardware=dict(job_id='4350254',node='a05u22g')
        def query(command,timeout):
            self.calls.append((command,timeout));self.now+=min(delay,timeout)
            return dict(command=command,returncode=rc,stdout=texts[min(len(self.calls)-1,len(texts)-1)],
                        stderr='rawstderr',timed_out=False)
        def sleep(seconds):self.now+=seconds
        def run():return ready.scheduler_readiness(hardware,lsf_gpu_id,
            lambda value:self.saved.append(copy.deepcopy(value)),query=query,clock=lambda:self.now,sleep=sleep)
        if fail:
            with self.assertRaises(RuntimeError):run()
            self.assertEqual(self.saved[-1]['status'],'FAIL')
            self.assertNotIn('gpu_scheduler_evidence',hardware)
            return hardware
        return hardware,run()

    def test_bare_then_complete_exact_raw_snapshot(self):
        hw,r=self.run_sequence([self.bare,self.published])
        self.assertEqual(r['successful_attempt'],2)
        self.assertEqual(hw['gpu_scheduler_evidence']['stdout'],self.published)
        self.assertEqual(self.saved[-1]['attempts'][0]['raw']['stdout'],self.bare)
        self.assertEqual(self.calls[0][0],['bjobs','-gpu','-l','4350254'])
        self.assertEqual(r['host_gpu_id'],5)

    def test_complete_once(self):
        _,r=self.run_sequence([self.published]);self.assertEqual(len(r['attempts']),1)

    def test_actual_race_table_without_external_exhausts(self):
        self.run_sequence([self.bare],fail=True)
        self.assertEqual((len(self.calls),self.now),(10,9.))

    def test_bare_job_only_exhausts(self):
        self.run_sequence(['Job <4350254>, Status <RUN>\n'],fail=True)
        self.assertEqual(len(self.calls),10)

    def test_query_error_exhausts(self):
        self.run_sequence(['unavailable'],rc=1,fail=True)
        self.assertEqual(len(self.calls),10)

    def test_deadline_and_each_query_timeout_bounded(self):
        self.run_sequence([''],rc=1,delay=3,fail=True)
        self.assertLessEqual(self.now,30.)
        self.assertLessEqual(len(self.calls),10)
        self.assertTrue(all(0<t<=3. for _,t in self.calls))

    def test_conflicts_fail_immediately_never_retry(self):
        for text in [self.published.replace('gpus=5;','gpus=5; a05u22g:gpus=6;'),
                     self.published.replace('gpus=5;','gpus=5,6;'),
                     self.published.replace('a05u22g:gpus','other:gpus'),
                     self.published.replace('1 5 -','1 6 -'),
                     self.published.replace('gpus=5;','gpus=6;'),
                     self.published.replace('Job <4350254>','Job <9>')]:
            self.run_sequence([text],fail=True);self.assertEqual(len(self.calls),1)

    def test_strict_parser_error_not_retried(self):
        self.run_sequence([self.published.replace('num=1/host','num=2/host')],fail=True)
        self.assertEqual(len(self.calls),1)


class Idle(unittest.TestCase):
    gpu={'uuid':'GPU-275a7a6e-3e02-c2e9-93ae-c0efa3f6e161'}

    def probe(self,stdout='',code=0,timed_out=False,fail=False):
        self.saved=[];self.commands=[]
        def query(command,timeout):
            self.commands.append((command,timeout))
            return dict(command=command,returncode=code,stdout=stdout,stderr='rawstderr',timed_out=timed_out)
        def run():return ready.idle_gpu(self.gpu,lambda v:self.saved.append(copy.deepcopy(v)),query=query)
        if fail:
            with self.assertRaises(Exception):run()
            return self.saved[-1]
        return run()

    def test_successful_empty_query_is_idle_not_zero_memory(self):
        r=self.probe();self.assertEqual(r['status'],'IDLE')
        self.assertEqual(r['observed_process_rows'],[])
        self.assertIsNone(r['memory_used_MiB'])
        self.assertEqual(self.commands[0][0][1:3],['-i',self.gpu['uuid']])

    def test_query_failure_or_timeout_preserves_unknown(self):
        for code,timeout in [(1,False),(None,True)]:
            r=self.probe(code=code,timed_out=timeout,fail=True)
            self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['observed_process_rows'])
            self.assertEqual(r['raw']['stderr'],'rawstderr')

    def test_active_process_rejected(self):
        r=self.probe(self.gpu['uuid']+', 1234, lmp, 99\n',fail=True)
        self.assertEqual(r['status'],'BUSY_OR_UNIDENTIFIABLE')

    def test_na_or_zero_is_not_idle(self):
        for text in ('N/A\n','0\n','permission denied\n'):
            self.assertEqual(self.probe(text,fail=True)['status'],'BUSY_OR_UNIDENTIFIABLE')

    def test_malformed_csv_fails_unknown(self):
        self.assertEqual(self.probe('"unterminated',fail=True)['status'],'UNKNOWN')

    def test_no_device_zero_fallback(self):
        for gpu in ({'index':0},{'uuid':'0'},{'uuid':'GPU-abcd'}):
            with self.assertRaises(RuntimeError):ready.idle_gpu(gpu,lambda v:None)


if __name__=='__main__':unittest.main()
