"""Offline ownership controls; real 174-event trace replay uses a fake idle query."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from kokkos_gpu_memory_test import fixture
import kokkos_gpu_ownership as own


class Ownership(unittest.TestCase):
    def setUp(self):
        self.gpu,_=fixture();self.path='/campaign/new-attempt/runinputs/lmp-old'
        self.saved=[];self.calls=[]

    def raw(self,name=None,pid=123):
        text='' if name is None else f'{self.gpu["uuid"]}, {pid}, {name}, 794\n'
        return dict(returncode=0,timed_out=False,stdout=text,stderr='')

    def event(self,name=None,pid=123,phase='running'):
        return dict(phase=phase,process_query=self.raw(name,pid))

    def memory(self):
        return dict(status='OBSERVED',identity_errors=0,events=[self.event(phase='baseline'),
            self.event(self.path),self.event('[No data]'),self.event(phase='final')])

    def idle(self,texts=None):
        texts=texts or [self.raw()]
        def query(command,timeout):
            self.calls.append((command,timeout));return texts[min(len(self.calls)-1,len(texts)-1)]
        return own.postrun_idle(self.gpu,lambda v:self.saved.append(copy.deepcopy(v)),query=query,sleep=lambda t:None)

    def verify(self,m=None,result=None,post=None):
        return own.verify_ownership(m or self.memory(),self.path,self.gpu,
            result or dict(returncode=0,timed_out=False),post or self.idle())

    def test_prior_bound_trailing_literal_and_post_idle_pass_without_rewriting(self):
        m=self.memory();before=copy.deepcopy(m);proof=self.verify(m)
        self.assertEqual(m,before);self.assertEqual(proof['status'],'VERIFIED')
        self.assertEqual(proof['trailing_name_unavailable'][0]['raw_process_name'],'[No data]')

    def test_exact_named_process_only_pass(self):
        m=self.memory();m['events'].pop(2);self.assertEqual(self.verify(m)['status'],'VERIFIED')

    def test_unknown_new_pid_rejected(self):
        m=self.memory();m['events'][2]=self.event('[No data]',999)
        with self.assertRaises(RuntimeError):self.verify(m)

    def test_same_basename_wrong_fullpath_rejected(self):
        m=self.memory();m['events'][1]=self.event('/foreign/runinputs/lmp-old')
        with self.assertRaises(RuntimeError):self.verify(m)

    def test_unknown_middle_then_any_named_process_rejected(self):
        for pid in (123,999):
            m=self.memory();m['events'].insert(3,self.event(self.path,pid))
            with self.assertRaises(RuntimeError):self.verify(m)

    def test_other_unknown_literals_rejected(self):
        for name in ('N/A','unknown','[Not Found]',''):
            m=self.memory();m['events'][2]=self.event(name)
            with self.assertRaises(RuntimeError):self.verify(m)

    def test_baseline_busy_and_baseline_queryerror_rejected(self):
        for raw in (self.raw(self.path),dict(returncode=1,timed_out=False,stdout='',stderr='permission denied')):
            m=self.memory();m['events'][0]['process_query']=raw
            with self.assertRaises(RuntimeError):self.verify(m)

    def test_running_queryerror_unknown_not_zero_rejected(self):
        m=self.memory();m['events'][2]['process_query'].update(returncode=1,stderr='permission denied')
        with self.assertRaises(RuntimeError):self.verify(m)

    def test_foreign_uuid_rejected(self):
        m=self.memory();m['events'][1]['process_query']['stdout']=m['events'][1]['process_query']['stdout'].replace(self.gpu['uuid'],'GPU-7216f1a0-2a8a-7dd2-5431-429f8efbe771')
        with self.assertRaises(RuntimeError):self.verify(m)

    def test_failed_or_timedout_process_rejected(self):
        for r in (dict(returncode=1,timed_out=False),dict(returncode=0,timed_out=True)):
            with self.assertRaises(RuntimeError):self.verify(result=r)

    def test_postrun_busy_retries_five_then_rejects(self):
        with self.assertRaises(RuntimeError):self.idle([self.raw('[No data]')])
        self.assertEqual(len(self.calls),5);self.assertEqual(self.saved[-1]['status'],'NOT_IDLE')

    def test_postrun_queryerror_immediate_unknown(self):
        raw=dict(returncode=1,timed_out=False,stdout='',stderr='denied')
        with self.assertRaises(RuntimeError):self.idle([raw])
        self.assertEqual(len(self.calls),1);self.assertEqual(self.saved[-1]['status'],'UNKNOWN')
        self.assertIsNone(self.saved[-1]['memory_used_MiB'])

    def test_postrun_bound_unknown_then_empty_pass_and_foreign_rejected(self):
        self.verify(post=self.idle([self.raw('[No data]'),self.raw()]))
        self.calls=[]
        with self.assertRaises(RuntimeError):self.verify(post=self.idle([self.raw('[No data]',999),self.raw()]))

    def test_independent_postrun_proof_required_not_sampler_final(self):
        with self.assertRaises(RuntimeError):self.verify(post=dict(status='IDLE',assigned_uuid=self.gpu['uuid'],attempts=[]))

    def test_actual174_samples_regression_replay(self):
        source=Path(os.environ.get('SUS2_OWNERSHIP_ACTUAL_FIXTURE',str(Path(__file__).with_name('ownership-actual174.json'))))
        self.assertTrue(source.is_file(),'Packager must supply actual174 trace for the regression test')
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         'bf89b97d58c54a7ae2da3fb28f8304751b3461204dc6861927a57a83d984862f')
        report=json.loads(source.read_text());m=report['planned'][0]['memory'];self.gpu=report['allocated_gpu']
        self.path=next(p['process_name'] for e in m['events'] for p in e['processes']['processes']
                       if p['process_name'].endswith('/lmp-old'))
        before=copy.deepcopy(m);proof=self.verify(m)
        self.assertEqual(len(m['events']),174);self.assertEqual(m,before)
        self.assertEqual(len(proof['trailing_name_unavailable']),1)
        self.assertEqual(proof['trailing_name_unavailable'][0]['pid'],1973134)


if __name__=='__main__':unittest.main()
