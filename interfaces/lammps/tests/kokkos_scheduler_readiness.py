"""Bounded current-job GPU readiness; no scheduler mutations or device guesses.

scheduler_query/readiness are extracted unchanged from frozen profile r3
(profile.py SHA f1a5428bdd1584aaee0944fff7d58de8a822fb7b85d6cfbaef4cf4ddec504bcc).
The caller supplies the original strict GPU parser and saves every raw attempt.
"""
import csv
import io
import re
import subprocess
import time


def require(ok,message):
    if not ok:raise RuntimeError(message)


def scheduler_query(command,timeout=10):
    record=dict(command=command,timeout_s=timeout,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,timed_out=False)
    except subprocess.TimeoutExpired as error:
        decode=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else (x or '')
        record.update(returncode=None,stdout=decode(error.stdout),stderr=decode(error.stderr),
            error=str(error),timed_out=True)
    except OSError as error:
        record.update(returncode=None,stdout='',stderr=str(error),timed_out=False)
    return record


def scheduler_readiness(hardware,parser,save,query=scheduler_query,clock=time.monotonic,sleep=time.sleep):
    """Wait only for asynchronously published current-job GPU evidence.

    The original strict parser sees the exact successful raw query. No ordinal
    inference, external-message synthesis, relaxed parsing or identity bypass.
    """
    job=str(hardware['job_id']);node=hardware['node'].split('.')[0]
    result=dict(status='WAITING',job_id=job,node=node,max_attempts=10,interval_s=1,total_deadline_s=30,attempts=[])
    deadline=clock()+30;save(result)
    for attempt in range(1,11):
        remaining=deadline-clock()
        if remaining<=0:break
        raw=query(['bjobs','-gpu','-l',job],timeout=min(3.,remaining))
        record=dict(attempt=attempt,raw=raw,status='WAITING');result['attempts'].append(record)
        try:
            if raw.get('returncode')!=0 or raw.get('timed_out'):
                record['reason']='Scheduler query unavailable'
            else:
                text=raw.get('stdout','')
                reported=re.search(r'\bJob\s+<(\d+)>',text)
                require(not reported or reported[1]==job,'Scheduler returned a different job identity')
                external=text.split('EXTERNAL MESSAGES:')[-1].split('RESOURCE REQUIREMENT DETAILS:')[0]
                hosts=re.findall(r'([A-Za-z0-9_.-]+):gpus=',external) if 'EXTERNAL MESSAGES:' in text else []
                require(not hosts or {h.split('.')[0] for h in hosts}=={node},'External GPU assignment conflicts with allocated host')
                assigned=re.findall(re.escape(node)+r':gpus=([^;\s]+)',external) if 'EXTERNAL MESSAGES:' in text else []
                require(len(set(assigned))<=1,'Conflicting external GPU assignments')
                require(not assigned or re.fullmatch(r'\d+',assigned[0]),'Multiple/malformed external GPU assignment')
                ids=[]
                if 'GPU_ALLOCATION:' in text:
                    for line in text.split('GPU_ALLOCATION:',1)[1].splitlines():
                        cols=line.split()
                        if len(cols)>=3 and cols[0]==node and cols[1].isdigit() and cols[2].isdigit():ids.append(int(cols[2]))
                        elif len(cols)>=2 and cols[0].isdigit() and cols[1].isdigit():ids.append(int(cols[1]))
                require(len(set(ids))<=1,'Multiple GPU_ID values in allocation table')
                require(not assigned or not ids or int(assigned[0])==ids[0],'External assignment conflicts with GPU_ID table')
                if not reported or not assigned or not ids or 'GPU REQUIREMENT DETAILS:' not in text:
                    record['reason']='Current job external assignment/table/requirements not yet complete'
                else:
                    host_id=parser(text,hardware['node'])
                    hardware['gpu_scheduler_evidence']=raw
                    record.update(status='READY',host_gpu_id=host_id)
                    result.update(status='READY',host_gpu_id=host_id,successful_attempt=attempt)
                    save(result);return result
        except Exception as error:
            record.update(status='FAIL',error=str(error));result.update(status='FAIL',error=str(error));save(result);raise
        save(result)
        if attempt<10:
            remaining=deadline-clock()
            if remaining>0:sleep(min(1.,remaining))
    result.update(status='FAIL',error='Scheduler GPU evidence not ready within ten reads/thirty seconds')
    save(result);raise RuntimeError(result['error'])


def idle_gpu(gpu,save,query=scheduler_query):
    """Independent UUID-scoped prelaunch absence proof. Unknown is not idle/zero.

    Successful empty NVML output establishes no visible compute processes at
    this instant. Failed/timeout/malformed/nonempty output fails closed and is
    retained verbatim; this is not a memory-usage measurement.
    """
    uuid=gpu.get('uuid','')
    require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid),
            'Idle proof requires the already verified full assigned GPU UUID')
    command=['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits']
    raw=query(command,timeout=3.)
    result=dict(status='UNKNOWN',assigned_uuid=uuid,raw=raw,observed_process_rows=None,
                memory_used_MiB=None,scope='Instantaneous prelaunch compute-process absence, not a zero-memory claim')
    try:
        require(raw.get('returncode')==0 and not raw.get('timed_out'),
                'Cannot verify absence of competing GPU processes')
        rows=list(csv.reader(io.StringIO(raw.get('stdout','')),skipinitialspace=True,strict=True))
        result['observed_process_rows']=rows
        if rows:
            result['status']='BUSY_OR_UNIDENTIFIABLE'
            raise RuntimeError('Assigned GPU has active or unidentifiable compute processes')
        result['status']='IDLE'
    except Exception as error:
        result['error']=str(error);save(result);raise
    save(result);return result
