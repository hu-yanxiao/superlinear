"""Conservative full-path/UUID/PID ownership proof after a completed launch.

Only a trailing literal '[No data]' name may be a teardown observation: that
same PID/UUID must previously have the exact used executable path, the launch
must finish successfully, no named process may reappear, and an independent
post-run UUID query must establish idle. Raw sampler observations are unedited.
"""
import time
from kokkos_gpu_memory import parse_apps_query
from kokkos_scheduler_readiness import scheduler_query


def require(ok,message):
    if not ok:raise RuntimeError(message)


def postrun_idle(gpu,save,query=scheduler_query,sleep=time.sleep):
    command=['nvidia-smi','-i',gpu['uuid'],'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits']
    result=dict(status='UNKNOWN',assigned_uuid=gpu['uuid'],max_attempts=5,interval_s=.25,
                memory_used_MiB=None,attempts=[])
    save(result)
    for attempt in range(1,6):
        raw=query(command,timeout=3.)
        parsed=parse_apps_query(raw,gpu)
        result['attempts'].append(dict(attempt=attempt,raw=raw,processes=parsed))
        if parsed['status']=='EMPTY':
            result.update(status='IDLE',successful_attempt=attempt);save(result);return result
        if parsed['status'] not in ('OK','PARTIAL_UNKNOWN'):
            result.update(status='UNKNOWN',error='Post-run UUID idle query failed or was unidentifiable')
            save(result);raise RuntimeError(result['error'])
        save(result)
        if attempt<5:sleep(.25)
    result.update(status='NOT_IDLE',error='Post-run UUID query still has processes after five reads')
    save(result);raise RuntimeError(result['error'])


def verify_ownership(memory,binary,gpu,process_result,postrun):
    expected=str(binary)
    require(expected.startswith('/'),'Exact absolute used executable path required')
    require(process_result.get('returncode')==0 and not process_result.get('timed_out'),
            'Teardown ownership requires actual successful process return')
    require(memory.get('identity_errors')==0 and memory.get('status') not in ('IDENTITY_ERROR','INCOMPLETE'),
            'Sampler device identity or completion failed')
    require(postrun.get('status')=='IDLE' and postrun.get('assigned_uuid')==gpu['uuid'] and postrun.get('attempts'),
            'Independent post-run UUID idle proof required')
    bound={};teardown=[];trailing=False
    records=[(e['phase'],i,e['process_query']) for i,e in enumerate(memory['events'])]
    records += [('postrun',i,e['raw']) for i,e in enumerate(postrun['attempts'])]
    for phase,index,raw in records:
        parsed=parse_apps_query(raw,gpu)
        require(parsed['status'] in ('EMPTY','OK','PARTIAL_UNKNOWN'),
                'Unknown/error/foreign-UUID process query cannot prove ownership')
        processes=parsed['processes']
        if phase=='baseline':
            require(parsed['status']=='EMPTY' and not processes,'Baseline has active or unknown compute processes')
        for p in processes:
            key=(p['gpu_uuid'],p['pid']);name=p['process_name']
            require(key[0]==gpu['uuid'],'Foreign compute-process UUID')
            if name==expected:
                require(not trailing,'Named process after unknown-name teardown: ownership cannot be proved')
                bound[key]=name
            else:
                require(name=='[No data]' and key in bound,'Foreign path or new/unsupported unknown-name PID')
                trailing=True
                teardown.append(dict(phase=phase,event_index=index,uuid=key[0],pid=key[1],
                                     raw_process_name=name,previously_observed_full_path=bound[key]))
    last=postrun['attempts'][-1]
    require(parse_apps_query(last['raw'],gpu)['status']=='EMPTY','Post-run final raw query is not empty')
    require(bound,'No exact executable/UUID/PID binding was actually observed')
    return dict(status='VERIFIED',expected_full_path=expected,assigned_uuid=gpu['uuid'],
                observed_bindings=[dict(uuid=u,pid=p,full_path=v) for (u,p),v in sorted(bound.items())],
                trailing_name_unavailable=teardown,raw_observations_modified=False,
                postrun_idle=True,explanation='Only prior-bound trailing literal [No data] accepted after successful return and independent UUID idle proof; unknown is not zero memory')
