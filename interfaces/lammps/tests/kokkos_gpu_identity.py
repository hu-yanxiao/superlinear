"""Cross-check LSF GPU assignment, NVML visibility and actual CUDA runtime."""
import csv
import io
import os
import re
from kokkos_benchmark import probe, require


def lsf_gpu_id(text, node):
    # Only actual scheduler sections count, never the echoed submitted script.
    detail = text.split('GPU REQUIREMENT DETAILS:')[-1]
    require('GPU REQUIREMENT DETAILS:' in text and re.search(r'Effective:.*?num=1(?:/host)?(?:\s|:)', detail, re.S),
            'LSF does not explicitly assign exactly one GPU per host')
    # LSF external message is an explicit physical GPU_ID assignment. Some
    # versions additionally print a GPU_ID table; never infer index zero.
    external = text.split('EXTERNAL MESSAGES:')[-1].split('RESOURCE REQUIREMENT DETAILS:')[0]
    assignments = re.findall(re.escape(node.split('.')[0]) + r':gpus=([^;\s]+)', external)
    require(assignments and len(set(assignments)) == 1, 'Missing/ambiguous LSF node GPU_ID assignment')
    ids = re.findall(r'\d+', assignments[0])
    require(len(ids) == 1 and assignments[0] == ids[0], 'LSF assignment is not a single physical GPU_ID')
    require('GPU_ALLOCATION:' in text, 'Missing authoritative bjobs GPU_ID allocation table')
    allocated = []
    for line in text.split('GPU_ALLOCATION:')[-1].splitlines():
        columns = line.split()
        if columns and columns[0] == node.split('.')[0]:
            require(len(columns) >= 3 and columns[1].isdigit() and columns[2].isdigit(), 'Malformed LSF GPU_ID row')
            allocated.append(int(columns[2]))
        elif len(columns) >= 2 and columns[0].isdigit() and columns[1].isdigit():
            allocated.append(int(columns[1]))
    require(allocated and set(allocated) == {int(ids[0])}, 'LSF GPU_ID table and scheduler external assignment disagree')
    return int(ids[0])


def assigned_gpu(hardware=None):
    result = probe(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.total', '--format=csv,noheader,nounits'])
    require(result['returncode'] == 0, 'Cannot inspect assigned GPU')
    rows = [[v.strip() for v in r] for r in csv.reader(io.StringIO(result['stdout']), skipinitialspace=True) if r]
    require(rows and all(len(r) == 5 for r in rows), 'Malformed NVML GPU inventory')
    require(len(rows) == 1, 'Require exactly one NVML-visible GPU; never choose device zero of a whole node')
    row = rows[0]
    scheduler, host_id = None, None
    if hardware is not None:
        require(os.environ.get('LSB_JOBID'), 'Missing LSF allocation identity')
        scheduler = hardware.get('gpu_scheduler_evidence') or probe(['bjobs', '-gpu', '-l', os.environ['LSB_JOBID']])
        require(scheduler['returncode'] == 0, 'Cannot inspect authoritative LSF GPU assignment')
        host_id = lsf_gpu_id(scheduler['stdout'], hardware['node'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible:
        require(',' not in visible and visible not in ('-1', 'NoDevFiles'), 'Require one explicit CUDA_VISIBLE_DEVICES assignment')
        if visible.startswith('GPU-'):
            require(row[1].startswith(visible), 'Explicit CUDA UUID disagrees with sole NVML UUID')
            method = 'EXPLICIT_CVD_UUID'
        else:
            require(visible.isdigit(), 'Explicit CUDA assignment must be one UUID or ordinal')
            # CUDA/MPS ordinals, NVML indices and LSF host IDs are different
            # enumeration scopes. A numeric CVD is recorded, never used as a
            # cross-namespace identity. Actual runtime UUID/count is mandatory.
            method = 'EXPLICIT_CVD_ORDINAL_NAMESPACE'
    else:
        require(hardware is not None and os.environ.get('LSB_JOBID'), 'Unset CVD requires scheduler allocation evidence')
        method = 'LSF_SINGLE_NAMESPACE_UUID'
    # Prefer additional scheduler UUID/PCI evidence when actually exposed.
    # Current cluster tables expose only host GPU_ID, so do not invent a UUID
    # mapping from numeric equality. The runtime must confirm the sole UUID.
    table = scheduler['stdout'].split('GPU_ALLOCATION:')[-1] if scheduler else ''
    host_uuids = set(re.findall(r'GPU-[0-9a-fA-F-]+', table))
    require(not host_uuids or host_uuids == {row[1]}, 'LSF UUID disagrees with sole NVML UUID')
    def pci(value):
        m = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
        require(m, 'Malformed PCI bus ID')
        return tuple(int(v,16) for v in m.groups())
    host_pci = {pci(v) for v in re.findall(r'\b[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b', table)}
    require(not host_pci or host_pci == {pci(row[2])}, 'LSF PCI identity disagrees with sole NVML PCI identity')
    pci(row[2])
    return dict(index=int(row[0]), nvml_visible_index=int(row[0]), lsf_host_gpu_id=host_id,
                uuid=row[1], pci_bus_id=row[2], model=row[3], memory_total_MiB=int(row[4]),
                cuda_visible_devices=visible or None, binding_method=method,
                scheduler_gpu_id=host_id, scheduler_evidence=scheduler,
                scheduler_uuid=next(iter(host_uuids)) if host_uuids else None,
                scheduler_pci_available=bool(host_pci), runtime_confirmation_required=True,
                cuda_mps_pipe_directory=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),
                enumeration_scope='LSF host GPU_ID, NVML visible index and CUDA runtime ordinal are independent; UUID is the device identity')


def runtime_device(text, expected):
    records = [dict(device=int(d), visible_devices=int(c), uuid=u, name=n.strip(), memory_bytes=int(m))
        for d,c,u,n,m in re.findall(r'SUS2_KOKKOS_DEVICE device=(\d+) visible_devices=(\d+) uuid=(GPU-[0-9a-f-]+) name=(.*?) memory_bytes=(\d+)', text)]
    require(records, 'Missing actual CUDA runtime device UUID/count/name marker')
    for r in records:
        require(r['visible_devices'] == 1 and r['device'] == 0, 'CUDA runtime must expose/select exactly one assigned local device')
        require(r['uuid'] == expected['uuid'] and r['name'] == expected['model'], 'CUDA runtime UUID/name does not match assigned NVML device')
        require(0 < r['memory_bytes'] <= expected['memory_total_MiB']*1024*1024, 'CUDA runtime memory does not fit assigned NVML hardware')
        r.update(identity_verified=True, lsf_host_gpu_id=expected.get('lsf_host_gpu_id'),
                 nvml_visible_index=expected.get('nvml_visible_index',expected.get('index')),
                 nvml_pci_bus_id=expected.get('pci_bus_id'),
                 identity_rule='LSF explicit single allocation + sole NVML UUID + CUDA count1/same UUID/name/memory; no index equality assumption')
    return records
