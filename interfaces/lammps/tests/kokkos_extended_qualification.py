#!/usr/bin/env python3
"""Execute the native-generated supplement before granting full GPU coverage.

All models run Direct/Table x MPI1/2/4. Additional prescribed trajectories
exercise both retained skin neighbors without rebuilding and new neighbors
after rebuilding, and actual non-identity atom sorting. Zero-ML ZBL controls
prove a nonzero CUDA repulsion, including fixed-cutoff extension past ML rc.
Serial/embedded-oracle controls never produce a CUDA PASS. Submits no jobs.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys
import numpy as np
import cpu_qualification as cpu
from kokkos_benchmark import allocation, sha
from kokkos_qualification import backend_evidence
from kokkos_extended_fixtures import assert_metadata, field, specifications

SCHEMA='sus2-kokkos-extension-v1'
CATEGORIES=['body_combinations','static_all_L','odd_R_K_ne_O','skin_md_sort','zbl_nonzero']


def validate_inputs(matrix, directory):
    cpu.require(matrix.get('status')=='FIXTURES_READY_NOT_QUALIFIED', 'Incomplete supplemental fixture generation')
    cases=matrix['cases']
    cpu.require([c['spec'] for c in cases]==list(specifications()), 'Supplement does not match the complete current generator plan')
    cpu.require(len({c['index'] for c in cases})==len(cases), 'Duplicate supplemental case indices')
    actual=[]
    for case in cases:
        path=directory/Path(case['model']).name
        cpu.require(sha(path)==case['sha256'], 'Supplement model changed: '+str(path))
        text=path.read_text()
        assert_metadata(text,case['spec'])
        actual.append(dict(index=case['index'],family=case['spec']['family'],
            l=int(field(text,'sh_l_max')),k=int(field(text,'sh_k_max')),
            o=int(field(text,'outer_k_rank','0')),r=int(field(text,'two_layer_gate_k_rank','0')),
            gate_mode=field(text,'two_layer_gate_mode'),requested_gate_mode=case['spec'].get('gate_mode'),
            projection=field(text,'adaptive_radial_projection'),zbl=field(text,'zbl_enabled','false'),
            inner_graph=field(text,'two_layer_gate_factor_pruning'),outer_graph=field(text,'outer_channel_mode','cross-channel'),
            readout=field(text,'readout_mode','shared'),gate_scaling=field(text,'two_layer_gate_scaling_mode','shared-scaling')))
    for family in ('static','plain'):
        cpu.require({x['l'] for x in actual if x['family']==family}==set(range(5)), 'Missing '+family+' L0..4')
    for family in ('legacy','pure','adaptive'):
        for l in range(1,5):
            rows=[x for x in actual if x['family']==family and x['l']==l]
            cpu.require(any(x['r']%2==1 and x['r']>1 and x['k']!=x['o'] for x in rows), 'Missing odd R/K!=O family/L cell')
            expected={'mu-scalar-full'} if family=='adaptive' else {'mu-scalar-full','mu-body-order','mu-body-linear-combo'}
            cpu.require({x['requested_gate_mode'] for x in rows}>=expected, 'Missing native gate mode or alias case')
    return actual


def snapshot(label, steps, sorting=False):
    return '\n'.join([f'dump observation all custom 1 {label}.dump id type proc x y z fx fy fz c_site',
        'dump_modify observation sort '+('off' if sorting else 'id')+' format float %.17g first yes',
        f'run {steps}', 'print "QUAL_THERMO '+label+' $(pe:%.17g) '+
        ' '.join(f'$(v_w{k}:%.17g)' for k in range(1,7))+' $(etotal:%.17g)"','undump observation'])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('lammps','native','matrix-dir','base-cfg','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backend',choices=('cuda','serial'),default='cuda')
    p.add_argument('--oracle-kind',choices=('native','embedded'),default='native')
    p.add_argument('--oracle-binding-audit',type=Path,
                   help='Independent portable-oracle bridge audit; mandatory for CUDA')
    p.add_argument('--ranks',default='1,2,4')
    p.add_argument('--mpiexec',default='mpirun')
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--fail-fast',action='store_true',help='Preserve failed and NOT_RUN slots and stop at the first failure')
    p.add_argument('--timeout',type=float,default=180)
    args=p.parse_args()
    args.lammps,args.native=args.lammps.resolve(),args.native.resolve()
    cpu.require(args.oracle_kind=='native' or args.backend=='serial', 'Embedded oracle is HOST_ONLY')
    cpu.require(args.backend!='cuda' or args.oracle_binding_audit is not None,
                'CUDA extension requires --oracle-binding-audit')
    reference_binding=None
    if args.oracle_binding_audit is not None:
        cpu.require(args.oracle_kind=='native', 'Portable binding requires the native oracle')
        from kokkos_oracle_binding import validate_reference
        reference_binding=validate_reference(args.native,args.oracle_binding_audit,
                                             args.matrix_dir/'matrix.json')
    if args.backend=='cuda':
        cpu.require(reference_binding.get('mode')=='VALIDATED_PORTABLE',
                    'CUDA extension requires the independently validated portable reference')
    ranks=[int(v) for v in args.ranks.split(',')]
    cpu.validate_ranks(ranks)
    if args.backend=='cuda':
        cpu.require(set(ranks)>={1,2,4}, 'CUDA extension requires MPI1/2/4')
        hardware=allocation('cuda1')
        from kokkos_gpu_identity import assigned_gpu
        gpu=assigned_gpu(hardware)
    else:
        hardware={'control':'HOST_ONLY'}
        gpu=None
    matrix=json.loads((args.matrix_dir/'matrix.json').read_text())
    actual=validate_inputs(matrix,args.matrix_dir)
    cpu.require(args.max_cases>=0,'Negative case limit')
    selected=matrix['cases'][:args.max_cases] if args.max_cases else matrix['cases']
    inherited=argparse.Namespace(lammps=args.lammps,native=args.native,out=args.out,
        pair_style='sus2/embedded/kk',mpiexec=args.mpiexec,timeout=args.timeout,
        direct_tolerance=1e-9,table_tolerance=2e-6,tabstep=.001)

    class Check(cpu.Qualification):
        def oracle(self,model,geometry,directory,label):
            if args.oracle_kind=='native':
                return super().oracle(model,geometry,directory,label)
            cfg=directory/(label+'.cfg')
            cpu.write_cfg(cfg,geometry)
            output=self.command([args.native,model,cfg,'1','direct'],directory,directory/(label+'.embedded.log'))
            return json.loads(next(line.split('NATIVE_EFS_JSON ',1)[1] for line in output.splitlines() if line.startswith('NATIVE_EFS_JSON ')))

        def process(self,argv,cwd,log,timeout):
            command=list(map(str,argv))
            candidate=str(args.lammps) in command
            if candidate:
                rec=self.report['runs'][-1]
                operation=rec['operation']
                script=Path(cwd)/'in.qualification'
                text=script.read_text()
                if operation in ('skin-cache','skin-rebuild','sorting'):
                    if operation=='sorting':
                        text=text.replace('atom_modify map array sort 0 0.0','atom_modify map array sort 1 0.2')
                        text=text.replace('dump_modify observation sort id','dump_modify observation sort off')
                        text+='timestep 0.00001\nfix prescribed all move linear 0.1 0.03 0.02 units box\n'
                        text+=snapshot('final',10,True)+'unfix prescribed\n'
                    else:
                        skin=2.0 if operation=='skin-cache' else .1
                        text=text.replace('neighbor 0.4 bin',f'neighbor {skin} bin')
                        text=text.replace('neigh_modify every 1 delay 0 check no','neigh_modify every 1 delay 0 check yes')
                        text+='group moving id 2\ntimestep 0.0005\nfix prescribed moving move linear -40 0 0 units box\n'
                        text+=snapshot('final',30)+'unfix prescribed\n'
                    script.write_text(text)
                elif operation.startswith('zbl-distance'):
                    # cpu.Qualification treats named exercises as initial/final
                    # pairs. Repeat run0 also checks force clearing in-place.
                    script.write_text(text+snapshot('final',0))
                index=command.index(str(args.lammps))+1
                launch=['-k','on','g','1'] if args.backend=='cuda' else ['-k','on','t','1']
                command[index:index]=launch+['-sf','kk','-pk','kokkos','neigh','half','newton','on','comm','device']
            result=super().process(command,cwd,log,timeout)
            if candidate and result['returncode']==0:
                evidence=backend_evidence((Path(cwd)/'screen.log').read_text(),args.backend,
                                          expected_device=gpu)
                self.report['runs'][-1]['device_evidence']=evidence
                self.report['runs'][-1]['device_backend']='CUDA' if evidence['gpu_execution_observed'] else 'Serial'
            return result

        def run(self,case,model,geometry,count,mode,operation='run0'):
            super().run(case,model,geometry,count,mode,operation)
            rec=self.report['runs'][-1]
            rec['native_comparison']='PASS' if rec['status']=='PASS' else 'FAIL'
            rec['model_sha256']=sha(model)
            rec['actual_model_metadata']=next(item for item in actual if item['index']==case['index'])
            rec['device_log']=str(self.out/rec['name']/'screen.log')
            rec['category']=('skin_md_sort' if operation in ('skin-cache','skin-rebuild','sorting') else
                'zbl_nonzero' if operation.startswith('zbl-distance') else
                'static_all_L' if case['spec']['family'] in ('static','plain') else 'body_combinations')
            if rec['status']=='PASS':
                try:
                    folder=self.out/rec['name']
                    if operation in ('skin-cache','skin-rebuild'):
                        builds=[int(v) for v in re.findall(r'Neighbor list builds =\s*(\d+)',(folder/'screen.log').read_text())]
                        cpu.require(builds,'Missing neighbor build count')
                        cpu.require(builds[-1]==0 if operation=='skin-cache' else builds[-1]>0,'Wrong observed CSR rebuild behavior')
                        initial=cpu.read_dump(folder/'initial.dump'); final=cpu.read_dump(folder/'final.dump')
                        distance=lambda d:float(np.linalg.norm([d[k][1]-d[k][0] for k in ('x','y','z')]))
                        cutoff=float(field(model.read_text(),'max_dist'))
                        cpu.require(distance(initial)>cutoff and distance(final)<cutoff,'Pair did not cross cutoff')
                        f=max(float(np.max(np.abs(final[k]))) for k in ('fx','fy','fz'))
                        cpu.require(f>1e-10,'Vacuous final skin force')
                        rec['skin_evidence']=dict(observed_rebuilds=builds[-1],expected_rebuilds='zero' if operation=='skin-cache' else 'positive',initial_distance_A=distance(initial),
                            final_distance_A=distance(final),cutoff_A=cutoff,final_force_abs=f,steps=30)
                    if operation=='sorting':
                        raw=(folder/'initial.dump').read_text().splitlines()
                        start=next(i for i,line in enumerate(raw) if line.startswith('ITEM: ATOMS '))
                        cols=raw[start].split()[2:]
                        ids=[int(line.split()[cols.index('id')]) for line in raw[start+1:start+1+len(geometry.types)]]
                        cpu.require(ids!=sorted(ids),'No non-identity atom storage order observed')
                        rec['sorting_evidence']=dict(sort_exercised=True,raw_initial_storage_ids=ids,sort_every=1,steps=10,
                            dump=str(folder/'initial.dump'))
                    if operation.startswith('zbl-distance'):
                        atoms=cpu.read_dump(folder/'initial.dump')
                        f=max(float(np.max(np.abs(atoms[k]))) for k in ('fx','fy','fz'))
                        enabled=case['spec']['zbl']
                        # Fixed cutoff extension is active at r=5.4 > ML rc=4.8.
                        active=enabled and (not case['spec']['typewise'] or float(operation.split('-')[-1])==.25)
                        if active:
                            cpu.require(f>1e-4,'ZBL probe is not strongly nonzero')
                        if not enabled:
                            cpu.require(f<1e-12,'Zero-ML disabled-ZBL control has nonzero force')
                        rec['zbl_evidence']=dict(nonzero_trigger=active,observed_max_force_abs=f,
                            zero_ml_control=True,typewise=case['spec']['typewise'],distance_A=float(operation.split('-')[-1]))
                except Exception as error:
                    rec.update(status='FAIL',error=str(error))
            self.save()
            if args.fail_fast and rec['status']!='PASS':
                raise RuntimeError('Extension fail-fast: '+rec['name']+': '+rec.get('error','unknown failure'))

    check=Check(inherited)
    check.report.update(schema=SCHEMA,backend='CUDA' if args.backend=='cuda' else 'Serial',hardware=hardware,allocated_gpu=gpu,
        coverage_categories=[],actual_model_metadata=actual,fixture_matrix_sha256=sha(args.matrix_dir/'matrix.json'),
        oracle_kind=args.oracle_kind,native_reference_binding=reference_binding,
        expected_model_cases=len(matrix['cases']),selected_model_cases=len(selected))
    geometry=cpu.restricted(cpu.read_cfg(args.base_cfg))
    schedule=[]
    for case in selected:
        for mode in ('direct','table'):
            for count in ranks:
                schedule.append((case,geometry,count,mode,'run0'))
    representatives=[next(c for c in selected if c['spec']['family']==f) for f in ('static','legacy','pure','adaptive') if any(c['spec']['family']==f for c in selected)]
    skin=cpu.Geometry(np.eye(3)*12,np.array([[3.,4.,4.],[7.95,4.,4.]]),np.array([0,1]))
    sorting=cpu.Geometry(np.eye(3)*12,np.array([[2.+(7-i)*.55,3.+.11*(i%3),4.] for i in range(8)]),np.arange(8)%2)
    for case in representatives:
        for mode in ('direct','table'):
            for count in ranks:
                for operation,g in (('skin-cache',skin),('skin-rebuild',skin),('sorting',sorting)):
                    schedule.append((case,g,count,mode,operation))
    for case in [c for c in selected if c['spec'].get('zbl_probe')]:
        for distance in (.25,.5,1.4) if case['spec']['typewise'] else (.65,1.4,5.4):
            g=cpu.Geometry(np.eye(3)*20,np.array([[4.,4.,4.],[4.+distance,4.,4.]]),np.array([0,1]))
            for mode in ('direct','table'):
                for count in ranks:
                    schedule.append((case,g,count,mode,'zbl-distance-'+str(distance)))
    slots=[dict(name=f'{case["index"]:04d}-{case["spec"]["family"]}-{operation}-{mode}-mpi{count}',
                case=case['index'],spec=case['spec'],ranks=count,mode=mode,operation=operation)
           for case,g,count,mode,operation in schedule]
    check.report.update(planned_execution_slots=slots,driver_sha256=sha(__file__),base_cfg_sha256=sha(args.base_cfg))
    check.save()
    fatal=None
    try:
        for case,g,count,mode,operation in schedule:
            model=args.matrix_dir/Path(case['model']).name
            cpu.require(sha(model)==case['sha256'],'Model changed before execution: '+str(model))
            check.run(case,model,g,count,mode,operation)
    except Exception as error:
        fatal=str(error)
    finally:
        completed={r['name'] for r in check.report['runs']}
        check.report['not_run']=[dict(slot,status='NOT_RUN') for slot in slots if slot['name'] not in completed]
        if sha(args.lammps)!=check.report['binary_sha256'] or sha(args.native)!=check.report['native_sha256']:
            fatal='Candidate or native binary changed during qualification'
        if fatal:
            check.report.update(status='FAIL',fatal_error=fatal)
        check.save()
    runs=check.report['runs']
    full=len(selected)==len(matrix['cases']) and set(ranks)>={1,2,4}
    passed=all(r['status']=='PASS' and r['native_comparison']=='PASS' for r in runs)
    rebuilds=sum(r.get('skin_evidence',{}).get('observed_rebuilds',0) for r in runs)
    sorted_ok=any(r.get('sorting_evidence',{}).get('sort_exercised') and r['ranks']==1 for r in runs)
    zbl_nonzero=any(r.get('zbl_evidence',{}).get('nonzero_trigger') for r in runs)
    expected=len(selected)*len(ranks)*2+len(representatives)*len(ranks)*2*3+len([c for c in selected if c['spec'].get('zbl_probe')])*3*len(ranks)*2
    passed=passed and not fatal and not check.report['not_run'] and len(runs)==expected and rebuilds>0 and sorted_ok and zbl_nonzero
    status=('PASS' if args.backend=='cuda' and args.oracle_kind=='native' and full else 'HOST_ONLY' if args.backend=='serial' else 'PARTIAL_GPU_PASS') if passed else 'FAIL'
    check.report.update(status=status,coverage_categories=CATEGORIES if passed and full else [],
        nonzero_trigger=zbl_nonzero,observed_rebuilds=rebuilds,sort_exercised=sorted_ok,
        expected_runs=expected,full_extension=full,failures=sum(r['status']!='PASS' for r in runs))
    check.save()
    print(json.dumps(dict(status=status,runs=len(runs),expected_runs=expected,report=str(check.out/'qualification.json'))))
    return 0 if passed else 1


if __name__=='__main__':
    sys.exit(main())
