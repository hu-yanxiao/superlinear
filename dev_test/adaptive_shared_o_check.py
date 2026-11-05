#!/usr/bin/env python3
"""Native shared-o qualification, to run inside a server LSF allocation.

Retains native analytic versus finite-difference checks and tied-full references.
The native helper writes every command and result into the output directory.
"""
import argparse
from pathlib import Path
import numpy as np
from adaptive_species_readout_check import assignment, numbers, put, put_numbers, asymmetric_shared_text, write_fixture, ZERO_REG
from adaptive_tensor_message_check import TensorCheck, topology_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpp', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    a = p.parse_args()
    check = TensorCheck(a.cpp, a.outdir)
    check.cfg = check.out/'asymmetric.cfg'
    write_fixture(check.cfg)
    for activation, amplitude, joint, outer, rank in [
        ('none','shared',False,'same-k',3),
        ('tanh','species',True,'q-total',3),
        ('none','species',False,'q-total',1),
    ]:
        label=f'{activation}-{amplitude}-{outer}-o{rank}'
        options=topology_args(gate='q-total',outer=outer,readout='species',message=0)
        options=[x for x in options if not x.startswith('--outer-k-rank=')]
        options += [f'--outer-k-rank={rank}','--adaptive-radial-projection=shared-o',
                    f'--adaptive-radial-amplitude-mode={amplitude}',
                    f'--adaptive-radial-activation={activation}',f'--joint-readout={str(joint).lower()}']
        top=check.out/(label+'.topology.sus2'); material=check.out/(label+'.material.sus2')
        check.run(label+'-init','init-sh',top,*options)
        check.run(label+'-materialize','roundtrip-model-dev',top,material)
        text=asymmetric_shared_text(material.read_text())
        # Make the nonlinear fixture non-vacuous while keeping finite differences stable.
        if activation=='tanh':
            text=put_numbers(text,'adaptive_radial_weights',numbers(text,'adaptive_radial_weights')*20)
        if joint:
            text=put_numbers(text,'joint_readout_coeffs',.04*np.sin(np.arange(len(numbers(text,'joint_readout_coeffs')))*.31+.23))
        model=check.out/(label+'.sus2');model.write_text(text)
        assert assignment(text,'version')=='1.10.0'
        L=int(assignment(text,'sh_l_max'))+1
        Q=int(assignment(text,'two_layer_gate_scalar_count'))
        U=numbers(text,'adaptive_radial_weights').reshape(L,-1,Q)
        full=put(text,'version','1.7.0')
        full=put(full,'adaptive_radial_projection','full')
        expanded=np.repeat(U[:,None,:,:],rank,axis=1).ravel()
        full=put(full,'adaptive_radial_weight_count',str(len(expanded)))
        full=put_numbers(full,'adaptive_radial_weights',expanded)
        tied=check.out/(label+'.tied-full.sus2');tied.write_text(full)
        np.testing.assert_allclose(check.calc(label,model),check.calc(label+'-tied',tied),atol=2e-10,rtol=2e-10)
        rt=check.out/(label+'.roundtrip.sus2')
        check.run(label+'-roundtrip','roundtrip-model-dev',model,rt)
        np.testing.assert_allclose(check.calc(label+'-rt',rt),check.calc(label+'-source',model),atol=2e-10,rtol=2e-10)
        fields=check.layout(label+'-layout',model)
        for name in ('adaptive_radial_weight','adaptive_radial_amplitude','radial'):
            start,end=int(fields[name+'_begin']),int(fields[name+'_end'])
            for weights,tag in [((1.,.1,.01),'mixed'),((0.,1.,0.),'force'),((0.,0.,1.),'virial')]:
                check.direction(label+'-'+tag+'-'+name,model,start,end,weights=weights)
            penalties=[x for x in ZERO_REG if not x.startswith(('--radial-smooth=','--gate-full-l2='))]
            check.run(label+'-prior-'+name,'check-loss-gradient-direction-dev',model,check.cfg,
                      '--max-configs=1','--energy-weight=0','--force-weight=0','--stress-weight=0',
                      *penalties,'--radial-smooth=1e-3','--radial-smooth-grid=32','--gate-full-l2=1e-2',
                      f'--coeff-start={start}',f'--coeff-end={end}','--displacement=5e-6',
                      '--abs-tolerance=5e-8','--rel-tolerance=5e-5')
        check.run(label+'-normalizer','check-radial-weight-norm-roundtrip-dev',model)
        check.run(label+'-normalizer-gradient','check-radial-weight-norm-gradient-dev',model,check.cfg,
                  '--radial-block=outer','--mu=0','--max-configs=1','--energy-weight=1',
                  '--force-weight=0.1','--stress-weight=0.01',*ZERO_REG,
                  '--displacement=2e-5','--abs-tolerance=5e-6','--rel-tolerance=5e-4')
        check.run(label+'-linear','check-linear-readout-dev',model,check.cfg,
                  '--max-configs=1','--max-atoms=3','--check-stress',
                  '--abs-tolerance=2e-9','--rel-tolerance=2e-9')
        check.run(label+'-geometry','check-efs-fd-dev',model,check.cfg,
                  '--max-configs=1','--max-atoms=3','--displacement=2e-5',
                  '--abs-tolerance=2e-6','--rel-tolerance=2e-4')
        check.record(label+'-PASS',True)
    check.report['status']='PASS'
    check.record('complete',True)


if __name__=='__main__':
    main()
