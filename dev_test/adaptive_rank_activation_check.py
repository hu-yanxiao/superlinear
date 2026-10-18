#!/usr/bin/env python3
"""Server-LSF-only rank/tanh mathematical regression; no local numerical runs.

Exercises native init/serialization, all gate/outer combinations, exact linear
folding, nonzero tanh mixed E/F/stress derivatives, source-species amplitudes,
joint/tensor routes, malformed metadata, and fresh/resume calibration.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpp', type=Path, required=True)
    parser.add_argument('--old-cpp', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--pytorch-root', type=Path, required=True,
                        help='Independent gate-feature evaluator for non-vacuous test-only tanh calibration')
    parser.add_argument('--stage', choices=('minimal', 'full'), default='minimal')
    args = parser.parse_args()
    if sys.platform == 'darwin' or not os.environ.get('LSB_JOBID'):
        raise SystemExit('Numerical checks require a server LSF allocation')
    import numpy as np
    sys.path.insert(0, str(args.pytorch_root.resolve()))
    import torch
    from sus2shpt.io.cfg import load_cfg
    from sus2shpt.io.sus2 import load_mtp_model
    from sus2shpt.model.sus2_sh import SUS2SHPyTorch
    torch.set_num_threads(1)
    from adaptive_species_readout_check import (
        assignment, numbers, put, put_numbers, asymmetric_shared_text, write_fixture, parse_prediction, ZERO_REG)
    from adaptive_tensor_message_check import TensorCheck, topology_args
    check = TensorCheck(args.cpp, args.outdir, args.old_cpp)
    check.cfg = check.out/'asymmetric.cfg'
    write_fixture(check.cfg)
    cases = {}

    def calibrate_test_signal(label, text):
        """Only fixtures are changed: make phi'' substantial, not near zero.

        PT supplies gate scalar values only. All E/F/S and mixed derivative
        acceptance checks below remain native analytic-vs-finite-difference.
        """
        tmp=check.out/(label+'.precalibration.sus2'); tmp.write_text(text)
        obj=SUS2SHPyTorch(load_mtp_model(tmp),device='cpu',dtype=torch.float64)
        cfg=load_cfg(check.cfg)[0]
        positions=torch.tensor([a.position for a in cfg.atoms],dtype=torch.float64)
        cell=torch.tensor(cfg.supercell,dtype=torch.float64)
        types=torch.tensor([a.type for a in cfg.atoms],dtype=torch.long)
        with torch.no_grad():
            selected,_,_=obj.raw_feature_values_from_tensors(positions,cell,types,outer_required=False)
            raw=obj._adaptive_project_scalars(selected,activate=False).reshape(len(types),-1).numpy()
        rms=np.sqrt(np.mean(raw*raw,axis=0))
        assert np.isfinite(rms).all() and (rms>1e-12).all(),(label,'degenerate fixture h',rms)
        factors=.7/rms
        q=int(assignment(text,'two_layer_gate_scalar_count'))
        weights=numbers(text,'adaptive_radial_weights').reshape(-1,q)*factors[:,None]
        text=put_numbers(text,'adaptive_radial_weights',weights.ravel())
        calibrated=raw*factors[None,:]
        phi=np.tanh(calibrated)
        second=-2*phi*(1-phi*phi)
        assert np.max(np.abs(second))>.3,(label,'vacuous phi second derivative')
        check.record(label+'-nonlinear-fixture',{'raw_rms_min':float(rms.min()),
            'raw_rms_max':float(rms.max()),'calibrated_rms':.7,
            'max_abs_phi_second':float(np.abs(second).max())})
        return text

    def case(label, projection='shared-n', rank=2, activation='tanh',
             gate='q-total', outer='same-k', readout='species', message=0, joint=False):
        top = check.out/(label+'.topology.sus2')
        materialized = check.out/(label+'.materialized.sus2')
        options = topology_args(gate=gate, outer=outer, readout=readout, message=message)
        options += ['--adaptive-radial-projection='+projection,
                    '--adaptive-radial-shared-n-rank='+str(rank),
                    '--adaptive-radial-activation='+activation,
                    '--adaptive-radial-amplitude-mode=species']
        if joint: options += ['--joint-readout=true']
        check.run(label+'-init', 'init-sh', top, *options)
        check.run(label+'-materialize', 'roundtrip-model-dev', top, materialized)
        text = put(asymmetric_shared_text(materialized.read_text()), 'readout_mode', readout)
        if message:
            text = put_numbers(text, 'tensor_message_coeffs',
                .06*np.cos(np.arange(len(numbers(text, 'tensor_message_coeffs')))*.39+.27))
        if joint:
            text = put_numbers(text, 'joint_readout_coeffs',
                .04*np.sin(np.arange(len(numbers(text, 'joint_readout_coeffs')))*.31+.23))
        if activation == 'tanh':
            text=calibrate_test_signal(label,text)
        q = int(assignment(text, 'two_layer_gate_scalar_count'))
        t = rank if projection == 'shared-n' else 1
        assert len(numbers(text, 'adaptive_radial_amplitudes')) == 2*27*t
        assert len(numbers(text, 'adaptive_radial_weights')) == (9*t if projection == 'shared-n' else 27)*q
        if rank > 1 or activation == 'tanh':
            assert assignment(text, 'version') == '1.7.0'
            assert assignment(text, 'adaptive_radial_activation') == activation
            assert int(assignment(text, 'adaptive_radial_shared_n_rank')) == rank
            assert assignment(text, 'joint_readout_enabled') == ('true' if joint else 'false')
        model = check.out/(label+'.sus2'); model.write_text(text)
        roundtrip = check.out/(label+'.roundtrip.sus2')
        check.run(label+'-roundtrip', 'roundtrip-model-dev', model, roundtrip)
        np.testing.assert_allclose(check.calc(label+'-efs', model),
            check.calc(label+'-roundtrip-efs', roundtrip), atol=2e-10, rtol=2e-10)
        if activation == 'tanh':
            identity=check.out/(label+'.identity.sus2')
            identity.write_text(put(text,'adaptive_radial_activation','none'))
            difference=float(np.abs(check.calc(label+'-nonlinear',model)
                -check.calc(label+'-identity',identity)).max())
            assert difference>1e-8,(label,'tanh has no measurable EFS effect',difference)
            check.record(label+'-nonlinear-EFS-effect',difference)
        layout = check.layout(label+'-layout', model)
        ranges = {name:(int(layout[name+'_begin']), int(layout[name+'_end']))
                  for name in ('adaptive_radial_amplitude', 'adaptive_radial_weight',
                               'gate_k_rank', 'gate_radial', 'gate_type', 'radial')}
        ranges['scaling'] = (2, int(layout['radial_begin']))
        if message: ranges['tensor_message'] = (int(layout['tensor_message_begin']), int(layout['tensor_message_end']))
        selected = ranges if label == 'shared_tanh' or joint else {
            name:ranges[name] for name in ('adaptive_radial_amplitude', 'adaptive_radial_weight')}
        for name, bounds in selected.items():
            check.direction(label+'-mixed-'+name, model, *bounds)
        if activation == 'tanh':
            for name in ('adaptive_radial_amplitude', 'adaptive_radial_weight', 'gate_k_rank'):
                check.direction(label+'-force-'+name, model, *ranges[name], weights=(0.,1.,0.))
                check.direction(label+'-stress-'+name, model, *ranges[name], weights=(0.,0.,1.))
        check.run(label+'-geometry', 'check-efs-fd-dev', model, check.cfg,
            '--max-configs=1', '--max-atoms=3', '--displacement=2e-5',
            '--abs-tolerance=2e-6', '--rel-tolerance=2e-4')
        check.run(label+'-linear-components', 'check-linear-readout-dev', model, check.cfg,
            '--max-configs=1', '--max-atoms=3', '--check-stress',
            '--abs-tolerance=2e-9', '--rel-tolerance=2e-9')
        # Dedicated response-prior check has zero data weights: all cross-t
        # terms in (sum_t B_t U_t)^2 must be differentiated, including scaling.
        for name in ('adaptive_radial_amplitude', 'adaptive_radial_weight', 'scaling'):
            start, end = ranges[name]
            penalty_opts = [x for x in ZERO_REG if not x.startswith(('--radial-smooth=', '--gate-full-l2='))]
            check.run(label+'-prior-'+name, 'check-loss-gradient-direction-dev', model, check.cfg,
                '--max-configs=1', '--energy-weight=0', '--force-weight=0', '--stress-weight=0',
                *penalty_opts, '--radial-smooth=1e-3', '--radial-smooth-grid=32', '--gate-full-l2=1e-2',
                f'--coeff-start={start}', f'--coeff-end={end}', '--displacement=5e-6',
                '--abs-tolerance=5e-8', '--rel-tolerance=5e-5')
        cases[label] = {'path':str(model), 'projection':projection, 'rank':rank,
                        'activation':activation, 'joint':joint, 'tensor_lmax':message}
        check.record(label+'-PASS', cases[label])
        return model, top, options

    target, top, options = case('shared_tanh')
    linear, _, _ = case('shared_linear', activation='none')
    full, _, _ = case('full_tanh', projection='full', rank=1)
    case('rank_N_joint_tensor', rank=3, gate='same-k', outer='q-total', joint=True, message=2)
    legacy, _, _ = case('legacy_rank_one', rank=1, activation='none')
    if args.old_cpp:
        np.testing.assert_allclose(check.calc('legacy-new',legacy),
            check.calc('legacy-old',legacy,binary=args.old_cpp),atol=2e-10,rtol=2e-10)
        check.record('legacy-rank-one-parity',True)
    angular = {}
    for backend in ('ry','rd-m0'):
        pred=check.out/('tanh-'+backend+'.cfg')
        check.run('tanh-'+backend,'calc-efs',target,check.cfg,pred,
            env={'SUS2_SH_ANGULAR_BACKEND':backend})
        angular[backend]=parse_prediction(pred)
    np.testing.assert_allclose(angular['ry'],angular['rd-m0'],atol=2e-10,rtol=2e-10)
    check.record('tanh-RY-RD-parity',True)
    zero=check.out/'zero-expansion.sus2'
    zero.write_text(put_numbers(target.read_text(),'adaptive_radial_amplitudes',
        np.zeros_like(numbers(target.read_text(),'adaptive_radial_amplitudes'))))
    zero_layout=check.layout('zero-expansion-layout',zero)
    for name in ('adaptive_radial_amplitude','adaptive_radial_weight'):
        check.direction('zero-expansion-'+name,zero,int(zero_layout[name+'_begin']),
            int(zero_layout[name+'_end']),nonzero=name=='adaptive_radial_amplitude',weights=(0.,1.,0.))
    check.record('zero-expansion-chain-rule',True)
    if args.stage == 'full':
        for gate, outer, readout in itertools.product(('same-k','q-total'), ('same-k','q-total'), ('shared','species')):
            case('combo_'+gate+'_'+outer+'_'+readout, gate=gate, outer=outer, readout=readout)

    # Exact identity-activation folding into full U with amplitude=1, same
    # source-species slices would be impossible in general; this fixture uses
    # a shared amplitude by collapsing species to slice zero before comparing.
    text = linear.read_text(); q = int(assignment(text, 'two_layer_gate_scalar_count'))
    a = numbers(text,'adaptive_radial_amplitudes').reshape(2,3,3,3,2)[0]
    u = numbers(text,'adaptive_radial_weights').reshape(3,3,2,q)
    compact = put(text,'adaptive_radial_amplitude_mode','shared')
    compact = put(put_numbers(compact,'adaptive_radial_amplitudes',a.ravel()),'adaptive_radial_amplitude_count',str(a.size))
    folded = put(compact,'adaptive_radial_projection','full')
    folded = put(folded,'adaptive_radial_shared_n_rank','1')
    effective = np.einsum('lont,lotq->lonq',a,u)
    folded = put(put_numbers(folded,'adaptive_radial_amplitudes',np.ones(27)),'adaptive_radial_amplitude_count','27')
    folded = put(put_numbers(folded,'adaptive_radial_weights',effective.ravel()),'adaptive_radial_weight_count',str(effective.size))
    # Drop stale optimizer amplitude scale vectors after changing topology.
    for field in ('adaptive_radial_amplitude_optimizer_scales',):
        compact = re.sub(r'(?m)^'+field+r'\s*=.*\n?', '', compact)
        folded = re.sub(r'(?m)^'+field+r'\s*=.*\n?', '', folded)
    cp=check.out/'compact-linear.sus2'; cp.write_text(compact)
    fp=check.out/'folded-linear.sus2'; fp.write_text(folded)
    np.testing.assert_allclose(check.calc('compact-linear',cp),check.calc('folded-linear',fp),atol=2e-10,rtol=2e-10)
    check.record('exact-linear-fold', True)

    original = target.read_text()
    malformed = {
        'rank_zero':put(original,'adaptive_radial_shared_n_rank','0'),
        'rank_too_large':put(original,'adaptive_radial_shared_n_rank','4'),
        'full_rank_two':put(original,'adaptive_radial_projection','full'),
        'activation_unknown':put(original,'adaptive_radial_activation','relu'),
        'old_version_new_fields':put(original,'version','1.5.0'),
        'missing_activation':re.sub(r'(?m)^adaptive_radial_activation\s*=.*\n','',original),
        'duplicate_rank':original+'\nadaptive_radial_shared_n_rank = 2\n',
        'missing_joint':re.sub(r'(?m)^joint_readout_enabled\s*=.*\n','',original),
        'bad_amplitude_count':put(original,'adaptive_radial_amplitude_count','27'),
    }
    for label, text in malformed.items():
        path=check.out/(label+'.sus2'); path.write_text(text)
        check.run(label,'roundtrip-model-dev',path,check.out/(label+'.unexpected.sus2'),fail=True)
    check.record('malformed-schema-rejected',list(malformed))

    fresh = check.out/'fresh-tanh.sus2'; resumed=check.out/'resume-tanh.sus2'
    train_opts=['--max-iter=0','--init-params=same','--init-random-seed=20260916',
                '--energy-weight=1','--force-weight=.1','--stress-weight=.01',
                '--std-weight=0','--stdd-weight=0']
    log=check.run('fresh-calibration','train',top,check.cfg,*train_opts,f'--trained-pot-name={fresh}')
    assert 'tanh_preactivation_rms_target=0.5' in log
    check.run('resume-calibration','train',fresh,check.cfg,*train_opts,f'--trained-pot-name={resumed}')
    np.testing.assert_allclose(check.calc('fresh-calibration-efs',fresh),check.calc('resume-calibration-efs',resumed),atol=2e-10,rtol=2e-10)
    check.run('reject-activation-resume','train',fresh,check.cfg,*train_opts,'--adaptive-radial-activation=none',fail=True)
    check.run('reject-rank-resume','train',fresh,check.cfg,*train_opts,'--adaptive-radial-shared-n-rank=1',fail=True)
    check.record('fresh-and-exact-resume',True)
    check.report.update(status='PASS',stage=args.stage,cases=cases)
    (check.out/'report.json').write_text(json.dumps(check.report,indent=2)+'\n')


if __name__ == '__main__':
    main()
