#!/usr/bin/env python3
"""Generate a NEW native-I/O fixture supplement; never rewrite the old 320.

Native interface: native-fixture-tool init-sh FILE --key=value ...;
roundtrip-model-dev INPUT OUTPUT; eval-efs MODEL CFG. The production initializer
canonicalizes mu-body-order to mu-scalar-full; Adaptive only accepts scalar-full.
No LAMMPS launch or scheduler submission is performed here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(text, key, default=None):
    match = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*([^\n]+)', text, re.M)
    return match[1].strip() if match else default


def replace_array(text, key, function):
    pattern = r'(?m)^([ \t]*' + re.escape(key) + r'[ \t]*=[ \t]*)\{([^}]*)\}'
    def replace(match):
        values = [float(v) for v in match[2].split(',') if v.strip()]
        return match[1] + '{' + ', '.join(f'{function(i):.17e}' for i in range(len(values))) + '}'
    return re.sub(pattern, replace, text)


def nonzero(text, zero_ml=False):
    text = re.sub(r'(?m)^\s*scaling\s*=.*$', 'scaling = 1.0', text)
    radial = re.search(r'(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)', text)
    require(radial, 'Missing radial coefficients')
    row = [0]
    def coefficients(match):
        row[0] += 1
        values = [float(v) for v in match[0][1:-1].split(',')]
        require(len(values) == 5, 'Expected radial width3 + species2')
        return '{' + ', '.join(f'{(.08+.009*row[0]+.017*math.sin(i+.4*row[0])) if i<3 else (1.05+.013*row[0]+.09*math.cos(i+row[0])):.17e}' for i in range(5)) + '}'
    text = text[:radial.start()] + radial[1] + re.sub(r'\{[^}]*\}', coefficients, radial[2]) + text[radial.end():]
    for key in ('moment_coeffs', 'outer_k_rank1_weights', 'two_layer_gate_weights',
                'two_layer_gate_body_mix_weights', 'two_layer_gate_additive_coeffs',
                'two_layer_gate_k_rank_weights', 'two_layer_gate_radial_coeffs',
                'two_layer_gate_type_coeffs', 'two_layer_gate_radial_mixing_coeffs',
                'adaptive_radial_weights', 'adaptive_radial_amplitudes',
                'tensor_message_coeffs', 'joint_readout_coeffs'):
        text = replace_array(text, key, lambda i: .09+.045*math.sin(.61*i+.3))
    text = replace_array(text, 'species_coeffs', lambda i: -.07 if i == 0 else .13)
    text = replace_array(text, 'scal_coeffs', lambda i: 1.8+.003*i)
    text = replace_array(text, 'two_layer_gate_scaling_coeffs', lambda i: 1.45+.007*i)
    if zero_ml:
        for key in ('moment_coeffs', 'species_coeffs', 'joint_readout_coeffs'):
            text = replace_array(text, key, lambda i: 0.)
    return text


def specifications():
    # Each family/angular pair has K != O where ranked. Rg=3 is odd and K=5.
    for family in ('static', 'plain'):
        for l in range(5):
            for head in ('shared', 'species'):
                yield dict(family=family, l=l, head=head, k=5, o=3 if family=='static' else 0,
                           gate='q-total', outer='same-k' if head=='species' else 'q-total')
    for family in ('legacy', 'pure', 'adaptive'):
        for l in range(1, 5):
            modes = ('mu-scalar-full',) if family=='adaptive' else ('mu-scalar-full', 'mu-body-order', 'mu-body-linear-combo')
            for gate_mode in modes:
                projections = ('full', 'shared-n') if family=='adaptive' else (None,)
                for projection in projections:
                    yield dict(family=family, l=l, head='shared' if family=='pure' else 'species', k=5, o=3, r=3,
                               gate='same-k', outer='same-k' if family=='pure' else 'q-total',
                               gate_mode=gate_mode, independent=True, projection=projection,
                               message=min(l, 2) if family=='adaptive' else 0,
                               joint=family=='adaptive', rank=3 if projection=='shared-n' else 1)
    # Pure-W alone legally supports O > K. Keep this exact native case too.
    yield dict(family='pure', l=2, head='shared', k=3, o=5, r=3,
               gate='same-k', outer='same-k', gate_mode='mu-body-linear-combo')
    # Paired zero-ML models make missing/doubled ZBL unambiguously observable.
    for typewise in (False, True):
        for enabled in (False, True):
            yield dict(family='plain', l=1, head='shared', k=5, o=0,
                       gate='q-total', outer='q-total', zbl=enabled,
                       zbl_probe=True, typewise=typewise, zero_ml=True)


def options(spec):
    l = spec['l']
    opts = ['--species-count=2', f'--l-max={l}', f'--k-max={spec["k"]}',
            '--body-order=3', f'--body-l-max={l},{l},{l},{l}',
            '--radial-basis-size=3', '--cutoff=4.8', '--radial-basis-type=RBChebyshev_sss',
            '--scaling-map=K', '--outer-k-rank='+str(spec['o']),
            '--sh-factor-pruning='+spec['outer'], '--readout-mode='+spec['head']]
    family = spec['family']
    if family in ('legacy', 'pure', 'adaptive'):
        opts += ['--two-layer-gate', '--two-layer-gate-mode='+spec['gate_mode'],
                 '--two-layer-gate-radial-mode=shared-radial', '--two-layer-gate-k-rank='+str(spec['r']),
                 '--two-layer-gate-factor-pruning='+spec['gate'], '--two-layer-gate-body-order=3']
    if spec.get('independent'):
        opts += ['--two-layer-gate-decoupled-scaling']
    if family == 'pure':
        opts += ['--superlinear-mode=pure-w-channelwise', '--two-layer-gate-radial-mixing=full']
    if family == 'adaptive':
        opts += ['--superlinear-mode=radial-coefficient-gate', '--adaptive-radial-projection='+spec['projection'],
                 '--adaptive-radial-shared-n-rank='+str(spec['rank']), '--adaptive-radial-activation=tanh',
                 '--adaptive-radial-amplitude-mode=species', '--tensor-message-lmax='+str(spec['message']),
                 '--joint-readout=true']
    if spec.get('zbl'):
        opts += ['--zbl-elements=H,O', '--zbl-inner=1.0', '--zbl-outer=6.0']
        if spec['typewise']:
            opts += ['--zbl-typewise-cutoff-factor=0.7']
    return opts


def assert_metadata(text, spec):
    expected = {'sh_l_max':str(spec['l']), 'sh_k_max':str(spec['k']),
                'readout_mode':spec['head'], 'outer_channel_mode':'same-channel' if spec['outer']=='same-k' else 'cross-channel'}
    defaults = {'readout_mode':'shared', 'outer_channel_mode':'cross-channel'}
    if spec['family']=='plain':
        expected.pop('outer_channel_mode')  # No ranked outer stage in plain.
    if spec['o']:
        expected['outer_k_rank'] = str(spec['o'])
    if 'gate_mode' in spec:
        expected['two_layer_gate_k_rank'] = str(spec['r'])
        expected['two_layer_gate_mode'] = 'mu-scalar-full' if spec['gate_mode']=='mu-body-order' else spec['gate_mode']
        expected['two_layer_gate_factor_pruning'] = spec['gate']
    if spec.get('independent'):
        expected['two_layer_gate_scaling_mode']='decoupled'
    if spec['family']=='adaptive':
        expected.update(adaptive_radial_projection=spec['projection'],adaptive_radial_shared_n_rank=str(spec['rank']),
                        adaptive_radial_activation='tanh',adaptive_radial_amplitude_mode='species',
                        tensor_message_lmax=str(spec['message']),joint_readout_enabled='true')
    if spec.get('zbl'):
        expected['zbl_typewise_cutoff_enabled']=str(spec['typewise']).lower()
    for key, value in expected.items():
        require(field(text,key,defaults.get(key)) == value, f'{key}: expected {value}')
    require(field(text,'zbl_enabled','false') == str(spec.get('zbl',False)).lower(), 'ZBL metadata mismatch')
    if spec.get('zero_ml'):
        for key in ('moment_coeffs','species_coeffs'):
            values=field(text,key)[1:-1].split(',')
            require(values and all(float(x)==0. for x in values),'ZBL control contains nonzero ML readout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    require(not args.out.exists(), 'Preserve existing fixtures; choose a NEW output directory')
    args.out.mkdir(parents=True)
    specs = list(specifications())
    report = dict(schema=1, status='PREPARED_NOT_RUN', cases=[], planned_specs=specs,
                  native=str(args.native.resolve()), native_sha256=sha(args.native),
                  generator_sha256=sha(__file__), native_source='dev_test/distributed_fixture_tool.cpp + dev_src/sh_model_init.cpp',
                  alias_contract='mu-body-order CLI input canonicalizes to mu-scalar-full; Adaptive scalar-full only')
    def save():
        (args.out/'matrix.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    if args.plan_only:
        return
    def run(name, argv):
        result = subprocess.run(list(map(str,argv)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        (args.out/(name+'.log')).write_text(result.stdout)
        require(result.returncode==0, name+' failed: '+result.stdout[-2000:])
    try:
        for index,spec in enumerate(specs,2000000):
            name=f'{index}-{spec["family"]}'
            top,material,model,roundtrip=[args.out/(name+suffix+'.sus2') for suffix in ('.topology','.material','','.roundtrip')]
            command=[args.native,'init-sh',top,*options(spec)]
            run(name+'-init',command)
            run(name+'-materialize',[args.native,'roundtrip-model-dev',top,material])
            text=nonzero(material.read_text(),spec.get('zero_ml',False))
            assert_metadata(text,spec)
            model.write_text(text)
            run(name+'-roundtrip',[args.native,'roundtrip-model-dev',model,roundtrip])
            assert_metadata(roundtrip.read_text(),spec)
            report['cases'].append(dict(index=index,spec=spec,model=str(model.resolve()),sha256=sha(model),
                                        init_command=list(map(str,command))))
            save()
            print(name,'FIXTURE_READY',flush=True)
        report['status']='FIXTURES_READY_NOT_QUALIFIED'
    except Exception as error:
        report.update(status='FAILED',error=str(error))
        raise
    finally:
        save()


if __name__=='__main__':
    main()
