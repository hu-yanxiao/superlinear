#!/usr/bin/env python3
"""Exact Legacy species heads: conversion, all E/F/S chains and training smoke."""
import argparse
import json
import math
import re
from pathlib import Path
import subprocess
import numpy as np
from adaptive_species_readout_check import numbers, put_numbers, parse_prediction, write_fixture, LOSS_ARGS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary', default='bin/mlp-sus2')
    ap.add_argument('--output', default='.codex_tmp/legacy-species-native')
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    cfg=out/'asymmetric.cfg'; write_fixture(cfg)
    records={}
    def run(label,*args):
        p=subprocess.run([a.binary,*map(str,args)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(label+'.log')).write_text(p.stdout)
        if p.returncode: raise AssertionError(label+'\n'+p.stdout[-4000:])
        return p.stdout
    cases=[('plain',False,'q-total','q-total')]+[(f'gate-{g}-{o}',True,g,o) for g in ('q-total','same-k') for o in ('q-total','same-k')]
    for name,gated,gate_pruning,outer_pruning in cases:
        init=out/(name+'-init.mtp')
        args=['--species-count=2','--l-max=1','--k-max=4','--body-order=3','--radial-basis-size=3',
              '--cutoff=5','--write-sh-scalar-info','--outer-k-rank=3','--sh-factor-pruning='+outer_pruning]
        if gated: args+=['--two-layer-gate','--two-layer-gate-mode=mu-scalar-full',
                        '--two-layer-gate-radial-mode=shared-radial','--two-layer-gate-k-rank=2', '--two-layer-gate-factor-pruning='+gate_pruning]
        run(name+'-init','init-sh',init,*args)
        materialized=out/(name+'-materialized.mtp')
        run(name+'-materialize','roundtrip-model-dev',init,materialized)
        text=materialized.read_text()
        radial=re.search(r'(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)',text)
        assert radial
        row=[0]
        def replace_radial(m):
            row[0]+=1
            return '{'+', '.join(f'{v:.17e}' for v in [.12+.017*row[0],-.09,.07,1.1+.08*row[0],.9+.11*row[0]])+'}'
        text=text[:radial.start()]+radial[1]+re.sub(r'\{[^}]*\}',replace_radial,radial[2])+text[radial.end():]
        for key in ['moment_coeffs','outer_k_rank1_weights']+(['two_layer_gate_weights','two_layer_gate_k_rank_weights','two_layer_gate_radial_coeffs'] if gated else []):
            v=numbers(text,key); text=put_numbers(text,key,[.13+.07*math.sin(i*.63+.3) for i in range(len(v))])
        text=put_numbers(text,'species_coeffs',[0.,-0.7])
        text=put_numbers(text,'scal_coeffs',1.8+.03*np.arange(len(numbers(text,'scal_coeffs'))))
        shared=out/(name+'-shared.mtp'); shared.write_text(text)
        converted=out/(name+'-converted.mtp')
        run(name+'-convert','convert-adaptive-readout-dev',shared,converted,'--readout-mode=species')
        pred=[]
        for tag,model in [('shared',shared),('converted',converted)]:
            p=out/(name+'-'+tag+'.cfg'); run(name+'-'+tag,'calc-efs',model,cfg,p); pred.append(parse_prediction(p))
        delta=float(np.max(np.abs(pred[0]-pred[1]))); assert delta<2e-9,delta
        free=out/(name+'-free.mtp'); tx=converted.read_text()
        v=numbers(tx,'moment_coeffs'); free.write_text(put_numbers(tx,'moment_coeffs',[.05+.09*math.sin(i*.37) for i in range(len(v))]))
        run(name+'-loss-fd','check-loss-gradient-dev',free,cfg,*LOSS_ARGS)
        for cmd,opts in [('check-linear-readout-dev',['--check-stress','--abs-tolerance=2e-8','--rel-tolerance=2e-8']),
                         ('check-linear-components-fd-dev',['--displacement=2e-5','--abs-tolerance=5e-8','--rel-tolerance=5e-5']),
                         ('check-efs-fd-dev',['--displacement=2e-5','--abs-tolerance=2e-6','--rel-tolerance=2e-4'])]:
            run(name+'-'+cmd,cmd,free,cfg,'--max-configs=1',*opts)
        run(name+'-train','train',free,cfg,'--max-iter=2','--skip-preinit','--trained-pot-name='+str(out/(name+'-trained.mtp')),
            '--energy-weight=1','--force-weight=0.1','--stress-weight=0.01','--std-weight=0','--stdd-weight=0',
            '--radial-smooth=0','--scalar-head-l2=0','--gate-full-l2=0')
        records[name]={'conversion_max_abs':delta,'status':'PASS'}
        if gated and gate_pruning=='q-total' and outer_pruning=='q-total':
            for label,bad in [('duplicate-readout',tx+'\nreadout_mode = species\n'),
                              ('missing-readout',re.sub(r'(?m)^readout_mode.*\n','',tx)),
                              ('adaptive-fields',tx+'\nadaptive_radial_weights = {0}\n'),
                              ('old-version',tx.replace('version = 1.8.0','version = 1.3.0'))]:
                invalid=out/(label+'.mtp'); invalid.write_text(bad)
                p=subprocess.run([a.binary,'roundtrip-model-dev',str(invalid),str(out/(label+'-out.mtp'))],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (out/(label+'.log')).write_text(p.stdout)
                assert p.returncode != 0,label
    (out/'report.json').write_text(json.dumps(records,indent=2)+'\n'); print(json.dumps(records))

if __name__=='__main__': main()
