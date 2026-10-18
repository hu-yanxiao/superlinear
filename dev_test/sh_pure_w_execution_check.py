#!/usr/bin/env python3
"""Exercise real EFS/mixed-FD regressions and force every execution backend.

The existing asymmetric physics suite provides nonzero W, gate and heads,
K/L/LK sharing, decoupled gate scaling, zero-W, O1 and R>K*O boundaries.
Check dispatch markers too: numerical parity through fallback alone is not a
successful test of a new kernel. No production or external data is needed.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT/'bin/mlp-sus2'
(ROOT/'.local_tmp').mkdir(exist_ok=True)
OUT = Path(tempfile.mkdtemp(prefix='pure-w-execution-', dir=ROOT/'.local_tmp'))
BASE = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
CFG = ROOT/'native_tests/pure_w_regularization.cfg'

def run(label, cmd, env):
    result = subprocess.run(list(map(str,cmd)), cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT/(label+'.log')).write_text(result.stdout)
    assert result.returncode == 0, (label, result.stdout[-3000:])
    return result.stdout

print(f'Evidence: {OUT}', flush=True)
for width in (1,2,4,8):
    env = dict(BASE, PURE_W_TEST_OUTER_RANK=str(width), SUS2_SH_PURE_W_KERNEL='basis',
               SUS2_SH_PURE_W_GRAPH='batched', SUS2_SH_PURE_W_PROFILE='1')
    output = run(f'O{width}-physics', [sys.executable, ROOT/'dev_test/sh_pure_w_physics_check.py',BIN], env)
    assert 'PASS pure-W physics regression' in output
    fixture = Path(re.search(r'^Evidence: (.*)$',output,re.M)[1])
    for mapping in ('K','L','LK'):
        model = fixture/f'{mapping}-k2-o{width}-r3-d0.sus2'
        predictions = []
        for mode, graph in [('legacy','legacy'),('auto','legacy'),('packed','legacy'),('packed','batched'),('basis','legacy'),('basis','batched')]:
            label = f'O{width}-{mapping}-{mode}-{graph}'
            path = OUT/(label+'.cfg')
            log = run(label, [BIN,'calc-efs',model,CFG,path],
                      dict(BASE,SUS2_SH_PURE_W_KERNEL=mode,SUS2_SH_PURE_W_GRAPH=graph,
                           SUS2_SH_PURE_W_PROFILE='1',SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE='0',
                           SUS2_SH_PRODUCT_ROWS='0'))
            marker = re.search(r'\[PURE_W_EXECUTION\] ([^\n]+)',log)
            assert marker, label
            fields = dict(re.findall(r'(\w+)=(\S+)',marker[1]))
            assert fields['basis'] == str(int(mode=='basis')), (label,fields)
            # This fixture has fewer than 16 neighbors/site: auto is legacy.
            assert fields['packed'] == str(int(mode not in ('legacy','auto'))), (label,fields)
            assert fields['graph'] == str(int(graph=='batched' and width>1)), (label,fields)
            if mode=='basis':
                assert fields['W_entries']=='0' and fields['W_storage_entries']=='0'
                assert int(fields['raw_dr_entries'])>0
            text = path.read_text()
            # All numeric CFG fields, including coordinates, E, F and stress.
            nums = list(map(float,re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',text)))
            predictions.append(nums)
        for actual in predictions[1:]:
            assert len(actual)==len(predictions[0])
            error=max(abs(a-b)/max(1,abs(a),abs(b)) for a,b in zip(actual,predictions[0]))
            assert error<2e-10,(width,mapping,error)
        if width>1:
            # Independent channel blocks can legally appear in another order
            # in a saved model. Labels must not be inferred from list position.
            src=model.read_text()
            block=re.search(r'(?m)^sh_products = \{(.*)\}$',src)
            terms=re.findall(r'\{[^{}]*\}',block[1])
            start=int(re.search(r'(?m)^outer_k_rank1_product_begin = (\d+)$',src)[1])
            per=(len(terms)-start)//width
            assert per*width==len(terms)-start
            reordered=terms[:start]+[t for lane in reversed(range(width))
                                    for t in terms[start+lane*per:start+(lane+1)*per]]
            src=src[:block.start()]+ 'sh_products = {'+', '.join(reordered)+'}'+src[block.end():]
            permuted=OUT/f'O{width}-{mapping}-reordered.sus2'
            permuted.write_text(src)
            for mode,graph in [('legacy','legacy'),('packed','batched'),('basis','batched')]:
                label=f'O{width}-{mapping}-reordered-{mode}'
                path=OUT/(label+'.cfg')
                log=run(label,[BIN,'calc-efs',permuted,CFG,path],dict(BASE,
                    SUS2_SH_PURE_W_KERNEL=mode,SUS2_SH_PURE_W_GRAPH=graph,
                    SUS2_SH_PURE_W_PROFILE='1',SUS2_SH_PRODUCT_ROWS='0'))
                actual=list(map(float,re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',path.read_text())))
                assert len(actual)==len(predictions[0])
                error=max(abs(a-b)/max(1,abs(a),abs(b)) for a,b in zip(actual,predictions[0]))
                assert error<2e-10,(label,error)
                assert re.search(r'\[PURE_W_EXECUTION\].* graph=0 ',log),label
    print(f'PASS O{width}: physics, K/L/LK, all six execution modes and dispatch',flush=True)
print('PASS pure-W execution matrix',flush=True)
