"""Source-bound normal finite-scale preparation on immutable whole2070.

Experiment bindings select evidence; generic legality/scanner/LLVM contracts
come from Merlin. Target mode guards and preserved target objects remain OOT.
"""
from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower import quant_hoist
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers, emit_finite_scale_helper
from merlin.llvmlower.source_expression_interval import IntervalEffectContract, build_source_interval_table, emit_source_interval_i8_lookup, find_closed_scalar_i8_observers
from merlin.runtime.backends.spike_model import _transform_host_ir
from merlin.runtime.dispatch_runtime import resolve_forward_args
from mlir_oot.late_quant_rne import merlin_host_llvm_transform

OLD=Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006')
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'out/artifacts/probes/finite-scale-normal-2070-v2-20261007'
CONTROL=OLD/'out/artifacts/probes/closed-i8-interval-normal-2062-20261007'
SOURCE=CONTROL.parent/'masked-contraction-normal-whole-v3-20261007/selected/lower/model.ll'
CORE=Path('/scratch/agustin/tmp/merlin-tiny-finite-domain-main-20261007')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
BUNDLE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/tiny_fresh_bundle')
BASE_NATIVE=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/native')
FROZEN=Path('/scratch/agustin/tmp/merlin-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/frozen_runtime')
CAPTURE=OLD/'out/artifacts/probes/source-continuation-contexts-v3-20261007'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    base=load('qualified_whole_base',OLD/'tests/source_continuation_whole_build.py')
    sys.modules['source_continuation_whole_build']=base
    qualifier=load('qualified_whole_native',OLD/'tests/source_continuation_whole_qualify.py')
    effects=IntervalEffectContract(True,True,True,True,True)
    typed=base.N/'typed_prepacket.generic.mlir'
    proofs,refused=find_closed_scalar_i8_observers(parse_mlir_text(typed.read_text()),effects=effects)
    assert len(proofs)==22 and not refused
    previous=json.loads((CONTROL/'selected/target/host_llvm/source_binding.json').read_text())
    helpers=bind_finite_scale_helpers(SOURCE.read_text(),routes=previous['source_bindings']['routes'],observers=proofs,effects=effects,immutable_inputs=True,fresh_disjoint_output=True)
    assert len(helpers)==22 and sum(len(h.scans) for h in helpers)==44
    table=build_source_interval_table(proofs[0].expression,effects=effects,leading_bits=16,max_table_bytes=512*1024)
    assert table.sha256==previous['source_table_sha256']
    tree=ast.parse((OLD/'tests/source_continuation_whole_build.py').read_text())
    callback=ast.unparse(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='transform'))+'\n'
    (OUT/'qualified_callback.py').write_text(callback)
    flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
    guards={};all_commands=[]
    for arm in ('control','selected'):
        assets=OUT/arm;assets.mkdir()
        for name in ('table.ll','source_activation.ll','source_quantize.ll'):os.link(CONTROL/arm/name if (CONTROL/arm/name).exists() else CONTROL/'selected'/name,assets/name)
        lookup=emit_source_interval_i8_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation',leading_bits=16,finite_inputs=helpers if arm=='selected' else ())
        (assets/'lookup.c').write_text(lookup)
        if arm=='control':assert lookup==(CONTROL/'selected/lookup.c').read_text()
        for native in (False,True):
            case=assets/('native' if native else 'target')
            seen=set()
            def run(argv):
                argv=list(map(str,argv));all_commands.append(argv)
                sources=[Path(p) for p in argv if p.endswith('mode_guard.c')]
                if arm=='selected' and sources and str(sources[0]) not in seen:
                    seen.add(str(sources[0]));path=sources[0];glue=path.read_text();add=[];decl=[]
                    for i,helper in enumerate(helpers):
                        symbol=f'prepared_{i}'
                        generated=emit_finite_scale_helper(helper,wrapper_symbol=symbol,finite_symbol=f'rne_{i}',fallback_symbol=f'source_{i}')
                        decl.append(f'void {symbol}('+','.join('void*' for _ in helper.arguments)+');')
                        old='else rne_'+str(i)+'('+','.join('a'+str(j) for j in range(len(helper.arguments)))+');'
                        new='else '+symbol+'('+','.join('a'+str(j) for j in range(len(helper.arguments)))+');'
                        assert glue.count(old)==1;glue=glue.replace(old,new)
                        add.append(generated)
                    path.write_text('\n'.join(decl)+'\n'+glue+'\n'+'\n'.join(add))
                result=subprocess.run(argv,capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stdout+result.stderr)
                return result
            namespace=dict(vars(base))
            namespace.update(O=assets,proofs=proofs,table=table,typed=typed,effects=effects,continuation=previous['continuation'],flags=flags,source=SOURCE,native=native,run=run)
            from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup
            namespace['rewrite_source_interval_lookup']=rewrite_source_interval_i8_lookup
            exec(compile(callback,str(OUT/'qualified_callback.py'),'exec'),namespace)
            selected,hook=_transform_host_ir(SOURCE,case/'host_llvm',namespace['transform'])
            save(case/'normal_hook_receipt.json',hook)
            bindings=json.loads((case/'host_llvm/source_binding.json').read_text())
            assert bindings['source_bindings']==previous['source_bindings']
            assert len(bindings['whole_helper_guards'])==22 and bindings['all_original_device155_references_conserved']
            suffix='.native' if native else ''
            if arm=='control':
                prior=CONTROL/'selected'/('native' if native else 'target')/'host_llvm'/f'expanded{suffix}.ll'
                assert selected.read_text().splitlines()[1:]==prior.read_text().splitlines()[1:]
            if not native:
                run([LLVM/'clang',*flags,'-c',selected,'-o',case/'model.o'])
                if arm=='control':assert sha(case/'model.o')==sha(CONTROL/'selected/target/model.o')
            else:
                host=case/'host';host.mkdir()
                run([LLVM/'clang','-O2','-fPIC','-c',selected,'-o',host/'model.o'])
                run(['cc','-O3','-march=native','-fPIC','-shared',host/'model.o',qualifier.BASE_NATIVE/'reference.c',qualifier.BASE_NATIVE/'device/device_catalog_shim.c',qualifier.FROZEN/'merlin/runtime/abi/mlir_runtime.c','-lm','-o',host/'model.so'])
                guards[arm]=bindings['whole_helper_guards']
            print('COMPILED',arm,native,flush=True)
    # Same complete physical inputs, dirty output guards and all native modes.
    feature=json.loads((CAPTURE/'features_v2/contexts.json').read_text())
    context_events=[];environment=ctypes.CDLL(None)
    oldmode,oldflags=environment.fegetround(),environment.fetestexcept(61)
    try:
        libraries={a:ctypes.CDLL(str(OUT/a/'native/host/model.so')) for a in ('control','selected')}
        for i,helper in enumerate(helpers):
            constant=helper.scans[0].proof.domain.constant_words[0]
            quant=helper._routes[0]['quant_factor_bits']
            matches=[r for r in feature['rows'] if r['preparation_factor_bits']==constant and r['quant_factor_bits']==quant];assert len(matches)==1
            ctx=matches[0]['context'];directory=CAPTURE/f'context_{ctx:02d}'
            arrays=[np.load(directory/(n+'.npy')) for n in ('a','scale_a','b','scale_b')]
            expected=np.load(directory/'expected.npy').reshape(-1)
            before=[sha(directory/(n+'.npy')) for n in ('a','scale_a','b','scale_b')]
            fns=[getattr(libraries[a],helper.symbol) for a in ('control','selected')]
            for fn in fns:fn.argtypes=[ctypes.c_void_p]*5;fn.restype=None
            for mode in (0,0x400,0x800,0xc00):
                assert environment.fesetround(mode)==0
                for sticky in (0,1,4,8,16,32,61):
                    outputs=[];observed=[]
                    for fn in fns:
                        dst=np.full(expected.size+128,73,np.int8)
                        environment.feclearexcept(61);environment.feraiseexcept(sticky)
                        fn(*[v.ctypes.data for v in arrays],dst.ctypes.data+64)
                        observed.append(environment.fetestexcept(61))
                        assert np.all(dst[:64]==73) and np.all(dst[-64:]==73)
                        outputs.append(dst[64:-64].copy())
                    assert np.array_equal(*outputs)
                    if not mode:assert np.array_equal(outputs[1],expected)
                    if mode:assert observed[0]==observed[1]
                    assert observed[1]&sticky==sticky
                    context_events.append({'context':ctx,'mode':mode,'sticky':sticky,'source_original_i8_words':expected.size,'words_exact':True,'unsupported_mode_flags_exact':bool(mode),'sticky_preserved':True})
            assert before==[sha(directory/(n+'.npy')) for n in ('a','scale_a','b','scale_b')]
            print('CONTEXT',i,ctx,flush=True)
    finally:
        environment.fesetround(oldmode);environment.feclearexcept(61);environment.feraiseexcept(oldflags)
    inputs=resolve_forward_args(qualifier.BUNDLE);plan=quant_hoist.read_plan(base.B);assert len(plan)==155
    values=quant_hoist.read_values(base.B);inputs.extend(np.ascontiguousarray(values[p.key]) for p in plan)
    original=np.load(CONTROL/'selected/native/host/output.npy');golden=np.load(qualifier.BUNDLE/'golden.npy')
    events=[]
    for arm in ('control','selected'):
        host=OUT/arm/'native/host';output=np.empty(original.shape,np.float32)
        model=HostModel.load(str(host/'model.so'));model([(a.ctypes.data,a.shape) for a in [*inputs,output]])
        assert np.array_equal(output.view(np.uint32),original.view(np.uint32)) and np.allclose(output,golden,atol=.03125,rtol=.02)
        np.save(host/'output.npy',output);events.append({'arm':arm,'all256000_original_words_exact':True,'torch_gate':True,'output_elements':output.size,'native_so_sha256':sha(host/'model.so')})
        print('WHOLE_NATIVE',arm,flush=True)
    save(OUT/'qualification.json',{'schema':'source_finite_broadcast_normal_whole_preparation_v1','status':'pass','events':events,'context_events':context_events,'all22contexts_991232_i8_exact':True,'control_model_object_byteexact2070':True,'all155devicebindings':True,'source_producer_chains':44,'scale_scan_words':sum(s.count for h in helpers for s in h.scans),'original_per_element_finite_checks':1982464,'whole_cycles':'UNKNOWN','native_only_no_hardware_claim':True,'commands':all_commands,'pins':{str(p):sha(p) for p in [Path(__file__),typed,SOURCE,*OUT.rglob('*'),*CORE.glob('src/merlin/llvmlower/scaled_integer_finite*.py'),CORE/'src/merlin/llvmlower/source_expression_interval.py'] if p.is_file()},'token_usage_available':False})
    print('NORMAL_FINITE_PREPARATION_PASS',flush=True)

if __name__=='__main__':main()
