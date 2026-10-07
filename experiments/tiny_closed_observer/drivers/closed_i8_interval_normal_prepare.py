"""Normal opt-in observer callback, preserving frozen2062 source and boundaries.

This prepares small model objects/native gates while the independent complete
M8 timing runs. No whole candidate ELF or stock arm is constructed here.
"""
from __future__ import annotations
import ast
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import source_continuation_whole_build as base

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract, build_source_interval_table,
    emit_source_interval_i8_lookup, find_closed_scalar_i8_observers,
)
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup
from merlin.llvmlower import quant_hoist
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.runtime.backends.spike_model import _transform_host_ir
from source_continuation_whole_qualify import BASE_NATIVE, BUNDLE, FROZEN
from mlir_oot.late_quant_rne import merlin_host_llvm_transform

Y=base.Y
CONTROL=Y/'out/artifacts/probes/masked-contraction-normal-whole-v3-20261007'
OUT=Y/'out/artifacts/probes/closed-i8-interval-normal-2062-20261007'
SOURCE=CONTROL/'selected/lower/model.ll'
C=Path('/scratch/agustin/tmp/merlin-closed-i8-interval-main-20261007')


def main():
    OUT.mkdir(exist_ok=False)
    source=SOURCE.read_text()
    typed=base.N/'typed_prepacket.generic.mlir'
    effects=IntervalEffectContract(True,True,True,True,True)
    proofs,refused=find_closed_scalar_i8_observers(parse_mlir_text(typed.read_text()),effects=effects)
    assert len(proofs)==22 and not refused
    table=build_source_interval_table(proofs[0].expression,effects=effects,leading_bits=16,max_table_bytes=512*1024)
    previous=json.loads((CONTROL/'selected/target/host_llvm/source_binding.json').read_text())
    assert table.sha256==previous['source_table_sha256']
    tree=ast.parse(Path(base.__file__).read_text())
    callback=ast.unparse(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='transform'))+'\n'
    (OUT/'qualified_callback.py').write_text(callback)
    flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin']
    guards={}
    for arm in ('control','selected'):
        assets=OUT/arm
        assets.mkdir()
        for name in ('table.ll','source_activation.ll','source_quantize.ll'):
            os.link(CONTROL/name,assets/name)
        if arm=='control':
            os.link(CONTROL/'lookup.c',assets/'lookup.c')
        else:
            (assets/'lookup.c').write_text(emit_source_interval_i8_lookup(table_name='source_interval_table',activation_name='source_activation',quantizer_name='source_quantize',lookup_name='source_lookup_activation',leading_bits=16))
        for native in (False,True):
            case=assets/('native' if native else 'target')
            namespace=dict(vars(base))
            namespace.update(O=assets,proofs=proofs,table=table,typed=typed,effects=effects,
                continuation=previous['continuation'],flags=flags,source=SOURCE,native=native)
            if arm=='selected':
                namespace['rewrite_source_interval_lookup']=rewrite_source_interval_i8_lookup
            exec(compile(callback,str(OUT/'qualified_callback.py'),'exec'),namespace)
            selected,hook=_transform_host_ir(SOURCE,case/'host_llvm',namespace['transform'])
            base.save(case/'normal_hook_receipt.json',hook)
            suffix='.native' if native else ''
            if arm=='control':
                old=CONTROL/'selected'/('native' if native else 'target')/'host_llvm'/f'expanded{suffix}.ll'
                # ModuleID/path provenance differs; every semantic LLVM line is
                # compared after the original ModuleID alone is removed.
                def module_content(text):
                    return '\n'.join(text.splitlines()[1:])
                assert module_content(selected.read_text())==module_content(old.read_text()),'control semanticLLVMchanged'
            bindings=json.loads((case/'host_llvm/source_binding.json').read_text())
            assert len(bindings['source_bindings']['routes'])==44
            assert len(bindings['whole_helper_guards'])==22
            assert bindings['all_original_device155_references_conserved']
            if not native:
                base.run([base.LLVM/'clang',*flags,'-c',selected,'-o',case/'model.o'])
                if arm=='control':
                    assert base.sha(case/'model.o')==base.sha(CONTROL/'selected/target/model.o'),'controlmodelobjectchanged'
            else:
                host=case/'host'
                host.mkdir()
                base.run([base.LLVM/'clang','-O2','-fPIC','-c',selected,'-o',host/'model.o'])
                base.run(['cc','-O3','-march=native','-fPIC','-shared',host/'model.o',BASE_NATIVE/'reference.c',BASE_NATIVE/'device/device_catalog_shim.c',FROZEN/'merlin/runtime/abi/mlir_runtime.c','-lm','-o',host/'model.so'])
                guards[arm]=bindings['whole_helper_guards']
    # All22 current captured complete compound inputs plus full original model.
    inputs=resolve_forward_args(BUNDLE)
    plan=quant_hoist.read_plan(base.B)
    assert len(plan)==155
    values=quant_hoist.read_values(base.B)
    inputs.extend(np.ascontiguousarray(values[item.key]) for item in plan)
    original=np.load(BASE_NATIVE/'output.npy')
    golden=np.load(BUNDLE/'golden.npy')
    events=[]
    for arm in ('control','selected'):
        host=OUT/arm/'native/host'
        output=np.empty(original.shape,np.float32)
        model=HostModel.load(str(host/'model.so'))
        model([(a.ctypes.data,a.shape) for a in [*inputs,output]])
        assert np.array_equal(output.view(np.uint32),original.view(np.uint32)),arm
        assert np.allclose(output,golden,atol=.03125,rtol=.02),arm
        np.save(host/'output.npy',output)
        events.append({'arm':arm,'all256000_original_words_exact':True,'torch_gate':True,'outputs':output.size,'native_so_sha256':base.sha(host/'model.so')})
    base.save(OUT/'qualification.json',{'schema':'normal_closed_integer_observer_native_preparation_v1','status':'pass','events':events,'all155devicebindings':True,'control_model_object_byteexact2062':True,'stock_cycles':'UNKNOWN','whole_candidate_ELF':'not built pending completeM8cost','pins':base.pin([SOURCE,typed,Path(__file__),*OUT.rglob('*')]),'commands':base.commands,'token_usage_available':False})
    print('NORMAL_INTEGER_RESULT_WHOLE_NATIVE_PASS',flush=True)


if __name__=='__main__':
    main()
