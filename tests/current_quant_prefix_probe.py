"""One source-bound exact quantization/padding experiment, no production selector."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, TensorType, UnitAttr, f32, i8
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, BlockArgument, Region

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.bounded_rne_maps import prove_scalar_bounded_rne, schedule_bounded_rne_maps
from merlin.llvmlower.bounded_rne_packet_llvm import rewrite_packet_helpers
from merlin.llvmlower.bounded_rne_interval import (
    localize_bounded_rne_observer, validate_localized_rne_observer,
    build_integer_observation_table, emit_integer_observation_lookup,
    replace_integer_observation_map,
)
from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.late_quant_rne import rewrite
from merlin.xdsl_dialects._common import text
from mlir_oot.frontend.parse import parse_module

WORK = Path(__file__).resolve().parents[1] / 'out/current_quant_prefix_v3'
SOURCE = Path('/scratch/agustin/tmp/gemmini-host-rne-eight-20261006/out/host_rne_eight/whole8/build_direct/device_catalog/catalog_source.mlir')
INPUT = Path('/scratch/agustin/tmp/gemmini-host-rne-eight-20261006/out/host_rne_eight/fixture_source_v2/input.bin')
EXPECTED = INPUT.with_name('expected.bin')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
EFFECTS = IntervalEffectContract(True, True, True, True, True)
BITS = 18
BUDGET = 524288


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def run(argv, stem, *, cwd=None):
    (WORK/(stem+'.argv.json')).write_text(json.dumps(list(map(str,argv)),indent=2)+'\n')
    with (WORK/(stem+'.log')).open('w') as log:
        completed = subprocess.run(list(map(str,argv)), stdout=log, stderr=subprocess.STDOUT, cwd=cwd, timeout=600)
    if completed.returncode:
        raise ValueError(f'{stem} failed; see retained log')


def extract():
    module = parse_module(SOURCE.read_text())
    target = [op for op in module.walk() if isinstance(op,GenericOp)
              and len(op.results)==1 and op.results[0].type==TensorType(i8,[1,224,224,3])
              and prove_scalar_bounded_rne(op.body.block) is not None and op.results[0].uses]
    if len(target)!=1: raise ValueError('unique live typed source quantization required')
    target = target[0]
    uses = tuple(target.results[0].uses)
    if len(uses)!=1 or uses[0].operation.name!='tensor.insert_slice':
        raise ValueError('one complete source padding consumer required')
    pad=uses[0].operation
    block=target.parent_block(); required={pad}; free=set(); pending=list(pad.operands)
    while pending:
        value=pending.pop()
        if isinstance(value,BlockArgument):
            if value.owner is not block: raise ValueError('unknown free scalar input')
            free.add(value);continue
        owner=value.owner
        if owner.parent_block() is not block: raise ValueError('unknown nested source input')
        if owner in required:continue
        scalar_constant=(isinstance(owner,GenericOp) and len(owner.inputs)==1
                         and owner.inputs[0].type==TensorType(f32,[]) and not owner.iterator_types)
        if owner is not target and not scalar_constant and owner.name not in (
            'arith.constant','tensor.splat','tensor.empty','linalg.fill','linalg.transpose','tensor.insert_slice'):
            raise ValueError('unproved source producer: '+owner.name)
        required.add(owner);pending.extend(owner.operands)
    if len(free)!=1:raise ValueError('one immutable source input required')
    argument=next(iter(free))
    if argument.type!=TensorType(f32,[1,3,224,224]):raise ValueError('source tensor shape/dtype differs')
    body=Block(arg_types=[argument.type]);mapping={argument:body.args[0]}
    for op in block.ops:
        if op in required:body.add_op(op.clone(mapping))
    body.add_op(func.ReturnOp(mapping[pad.results[0]]))
    forward=func.FuncOp('forward',([argument.type],[pad.results[0].type]),Region(body))
    forward.attributes['llvm.emit_c_interface']=UnitAttr()
    forward.attributes.update(block.parent_op().attributes)
    fixture=ModuleOp([forward]);fixture.attributes.update(module.attributes);fixture.verify()
    return fixture,argument.index,text(target,generic=True),text(pad,generic=True)


def main():
    WORK.mkdir(parents=True,exist_ok=False)
    assert pin(SOURCE)['sha256']=='dcd3cbdb367828da2c47de2ffe6f570dc3afc4f4fc0b2c868b6c8d98dd2bb474'
    module,index,quant_source,pad_source=extract()
    (WORK/'source.mlir').write_text(text(module,generic=True))
    control=module.clone()
    routes=schedule_bounded_rne_maps(control,lanes=8)
    if len(routes)!=1:raise ValueError('current eight-lane control must close')
    (WORK/'control.mlir').write_text(text(control,generic=True))
    run([LLVM/'mlir-opt',WORK/'source.mlir','--canonicalize','--linalg-inline-scalar-operands','--canonicalize','--mlir-print-op-generic','-o',WORK/'constant_folded.mlir'],'upstream_constant_fold')
    candidate=parse_module((WORK/'constant_folded.mlir').read_text())
    targets=[op for op in candidate.walk() if isinstance(op,GenericOp) and len(op.results)==1 and op.results[0].type==TensorType(i8,[1,224,224,3])]
    if len(targets)!=1:raise ValueError('unique current typed constant map required')
    target=targets[0]
    observer=localize_bounded_rne_observer(target.body.block,target.body.block.args[0],symbol='original_scalar',effects=EFFECTS)
    table=build_integer_observation_table(observer.proof,effects=EFFECTS,leading_bits=BITS,max_table_bytes=BUDGET)
    validate_localized_rne_observer(observer)
    (WORK/'table.bin').write_bytes(table.data)
    report=replace_integer_observation_map(candidate,target,observer,lookup_symbol='exact_lookup',rounding_symbol='runtime_rne_admission')
    (WORK/'candidate.mlir').write_text(text(candidate,generic=True))
    cells=np.frombuffer(table.data,np.uint8).reshape(-1,2)
    data=','.join('{%d,%d}'%tuple(row) for row in cells)
    lookup=emit_integer_observation_lookup(table_name='observation_cells',source_name='original_scalar',lookup_name='exact_lookup',leading_bits=BITS)
    lookup+='\nconst unsigned char observation_cells[%d][2]={%s};\n'%(1<<BITS,data)
    native_lookup=lookup+'\n#include <fenv.h>\nint runtime_rne_admission(void){return fegetround()==FE_TONEAREST;}\n'
    target_lookup=lookup+'\nint runtime_rne_admission(void){unsigned long frm;__asm__ volatile("csrr %0,frm":"=r"(frm)::"memory");return frm==0;}\n'
    (WORK/'lookup_native.c').write_text(native_lookup);(WORK/'lookup_target.c').write_text(target_lookup)
    actual=np.frombuffer(INPUT.read_bytes(),np.float32).reshape(1,3,224,224)
    expected=np.zeros((1,230,230,3),np.int8)
    expected[:,3:227,3:227,:]=np.frombuffer(EXPECTED.read_bytes(),np.int8).reshape(1,224,224,3)
    (WORK/'input.bin').write_bytes(actual.tobytes());(WORK/'expected.bin').write_bytes(expected.tobytes())
    arm_records={}
    for name,graph in [('control',control),('candidate',candidate)]:
        arm=WORK/name;arm.mkdir()
        llvm=lower_to_llvm_ir(text(graph,generic=True),workdir=arm/'lower',vectorize=False)
        (arm/'native.ll').write_text(llvm)
        legalized,proof=(rewrite_packet_helpers(llvm,host_isa='rv64gc',max_lanes=8) if name=='control' else rewrite(llvm,host_isa='rv64gc',combine_clamp=True))
        (arm/'target.ll').write_text(legalized)
        (arm/'legalization.json').write_text(json.dumps(proof,indent=2)+'\n')
        run([LLVM/'clang','-O2','-fPIC','-c',arm/'native.ll','-o',arm/'native.o'],name+'_native_compile')
        run([LLVM/'clang','--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-c',arm/'target.ll','-o',arm/'kernel.o'],name+'_target_compile')
        native_extra=[]
        if name=='candidate':
            run(['cc','-O2','-fPIC','-fno-fast-math','-ffp-contract=off','-c',WORK/'lookup_native.c','-o',arm/'lookup_native.o'],name+'_lookup_native_compile')
            run([LLVM/'clang','--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off','-c',WORK/'lookup_target.c','-o',arm/'lookup.o'],name+'_lookup_target_compile')
            native_extra=[arm/'lookup_native.o']
        run(['cc','-O2','-fPIC','-shared',arm/'native.o',*native_extra,mlir_runtime_c(),'-lm','-o',arm/'native.so'],name+'_native_link')
        out=np.full(expected.shape,-91,np.int8);before=actual.view(np.uint32).copy()
        HostModel.load(str(arm/'native.so'))([(actual.ctypes.data,actual.shape),(out.ctypes.data,out.shape)])
        if not np.array_equal(out,expected) or not np.array_equal(actual.view(np.uint32),before):
            raise ValueError(name+' native source/layout/output mismatch')
        (arm/'native_output.bin').write_bytes(out.tobytes())
        arm_records[name]={'native_exact':True,'output':pin(arm/'native_output.bin'),'object':pin(arm/'kernel.o'),'llvm':pin(arm/'target.ll'),'legalization':proof}
    words=actual.view(np.uint32).reshape(-1)
    certified=cells[words>>(32-BITS),1].astype(bool)
    if not np.array_equal(cells[words[certified]>>(32-BITS),0],np.frombuffer(EXPECTED.read_bytes(),np.uint8).reshape(1,224,224,3).transpose(0,3,1,2).reshape(-1)[certified]):
        raise ValueError('independent original captured quantized byte check failed')
    record={'schema':'source_integer_quant_prefix_prelabel_v1','source':pin(SOURCE),'source_argument_index':index,
            'source_quantization_sha256':hashlib.sha256(quant_source.encode()).hexdigest(),'source_padding_sha256':hashlib.sha256(pad_source.encode()).hexdigest(),
            'expression_sha256':table.expression_sha256,'bounds':table.bounds,'partition_bits':BITS,'table_bytes':len(table.data),
            'certified_cells':table.certified_cells,'zero_cells':table.zero_cells,'saturated_cells':table.saturated_cells,
            'original_fixture_certified':int(certified.sum()),'original_fixture_fallback':int((~certified).sum()),
            'elements':actual.size,'output_elements':expected.size,'table':pin(WORK/'table.bin'),'input':pin(WORK/'input.bin'),'expected':pin(WORK/'expected.bin'),
            'source_binding':report,'arms':arm_records,'scope':'complete source quantization, transpose, padding/initialization and ranked output publication',
            'hypothesis':'certified integer cells can bypass original multiply/clamp/convert; immutable table requests and misses/fallback may outweigh arithmetic savings',
            'model_status':'UNPRICED: existing FP/service domains do not price table gather, branch, allocator and fallback mixture',
            'numeric_contract':'original integer observation exact; no floating escape; RNE/gradual/nontrapping and flags/signed-zero unobservable explicit; unsupported cells/modes use original scalar callback',
            'producer':pin(__file__)}
    (WORK/'prelabel.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['elements','output_elements','certified_cells','original_fixture_certified','original_fixture_fallback']}),flush=True)


if __name__=='__main__':main()
