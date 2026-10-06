"""Read actual source DAG; extract unchanged first quantized frontier oracle."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from xdsl.dialects import arith,func,tensor
from xdsl.dialects.builtin import ModuleOp,TensorType,bf16,i8
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block,Region,OpResult
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.abi import HostModel
w=Path(__file__).resolve().parent;out=w/'quant_frontier';out.mkdir(exist_ok=True)
source=w/'bounded_native/build/device_prepared/source_control/source_groups.mlir'
module=parse_mlir_text(source.read_text());function=next(op for op in module.body.block.ops if isinstance(op,func.FuncOp) and op.sym_name.data=='forward');block=function.body.block
ops=list(block.ops);positions={op:i for i,op in enumerate(ops)}
calls=[op for op in ops if isinstance(op,func.CallOp) and 'merlin.source_group_binding_sha256' in op.attributes]
assert len(calls)==48
# Follow the original use chain, not source names or an invented layout.
concats=[];values=[]
for call in calls[:4]:
    uses=list(call.results[0].uses)
    assert len(uses)==1 and uses[0].operation.name=='tensor.insert_slice' and uses[0].index==0
    concats.append(uses[0].operation)
    values.append(call.results[0])
assert all(nextop.operands[1] is previous.results[0] for previous,nextop in zip(concats,concats[1:]))
boundary=concats[-1].results[0]
assert boundary.type==TensorType(bf16,[1,12,1024,64])
# Backward DAG closure for each candidate source quantization cast.
def dependencies(value,seen=None):
    seen=set() if seen is None else seen
    if value in seen:return set()
    seen.add(value)
    if value is boundary:return {boundary}
    if not isinstance(value,OpResult):return {value}
    result={value}
    for operand in value.owner.operands:result.update(dependencies(operand,seen))
    return result
candidates=[]
for operation in ops[positions[concats[-1]]+1:]:
    if (isinstance(operation,GenericOp) and operation.results[0].type.element_type==i8
        and any(isinstance(child,arith.FPToSIOp) for child in operation.body.walk())
        and boundary in dependencies(operation.results[0])):candidates.append(operation)
assert candidates
quant=candidates[0];qvalue=quant.results[0]
needed=set();externals=set()
def need(value):
    if value is boundary:return
    if not isinstance(value,OpResult):externals.add(value);return
    op=value.owner
    if op in needed:return
    if op.parent is not block:raise ValueError('nonlocal source producer')
    if positions[op]<=positions[concats[-1]] and op.name not in ('arith.constant','tensor.empty','tensor.splat'):
        raise ValueError('unexpected source dependency before attention endpoint')
    needed.add(op)
    for v in op.operands:need(v)
need(qvalue)
assert not externals
# Retain every non-i8 value that escapes this exact quantization closure.
escapes=[]
for operation in sorted(needed,key=positions.__getitem__):
    for result in operation.results:
        if result is qvalue:continue
        external_uses=[use for use in result.uses if use.operation.parent is block and use.operation not in needed]
        # Empty storage or arithmetic constants are not an attention data escape.
        if external_uses and boundary in dependencies(result):escapes.append(result)
assert escapes
body=Block(arg_types=[boundary.type]);mapping={boundary:body.args[0]}
for operation in sorted(needed,key=positions.__getitem__):
    clone=operation.clone(value_mapper=mapping);body.add_op(clone)
    mapping.update(zip(operation.results,clone.results))
results=[qvalue,*escapes]
body.add_op(func.ReturnOp(*[mapping[v] for v in results]))
f=func.FuncOp('source_quant_frontier',([boundary.type],[v.type for v in results]),Region(body))
f.attributes.update(function.attributes);f.attributes['llvm.emit_c_interface']=__import__('xdsl.dialects.builtin',fromlist=['UnitAttr']).UnitAttr()
extracted=ModuleOp([f]);extracted.verify();ir=text(extracted);(out/'source_quant_frontier.mlir').write_text(ir)
receipt=dict(schema='actual_source_quantized_frontier_v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_groups=48,first_rows_groups=4,source_quant_operations=len(needed),input_type=str(boundary.type),output_types=[str(v.type) for v in results],extra_live_escape_types=[str(v.type) for v in escapes],source_operation_names=[op.name for op in sorted(needed,key=positions.__getitem__)],frontier_source_sha256=hashlib.sha256(ir.encode()).hexdigest(),extra_live_uses=[[use.operation.name for use in v.uses if use.operation.parent is block and use.operation not in needed] for v in escapes],token_usage_available=False)
(out/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
llvm=lower_to_llvm_ir(ir,workdir=out/'lower');(out/'frontier.ll').write_text(llvm)
clang='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/clang'
subprocess.run([clang,'-O2','-march=native','-ffp-contract=off','-fPIC','-c',str(out/'frontier.ll'),'-o',str(out/'frontier.o')],check=True)
provider=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/host_outlined/model.so')
subprocess.run(['cc','-shared',str(out/'frontier.o'),str(provider),'-lm','-o',str(out/'frontier.so')],check=True)
run=HostModel.load(str(out/'frontier.so'),name='source_quant_frontier')
outputs=[]
for directory in ['exact_native_control','center_native_screen']:
    data=np.concatenate([np.load(w/directory/f'group_{i}_output.npy') for i in range(4)],axis=2)
    output=[]
    for value in results:
        dtype=np.int8 if value.type.element_type==i8 else np.uint16
        assert value.type.element_type in (i8,bf16)
        output.append(np.full(value.type.get_shape(),17,dtype))
    run([(data.ctypes.data,data.shape)]+[(v.ctypes.data,v.shape) for v in output])
    outputs.append(output)
    for i,value in enumerate(output):np.save(out/(directory+'_output_'+str(i)+'.npy'),value)
comparison=[]
for i,(exact,center) in enumerate(zip(*outputs)):
    comparison.append(dict(index=i,type=str(results[i].type),elements=int(exact.size),changed_words=int(np.count_nonzero(exact!=center)),changed_rows=int(np.count_nonzero(np.any(exact!=center,axis=-1))),max_integer_difference=int(np.max(np.abs(exact.astype(np.int32)-center.astype(np.int32))))))
receipt['same_original_first4_inputs']=True
receipt['first4_comparison']=comparison
receipt['llvm_sha256']=hashlib.sha256(llvm.encode()).hexdigest()
receipt['library_sha256']=hashlib.sha256((out/'frontier.so').read_bytes()).hexdigest()
(out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
