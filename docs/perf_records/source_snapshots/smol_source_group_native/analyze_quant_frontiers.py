"""Validate live typed producer-to-integer observations for the full source."""
from pathlib import Path
import hashlib,json
from xdsl.dialects import arith,func,tensor
from xdsl.dialects.builtin import i8
from xdsl.dialects.linalg.ops import GenericOp
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_consumer_frontier import analyze_quantized_consumer_frontier,validate_quantized_consumer_frontier,quantized_consumer_semantic_sha256
w=Path(__file__).resolve().parent;path=w/'bounded_native/build/device_prepared/source_control/source_groups.mlir'
module=parse_mlir_text(path.read_text());function=next(op for op in module.body.block.ops if isinstance(op,func.FuncOp) and op.sym_name.data=='forward');ops=list(function.body.block.ops);positions={op:i for i,op in enumerate(ops)}
calls=[op for op in ops if isinstance(op,func.CallOp) and 'merlin.source_group_binding_sha256' in op.attributes]
records=[]
# Explicit source instance order is receipt binding, never provider policy.
for index in range(0,len(calls),4):
    producers=tuple(call.results[0] for call in calls[index:index+4]);last=calls[index+3]
    end=positions[calls[index+4]] if index+4<len(calls) else len(ops)
    found=None;refusals=[]
    for op in ops[positions[last]+1:end]:
        if isinstance(op,GenericOp) and op.results[0].type.element_type==i8 and any(isinstance(child,arith.FPToSIOp) for child in op.body.walk()):
            try:found=analyze_quantized_consumer_frontier(op.results[0],source_values=producers)
            except ValueError as error:refusals.append(str(error));continue
            break
    assert found is not None,(index,refusals)
    validate_quantized_consumer_frontier(found)
    assert found.source_uses_closed
    records.append(dict(first_source_instance=index,source_instances=4,source_producer_shapes=[list(v.type.get_shape()) for v in producers],consumer_semantic_sha256=quantized_consumer_semantic_sha256(found),operations=len(found.operations),observed_types=[str(v.type) for v in found.observation_outputs],floating_escape_types=[str(v.type) for v in found.floating_escapes],unquantized_source_escapes=len(found.unquantized_source_escapes),assembly_coordinates=[dict(source_ordinal=producers.index(v),offsets=list(o),sizes=list(n)) for v,o,n in found.assembly_boxes],coordinate_operation_names=[op.name for op in found.operations if op.name.startswith('tensor.') or op.name=='linalg.transpose'],scope='Live original scalar DAG and exact coordinate/use closure only; external numeric certificate remains required'))
    print('QUANT_FRONTIER',index,len(found.operations),[str(v.type) for v in found.observation_outputs],flush=True)
r=dict(schema='whole_source_integer_observation_frontier_v1',prepared_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_group_calls=len(calls),quantized_consumer_frontiers=len(records),covered_source_groups=sum(row['source_instances'] for row in records),unquantized_bf16_escapes=sum(row['unquantized_source_escapes'] for row in records),records=records,numerical_certificate=False,compiler_mutation=False,target_admission=False,token_usage_available=False)
(w/'quant_frontier/full_source_closure_semantic.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True);print('SEMANTIC_FINGERPRINTS',sorted({row['consumer_semantic_sha256'] for row in records}),flush=True)
