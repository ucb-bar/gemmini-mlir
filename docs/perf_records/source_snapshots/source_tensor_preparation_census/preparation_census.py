"""Typed read-only preparation census of accepted original source bindings."""
import hashlib,json,math
from pathlib import Path

from xdsl.dialects import builtin,func
from xdsl.ir import Block
from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_bf16_group_binding import verify_group_module_coverage
from merlin.llvmlower.ordered_fma_rewrite import _match
from merlin.llvmlower.tensor_preparation_identity import (
    TensorPreparationRequest,canonical_tensor_read_view,
    find_tensor_preparation_opportunities,validate_tensor_preparation_opportunity,
)
from merlin.xdsl_dialects._common import text

WORK=Path(__file__).resolve().parent
source=WORK/'bounded_native/build/device_prepared/source_control/source_groups.mlir'
receipt=WORK/'bounded_native/build/device_prepared/source_control/source_group_calls.json'
proof=json.loads(receipt.read_text())
assert sha256_file(source)==proof['selected_sha256']=='1056ae3279d19fbfd82f3d9abee376fdf462ba7c4b3b734048c0d42cbfd461a8'
module=parse_mlir_text(source.read_text());verify_group_module_coverage(module,proof)
before=text(module,generic=True)
bindings={record['symbol']:record for record in proof['records']}
calls=[operation for operation in module.walk() if isinstance(operation,func.CallOp)
       and operation.callee.root_reference.data in bindings]
assert len(calls)==48
codec_header=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/numeric_capability_frozen/bf16_radix_pack.h')
format_contract=dict(domain='source BF16 immutable logical rows',layout='signed i8 plane-major row encoding',
    scalar_header_sha256=sha256_file(codec_header),radix_bits=7,digits=3,
    row_scale='runtime power-of-two source-derived RNE',prepared_provider_contract_installed=False)
functions={operation.sym_name.data:operation for operation in module.body.block.ops
           if isinstance(operation,func.FuncOp)}
domains={};format_contracts={}
for symbol in bindings:
    function=functions[symbol]
    uses={}
    for operation in function.walk():
        transposed=_match(operation)
        if transposed is None:continue
        for role,value in enumerate(operation.inputs):
            if value not in function.body.block.args:continue
            argument=list(function.body.block.args).index(value)
            rank=len(value.type.get_shape())
            signature=(tuple(range(rank-2)),rank-2 if role==0 or transposed else rank-1,
                       rank-1 if role==0 or transposed else rank-2,role==1)
            uses.setdefault(argument,set()).add(signature)
    domains[symbol]={}
    for argument,signatures in uses.items():
        assert len(signatures)==1,'one source argument has incompatible preparation domains'
        batch,row,vector,transpose=next(iter(signatures))
        contract=dict(format_contract,batch_axes=batch,row_axis=row,vector_axis=vector,
            reconstructed_f32_f64_layout='[batch,row,vector]',
            signed_planes_layout='[plane,batch,vector,row]' if transpose else '[plane,batch,row,vector]',
            source_replay_widening='retain original source f32 separately from reconstructed f32')
        pin=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        domains[symbol][argument]=pin;format_contracts[pin]=contract
requests=[];rows=[];ordinals={operation:index for index,operation in enumerate(module.walk())}
positions={call:index for index,call in enumerate(calls)}
for call in calls:
    for index,value in enumerate(call.operands):
        if isinstance(value.type,builtin.TensorType) and value.type.element_type==builtin.bf16:
            format_pin=domains[call.callee.root_reference.data][index]
            requests.append(TensorPreparationRequest(value,call,format_pin))
            view=canonical_tensor_read_view(value)
            rows.append(dict(call_index=positions[call],argument_index=index,
                root_operation_ordinal=ordinals[view.root.owner] if not isinstance(view.root.owner,Block) else None,
                offsets=view.offsets,sizes=view.sizes,strides=view.strides,
                logical_bf16_words=math.prod(view.sizes),identity=view.identity))
opportunities=find_tensor_preparation_opportunities(requests)
for opportunity in opportunities:validate_tensor_preparation_opportunity(opportunity)
assert text(module,generic=True)==before
unique={row['identity']:row['logical_bf16_words'] for row in rows}
reuse=[]
for opportunity in opportunities:
    view=opportunity.views[0]
    reuse.append(dict(root_operation_ordinal=ordinals[view.root.owner],offsets=view.offsets,
        sizes=view.sizes,strides=view.strides,logical_bf16_words=math.prod(view.sizes),
        source_uses=len(opportunity.views),distinct_view_ssa=len({v.value for v in opportunity.views}),
        consumer_call_indices=[positions[consumer] for consumer in opportunity.consumers],
        format_sha256=opportunity.format_sha256))
for row in rows:del row['identity']
result=dict(schema='source_tensor_preparation_census_v1',source_sha256=sha256_file(source),
    original_full_source_sha256='4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc',
    source_binding_receipt_sha256=sha256_file(receipt),source_call_count=len(calls),
    requests=len(rows),unique_logical_views=len(unique),equivalent_view_groups=len(reuse),
    logical_input_words_before=sum(row['logical_bf16_words'] for row in rows),
    logical_input_words_once_per_unique_view=sum(unique.values()),
    potential_repeated_logical_input_words=sum(row['logical_bf16_words'] for row in rows)-sum(unique.values()),
    format_contracts=format_contracts,opportunities=reuse,source_requests=rows,
    source_unchanged=True,token_usage_available=False,
    scope='Read-only compiler opportunity census. One read/preparation per source-call BF16 argument is a logical counter, not current emitted execution. Q reused inside two contractions, probability preparations, absolute/error plane formats, storage/copy/metadata work and physical provider costs are excluded. Format/lifetime/effect admission and actual performance remain pending.',
    actual_preparation_calls_saved=None,actual_device_calls_saved=None,actual_cycles_saved=None,
    implementation_pin=sha256_file(Path(__import__('merlin.llvmlower.tensor_preparation_identity',fromlist=['x']).__file__)),
    script_sha256=sha256_file(Path(__file__)))
output=WORK/'preparation_census.json';output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key not in ('opportunities','source_requests')},indent=2))
