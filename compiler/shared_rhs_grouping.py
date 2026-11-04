"""Default-off frontend adapter for checked invocation-local row packing.

The original source and every logical input/output retain their exact identity.
Compiler-created allocations derive their shape and lifetime from the shared
constructive preparation checker. Existing lowering emits all concat/slice bytes.
"""
from copy import deepcopy
from hashlib import sha256
from merlin.transforms.input_row_packing import plan_shared_rhs_grouping,_text


def setting():
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    return knob('shared_rhs_row_packing',default='off',values=('off','region_local'),root=Path(__file__).parent)


def lower_owned_source(source,*,max_members=8,max_rows=16):
    """Consume checked candidate but bind emission to the untouched original.

    The complete original pure region becomes one task. This does not authorize
    crossing arbitrary effects: packing checks the actual closed tensor block,
    row batching checks every scalar recurrence and exact live result slice.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from linalg_lowering import lower_linalg
    from source_attribution import prepare_source_bindings
    original=lower_linalg(source);prepare_source_bindings(source,original)
    plan=plan_shared_rhs_grouping(parse_mlir_text(source),max_members=max_members,max_rows=max_rows)
    if plan is None:return None
    plan.revalidate();candidate=lower_linalg(_text(plan.candidate));plan.revalidate()
    original_params=original['params'];params=candidate['params'];candidate_bindings=deepcopy(params['tensor_source_map']);owned=list(range(original_params['source_op_count']))
    anchor=next(row['source_primary_op_index'] for row in (c.get('attributes',{}) for c in original['commands']) if 'source_primary_op_index' in row)
    original_bindings=deepcopy(original_params['tensor_source_map'])
    for key in ('source_sha256','source_op_count','source_ops','source_direct_ops','source_input_names','source_output_names','source_return_op_index','source_level','source_entry'):
        if key in original_params:params[key]=deepcopy(original_params[key])
    # Original ABI inputs/outputs keep their original identities and ordering.
    # Compiler-created packed/batched buffers are explicit private temporaries.
    for name,origin in original_bindings.items():
        if name not in candidate['tensors'] or candidate['tensors'][name]!=original['tensors'][name]:
            raise ValueError('row batching changed an original materialized source binding')
    params['tensor_source_map']=original_bindings
    temps=[{'tensor':name,'source_op_index':anchor,'source_result_index':0,'purpose':'checked region-local shared RHS input packing/batched output'}
        for name in candidate['tensors'] if name not in original_bindings]
    from math import prod
    allocations=[{'tensor':row['tensor'],'candidate_source':candidate_bindings[row['tensor']],
        'shape':candidate['tensors'][row['tensor']]['shape'],'dtype':candidate['tensors'][row['tensor']]['dtype'],
        'storage_bytes':prod(candidate['tensors'][row['tensor']]['shape'])*(1 if candidate['tensors'][row['tensor']]['dtype']=='i8' else 4),
        'lifetime_task_index':0,'allocation':'distinct_invocation_scratch'} for row in temps]
    params['source_preparation_storage']={'schema':'merlin.input_row_packing_storage.v1',
        'original_source':source,'candidate_source':_text(plan.candidate),'max_members':max_members,'max_rows':max_rows,
        'candidate_tensor_sources':candidate_bindings,'allocations':allocations}
    params['compiler_temporaries']=temps
    for command in candidate['commands']:
        attrs=command.setdefault('attributes',{});attrs.update(source_task_index=0,source_op_indices=owned,source_primary_op_index=anchor)
    for transport in params.get('source_transport_tasks',[]):
        transport.update(source_task_index=0,source_op_indices=owned,source_primary_op_index=anchor)
    params['source_task_definitions']=[{'task_index':0,'kind':'source_tensor_computation','source_op_indices':owned,'command_indices':list(range(len(candidate['commands']))),
        'reads':list(original_params['source_input_names']),'writes':list(original_params['source_output_names'])+[x['tensor'] for x in temps]}]
    params.pop('command_source_attribution',None)
    params['shared_rhs_row_packing']=plan.record();params['compiler_source_sha256']=sha256(source.encode()).hexdigest()
    prepare_source_bindings(source,candidate);plan.revalidate()
    return candidate,plan
