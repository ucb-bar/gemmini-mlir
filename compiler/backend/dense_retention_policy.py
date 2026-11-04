"""Optional whole-RHS retention over current-source checked dense plans.

The resource class changes immutable operand reuse only. It grants no execution
or configuration premise and keeps the incumbent on a refused optional plan.
"""
from dataclasses import replace
from hashlib import sha256
import json

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext
from .dense_pipeline_policy import DensePipelineSource, pipeline_policy_session, LEVEL
from . import isa

OUTPUT='checked_dense_retention_schedule'


def execute_capacity_retention(m,k,n,bias=False,column_scales=False,*,
        geometry='wide',storage_split='half',incumbent=None):
    from .dense_pipeline import _retained_schedule,_retained_load_counts
    from .dense import _adapter
    source=DensePipelineSource(m,k,n,bias,column_scales,geometry,storage_split)
    resource_identity=sha256(json.dumps(dict(dim=isa.DIM,spad_rows=isa.SPAD_ROWS,
        acc_rows=isa.ACC_ROWS,geometry=geometry,storage_split=storage_split),sort_keys=True).encode()).hexdigest()
    session=pipeline_policy_session(source,context=PipelineContext(resources=resource_identity))
    # Adaptive supplies its actual checked retained fallback. This preserves
    # the selected source/resource plan rather than rebuilding a surrogate.
    if incumbent is None:
        incumbent=_retained_schedule(m,k,n,bias,column_scales,False,geometry,storage_split,'off','lhs')
    try:
        alternate=_retained_schedule(m,k,n,bias,column_scales,False,geometry,storage_split,'off','weight')
    except (ValueError,TimeoutError) as error:
        alternate=(None,'retained weight admission refused: '+str(error))
    candidates={'lhs':incumbent,'weight':alternate}
    identities={name:None if result is None else result.kernel.digest()
        for name,(result,_) in candidates.items()}
    adapter=_adapter()
    def validate():
        for name,(result,refusal) in candidates.items():
            if result is None:
                if not refusal:raise ValueError('retention candidate lost its original refusal')
                continue
            result.revalidate()
            if (refusal is not None or result.kernel.digest()!=identities[name]
                    or result.realization['retained_operand']!=name
                    or result.realization['panel_geometry']['selection']!=geometry):
                raise ValueError('retention candidate changed its checked role, source or resource bounds')
            expected=[('lhs',(m,k),'i8','read'),('rhs',(k,n),'i8','read')]
            if bias:expected.append(('bias',(n,),'i32','read'))
            expected.append(('output',(m,n),'i32','write'))
            if [(a.name,a.shape,a.dtype,a.access) for a in result.kernel.args]!=expected:
                raise ValueError('retention candidate changed the original source ABI')
    validate()
    lhs,weight=incumbent[0],alternate[0]
    counts={name:None if result is None else _retained_load_counts(result.kernel,adapter)
        for name,(result,_) in candidates.items()}
    tile_n=1 if column_scales else isa.DIM
    whole_rhs=False
    if lhs is not None and weight is not None:
        block=weight.realization['panel_geometry']['selected_block_tiles']
        whole_rhs=(block[2]==(n+tile_n-1)//tile_n and block[0]<(m+isa.DIM-1)//isa.DIM
            and counts['weight'][0]<=counts['lhs'][0] and counts['weight'][1]<counts['lhs'][1])
    facts=PolicyFacts(session.stamp(),shapes=((m,k),(k,n)),
        element_types=('i8','i8'),layouts=('row_major','row_major'),operations=frozenset({'dense'}),
        resources=(('whole_rhs_reuse',int(whole_rhs)),))
    policy=PipelinePolicy((
        PipelineVariant('weight',('dense_retention_weight',),(ResourcePredicate('whole_rhs_reuse',1),)),
        PipelineVariant('lhs',('dense_retention_lhs',))), 'lhs')
    def entry(name):
        def action(current):
            if current.module!=source:raise ValueError('retention registry cannot reuse another source')
            validate()
            return {OUTPUT:candidates[name]}
        key='dense_retention_'+name
        return key,RegisteredPass(PassSpec(key,0,produces=(OUTPUT,),mutates_ir=False,ir_level=LEVEL),action)
    selected=execute_policy(session,policy,dict(entry(name) for name in candidates),facts,output=OUTPUT)
    result,refusal=selected.artifact
    if result is not None:
        receipt=dict(schema='authored_dense_capacity_retention.v1',policy=selected.receipt,
            candidate_kernels=identities,actual_operand_loads=counts,
            candidate_refusals={name:value[1] for name,value in candidates.items()},
            whole_rhs_reuse=whole_rhs,
            evidence_basis='paired complete provisional native timing; exact numerical checks')
        result=replace(result,realization={**result.realization,'capacity_retention':receipt})
        result.revalidate()
    return result,refusal
