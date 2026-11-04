"""Explicit multi-M complete-K panels with selected ordering and real drains.

The shared group planner supplies disjoint LHS/ACC slots. The selected epoch
law supplies RAW/WAR/WAW, result visibility, adjacent fresh descriptor/compute,
ordered execution and finite-queue progress independently of that capacity.
Its authored issued-work cap is rebound to exact source/group work; its flags,
identities and finite pending capacity are unchanged. No STAY is emitted.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group
from merlin.transforms.pass_exploration import PassFamily,generate_candidates
from merlin.transforms.pipeline_policy import PipelinePolicy,PolicyFacts,RegisteredPass,execute_policy
from merlin.transforms.pipeline_state import PassSpec,PipelineContext,PipelineSession

from . import isa


@dataclass(frozen=True)
class SpatialGroups:
    group: object
    epoch: object
    receipt: dict


def _reduction_order(prepared):
    order=prepared.attributes.get('conv_spatial_reduction_order',knob(
        'CONV_SPATIAL_REDUCTION_ORDER',default='km',values=('km','mk'),
        root=Path(__file__).resolve().parents[1]))
    if order not in ('km','mk'):
        raise ValueError('CONV_SPATIAL_REDUCTION_ORDER must be km or mk')
    return order


def plan(prepared,request):
    if request not in ('two','four'):raise ValueError('unknown spatial group request')
    reduction_order=_reduction_order(prepared)
    p=prepared.source.plan;w=p.window;panel=prepared.group.panel
    pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
    output='convolution_spatial_group';level='counted_convolution'
    def bind(options):
        blocks=options['max_row_blocks']
        return RegisteredPass(PassSpec('spatial_group_'+str(blocks),0,produces=(output,),
            mutates_ir=False,ir_level=level),lambda session:{output:
                plan_convolution_output_group(panel,pixels,max_row_blocks=blocks)})
    family=PassFamily('complete_k_spatial_groups',bind,(('max_row_blocks',(1,2,4)),))
    space=generate_candidates((family,),output=output,max_candidates=3)
    bound={'two':2,'four':4}[request]
    variant=next(v for v in space.variants
        if space.bindings[v.passes[0]]['knobs']['max_row_blocks']==bound)
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources=prepared.epoch.hardware_identity)
    session=PipelineSession(prepared.source,context=context,ir_level=level,
        fingerprint=lambda source:source.identity,verify_ir=lambda source:source.__post_init__())
    facts=PolicyFacts(session.stamp(),shapes=(w.input_shape,),
        resources=(('spad_rows',isa.SPAD_ROWS),('acc_rows',isa.ACC_ROWS)))
    result=execute_policy(session,PipelinePolicy((variant,),variant.name),space.registry,facts,output=output)
    group=result.artifact;group.__post_init__()
    if group.row_blocks<=1:return None
    epoch=prepared.epoch
    if epoch.source_identity!=prepared.source.identity:raise ValueError('spatial group changed epoch source')
    try:epoch.require(2) # selected ordering/progress premises, before extending work
    except ValueError:return None
    work=panel.reduction_tiles*panel.column_blocks*group.row_blocks
    if work>=2**31:return None
    try:
        grouped_epoch=replace(epoch,max_compute_pairs=max(epoch.max_compute_pairs,work))
        grouped_epoch.require(max(2,work))
    except ValueError:return None
    receipt=dict(requested_max_row_blocks=bound,actual_row_blocks=group.row_blocks,
        spatial_reduction_order=reduction_order,
        source_sha256=prepared.source.identity,group=asdict(group),
        pass_family_bindings=dict(space.bindings),pipeline=result.receipt,
        selected_ordering=prepared.ordering,
        original_authored_issued_cap=epoch.max_compute_pairs,
        grouped_authored_issued_cap=grouped_epoch.max_compute_pairs,
        max_group_compute_pairs=work,pending_compute_pairs=epoch.pending_compute_pairs,
        budget_basis='authored complete source K*N*M issue envelope; no added hardware queue capacity',
        law_flags_and_identities_preserved=True,
        lhs_ranges=[[group.lhs_row(m),group.lhs_row(m)+isa.DIM]for m in range(group.row_blocks)],
        accumulator_ranges=[[group.accumulator_row(m,0),
            group.accumulator_row(m,panel.column_blocks-1)+isa.DIM]for m in range(group.row_blocks)],
        readout='original terminal encoder, before-config/store/output/final drains retained',
        order='each original output retains ky/kx/channel order; every M/N pair gets fresh PRELOAD')
    return SpatialGroups(group,grouped_epoch,receipt)


def emit(prepared,proposal,builder,lower,complete,*,hoist_gather=False,interior_runs=None,store_epochs=False):
    """Consume admitted disjoint slots; one complete K/N/M producer epoch."""
    from .conv_spatial_panels import (_gather_coordinates,_gather_bands,
        _emit_gather_bands,_input_fragments,_predicate_and)
    proposal.group.__post_init__();prepared.gather.__post_init__()
    if prepared.group!=proposal.group or prepared.epoch!=proposal.epoch:
        raise ValueError('spatial group emission differs from admitted storage/law')
    p=prepared.source.plan;w=p.window;group=prepared.group;panel=group.panel
    reduction_order=_reduction_order(prepared)
    if proposal.receipt['spatial_reduction_order']!=reduction_order:
        raise ValueError('spatial reduction order differs from admitted proposal')
    interior_branch=prepared.attributes.get('conv_group_interior_branch',knob(
        'CONV_GROUP_INTERIOR_BRANCH',default='per_tile',values=('per_tile','group_once'),
        root=Path(__file__).resolve().parents[1]))
    if interior_branch not in ('per_tile','group_once'):
        raise ValueError('CONV_GROUP_INTERIOR_BRANCH must be per_tile or group_once')
    group_interior=interior_branch=='group_once'and interior_runs is not None and reduction_order=='km'
    batch,_,_,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
    pixels=batch*oh*ow;chunks=panel.channel_tiles;max_m=group.row_blocks
    def phase(column,widths):
        active=len(widths)
        def reduction(callback):
            def y(ky):
                def x(kx):
                    full,tail=divmod(ci,isa.DIM)
                    if full:builder.loop(0,full,1,lambda c:callback(ky,kx,c,isa.DIM))
                    if tail:callback(ky,kx,full,tail)
                builder.loop(0,kw,1,x)
            builder.loop(0,kh,1,y)
        def location(ky,kx,chunk):
            tap=builder.op('add',builder.op('mul',ky,kw),kx)
            slot=builder.op('add',builder.op('mul',tap,chunks),chunk)
            start=builder.op('add',builder.op('mul',tap,ci),builder.op('mul',chunk,isa.DIM))
            return slot,start
        def weights(ky,kx,chunk,depth):
            slot,start=location(ky,kx,chunk)
            for n,columns in enumerate(widths):
                lower('load_weight',dict(reduction_row=start,output_column=builder.op('add',column,n*isa.DIM),
                    local_row=builder.op('add',panel.rhs_base,
                        builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                    rows=depth,columns=columns))
        reduction(weights);complete('spatial_phase_rhs_ready',{})
        def spatial(first,extents):
            prepared.epoch.require(max(2,panel.reduction_tiles*active*len(extents)))
            tiles=[]
            for m,rows in enumerate(extents):
                origin=builder.op('add',first,m*isa.DIM)
                outputs=tuple(dict(output_row=origin,output_column=builder.op('add',column,n*isa.DIM),
                    rows=rows,columns=columns,
                    accumulator_row=group.accumulator_row(m,n,active_column_blocks=active))
                    for n,columns in enumerate(widths))
                if p.bias:
                    for out in outputs:lower('initialize_bias',out)
                    complete('spatial_initializer_ready',{})
                coordinates=_gather_coordinates(prepared,builder,origin,rows)if hoist_gather else None
                interior=interior_runs.guard(prepared,builder,origin,rows)if interior_runs is not None else None
                tiles.append((m,origin,rows,outputs,coordinates,interior))
            def products(ky,kx,chunk,depth,interior_mode='per_tile',members=None):
                if members is None:members=tiles
                channel=builder.op('mul',chunk,isa.DIM)
                # Every private LHS slot is loaded before this K epoch reads it.
                for m,origin,rows,outputs,coordinates,interior in members:
                    def transfer(fields):
                        lower('load_input',dict(fields,local_row=builder.op('add',fields['local_row'],group.lhs_row(m))))
                    def gather():
                        if coordinates is None:
                            _input_fragments(prepared,builder,origin,rows,ky,kx,channel,depth,transfer)
                        else:
                            bands=_gather_bands(prepared,builder,coordinates,ky,kx)
                            _emit_gather_bands(prepared,builder,bands,channel,depth,transfer)
                    if interior_mode=='gather'or interior is None:gather()
                    elif interior_mode=='direct':
                        transfer(interior_runs.transfer(
                            prepared,builder,interior[1],rows,ky,kx,channel,depth))
                    else:
                        predicate,base=interior
                        builder.branch(predicate,lambda:transfer(interior_runs.transfer(
                            prepared,builder,base,rows,ky,kx,channel,depth)),gather)
                slot,_=location(ky,kx,chunk)
                initial=_predicate_and(builder,builder.cmp('eq',ky,0),builder.cmp('eq',kx,0))
                initial=_predicate_and(builder,initial,builder.cmp('eq',chunk,0))
                for m,origin,rows,outputs,coordinates,interior in members:
                    for n,out in enumerate(outputs):
                        lower('compute',dict(out,lhs_row=group.lhs_row(m),rhs_row=builder.op('add',panel.rhs_base,
                            builder.op('mul',builder.op('add',builder.op('mul',slot,active),n),isa.DIM)),
                            depth=depth,first_reduction=initial,bias=p.bias,fresh_weights=True))
            if reduction_order=='mk':
                # Disjoint M ownership permits a complete original K traversal
                # before the next member. Each private LHS rewrite still follows
                # its prior consuming compute under the unchanged WAR law.
                for tile in tiles:
                    def member(mode):
                        reduction(lambda *args:products(*args,interior_mode=mode,members=(tile,)))
                    interior=tile[-1]
                    if interior is None:member('gather')
                    else:builder.branch(interior[0],lambda:member('direct'),lambda:member('gather'))
            elif group_interior:
                # Existing source endpoint guards prove each original M tile;
                # their conjunction admits direct loads for every live member.
                # One failed member retains gather for the complete group.
                predicate=tiles[0][-1][0]
                for tile in tiles[1:]:
                    predicate=_predicate_and(builder,predicate,tile[-1][0])
                builder.branch(predicate,
                    lambda:reduction(lambda *args:products(*args,interior_mode='direct')),
                    lambda:reduction(lambda *args:products(*args,interior_mode='gather')))
            else:reduction(products)
            complete('spatial_reduction_epoch_complete',dict(
                outputs=tuple(out for tile in tiles for out in tile[3]),
                first_output_row=first,first_output_column=column,row_extents=extents,
                column_widths=widths)if store_epochs else {})
            for m,origin,rows,outputs,coordinates,interior in tiles:
                for out in outputs:lower('store',out)
                if not store_epochs:complete('spatial_output_last_use',{})
            if store_epochs:complete('spatial_group_output_last_use',{})
        span=max_m*isa.DIM;full,tail=divmod(pixels,span)
        if full:builder.loop(0,full,1,lambda m:spatial(builder.op('mul',m,span),(isa.DIM,)*max_m))
        if tail:
            blocks,remainder=divmod(tail,isa.DIM)
            spatial(full*span,(isa.DIM,)*blocks+((remainder,)if remainder else ()))
        complete('spatial_phase_rhs_last_use',{})
    full_columns,last_columns=divmod(p.output_channels,isa.DIM)
    phases,tail=divmod(full_columns,panel.column_blocks)
    if phases:builder.loop(0,phases,1,lambda n:phase(
        builder.op('mul',n,panel.column_blocks*isa.DIM),(isa.DIM,)*panel.column_blocks))
    widths=(isa.DIM,)*tail+((last_columns,)if last_columns else ())
    if widths:phase(phases*panel.column_blocks*isa.DIM,widths)
    return dict(mechanism='complete_k_multi_m_spatial_groups',source_sha256=prepared.source.identity,
        original_output_hw=[oh,ow],spatial_groups=proposal.receipt,
        gather_hoisted=hoist_gather,interior_runs=interior_runs.receipt()if interior_runs else None,
        requested_group_interior_branch=interior_branch,group_interior_branch_applied=group_interior,
        spatial_reduction_order=reduction_order,
        spatial_product_traversal=(['M','kernel_y','kernel_x','input_channel','N']if reduction_order=='mk'
            else ['kernel_y','kernel_x','input_channel','M','N']),
        group_interior_branch_declined=('group_once applies only to km; mk uses independent per-M guards'
            if interior_branch=='group_once'and reduction_order=='mk'else None),
        group_interior_branch=('one conjunction of existing per-M endpoint guards outside K; '
            'two complete reduction bodies: all direct or all original gather'if group_interior else None),
        reduction_order=['kernel_y','kernel_x','input_channel'],
        compute_epoch=asdict(prepared.epoch),ordering=prepared.ordering,
        completion='RHS preparation; each initializer; grouped complete K/N/M; original per-M terminal readout/output last-use; phase last-use',
        fresh_preload_per_m=True,stay_emitted=False,physical_mapping_verified=False,target_execution=False,timing=None)
