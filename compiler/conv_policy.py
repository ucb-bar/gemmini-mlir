"""Finite convolution alternatives; timing feedback owns any promotion.

Storage alternatives reuse loaded windows and capacity-proved resident RHS.
An independent opt-in completion choice consumes selected RTL dependency facts
through the shared library; its remaining epoch and output drains stay real.
"""
from pathlib import Path
import json
from hashlib import sha256
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource, ConvolutionScheduleChoice,
    ConvolutionRhsResidencyLaw, plan_convolution_schedule,
    plan_convolution_rhs_residency, emit_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionInputRetentionLaw, ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw,
)

VALUES=('synchronous','retained_input','resident_weights')
DECLARATION={
    'values':list(VALUES),
    'mechanism':'convolution input retention and invocation-local RHS residency',
    'note':'Synchronous incumbent; alternatives retain exact source reduction/readout. Paired numeric/cycle evidence required.',
}
EPOCH_VALUES=('off','load_issue','pairs16','pairs64')
EPOCH_DECLARATION={
    'values':list(EPOCH_VALUES),
    'mechanism':'fact-backed convolution load issue and bounded compute epochs',
    'note':'Off preserves synchronous incumbent. Actual shared-library epoch consumer removes only fact-ordered drains; compare complete-call numerics/cycles.',
}
ACCUMULATOR_INIT_VALUES=('overwrite','zero_accumulate','zero_single_slice')
ACCUMULATOR_INIT_DECLARATION={
    'values':list(ACCUMULATOR_INIT_VALUES),
    'mechanism':'explicit source-zero accumulator DMA initialization before real products',
    'note':'Overwrite incumbent; zero_accumulate initializes source zeros before real products. zero_single_slice selects it only without bias and with total source K at most the selected array edge. Existing bias initialization is preserved.',
}


def accumulator_init_selection():
    return knob('conv_accumulator_init',default='overwrite',values=ACCUMULATOR_INIT_VALUES,root=Path(__file__).parent)


def epoch_selection():
    return knob('conv_completion_epochs',default='off',values=EPOCH_VALUES,root=Path(__file__).parent)


def completion_laws(plan,source_identity,live_tiles,*,mode=None):
    """Bind original selected RAW/WAR/configuration/queue evidence, or decline."""
    mode=epoch_selection() if mode is None else mode
    if mode not in EPOCH_VALUES: raise ValueError('unknown convolution completion mode')
    if mode=='off': return None
    root=Path(__file__).resolve().parent
    path=next((p for p in (root/'command_ordering.json',root.parent/'COMMAND_ORDERING_SELECTION.json') if p.exists()),None)
    if path is None: return None
    selection=json.loads(path.read_text()); record=selection['ordering']
    digest=sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    facts_bytes=(root/'facts.json').read_bytes()
    if (record.get('status')!='derived' or digest!=selection.get('ordering_sha256')
        or sha256(facts_bytes).hexdigest()!=selection['base_facts']['sha256']):
        raise ValueError('convolution ordering selection disagrees with selected RTL facts')
    cross={(r['older'],r['younger'],r['space']):r for r in record['cross_queue']}
    table={}
    def pairs(name,relations):
        rows=[cross.get(relation) for relation in relations]
        okay=all(r and r.get('overlap')=='completion' and
            (r.get('tracked') is True or r.get('disjoint')=='completion') for r in rows)
        table[name]={'established':okay,'evidence':rows}
        return okay
    loads=('MVIN.spad_addr','MVIN2.spad_addr')
    readers=('COMPUTE_PRELOADED.a_spad','PRELOAD.weight_spad')
    raw=pairs('local_read_after_write',[(l,c,'scratchpad') for l in loads for c in readers])
    consumed=pairs('operand_consumption_retirement',[(c,l,'scratchpad') for l in loads for c in readers])
    war=pairs('readout_before_accumulator_reuse',[('MVOUT.acc_addr','PRELOAD.c_addr','accumulator')])
    visible=pairs('accumulator_result_visibility',[('PRELOAD.c_addr','MVOUT.acc_addr','accumulator')])
    initialized=not plan.bias or pairs('bias_before_accumulator_update',[('MVIN3.spad_addr','PRELOAD.c_addr','accumulator')])
    same=record['same_queue']; order=record['completion_order']
    ordered=order.get('execute',{}).get('order')=='in_order'
    config={int(code):queue for row in record['funct_queue'] for code,queue in row.get('by_code',{}).items()}
    captured=all(config.get(code)==queue and same.get(queue,{}).get('release') in ('issue','completion')
                 for code,queue in ((0,'execute'),(1,'load'),(2,'store')))
    classes=json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']
    entries={q:classes[q]['entries'] for q in ('load','execute','store')}
    finite=all(e.get('status')=='derived' and type(e.get('value')) is int and e['value']>0 for e in entries.values())
    finite=finite and entries['execute']['value']>=2
    paired=same.get('execute',{}).get('release') in ('issue','completion')
    for name,value,evidence in (('ordered_accumulator_updates',ordered,order.get('execute')),
        ('ordered_configuration_capture',captured,config),('finite_queue_progress',finite,entries),
        ('adjacent_descriptor_compute',paired,same.get('execute'))):
        table[name]={'established':value,'evidence':evidence}
    if not all((raw,consumed,war,visible,initialized,ordered,captured,finite,paired)): return None
    ids=('selected-gemmini-private-stores-v1','i8xi8-i32-source-readout-v1',source_identity,'selected-rtl-convolution-epochs:'+digest)
    issue=ConvolutionLoadIssueLaw(*ids,plan.row_tile+1+live_tiles*int(plan.bias),True,True,True,True,True,True)
    pairs_count={'load_issue':1,'pairs16':16,'pairs64':64}[mode]
    epoch=None if pairs_count==1 else ConvolutionComputeEpochLaw(*ids,pairs_count,
        raw,consumed and war,ordered and raw and consumed and initialized,visible,consumed,captured,
        paired,finite,True,True,pending_compute_pairs=max(1,entries['execute']['value']//2))
    return issue,epoch,pairs_count,table


def selected():
    return knob('conv_schedule',default=VALUES[0],values=VALUES,root=Path(__file__).parent)


def emit_alternative(plan,builder,lower,*,complete,issue=None,selection=None,pooling=False,completion_mode=None):
    """Emit the selected checked alternative, or return None for the incumbent.

    Shape admission limits compile expansion; it does not predict a timing
    winner. ACC ownership derives from actual row capacity. Readout scales must
    already be homogeneous in every original column tile. Pool observations
    use the separate private-patch adapter below.
    """
    selection=selected() if selection is None else selection
    completion_mode=epoch_selection() if completion_mode is None else completion_mode
    if selection not in VALUES: raise ValueError('unknown convolution alternative')
    if completion_mode not in EPOCH_VALUES: raise ValueError('unknown convolution completion mode')
    if (selection=='synchronous' and completion_mode=='off') or pooling: return None
    # This emitter statically enumerates original column partitions. Keep its
    # expansion bounded independently of spatial size and K iteration count.
    if plan.column_tile!=plan.primitive or (plan.output_channels+plan.column_tile-1)//plan.column_tile>32:
        return None
    intervals=tuple((col,min(plan.column_tile,plan.output_channels-col))
                    for col in range(0,plan.output_channels,plan.column_tile))
    source=ConvolutionExplorationSource(plan,intervals,plan.identity)
    ids=('selected-gemmini-private-stores-v1','i8xi8-i32-source-readout-v1',
         source.identity,'logical-disjoint-immutable-inputs-full-fence-v1')
    # Every distinct output column is observed once after full K; immutable
    # IFM/weights and private ACC intervals permit sharing the completed LHS.
    retention=ConvolutionInputRetentionLaw(*ids,max_live_tiles=len(intervals),
        input_preserved=True,independent_outputs=True,immutable_mapped_disjoint_buffers=True)
    laws=completion_laws(plan,source.identity,len(intervals),mode=completion_mode)
    if selection=='synchronous' and laws is None: return None
    if laws is not None and not callable(issue):
        raise ValueError('convolution epochs require a non-draining terminal effect encoder')
    issue_law,epoch_law,pair_count,evidence=laws if laws else (None,None,1,None)
    schedule=plan_convolution_schedule(source,ConvolutionScheduleChoice(
        live_tiles=1 if selection=='synchronous' else None,load_issue=laws is not None,compute_pairs=pair_count),
        retention_law=retention,issue_law=issue_law,epoch_law=epoch_law)
    rhs=None
    if selection=='resident_weights':
        # The encoder's weight transfer writes only the assigned RHS rows. The
        # private LHS range and every accumulator/readout preserve them; entry
        # and preparation completion are implemented by actual fences.
        rhs_law=ConvolutionRhsResidencyLaw(*ids,True,True,True,True,True)
        rhs=plan_convolution_rhs_residency(schedule,operand_rows=plan.operand_rows,law=rhs_law)
    emitted=emit_convolution_schedule(schedule,builder,issue if laws else lower,complete=complete,
                                     synchronous_completion=True,rhs=rhs)
    return dict(emitted,conv_schedule=selection,rhs_residency_applied=rhs is not None,
                source_sha256=source.identity,checked_schedule_sha256=schedule.identity,
                completion_epochs=completion_mode if laws else 'off',ordering_evidence=evidence,
                timing='unmeasured alternative; incumbent unchanged')


def pool_retained_tiles(plan,*,patch_rows,selection=None):
    """Reserve a complete pool patch per N tile, or decline without changes."""
    selection=selected() if selection is None else selection
    if selection=='synchronous' or plan.column_tile!=plan.primitive:
        return None
    count=(plan.output_channels+plan.column_tile-1)//plan.column_tile
    if count>32 or count*patch_rows>plan.accumulator_rows: return None
    from merlin.compile.scheduling.window_transfer import ConvolutionOutputTile
    return tuple(ConvolutionOutputTile(col,min(plan.column_tile,plan.output_channels-col),i*patch_rows)
        for i,col in enumerate(range(0,plan.output_channels,plan.column_tile)))


def emit_pool_retained_row(plan,builder,lower,*,tiles,patch_offset,output_row,rows,issue=None,complete=None,completion_mode=None):
    """Share a checked window; library owns selected primitive/epoch drains."""
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionOutputTile, emit_convolution_column_block,
    )
    shifted=tuple(ConvolutionOutputTile(t.output_column,t.columns,t.accumulator_row+patch_offset) for t in tiles)
    ids=('selected-gemmini-private-stores-v1','i8xi8-i32-source-readout-v1',
         plan.identity,'private-complete-pool-patches-full-fence-v1')
    law=ConvolutionInputRetentionLaw(*ids,max_live_tiles=len(tiles),input_preserved=True,
        independent_outputs=True,immutable_mapped_disjoint_buffers=True)
    laws=completion_laws(plan,plan.identity,len(tiles),mode=completion_mode)
    if laws and issue is not None and complete is not None:
        issue_law,epoch_law,pair_count,_=laws
        from merlin.compile.scheduling.window_transfer import (
            emit_convolution_load_issue_column_block, emit_convolution_compute_epoch_column_block,
        )
        args=dict(output_row=output_row,rows=rows,output_tiles=shifted,law=law,issue_law=issue_law)
        if epoch_law is not None:
            return emit_convolution_compute_epoch_column_block(plan,builder,issue,complete,
                **args,epoch_law=epoch_law,compute_pairs=pair_count)
        return emit_convolution_load_issue_column_block(plan,builder,issue,complete,**args)
    return emit_convolution_column_block(plan,builder,lower,output_row=output_row,rows=rows,
        output_tiles=shifted,law=law,synchronous_completion=True)


def emit_pool_epoch_tile(plan,builder,lower,*,output_row,output_column,rows,columns,accumulator_row,issue,complete,completion_mode=None):
    """One original private patch row, retaining all store-controller fences."""
    completion_mode=epoch_selection() if completion_mode is None else completion_mode
    if completion_mode=='off' or type(output_column) is not int or type(accumulator_row) is not int: return None
    from merlin.compile.scheduling.window_transfer import ConvolutionOutputTile
    tiles=(ConvolutionOutputTile(output_column,columns,accumulator_row),)
    return emit_pool_retained_row(plan,builder,lower,tiles=tiles,patch_offset=0,
        output_row=output_row,rows=rows,issue=issue,complete=complete,completion_mode=completion_mode)
