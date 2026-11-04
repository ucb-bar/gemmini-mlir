"""Source-bound scalar readout batching for checked flattened private groups.

The shared convolution emitter requires row-local M tiles. Flattened M tiles
cross source rows, so this bounded adapter checks its actual native readout law
on every ACC row and consumes a native effect program with a real final drain.
No compute, transfer extent, output order or configuration word changes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.llvm_program import emit_index_effect_program
from merlin.compile.scheduling.window_transfer import (
    ConvolutionOutputTile, ConvolutionReadoutEpochLaw,
)
from merlin.sched.ir import Call, Const, Kernel, check_structure
from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _prepared_identity(p):
    return _digest(dict(source=p.source.identity,gather=asdict(p.gather),
        group=asdict(p.group),attributes=p.attributes,
        rhs_pitch=p.rhs_pitch,output_pitch=p.output_pitch))


@dataclass(frozen=True)
class PrivateGroupReadoutEpoch:
    prepared: object
    law: ConvolutionReadoutEpochLaw
    ordering_sha256: str
    source_group_sha256: str

    def validate(self):
        p=self.prepared
        p.source.plan.__post_init__();p.gather.__post_init__();p.group.__post_init__()
        if self.source_group_sha256 != _prepared_identity(p):
            raise ValueError('readout epoch source or physical mapping changed')
        if self.law.source_identity != p.source.identity:
            raise ValueError('readout epoch changed original source identity')
        self.law.__post_init__()

    def emit(self,builder,outputs,lower,complete):
        """Consume every original scalar store, then the original group drain."""
        self.validate()
        p=self.prepared;g=p.group;plan=p.source.plan
        if (type(outputs)is not tuple or not outputs or len(outputs)>g.row_blocks
                or any(type(tile)is not tuple or not tile for tile in outputs)):
            raise ValueError('readout epoch requires the complete original private group')
        active=len(outputs[0]);stores=[];cells=set()
        if sum(map(len,outputs))>self.law.max_issued_readouts:
            raise ValueError('readout epoch exceeds its issued bound')
        for m,tile in enumerate(outputs):
            if len(tile)!=active or active>g.panel.column_blocks:
                raise ValueError('readout epoch changed original column partition')
            rows=tile[0]['rows']
            if type(rows)is not int or not 0<rows<=g.panel.tile:
                raise ValueError('readout epoch changed original M extent')
            column=0
            for n,out in enumerate(tile):
                if (out['rows']!=rows or type(out['columns'])is not int
                        or not 0<out['columns']<=plan.column_tile
                        or out['accumulator_row']!=g.accumulator_row(m,n,active_column_blocks=active)):
                    raise ValueError('readout epoch changed original private ACC mapping')
                stores.append(out)
                # Every scalar readout row is a valid original row-local law
                # instance, including tiles crossing a source-row/image edge.
                # The full physical transfer keeps its original rows/columns.
                for i in range(rows):
                    proof=ConvolutionOutputTile(column,out['columns'],out['accumulator_row']+i)
                    self.law.require(plan,1,(proof,))
                    for j in range(out['columns']):
                        cell=(out['accumulator_row']+i,j)
                        if cell in cells:raise ValueError('readout epoch aliases live ACC cells')
                        cells.add(cell)
                column+=out['columns']
        # Native calls refer to the bounded, checked original store roster;
        # dynamic source coordinates remain the actual original SSA operands.
        before=[];after=[]
        for ordinal in range(len(stores)):
            call=Call('original_scalar_readout',(('ordinal',Const(ordinal)),))
            before.extend((Call('original_before_config',()),call,Call('original_after_store',())))
            after.append(call)
        final=Call('private_group_last_use',())
        source=Kernel('original_private_group_scalar_readouts',(),tuple(before)+(final,))
        program=Kernel('private_group_scalar_readout_epoch',(),tuple(after)+(final,))
        if check_structure(source) or check_structure(program):
            raise ValueError('invalid native readout effect program')
        def consume(tag,fields):
            if tag!='original_scalar_readout' or set(fields)!={'ordinal'}:
                raise ValueError('readout epoch emitted a non-original effect')
            ordinal=fields['ordinal']
            if type(ordinal)is not int or not 0<=ordinal<len(stores):
                raise ValueError('readout epoch changed original scalar store roster')
            lower('store',stores[ordinal])
        emitted=emit_index_effect_program(program,builder,consume,
            complete_epoch=lambda:complete('spatial_output_last_use',{}),
            completion_instruction='private_group_last_use')
        self.validate()
        if emitted['kernel_sha256']!=program.digest():
            raise ValueError('readout consumer did not emit the selected actual program')
        return dict(source_kernel_sha256=source.digest(),consumed_kernel_sha256=program.digest(),
            source_group_sha256=self.source_group_sha256,law=asdict(self.law),
            ordering_sha256=self.ordering_sha256,source_program=source.text(),
            consumed_program=program.text(),emission=emitted,
            original_stores=len(stores),original_acc_cells=len(cells),
            removed_store_boundary_drains=2*len(stores),
            retained_completion_edges=['complete original reduction before all readouts',
                'real private group last-use drain before any reuse',
                'real phase last-use drain before RHS replacement','real whole-call final drain'],
            geometry=[dict(rows=t[0]['rows'],columns=[o['columns']for o in t],
                accumulator_rows=[o['accumulator_row']for o in t])for t in outputs],
            timing='UNKNOWN',target_execution=False)


def prepare_private_group_readout_epoch(prepared):
    """Read-only admission; selected RTL ordering and owned source only."""
    try:
        record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
        ordering=record['ordering'];digest=_digest(ordering)
        if (record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
                or digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
                or ordering['status']!='derived'):
            return None
        cross={(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
        if cross['PRELOAD.c_addr','MVOUT.acc_addr','accumulator']['overlap']!='completion':return None
        if (ordering['completion_order']['execute']['order']!='in_order'
                or ordering['completion_order']['store']['order']!='in_order'
                or ordering['same_queue']['store']['release']!='completion'):
            return None
        # CONFIG_ST and MVOUT share the original ordered store lane. Every
        # original CONFIG_ST is retained with one invocation-invariant value;
        # no varying-scale configuration snapshot premise is introduced.
        queue={}
        for row in ordering['funct_queue']:
            if row.get('funct')==0:queue.update({int(k):v for k,v in row['by_code'].items()})
        if queue.get(2)!='store' or not any(r.get('funct')==3 and r['queue']=='store'for r in ordering['funct_queue']):return None
        entry=record['finite_entries']['store']
        if entry['status']!='derived' or type(entry['value'])is not int or entry['value']<1:return None
        scale=prepared.attributes.get('acc_scale_by_column')
        if scale is not None and len(set(map(float,scale)))!=1:return None
        law=ConvolutionReadoutEpochLaw(prepared.epoch.hardware_identity,
            prepared.epoch.numerical_policy_identity,prepared.source.identity,
            'private-flattened-original-scalar-readout-epoch-v1',
            max_issued_readouts=prepared.group.row_blocks*prepared.group.panel.column_blocks,
            pending_readouts=entry['value'],local_read_after_write=True,
            accumulator_result_visibility=True,ordered_configuration_capture=True,
            finite_queue_progress=True,complete_device_drain=True,exclusive_private_state=True)
        source_group=_prepared_identity(prepared)
        result=PrivateGroupReadoutEpoch(prepared,law,digest,source_group)
        result.validate()
        return result
    except(KeyError,ValueError,TypeError,OverflowError):
        return None


def prepare_guarded_readout_epoch(prepared):
    """Select a prepared lawful epoch only in the admitted guarded4 region.

    The resource predicates are performance applicability, never legality.
    Unknown law premises and small/limited groups retain the original stores.
    No timing, named workload lookup or startup experiment occurs here.
    """
    epoch=prepare_private_group_readout_epoch(prepared)
    level='authored_flattened_conv_readout_source'
    output='prepared_convolution_readout_epoch'
    identity=_prepared_identity(prepared)
    def verify(p):
        p.source.plan.__post_init__();p.gather.__post_init__();p.group.__post_init__()
        if _prepared_identity(p)!=identity:
            raise ValueError('readout policy source or actual private group changed')
    context=PipelineContext(hardware=prepared.epoch.hardware_identity,
        numerical_policy=prepared.epoch.numerical_policy_identity,
        resources='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
        execution=prepared.ordering['ordering_sha256'])
    session=PipelineSession(prepared,context=context,ir_level=level,
        fingerprint=_prepared_identity,verify_ir=verify)
    w=prepared.source.plan.window
    pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
    resources=dict(actual_group_blocks=prepared.group.row_blocks,
        complete_m_tiles=pixels//prepared.group.panel.tile,
        admitted_readout_law=int(epoch is not None))
    facts=PolicyFacts(session.stamp(),operations=frozenset({'convolution'}),
        layouts=('nhwc',),resources=tuple(sorted(resources.items())))
    def entry(name,artifact):
        def action(active):
            verify(active.module)
            if artifact is not False:artifact.validate()
            return {output:artifact}
        return name,RegisteredPass(PassSpec(name,0,produces=(output,),
            mutates_ir=False,ir_level=level),action)
    registry=dict((entry('original_scalar_readouts',False),
        entry('checked_private_readout_epoch',epoch if epoch is not None else False)))
    policy=PipelinePolicy((
        PipelineVariant('readout_epoch',('checked_private_readout_epoch',),(
            ResourcePredicate('actual_group_blocks',4),
            ResourcePredicate('complete_m_tiles',8),
            ResourcePredicate('admitted_readout_law',1))),
        PipelineVariant('original',('original_scalar_readouts',))), 'original')
    result=execute_policy(session,policy,registry,facts,output=output)
    receipt=dict(selection='capacity_guarded',policy=result.receipt,
        source_group_sha256=identity,resources=resources,
        performance_class=dict(actual_group_blocks_at_least=4,
            complete_m_tiles_at_least=8,boundary_status='UNKNOWN',
            missing_neighbor_values=[7,8],observed_complete_m_tiles=[3,49,196,784],
            evidence='complete matched Spike call pairs; original on tiny single group; batching on admitted guarded4',
            qualification='performance applicability only; exact native readout law independently required'))
    return (None if result.artifact is False else result.artifact),receipt
