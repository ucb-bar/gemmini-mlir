"""Uniform complete-K convolution stores between real readiness/reuse drains.

The selected store lane orders one unchanged literal CONFIG_ST and its MVOUTs.
No varying-config snapshot or asynchronous accumulator overwrite is assumed.
Actual source-relative store operands are consumed at the producer completion.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import ConvolutionOutputTile, ConvolutionReadoutEpochLaw, FlattenedConvolutionReadout
from merlin.compile.scheduling.configuration import ConfigurationEffects
from . import isa


def _linear(value):
    """Normalize actual integer/LLVM affine additions for operand projection."""
    if type(value)is int:return {},value
    owner=getattr(value,'owner',None);name=getattr(owner,'name',None)
    if name=='llvm.mlir.constant':return {},int(owner.value.value.data)
    if name=='llvm.add':
        a,x=_linear(owner.operands[0]);b,y=_linear(owner.operands[1]);terms=dict(a)
        for term,c in b.items():terms[term]=terms.get(term,0)+c
        return {t:c for t,c in terms.items()if c},x+y
    if name=='llvm.mul':
        a,x=_linear(owner.operands[0]);b,y=_linear(owner.operands[1])
        if not a:return {t:c*x for t,c in b.items()if c*x},x*y
        if not b:return {t:c*y for t,c in a.items()if c*y},x*y
    return {value:1},0


def _relative(value,base,offset):
    terms,constant=_linear(base)
    return _linear(value)==(terms,constant+offset)


def _configuration(prepared,dtype):
    attrs=prepared.attributes;p=prepared.source.plan;scales=attrs.get('acc_scale_by_column')
    if dtype not in ('i8','i32')or any('pool'in str(s).lower()for s in attrs.get('epilogue',())):
        raise ValueError('uniform store epochs require nonpooling integer readout')
    if scales is not None and(len(scales)!=p.output_channels or len(set(map(float,scales)))!=1):
        raise ValueError('column-varying readout configuration')
    scale=scales[0]if scales else attrs.get('acc_scale',1.0)
    relu=bool(attrs.get('relu',False)or'relu'in attrs.get('epilogue',()))
    if dtype=='i32'and(scales is not None or float(scale)!=1.0 or relu):
        raise ValueError('full-width readout cannot change numeric order')
    command=isa.config_st(prepared.output_pitch*(4 if dtype=='i32'else 1),scale=scale,activation=int(relu))
    if command[1]&~15:raise ValueError('store epochs require zero pooling fields')
    return command


@dataclass(frozen=True)
class ReadoutEpochPlan:
    source: str
    group: object
    epoch: object
    law: object
    command: tuple
    pixels: int
    channels: int
    row_classes: tuple
    column_classes: tuple
    report: dict

    def bind(self,ctx,readout_ctx,prepared):
        if prepared.source.identity!=self.source or prepared.group!=self.group or prepared.epoch!=self.epoch:
            raise ValueError('readout epoch source/storage/ordering changed after admission')
        return ReadoutEpoch(self,ctx,readout_ctx)


def plan(prepared,dtype,*,complete_k):
    """Read-only admission of every possible full/peeled spatial/column class."""
    selection=prepared.attributes.get('conv_store_epochs',knob('CONV_STORE_EPOCHS',
        default='off',values=('off','uniform'),root=Path(__file__).resolve().parents[1]))
    if selection not in ('off','uniform'):raise ValueError('CONV_STORE_EPOCHS must be off or uniform')
    report=dict(selection=selection,applied=False)
    if selection=='off':return None,report
    if not complete_k:return None,dict(report,refusal='segmented reduction route unchanged')
    try:
        command=_configuration(prepared,dtype)
        from .column_batches import _admission
        selected=_admission()
        epoch=prepared.epoch;epoch.require(2)
        if (epoch.hardware_identity!='gemmini-fire-aligned-'+selected['base_facts_sha256'][:16]
                or epoch.source_identity!=prepared.source.identity):
            raise ValueError('readout/compute selected source or target identities differ')
        prepared.source.__post_init__();group=prepared.group;group.__post_init__()
        p=prepared.source.plan;w=p.window;pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
        if group.panel.tile!=isa.DIM or prepared.output_pitch<p.output_channels:
            raise ValueError('readout requires original disjoint primitive output layout')
        max_m=group.row_blocks;max_n=group.panel.column_blocks;span=max_m*isa.DIM
        full,tail=divmod(pixels,span);row_classes=[]
        if full:row_classes.append((isa.DIM,)*max_m)
        if tail:
            blocks,last=divmod(tail,isa.DIM);row_classes.append((isa.DIM,)*blocks+((last,)if last else ()))
        columns,last=divmod(p.output_channels,isa.DIM);phases,tail=divmod(columns,max_n);column_classes=[]
        if phases:column_classes.append(((isa.DIM,)*max_n,0,(phases-1)*max_n*isa.DIM,phases))
        widths=(isa.DIM,)*tail+((last,)if last else ())
        if widths:column_classes.append((widths,phases*max_n*isa.DIM,phases*max_n*isa.DIM,1))
        law=ConvolutionReadoutEpochLaw(epoch.hardware_identity,epoch.numerical_policy_identity,
            epoch.source_identity,epoch.qualification_identity,max_m*max_n,selected['finite_store_entries'],
            local_read_after_write=epoch.local_read_after_write,
            accumulator_result_visibility=epoch.accumulator_result_visibility,
            ordered_configuration_capture=epoch.ordered_configuration_capture,
            finite_queue_progress=True,complete_device_drain=epoch.complete_device_drain,
            exclusive_private_state=epoch.exclusive_private_state)
        regions=[]
        for extents in row_classes:
            for widths,first_column,last_column,occurrences in column_classes:
                cells=[]
                for m,rows in enumerate(extents):
                    for column in {first_column,last_column}:
                        tiles=tuple(ConvolutionOutputTile(column+n*isa.DIM,width,
                            group.accumulator_row(m,n,active_column_blocks=len(widths)))for n,width in enumerate(widths))
                        # Endpoint representatives bind each full/peeled
                        # interval class to the original flattened tensor.
                        # The affine pixel loop supplies every intermediate
                        # in-range base; _outputs consumes its actual operands.
                        for first_pixel in {m*isa.DIM,pixels-sum(extents)+m*isa.DIM}:
                            law.require_flattened(FlattenedConvolutionReadout(p,
                                prepared.source.identity,first_pixel,rows,prepared.output_pitch,tiles))
                    for n,width in enumerate(widths):
                        acc=group.accumulator_row(m,n,active_column_blocks=len(widths))
                        # Actual physical ACC cells and source-relative rectangles.
                        cell=(acc,acc+rows,m*isa.DIM,m*isa.DIM+rows,n*isa.DIM,n*isa.DIM+width)
                        for a,b,y0,y1,x0,x1 in cells:
                            if acc<b and a<acc+rows:raise ValueError('live ACC readout rows overlap')
                            if cell[2]<y1 and y0<cell[3]and cell[4]<x1 and x0<cell[5]:
                                raise ValueError('live original output cells overlap')
                        cells.append(cell)
                regions.append(dict(row_extents=list(extents),column_widths=list(widths),
                    first_column=first_column,last_column=last_column,column_phase_occurrences=occurrences,
                    cells=[list(c)for c in cells],issued_readouts=len(cells)))
        selected=dict(selected,store_configuration_snapshot='not assumed; exact uniform state through real closing drain')
        report=dict(report,applied=True,source_sha256=prepared.source.identity,
            uniform_config=list(command),configuration_effects=asdict(ConfigurationEffects(
                writes=(('store.rs1',command[1]),('store.rs2',command[2])),ordered=True,pure_write=True)),
            selected_admission=selected,readout_law=asdict(law),regions=regions,
            output_layout_proof=dict(api='ConvolutionReadoutEpochLaw.require_flattened',
                source_plan_sha256=p.identity,total_pixels=pixels,output_hw=list(w.output_spatial),
                batch=w.input_shape[0],channels=p.output_channels,output_pitch=prepared.output_pitch,
                primitive=p.primitive,original_window_row_tile=p.row_tile,
                mapping='pixel=((batch*OH)+y)*OW+x; physical element=pixel*pitch+column',
                interval='0<=first_pixel; 0<rows<=primitive; first_pixel+rows<=total_pixels',
                actual_operand_projection='existing _outputs checks exact group row/column offsets and ACC cells'),
            readiness='retained real complete K/N/M producer drain before CONFIG_ST and all stores',
            reuse='one real group output-last-use full drain before next producer, LHS/ACC reuse or phase change',
            projection='actual source-relative M/N operands checked; original common terminal encoder and store order',
            no_async_configuration_capture_assumed=True,timing=None)
        return ReadoutEpochPlan(prepared.source.identity,group,epoch,law,command,pixels,p.output_channels,
            tuple(row_classes),tuple(column_classes),report),report
    except(ValueError,TypeError,KeyError,OverflowError)as exc:
        return None,dict(report,refusal=str(exc))


class ReadoutEpoch:
    def __init__(self,plan,ctx,readout_ctx):
        self.plan=plan;self.ctx=ctx;self.readout_ctx=readout_ctx
        self.active=False;self.outputs=();self.position=0;self.phase=None
        self.configurations=0;self.stores=0;self.groups=0

    def __getattr__(self,name):return getattr(self.readout_ctx,name)

    def _outputs(self,fields):
        outputs=tuple(fields['outputs']);extents=tuple(fields['row_extents']);widths=tuple(fields['column_widths'])
        first=fields['first_output_row'];column=fields['first_output_column'];group=self.plan.group
        if (extents not in self.plan.row_classes or not any(widths==c[0]for c in self.plan.column_classes)
                or len(outputs)!=len(extents)*len(widths)or len(outputs)>self.plan.law.max_issued_readouts):
            raise ValueError('actual readout group differs from admitted source/layout classes')
        if type(first)is int and not 0<=first<=self.plan.pixels-sum(extents):
            raise ValueError('actual readout exceeds original output rows')
        if type(column)is int and not 0<=column<=self.plan.channels-sum(widths):
            raise ValueError('actual readout exceeds original output columns')
        for i,out in enumerate(outputs):
            m,n=divmod(i,len(widths))
            if (out['rows']!=extents[m]or out['columns']!=widths[n]
                    or out['accumulator_row']!=group.accumulator_row(m,n,active_column_blocks=len(widths))
                    or not _relative(out['output_row'],first,m*isa.DIM)
                    or not _relative(out['output_column'],column,n*isa.DIM)):
                raise ValueError('actual terminal operands changed source-relative owned cells')
        return outputs

    def complete(self,kind,fields):
        if kind=='spatial_reduction_epoch_complete':
            if self.active:raise ValueError('producer reaches unfinished output epoch')
            self.outputs=self._outputs(fields);self.position=0;self.phase=None
            self.ctx.fence() # actual readiness, before any readout/configuration
            self.readout_ctx.emit(*self.plan.command);self.configurations+=1
            self.active=True;self.groups+=1
        elif kind in ('spatial_output_last_use','spatial_group_output_last_use'):
            if not self.active or self.position!=len(self.outputs)or self.phase is not None:
                raise ValueError('output epoch closes before every admitted MVOUT')
            self.ctx.fence() # all stores retire before any private state reuse
            self.active=False;self.outputs=()
        else:
            if self.active:raise ValueError('non-store effect crosses live readout epoch')
            self.ctx.fence()

    def store(self,fields,encode):
        if not self.active or self.phase is not None or self.position>=len(self.outputs):
            raise ValueError('terminal store has no completed original producer')
        expected=self.outputs[self.position]
        if any(fields.get(k)is not v and fields.get(k)!=v for k,v in expected.items()):
            raise ValueError('terminal store differs from consumed ready output')
        self.phase='before';encode(self,self.completion)
        if self.phase!='finished':raise ValueError('terminal helper omitted exact store completion callback')
        self.phase=None;self.position+=1

    def completion(self,phase):
        if phase=='before_config'and self.phase=='before':self.phase='config'
        elif phase=='after_store'and self.phase=='stored':self.phase='finished'
        else:raise ValueError('terminal completion order changed inside store epoch')

    def emit(self,*command):
        if (len(command)==3 and command[0]==isa.FUNCT['config']and type(command[1])is int and command[1]&3==2):
            if tuple(command)!=self.plan.command or self.phase!='config':
                raise ValueError('store epoch terminal helper changes uniform configuration')
            self.phase='configured';return
        if command[0]!=isa.FUNCT['mvout']or self.phase!='configured':
            raise ValueError('unexpected terminal effect in uniform readout epoch')
        self.readout_ctx.emit(*command);self.stores+=1;self.phase='stored'

    def receipt(self):
        if self.active:raise ValueError('convolution returns before all MVOUTs retire')
        return dict(self.plan.report,static_groups=self.groups,static_stores=self.stores,
            static_establish_configuration_requests=self.configurations,
            counts='static callback/encoder requests; invocation reuse may filter config; not runtime counts or timing',
            existing_invocation_config_reuse=type(self.readout_ctx).__name__=='UniformReadoutContext')
