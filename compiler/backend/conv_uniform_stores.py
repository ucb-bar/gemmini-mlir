"""Identical-configuration stores inside an already drained convolution group.

The actual source group owns disjoint ACC/output rectangles. Its complete-K
producer drain precedes this consumer and its original last-use drain follows
it. No compute, load, configuration change or ownership transition is moved.
Only exact repeated CONFIG_ST and per-store completion are elided, under the
existing selected identical-config ordered-store law.
"""
from dataclasses import asdict
from hashlib import sha256

from merlin.compile.scheduling.convolution_output_groups import ConvolutionOutputGroup
from . import isa
from .column_batches import _admission
from .dense_ready import uniform_store_command
from .dense_readout import store_tile


def _affine(value):
    """Normalize actual LLVM address operands; unknown SSA values stay atoms."""
    if type(value) is int:return value,{}
    owner=getattr(value,'owner',None)
    name=getattr(owner,'name',None)
    if name=='llvm.mlir.constant':
        literal=getattr(owner,'value',None)
        if hasattr(literal,'value') and hasattr(literal.value,'data'):
            return int(literal.value.data),{}
    if name in ('llvm.add','llvm.sub','llvm.mul'):
        a,b=map(_affine,owner.operands)
        if name=='llvm.mul':
            if a[1] and b[1]:return 0,{value:1}
            factor,other=(a[0],b) if not a[1] else (b[0],a)
            return factor*other[0],{key:factor*n for key,n in other[1].items()}
        sign=-1 if name=='llvm.sub' else 1
        terms=dict(a[1])
        for key,n in b[1].items():terms[key]=terms.get(key,0)+sign*n
        return a[0]+sign*b[0],{key:n for key,n in terms.items() if n}
    return 0,{value:1}


def _offset(actual,base,expected):
    a,b=_affine(actual),_affine(base)
    return a[1]==b[1] and a[0]-b[0]==expected


class _Terminal:
    def __init__(self,ctx,command,first):
        self.ctx,self.command,self.first=ctx,command,first
        self.configs=self.stores=0;self.phases=[]
    def __getattr__(self,name):return getattr(self.ctx,name)
    def emit(self,funct,rs1,rs2):
        if funct==isa.FUNCT['config']:
            if (funct,rs1,rs2)!=self.command or self.configs or self.stores:
                raise ValueError('uniform convolution terminal changed complete CONFIG_ST')
            self.configs+=1
            if self.first:self.ctx.emit(funct,rs1,rs2)
        else:
            if funct!=isa.FUNCT['mvout'] or self.configs!=1 or self.stores:
                raise ValueError('uniform convolution terminal is not one unchanged configured MVOUT')
            self.stores+=1;self.ctx.emit(funct,rs1,rs2)
    def complete(self,phase):self.phases.append(phase)
    def finish(self):
        if (self.configs,self.stores,self.phases)!=(1,1,['before_config','after_store']):
            raise ValueError('uniform convolution terminal changed its completion protocol')


class UniformConvolutionStores:
    def __init__(self,prepared,dtype):
        if type(prepared.group) is not ConvolutionOutputGroup:
            raise ValueError('uniform stores require the original complete output group')
        prepared.group.__post_init__()
        self.prepared,self.dtype=prepared,dtype
        self.command=uniform_store_command(prepared.attributes,prepared.output_pitch,dtype)
        self.identity=self._identity()
        admission=_admission()
        admission['store_configuration_snapshot']='not assumed; exact complete words remain unchanged until original group-end drain'
        self.report=dict(mechanism='checked_uniform_convolution_store_group',
            source_identity=prepared.source.identity,output_group=asdict(prepared.group),
            actual_config_st_words=self.command,admission=admission,
            static_emitted_groups=0,static_emitted_stores=0,maximum_stores_per_group=0,
            source_and_addresses='same actual output rectangles and store_tile encoder; no data arithmetic or address rewrite',
            completion='original complete-K/N producer drain and spatial_output_last_use group drain retained',
            execution='identical configuration with selected ordered store completion and finite ready/valid backpressure')
    def _identity(self):
        p=self.prepared
        return sha256(repr((p.source.identity,asdict(p.group),p.attributes,p.output_pitch,self.dtype,self.command)).encode()).hexdigest()
    def revalidate(self):
        self.prepared.group.__post_init__()
        if self.identity!=self._identity() or self.command!=uniform_store_command(
                self.prepared.attributes,self.prepared.output_pitch,self.dtype):
            raise ValueError('uniform convolution source, group or configuration changed')
    def check(self,outputs):
        self.revalidate()
        group=self.prepared.group;panel=group.panel
        if not outputs or len(outputs)>group.row_blocks*panel.column_blocks:
            raise ValueError('uniform convolution group exceeds original output domain')
        required={'output_row','output_column','rows','columns','accumulator_row'}
        if any(set(out)!=required for out in outputs):
            raise ValueError('uniform convolution store fields differ from original output tile')
        first=outputs[0]
        active=next((i for i,out in enumerate(outputs) if not _offset(out['output_row'],first['output_row'],0)),len(outputs))
        if active>panel.column_blocks or len(outputs)%active or len(outputs)//active>group.row_blocks:
            raise ValueError('uniform convolution row/column partition differs from original group')
        for i,out in enumerate(outputs):
            m,n=divmod(i,active)
            rows,cols,acc=(out[key] for key in ('rows','columns','accumulator_row'))
            if (any(type(v)is not int for v in (rows,cols,acc))
                    or not 0<rows<=panel.tile or not 0<cols<=panel.tile
                    or acc!=group.accumulator_row(m,n,active_column_blocks=active)
                    or acc+rows>panel.accumulator_capacity_rows
                    or rows!=outputs[m*active]['rows'] or cols!=outputs[n]['columns']
                    or (m+1<len(outputs)//active and rows!=panel.tile)
                    or (n+1<active and cols!=panel.tile)
                    or not _offset(out['output_row'],first['output_row'],m*panel.tile)
                    or not _offset(out['output_column'],first['output_column'],n*panel.tile)):
                raise ValueError('uniform convolution actual rectangles change partition or ACC ownership')
        return outputs
    def emit(self,ctx,dst,outputs):
        outputs=self.check(tuple(outputs))
        if ctx.spec(dst)['dtype']!=self.dtype:
            raise ValueError('uniform convolution terminal output dtype changed')
        for i,out in enumerate(outputs):
            terminal=_Terminal(ctx,self.command,i==0)
            store_tile(terminal,dst,out['accumulator_row'],out['rows'],out['columns'],
                self.prepared.output_pitch,out['output_row'],out['output_column'],
                self.prepared.attributes,completion=terminal.complete)
            terminal.finish()
        self.report['static_emitted_groups']+=1
        self.report['static_emitted_stores']+=len(outputs)
        self.report['maximum_stores_per_group']=max(self.report['maximum_stores_per_group'],len(outputs))
        self.revalidate()


def prepare_uniform_stores(prepared,dtype):
    """Read-only admission after the source's uniform-scale/alias checks."""
    attrs=prepared.attributes
    scales=attrs.get('acc_scale_by_column')
    if dtype not in ('i8','i32') or (scales is not None and
            (len(scales)!=prepared.source.plan.output_channels or len(set(map(float,scales)))!=1)):
        return None
    if attrs.get('requant_shift') is not None or any(x not in ('bias_add','acc_scale','relu') for x in attrs.get('epilogue',())):
        return None
    if dtype=='i32' and (scales is not None or float(attrs.get('acc_scale',1))!=1
            or attrs.get('relu') or 'relu' in attrs.get('epilogue',())):
        return None
    return UniformConvolutionStores(prepared,dtype)
