"""Exact NHWC nonoverlapping windows as a dense operand address view.

No im2col tensor exists. The original dense terminal MVIN reads each 16x16
logical rectangle directly with the original contiguous-window row pitch.
This proves source coordinate correspondence, not new hardware semantics.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod,isfinite
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.tile_load_coalescing import _linear
from merlin.sched.ir import Const, Loop
from merlin.sched.ir.expr import Select, Sum, Scaled
from . import isa


class DenseWindowMappingRefusal(ValueError):
    """Actual schedule geometry cannot consume this source address view."""


@dataclass(frozen=True)
class DenseWindowInputMapping:
    window: Window2D
    output_channels: int
    row_run: bool = False

    def __post_init__(self):
        w=self.window
        if type(w) is not Window2D or type(self.output_channels) is not int or self.output_channels<=0:
            raise ValueError('dense window mapping requires typed original source geometry')
        b,h,width,ci=w.input_shape;kh,kw=w.kernel;oh,ow=w.output_spatial
        if (w.stride!=w.kernel or w.dilation!=(1,1) or any(w.padding)
                or h%kh or width%kw or (kw*ci)%isa.DIM
                or type(self.row_run) is not bool
                or not self.row_run and (ow%isa.DIM or self.m%isa.DIM)
                or self.k%isa.DIM
                or max(prod(w.input_shape),self.m*self.k,self.k*self.output_channels,
                       b*oh*ow*self.output_channels*4)>=2**63):
            raise ValueError('dense window view needs complete aligned nonoverlapping NHWC windows')

    @property
    def m(self):
        return (self.window.output_spatial[1] if self.row_run else
                self.window.input_shape[0]*prod(self.window.output_spatial))
    @property
    def k(self):
        return prod(self.window.kernel)*self.window.input_shape[-1]
    @property
    def source_row_pitch(self):
        return self.window.kernel[1]*self.window.input_shape[-1]
    @property
    def identity(self):
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def source_offset(self,row,column):
        if not 0<=row<self.m or not 0<=column<self.k:
            raise ValueError('logical dense coordinate leaves original convolution')
        b,h,width,ci=self.window.input_shape;kh,kw=self.window.kernel
        oh,ow=self.window.output_spatial
        image,spatial=divmod(row,oh*ow);y,x=divmod(spatial,ow)
        ky,inner=divmod(column,self.source_row_pitch)
        return ((image*h+y*kh+ky)*width+x*kw)*ci+inner

    def emit_offset(self,fb,row,column):
        _,h,width,ci=self.window.input_shape;kh,kw=self.window.kernel
        oh,ow=self.window.output_spatial
        image=fb.udiv_i(row,fb.const(oh*ow))
        y=fb.urem_i(fb.udiv_i(row,fb.const(ow)),fb.const(oh))
        x=fb.urem_i(row,fb.const(ow))
        ky=fb.udiv_i(column,fb.const(self.source_row_pitch))
        inner=fb.urem_i(column,fb.const(self.source_row_pitch))
        input_y=fb.add_i(fb.mul_i(y,fb.const(kh)),ky)
        image_y=fb.add_i(fb.mul_i(image,fb.const(h)),input_y)
        spatial=fb.add_i(fb.mul_i(image_y,fb.const(width)),fb.mul_i(x,fb.const(kw)))
        return fb.add_i(fb.mul_i(spatial,fb.const(ci)),inner)

    def validate_source(self,ctx,lhs,rhs,dst,m,k,n,attrs):
        self.__post_init__()
        from .context import EmissionContext
        source_ctx=ctx.base if type(ctx) is _OutputRowContext else ctx
        if type(source_ctx) is not EmissionContext or getattr(source_ctx,'physical_views',None) is not None:
            raise ValueError('dense window source requires the original contiguous emission context')
        if self.row_run and (type(ctx) is not _OutputRowContext or ctx.mapping is not self
                             or ctx.lhs!=lhs or ctx.dst!=dst):
            raise ValueError('row-run address mapping requires its checked output-row context')
        output_rows=self.window.input_shape[0]*prod(self.window.output_spatial)
        if (tuple(ctx.spec(lhs)['shape'])!=self.window.input_shape or ctx.spec(lhs)['dtype']!='i8'
                or ctx.spec(rhs)['dtype']!='i8' or tuple(ctx.spec(rhs)['shape'])!=(self.k,n)
                or (m,k,n)!=(self.m,self.k,self.output_channels)
                or prod(ctx.spec(dst)['shape'])!=output_rows*n):
            raise ValueError('dense window input view differs from actual source/output tensors')
        bias=attrs.get('bias')
        involved={lhs,rhs,dst,*((bias,) if bias else ())}
        aliases=getattr(ctx,'cb',{}).get('params',{}).get('tensor_aliases',{})
        if not isinstance(aliases,dict) or any(
                name in involved or not isinstance(view,dict) or view.get('source') in involved
                for name,view in aliases.items()):
            raise ValueError('dense window mapping requires original unaliased logical tensor views')
        forbidden_spec=('strides','stride','strides_bytes','row_stride','offset','base_offset',
                        'storage_offset','byte_offset','physical_views','physical_strides','layout','view','alias')
        if (any(any(key in ctx.spec(name) for key in forbidden_spec) for name in involved)
                or any(key in getattr(ctx,'cb',{}).get('params',{}) for key in
                       ('physical_views','physical_strides','tensor_offsets'))
                or any(key in attrs for key in ('physical_views','rhs_row_stride','output_row_stride',
                                                'input_strides','input_offset'))):
            raise ValueError('dense window mapping requires original contiguous tensor storage')
        if dst in (lhs,rhs,bias):
            raise ValueError('dense window convolution needs disjoint output')
        pointers=getattr(ctx,'buffers',{})
        if dst in pointers and any(name in pointers and pointers[name] is pointers[dst]
                                   for name in (lhs,rhs,bias) if name is not None):
            raise ValueError('dense window convolution output aliases an operand or bias')
        if (attrs.get('accumulator_init') is not None or 'requant_shift' in attrs
                or any(stage not in ('bias_add','acc_scale','relu') for stage in attrs.get('epilogue',()))):
            raise ValueError('dense window mapping requires the existing bias/scale/relu readout contract')
        if 'bias_add' in attrs.get('epilogue',()) and not bias:
            raise ValueError('dense window bias stage requires its named source tensor')
        if bias and (ctx.spec(bias)['dtype']!='i32' or prod(ctx.spec(bias)['shape'])!=n):
            raise ValueError('dense window bias must contain one original i32 per output channel')
        scales=attrs.get('acc_scale_by_column')
        if scales is not None and (len(scales)!=n or len(set(map(float,scales)))>1):
            raise ValueError('dense window mapping requires one uniform existing readout scale')
        dtype=attrs.get('output_dtype',ctx.spec(dst)['dtype'])
        relu=bool(attrs.get('relu',False) or 'relu' in attrs.get('epilogue',()))
        scale=float(attrs.get('acc_scale',1.0))
        if (dtype not in ('i8','i32') or ctx.spec(dst)['dtype']!=dtype
                or 'requant' in attrs.get('epilogue',())
                or not isfinite(scale) or scales is not None and not all(isfinite(float(s)) for s in scales)
                or dtype=='i32' and (scales is not None or scale!=1.0 or relu)):
            raise ValueError('dense window source must obey the shared exact readout contract')
        # Validate the actual binary32 coefficient before any target effects.
        isa.f32_bits(scales[0] if scales else scale)

    def validate_kernel(self,kernel,adapter):
        """Check actual final load geometry before any terminal effect is issued."""
        count=0
        def aligned(expr):
            try:
                constant,terms=_linear(expr)
            except ValueError:
                return False
            return constant%isa.DIM==0 and all(v%isa.DIM==0 for v in terms.values())
        def walk(body,domains=None):
            domains={} if domains is None else domains
            nonlocal count
            for item in body:
                if type(item) is Loop:
                    walk(item.body,{**domains,item.var:(0,item.extent-1)})
                elif item.instr=='gemmini_mvin_lhs':
                    op=adapter.project(item)
                    if (type(op) is not Load or adapter.encode(op)!=item
                            or not self.row_run and (type(op.rows) is not Const or not 0<op.rows.value<=isa.DIM
                            or type(op.cols) is not Const or not 0<op.cols.value<=isa.DIM)
                            or not aligned(op.dram_row) or not aligned(op.dram_col)):
                        raise DenseWindowMappingRefusal('actual dense LHS transfer crosses admitted window tile geometry')
                    if self.row_run:
                        _check_rectangle(op,domains,self.m,self.k)
                    count+=1
        walk(kernel.body)
        if not count:raise DenseWindowMappingRefusal('dense window mapping found no actual source loads')
        return dict(mapping_sha256=self.identity,kernel_sha256=kernel.digest(),
            checked_lhs_load_sites=count,source_row_pitch_bytes=self.source_row_pitch,
            source_axes=['image','output_y','output_x','kernel_y','kernel_x','input_channel'],
            coordinate_map='((image*H+output_y*KH+kernel_y)*W+output_x*KW)*CI+inner',
            dense_reduction='column=kernel_y*(KW*CI)+inner',
            geometry_proof=('counted output-row slices; actual load rectangles proven inside OW and K; aligned origins and extents at most DIM' if self.row_run else
                            'OW and KW*CI multiples DIM; actual row/column origins multiples DIM; extents at most DIM'),
            counted_output_row_run=self.row_run,
            payload_materialization=False,source_only_proof=True)


def prepare(ctx,ifm,rhs,dst,attributes):
    """Refuse unsupported source windows before any emission or allocation."""
    try:
        attrs=dict(attributes or {});shape=tuple(map(int,ctx.spec(ifm)['shape']))
        kh,kw,ci,co=tuple(map(int,attrs['kernel']))
        if (len(shape)!=4 or shape[-1]!=ci or attrs.get('layout','nhwc').lower()!='nhwc'):
            return None
        padding=tuple(map(int,attrs.get('padding',(0,0,0,0))))
        if len(padding)==2:padding=(*padding,*padding)
        window=Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),
                        tuple(attrs.get('dilation',(1,1))),padding)
        mapping=DenseWindowInputMapping(window,co)
        mapping.validate_source(ctx,ifm,rhs,dst,mapping.m,mapping.k,co,attrs)
        return mapping
    except (KeyError,TypeError,ValueError,OverflowError):
        return None


def lower(ctx,ifm,rhs,dst,attributes):
    mapping=prepare(ctx,ifm,rhs,dst,attributes)
    if mapping is None:return False
    from .dense import lower_dense
    try:
        receipt=lower_dense(ctx,ifm,rhs,dst,m=mapping.m,k=mapping.k,n=mapping.output_channels,
                            attributes=attributes,lhs_mapping=mapping)
    except DenseWindowMappingRefusal:
        # The dense hook checks its final actual Kernel before its first
        # fence/configuration/command, so this refusal leaves ctx untouched.
        return False
    ctx.metadata.setdefault('convolution_dense_windows',[]).append(dict(
        mechanism='direct_nonoverlapping_nhwc_dense_windows',mapping_sha256=mapping.identity,
        input_shape=list(mapping.window.input_shape),kernel=list(mapping.window.kernel),
        dense_shape=[mapping.m,mapping.k,mapping.output_channels],
        source_row_pitch_bytes=mapping.source_row_pitch,payload_materialization=False,
        selected_dense_pipeline=receipt['selected_pipeline'],target_execution=False,timing=None))
    return True


def _check_rectangle(op, domains, m, k):
    """Bound affine/last-tile expressions without enumerating loop iterations.

    Splitting at every Select equality preserves the correlation between an
    origin and its final-tile extent. Each remaining expression is affine on
    a rectangular integer domain, so endpoint extrema are exact.
    """
    from itertools import product
    expressions=(op.dram_row,op.dram_col,op.rows,op.cols)
    cuts={name:{lo,hi+1} for name,(lo,hi) in domains.items()}
    def collect(e):
        if type(e) is Select:
            if e.var not in domains:raise DenseWindowMappingRefusal('unbound tail selection')
            lo,hi=domains[e.var]
            cuts[e.var].update(v for v in (e.equals,e.equals+1) if lo<=v<=hi+1)
            collect(e.then);collect(e.other)
        elif type(e) is Sum:
            for term in e.terms:collect(term)
        elif type(e) is Scaled:collect(e.expr)
    for expr in expressions:collect(expr)
    names=list(cuts)
    partitions=[list(zip(sorted(cuts[name])[:-1],sorted(cuts[name])[1:])) for name in names]
    def simplify(e,env):
        if type(e) is Select:return simplify(e.then if env[e.var]==e.equals else e.other,env)
        if type(e) is Sum:
            return sum((simplify(t,env) for t in e.terms),Const(0))
        if type(e) is Scaled:return simplify(e.expr,env)*e.factor
        return e
    def bounds(e,box):
        try:constant,terms=_linear(e)
        except ValueError as error:
            raise DenseWindowMappingRefusal('unsupported actual rectangle expression') from error
        lo=hi=constant
        for name,value in terms.items():
            if name not in box:raise DenseWindowMappingRefusal('unbound load coordinate')
            a,b=box[name];lo+=min(value*a,value*b);hi+=max(value*a,value*b)
        return lo,hi
    for intervals in product(*partitions):
        box=dict(zip(names,((a,b-1) for a,b in intervals)))
        env={name:lo for name,(lo,_) in box.items()}
        row,col,rows,cols=(simplify(e,env) for e in expressions)
        if (bounds(row,box)[0]<0 or bounds(col,box)[0]<0
                or not 1<=bounds(rows,box)[0]<=bounds(rows,box)[1]<=isa.DIM
                or not 1<=bounds(cols,box)[0]<=bounds(cols,box)[1]<=isa.DIM
                or bounds(row+rows,box)[1]>m or bounds(col+cols,box)[1]>k):
            raise DenseWindowMappingRefusal('actual load rectangle leaves its output-row or kernel slice')


class _OutputRowContext:
    """Typed address slice; only IFM and output pointers acquire row offsets."""
    def __init__(self,base,mapping,lhs,dst,row):
        self.base=base;self.mapping=mapping;self.lhs=lhs;self.dst=dst;self.row=row
    def __getattr__(self,name):return getattr(self.base,name)
    def ptr(self,name,elem_offset=0):
        if name not in (self.lhs,self.dst):return self.base.ptr(name,elem_offset)
        fb=self.fb;w=self.mapping.window;b,h,width,ci=w.input_shape
        kh,_=w.kernel;oh,ow=w.output_spatial
        row=fb.const(self.row) if type(self.row) is int else self.row
        if name==self.lhs:
            image=fb.udiv_i(row,fb.const(oh));y=fb.urem_i(row,fb.const(oh))
            offset=fb.mul_i(fb.add_i(fb.mul_i(image,fb.const(h)),fb.mul_i(y,fb.const(kh))),fb.const(width*ci))
        else:offset=fb.mul_i(row,fb.const(ow*self.mapping.output_channels))
        index=fb.const(elem_offset) if type(elem_offset) is int else elem_offset
        return self.base.ptr(name,fb.add_i(offset,index))


def lower_row_runs(ctx,ifm,rhs,dst,attributes):
    """Opt-in OW-tail variant: one counted dense call per original output row.

    Preflight uses the same lowerer, knobs, original shapes and actual terminal
    encoder in a private LLVM module. Any unsupported final rectangle therefore
    refuses before a block, command, pointer or allocation is added to ctx.
    Every runtime iteration ends with completion before operand rows are reused.
    """
    from .dense import lower_dense
    from .context import EmissionContext
    from copy import deepcopy
    try:
        attrs=dict(attributes or {});shape=tuple(ctx.spec(ifm)['shape'])
        kh,kw,ci,co=map(int,attrs['kernel'])
        if len(shape)!=4 or shape[-1]!=ci or attrs.get('layout','nhwc').lower()!='nhwc':return False
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        window=Window2D(shape,(kh,kw),tuple(attrs.get('stride',(1,1))),tuple(attrs.get('dilation',(1,1))),padding)
        # Existing complete-tile sources retain their existing lowering.
        if window.output_spatial[1]%isa.DIM==0:return False
        mapping=DenseWindowInputMapping(window,co,row_run=True)
        view=_OutputRowContext(ctx,mapping,ifm,dst,0)
        mapping.validate_source(view,ifm,rhs,dst,mapping.m,mapping.k,co,attrs)
        names=list(dict.fromkeys([ifm,rhs,dst]+([attrs['bias']] if attrs.get('bias') else [])))
        probe_cb={'tensors':{name:deepcopy(ctx.spec(name)) for name in names},'commands':[],
                  'kernel_abi':{'kind':'whole_program','args':[{'tensor':name,'access':'write' if name==dst else 'read'} for name in names],'outputs':[dst]}}
        probe=EmissionContext(probe_cb,symbol='dense_window_row_preflight')
        probe_view=_OutputRowContext(probe,mapping,ifm,dst,0)
        preflight=lower_dense(probe_view,ifm,rhs,dst,m=mapping.m,k=mapping.k,n=co,attributes=attrs,lhs_mapping=mapping)
        verified=probe.finish()
    except (KeyError,TypeError,ValueError,OverflowError):
        return False
    receipts=[]
    def body(row):
        view=_OutputRowContext(ctx,mapping,ifm,dst,row)
        receipt=lower_dense(view,ifm,rhs,dst,m=mapping.m,k=mapping.k,n=co,attributes=attrs,lhs_mapping=mapping)
        receipts.append(receipt)
        ctx.fence()
    rows=shape[0]*window.output_spatial[0]
    ctx.fb.for_loop(0,rows,1,body,retain_loop=True)
    ctx.metadata.setdefault('convolution_dense_windows',[]).append(dict(
        mechanism='direct_nonoverlapping_nhwc_counted_output_rows',mapping_sha256=mapping.identity,
        input_shape=list(shape),kernel=[kh,kw],dense_shape_per_row=[mapping.m,mapping.k,co],
        counted_output_rows=rows,source_row_pitch_bytes=mapping.source_row_pitch,
        input_row_base='((row//OH)*H+(row%OH)*KH)*W*CI',
        output_row_base='row*OW*CO',completion='fence at end of each counted output row',
        preflight_llvm_sha256=sha256(verified.encode()).hexdigest(),
        preflight_kernel_sha256=preflight['lhs_input_mapping']['kernel_sha256'],
        emitted_kernel_sha256=receipts[0]['lhs_input_mapping']['kernel_sha256'],
        payload_materialization=False,target_execution=False,timing=None))
    return True
