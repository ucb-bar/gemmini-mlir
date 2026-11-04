"""Callback-bound column layout preparation, separate from policy selection.

The checked bijection and shared host-copy adapter retain the frozen
experiment implementation. This entry never imports, captures or patches an
ordinary convolution emitter at module initialization.
"""
from math import prod
from dataclasses import dataclass
import hashlib
import struct
from gemmini_readout import _scale_bits, column_scale_runs, source_scales


@dataclass(frozen=True)
class ColumnGrouping:
    source_bits: tuple[int, ...]
    permutation: tuple[int, ...]  # grouped column -> original column
    inverse: tuple[int, ...]      # original column -> grouped column

    def verify(self):
        n = len(self.source_bits)
        if (not n or len(self.inverse) != n or len(self.permutation) != n
                or any(type(i) is not int for i in (*self.permutation,*self.inverse))
                or sorted(self.permutation) != list(range(n))
                or any(self.inverse[c] != j for j,c in enumerate(self.permutation))):
            raise ValueError('column grouping requires a complete bounded bijection')
        for bits in self.source_bits:
            if type(bits) is not int or not 0 <= bits < 2**32 or _scale_bits(_float(bits)) != bits:
                raise ValueError('column grouping needs unchanged positive finite f32 source bits')
        # One contiguous run per bit pattern, retaining first-occurrence order.
        expected = tuple(c for bits in dict.fromkeys(self.source_bits)
                         for c,value in enumerate(self.source_bits) if bits == value)
        if self.permutation != expected:
            raise ValueError('column grouping differs from exact source bit classes')

    @property
    def grouped_bits(self):
        self.verify()
        return tuple(self.source_bits[c] for c in self.permutation)

    @property
    def identity(self):
        self.verify()
        return hashlib.sha256(repr((self.source_bits,self.permutation,self.inverse)).encode()).hexdigest()


def _float(bits):
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def plan_grouping(scales, *, columns):
    if type(columns) is not int or not 0 < columns < 2**31:
        raise ValueError('column grouping needs a bounded source column count')
    values = tuple(scales) if isinstance(scales,(list,tuple)) else (scales,)*columns
    if len(values) != columns:
        raise ValueError('column scale vector differs from source columns')
    bits = tuple(_scale_bits(s) for s in values)
    order = tuple(c for b in dict.fromkeys(bits) for c,x in enumerate(bits) if x == b)
    inverse = [0]*columns
    for j,c in enumerate(order): inverse[c] = j
    result = ColumnGrouping(bits,order,tuple(inverse)); result.verify()
    return result


def _index_table(fn, permutation):
    from xdsl.dialects import builtin,llvm
    from merlin.llvmlower.host_builder import PTR
    globals_ = getattr(fn,'_gemmini_globals',None)
    if globals_ is None: globals_ = fn._gemmini_globals = []
    name = f'gemmini_column_bijection_{len(globals_)}'
    value = builtin.DenseIntOrFPElementsAttr.from_list(
        builtin.TensorType(builtin.i32,(len(permutation),)),list(permutation))
    globals_.append(llvm.GlobalOp(llvm.LLVMArrayType(len(permutation),builtin.i32),
        name,'internal',constant=True,value=value,alignment=64))
    return fn.add(llvm.AddressOfOp(name,PTR)).results[0]


def _copy_columns_indexed(host, source, destination, *, rows, permutation, dtype):
    """Exact bounded gather via shared StoredTensor -> counted copy lowering."""
    from gemmini_host import StoredTensor
    fn=host.fb; columns=len(permutation)
    if sorted(permutation) != list(range(columns)) or type(rows) is not int or rows <= 0:
        raise ValueError('column copy requires a complete bijection and static rows')
    table=_index_table(fn,permutation)
    def read(coordinates):
        row,column=coordinates
        original=fn.load_i64(table,column,'i32')
        offset=fn.add_i(fn.mul_i(row,fn.const(columns)),original)
        return host._read(source,offset,dtype)
    logical=StoredTensor(host,(rows,columns),dtype,lambda iv:read(host._coordinates(iv,(rows,columns))),
        coordinate_reader=read,traversal=((rows,columns),read))
    host._copy(logical,host._stored((rows,columns),dtype,destination))




def _copy_columns(host, source, destination, *, rows, permutation, dtype,
                  external_buffers_disjoint=False):
    from merlin.compile.tuning_knobs import knob
    from pathlib import Path
    mode=knob('conv_column_copy',default='affine',values=('indexed','affine'),root=Path(__file__).resolve().parent)
    if mode == 'affine':
        from gemmini_column_copy import copy_columns
        factor=knob('conv_column_copy_factor',default=1,values=(1,2,4,8),root=Path(__file__).resolve().parent)
        if factor != 1 and external_buffers_disjoint:
            from gemmini_column_copy import plan_column_copy
            from gemmini_copy_loops import copy_affine_columns
            short=knob('conv_column_copy_short',default='off',values=('off','on'),root=Path(__file__).resolve().parent)
            outer=knob('conv_column_copy_outer',default='off',values=('off','on'),root=Path(__file__).resolve().parent)
            plan=plan_column_copy(permutation)
            if plan is not None:
                return copy_affine_columns(host,source,destination,rows=rows,
                    plan=plan,dtype=dtype,factor=factor,
                    external_buffers_disjoint=True,short_inner=short == 'on',
                    outer_lanes=outer == 'on')
        return copy_columns(host,source,destination,rows=rows,permutation=permutation,
            dtype=dtype,generic_copy=_copy_columns_indexed)
    return _copy_columns_indexed(host,source,destination,rows=rows,permutation=permutation,dtype=dtype)


def select_column_grouping(*, attrs, columns, output_dtype, max_columns=16):
    """Return (checked grouping or None, structural selection diagnostics).

    Fewer hardware intervals is an admission diagnostic, not a cycle prediction.
    Policy remains with the caller; raw, uniform, identity and nonreducing cases
    retain the ordinary emitter without preparation.
    """
    report=dict(selected=False,reason='source output has no narrowing readout')
    if str(output_dtype) not in ('i8','int8'):
        return None,report
    attrs=dict(attrs or {})
    grouping=plan_grouping(source_scales(attrs),columns=columns)
    original=column_scale_runs(attrs,columns,max_columns=max_columns)
    prepared=dict(attrs,acc_scale_by_column=[_float(b) for b in grouping.grouped_bits])
    grouped=column_scale_runs(prepared,columns,max_columns=max_columns)
    report.update(original_intervals=len(original),grouped_intervals=len(grouped),
        scale_classes=len(set(grouping.source_bits)),grouping_identity=grouping.identity)
    if len(set(grouping.source_bits)) <= 1:
        report['reason']='uniform source scale bits'
    elif grouping.permutation == tuple(range(columns)):
        report['reason']='source columns already have grouped scale bits'
    elif len(grouped) >= len(original):
        report['reason']='grouping does not reduce original hardware intervals'
    else:
        report.update(selected=True,reason='exact mixed source bit classes reduce hardware intervals')
        return grouping,report
    return None,report


def emit_grouped_conv(fn,input_ptr,weight_ptr,dst_ptr,*,input_shape,weight_shape,
                      attrs=None,bias=None,output_dtype='i32',ordinary_emitter,prepared_emitter=None):
    """Prepare a checked layout and call the independently supplied emitter.

    The callback must implement the original complete synchronous convolution
    entry. All copies use private initialized storage and retain the original
    reduction coordinates; original device readout owns the numerical pipeline.
    """
    if not callable(ordinary_emitter):
        raise ValueError('column grouping requires the ordinary convolution emitter callback')
    from gemmini_conv import _geometry, _resources
    from merlin.llvmlower.host_scalar_builder import FpBuilder
    from gemmini_host import CompactHostLinalg
    from gemmini_support import scratch_buffer
    from gemmini_encode import fence
    attrs=dict(attrs or {})
    grouping,selection=select_column_grouping(attrs=attrs,columns=weight_shape[-1],
        output_dtype=output_dtype,max_columns=_resources()[0])
    if grouping is None:
        return ordinary_emitter(fn,input_ptr,weight_ptr,dst_ptr,input_shape=input_shape,
            weight_shape=weight_shape,attrs=attrs,bias=bias,output_dtype=output_dtype)
    window,columns=_geometry(input_shape,weight_shape,attrs)
    host=CompactHostLinalg(FpBuilder(fn),[],[])
    rhs=scratch_buffer(fn,prod(weight_shape),'column_group_rhs')
    prepared_bias=scratch_buffer(fn,columns*4,'column_group_bias') if bias is not None else None
    rows=window.input_shape[0]*prod(window.output_spatial)
    target=scratch_buffer(fn,rows*columns,'column_group_output')
    fence(fn)
    # Each pair contains a fresh internal scratch allocation. The external ABI
    # cannot alias these private symbols; distinct pointer values alone would
    # not establish the copy adapter's nonoverlap premise.
    copies={}
    copies['rhs']=_copy_columns(host,weight_ptr,rhs,rows=prod(weight_shape)//columns,
        permutation=grouping.permutation,dtype='i8',external_buffers_disjoint=True)
    if bias is not None:
        copies['bias']=_copy_columns(host,bias,prepared_bias,rows=1,
            permutation=grouping.permutation,dtype='i32',external_buffers_disjoint=True)
    prepared_attrs=dict(attrs,acc_scale_by_column=[_float(b) for b in grouping.grouped_bits])
    receipt=(prepared_emitter or ordinary_emitter)(fn,input_ptr,rhs,target,input_shape=input_shape,
        weight_shape=weight_shape,attrs=prepared_attrs,bias=prepared_bias,output_dtype=output_dtype)
    fence(fn)
    copies['output']=_copy_columns(host,target,dst_ptr,rows=rows,
        permutation=grouping.inverse,dtype='i8',external_buffers_disjoint=True)
    copy_recipe={}
    if any(isinstance(c,dict) and c.get('factor',1)!=1 for c in copies.values()):
        copy_recipe['copy_recipe']=copies
    return dict(receipt,column_grouping=dict(identity=grouping.identity,
        source_columns=columns,equal_scale_classes=len(set(grouping.source_bits)),
        rhs_copy_bytes=prod(weight_shape),bias_copy_bytes=columns*4 if bias is not None else 0,
        output_copy_bytes=rows*columns,original_reduction_order=True,
        selection=selection,**copy_recipe))
