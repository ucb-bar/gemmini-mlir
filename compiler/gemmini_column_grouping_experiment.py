"""Isolated exact-bit output-column grouping; never installed by default.

Only independent N coordinates change. Original kh/kw/CI order is unchanged,
RHS and bias follow the same checked bijection, and output observes its inverse.
Preparation uses the established counted host tensor-copy owner; contractions
and their full readout remain owned by the ordinary convolution emitter.
"""
from dataclasses import dataclass
from math import prod
import hashlib
import struct

from gemmini_conv import emit_conv as _original_emit_conv, _geometry
from gemmini_readout import _scale_bits


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


def _copy_columns(host, source, destination, *, rows, permutation, dtype):
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


def emit_grouped_conv(fn,input_ptr,weight_ptr,dst_ptr,*,input_shape,weight_shape,
                      attrs=None,bias=None,output_dtype='i32'):
    """Experimental emitter compatible with the normal emit_conv call boundary.

    Source operands are copied on every invocation into disjoint private BSS.
    No fixed-input cache, pointer-alias promise, scale arithmetic, or implicit
    asynchronous preparation is introduced. The ordinary owner validates all
    source epilogue and convolution admission before final module publication.
    """
    from merlin.llvmlower.host_scalar_builder import FpBuilder
    from gemmini_host import CompactHostLinalg
    from gemmini_support import scratch_buffer
    from gemmini_encode import fence
    attrs=dict(attrs or {})
    if str(output_dtype) not in ('i8','int8'):
        raise ValueError('column grouping experiment requires source hardware i8 readout')
    window,columns=_geometry(input_shape,weight_shape,attrs)
    scales=attrs.get('acc_scale_by_column',attrs.get('acc_scale',attrs.get('output_scale',attrs.get('scale',1.0))))
    grouping=plan_grouping(scales,columns=columns)
    if grouping.permutation == tuple(range(columns)):
        return _original_emit_conv(fn,input_ptr,weight_ptr,dst_ptr,input_shape=input_shape,
            weight_shape=weight_shape,attrs=attrs,bias=bias,output_dtype=output_dtype)
    host=CompactHostLinalg(FpBuilder(fn),[],[])
    rhs=scratch_buffer(fn,prod(weight_shape),'column_group_rhs')
    prepared_bias=scratch_buffer(fn,columns*4,'column_group_bias') if bias is not None else None
    rows=window.input_shape[0]*prod(window.output_spatial)
    target=scratch_buffer(fn,rows*columns,'column_group_output')
    fence(fn)
    _copy_columns(host,weight_ptr,rhs,rows=prod(weight_shape)//columns,
        permutation=grouping.permutation,dtype='i8')
    if bias is not None:
        _copy_columns(host,bias,prepared_bias,rows=1,permutation=grouping.permutation,dtype='i32')
    prepared_attrs=dict(attrs,acc_scale_by_column=[_float(b) for b in grouping.grouped_bits])
    receipt=_original_emit_conv(fn,input_ptr,rhs,target,input_shape=input_shape,
        weight_shape=weight_shape,attrs=prepared_attrs,bias=prepared_bias,output_dtype=output_dtype)
    fence(fn)
    _copy_columns(host,target,dst_ptr,rows=rows,permutation=grouping.inverse,dtype='i8')
    return dict(receipt,column_grouping=dict(identity=grouping.identity,
        source_columns=columns,equal_scale_classes=len(set(grouping.source_bits)),
        rhs_copy_bytes=prod(weight_shape),bias_copy_bytes=columns*4 if bias is not None else 0,
        output_copy_bytes=rows*columns,original_reduction_order=True))


def install_experiment():
    """Explicit isolated manifest-tool hook; no import-time production mutation."""
    import gemmini_conv
    gemmini_conv.emit_conv=emit_grouped_conv
