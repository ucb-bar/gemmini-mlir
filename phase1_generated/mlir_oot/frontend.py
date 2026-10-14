"""Declarative xDSL grammar for the frozen interface, with SSA verification."""
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute, Attribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, operand_def, result_def, var_operand_def, region_def
from xdsl.dialects.builtin import Builtin, TensorType, MemRefType, StringAttr, IntegerAttr, IntegerType, FloatAttr, ArrayAttr, DenseArrayBase, DenseIntOrFPElementsAttr, NoneAttr, IndexType, DYNAMIC_INDEX
from xdsl.dialects import func, arith, tensor, math, scf, cf
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.ops import BatchMatmulOp
from xdsl.dialects.utils import EnumAttribute
from xdsl.utils.str_enum import StrEnum
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class GenerateOp(IRDLOperation):
    """Structural tensor.generate grammar missing from the bundled xDSL dialect.

    The signature and assembly format follow installed upstream TensorOps.td.
    Its rank and element verifier accepts general valid source generation;
    initializer proof below separately restricts accelerator applicability.
    """
    name = 'tensor.generate'
    dynamic_extents = var_operand_def(IndexType)
    body = region_def('single_block')
    result = result_def(TensorType)
    assembly_format = '$dynamic_extents $body attr-dict `:` type($result)'

    def verify_(self):
        shape=self.result.type.get_shape()
        if len(self.dynamic_extents)!=sum(d==DYNAMIC_INDEX for d in shape):
            raise VerifyException('tensor.generate requires one index operand per dynamic extent')
        block=self.body.block
        if len(block.args)!=len(shape) or any(not isinstance(arg.type,IndexType) for arg in block.args):
            raise VerifyException('tensor.generate requires one index region argument per result dimension')
        terminator=block.last_op
        if (terminator is None or terminator.name!='tensor.yield'
                or len(terminator.operands)!=1
                or terminator.operands[0].type!=self.result.type.element_type):
            raise VerifyException('tensor.generate must yield exactly one result element')


class TypeFn(StrEnum):
    CAST_SIGNED = 'cast_signed'
    CAST_UNSIGNED = 'cast_unsigned'


@irdl_attr_definition
class TypeFnAttr(EnumAttribute[TypeFn]):
    name = 'linalg.type_fn'

    @classmethod
    def parse_parameter(cls, parser):
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer):
        with printer.in_angle_brackets():
            super().print_parameter(printer)

@irdl_attr_definition
class Resident(ParametrizedAttribute, TypeAttribute):
    name = 'merlin_iface.resident'

@irdl_attr_definition
class Accumulator(ParametrizedAttribute, TypeAttribute):
    name = 'merlin_iface.acc'
    element: Attribute

ARITY = {'tensor':0,'resident_pack':1,'matmul':2,'commit':1,'evict':1,'movement':1,'conv2d':2,'residual_add':2,'bias_add':2,'matmul_batched':2,'attention_qk':2,'attention_pv':2,'softmax':1,'rmsnorm':2,'rope':1}
OPCODES = {'resident_pack':'RES_PACK','matmul':'MATMUL_RESIDENT','commit':'COMMIT','evict':'EVICT','movement':'MOVEMENT','conv2d':'CONV2D','residual_add':'RESIDUAL_ADD','bias_add':'BIAS_ADD','matmul_batched':'BATCHED_MATMUL','attention_qk':'ATTENTION_QK','attention_pv':'ATTENTION_PV','softmax':'SOFTMAX','rmsnorm':'RMSNORM','rope':'ROPE'}
KEYS = {'RES_PACK':['src'],'MATMUL_RESIDENT':['lhs','rhs'],'COMMIT':['src'],'EVICT':['handle'],'MOVEMENT':['src'],'CONV2D':['ifm','weight'],'RESIDUAL_ADD':['lhs','rhs'],'BIAS_ADD':['src','bias'],'BATCHED_MATMUL':['a','w'],'ATTENTION_QK':['q','k'],'ATTENTION_PV':['p','v'],'SOFTMAX':['src'],'RMSNORM':['src','gamma'],'ROPE':['src']}

def dialect():
    classes = []
    for mnemonic,n in ARITY.items():
        fields = {'name':'merlin_iface.'+mnemonic}
        for i in range(n): fields['arg'+str(i)] = operand_def()
        if mnemonic != 'evict': fields['out'] = result_def()
        if mnemonic == 'tensor': fmt = 'attr-dict `:` type($out)'
        elif mnemonic == 'evict': fmt = '$arg0 attr-dict `:` `(` type($arg0) `)` `->` `(` `)`'
        else: fmt = ' `,` '.join('$arg'+str(i) for i in range(n)) + ' attr-dict `:` functional-type(operands, results)'
        fields['assembly_format'] = fmt
        classes.append(irdl_op_definition(type(mnemonic.title()+'Op',(IRDLOperation,),fields)))
    return Dialect('merlin_iface',classes,[Resident,Accumulator])

IFACE = dialect()
def parse(text):
    ctx = Context(allow_unregistered=True)
    for d in (Builtin,IFACE,func.Func,arith.Arith,tensor.Tensor,math.Math,scf.Scf,cf.Cf,Linalg): ctx.load_dialect(d)
    ctx.load_op(BatchMatmulOp)
    ctx.load_op(GenerateOp)
    ctx.load_attr_or_type(TypeFnAttr)
    module = Parser(ctx,text).parse_module()
    for op in module.walk():
        if op.name=='linalg.transpose':
            verify_transpose_signature(op)
        if op.name not in ('linalg.matmul','linalg.batch_matmul'):continue
        cast=op.properties.get('cast',op.attributes.get('cast'))
        if cast is not None and (not isinstance(cast,TypeFnAttr)
                                 or cast.data not in (TypeFn.CAST_SIGNED,TypeFn.CAST_UNSIGNED)):
            raise ValueError('Named contraction requires a valid linalg.type_fn cast')
    module.verify()
    return module

def attr_value(a):
    if isinstance(a,StringAttr): return a.data
    if isinstance(a,IntegerAttr): return a.value.data
    if isinstance(a,FloatAttr): return a.value.data
    if isinstance(a,ArrayAttr): return [attr_value(v) for v in a]
    raise ValueError('Unsupported attribute '+str(a))

def source_index(op, local_index):
    """Keep the original owner across extracted source-region boundaries.

    An unannotated original function uses its direct-operation order. A provided
    host extraction may already bind that order; never renumber such an owner
    merely because constants or other regions were removed from the leaf.
    """
    index=op.attributes.get('merlin.source_op_index')
    if index is None:return local_index
    if not isinstance(index,IntegerAttr) or index.value.data<0:
        raise ValueError('Invalid original source operation index')
    return index.value.data

def inventory(module):
    if any(op.name=='func.func' for op in module.body.block.ops): return inventory_function(module)
    if module.attributes.get('merlin_iface.version') != StringAttr('0.1'): raise ValueError('Unsupported interface grammar')
    cb = {'abi_version':'0.1','target':'gemmini','tensors':{},'commands':[]}
    bindings = {}
    encoded_tensor=False
    for index,op in enumerate(module.body.block.ops):
        # Neutral interface tensors use the same dense pointer boundary as
        # source contractions. Logical dimensions do not describe encoded
        # physical storage, including movement-only operands and declarations.
        # Keep all original tensor declarations and source attribution while
        # inventorying, then refuse the complete program without device work.
        encoded_tensor|=any(isinstance(v.type,TensorType)
                            and not isinstance(v.type.encoding,NoneAttr)
                            for v in [*op.operands,*op.results])
        mnemonic=op.name.split('.')[-1]
        attrs={k:attr_value(v) for k,v in op.attributes.items()}
        if mnemonic not in ARITY: raise ValueError('Unsupported source '+op.name)
        if op.results:
            name=attrs.pop('name',op.results[0].name_hint or 'v'+str(len(bindings)))
            bindings[op.results[0]]=name
            typ=op.results[0].type
            if isinstance(typ,TensorType):
                cb['tensors'][name]={'shape':list(typ.get_shape()),'dtype':str(typ.element_type),'role':attrs.pop('role','output')}
        if mnemonic=='tensor': continue
        attrs['merlin.source_op_index']=source_index(op,index)
        opcode=OPCODES[mnemonic]
        operands=dict(zip(KEYS[opcode],[bindings[v] for v in op.operands]))
        if op.results: operands['dst']=name
        cb['commands'].append({'opcode':opcode,'operands':operands,'attributes':attrs})
    if encoded_tensor:
        cb['commands']=[]
        cb['declined']={'reason':'Encoded interface tensor requires explicit physical-layout ABI and lowering; dense row-major pointer scheduling admits only unencoded TensorType operands and results','op':'interface_graph'}
        return cb
    if not cb['commands']: raise ValueError('No computation in source')
    return cb


def identity_slice_source(value):
    """Prove a full-extent static tensor.extract_slice is a pointer identity.

    Exact TensorType equality includes rank, element type, shape and encoding.
    Every dimension must have offset zero, its complete source extent as size,
    and stride one. Dynamic indices (even constant SSA indices), rank reduction,
    partial slices and type/encoding changes are refused. This introduces no
    storage, address adjustment or source operation rewriting.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.extract_slice'
            or len(owner.operands)!=1 or len(owner.results)!=1
            or value is not owner.results[0]
            or not isinstance(value.type,TensorType)):
        return None
    src=owner.operands[0]
    if (src.type!=value.type or str(value.type.element_type) not in ('i8','i32')
            or any(d<0 for d in value.type.get_shape())):
        return None
    shape=list(value.type.get_shape())
    expected={'static_offsets':[0]*len(shape),'static_sizes':shape,
              'static_strides':[1]*len(shape)}
    for key,indices in expected.items():
        attr=owner.properties.get(key,owner.attributes.get(key))
        if not isinstance(attr,DenseArrayBase) or list(attr.get_values())!=indices:
            return None
    return src


def tensor_storage_source(value):
    """Resolve proved static i8/i32 views to their original storage owner.

    Casts and full slices require exact TensorType equality. Reassociation and
    direct constant-shape reshapes preserve the contiguous element sequence
    after checking complete geometry and exact unencoded element types. Dynamic or malformed
    views remain opaque. The original operations are retained in the IR,
    so stripping a pointer alias never renumbers the contraction's owner.
    """
    seen=set()
    while value not in seen:
        seen.add(value)
        owner=value.owner
        if getattr(owner,'name','')=='linalg.transpose':
            source=identity_transpose_source(value)
            if source is None:break
            value=source
            continue
        if getattr(owner,'name','') in ('tensor.collapse_shape','tensor.expand_shape'):
            source=tensor_reshape_source(value)
            if source is None:break
            value=source
            continue
        if getattr(owner,'name','')=='tensor.reshape':
            source=tensor_shape_reshape_source(value)
            if source is None:break
            value=source
            continue
        if getattr(owner,'name','')=='tensor.extract_slice':
            source=identity_slice_source(value)
            if source is None:break
            value=source
            continue
        if getattr(owner,'name','')!='tensor.cast':break
        if (len(owner.operands)!=1 or len(owner.results)!=1
                or value is not owner.results[0]
                or not isinstance(value.type,TensorType)
                or owner.operands[0].type!=value.type
                or str(value.type.element_type) not in ('i8','i32')
                or any(d<0 for d in value.type.get_shape())):break
        value=owner.operands[0]
    return value


def identity_tensor_source(value):
    """Resolve storage-view chains only when the final boundary type is exact.

    Each step may change rank through a proved contiguous reshape, but
    the complete chain must restore the original shape, dtype and encoding.
    This keeps command-buffer tensor shapes identical to boundary declarations.
    A rank-changing final boundary remains opaque and is refused as before.
    """
    source=tensor_storage_source(value)
    return source if source.type==value.type else value


def verify_transpose_signature(op):
    """Complete the bundled upstream linalg.transpose signature verification.

    LinalgStructuredOps.td defines result axis i as input axis permutation[i].
    The bundled verifier checks dimensions but omits permutation bijectivity
    and precise element/result types. Check those structurally before invoking
    it, so malformed axes never reach its unchecked indexing operations.
    This verifies source grammar; static dense applicability is proved below.
    """
    inputs=list(op.inputs);outputs=list(op.outputs)
    if len(inputs)!=1 or len(outputs)!=1:
        raise ValueError('linalg.transpose requires exactly one input and one init')
    source,init=inputs[0],outputs[0]
    if (not isinstance(source.type,(TensorType,MemRefType))
            or not isinstance(init.type,type(source.type))
            or source.type.element_type!=init.type.element_type):
        raise ValueError('linalg.transpose requires matching shaped element types')
    rank=len(source.type.get_shape())
    permutation=op.properties.get('permutation',op.attributes.get('permutation'))
    if (not isinstance(permutation,DenseArrayBase) or str(permutation.elt_type)!='i64'
            or sorted(permutation.get_values())!=list(range(rank))
            or len(init.type.get_shape())!=rank):
        raise ValueError('linalg.transpose requires a complete rank permutation')
    if isinstance(init.type,TensorType):
        if len(op.results)!=1 or op.results[0].type!=init.type:
            raise ValueError('linalg.transpose tensor result must exactly match init type')
    elif op.results:
        raise ValueError('linalg.transpose memref form has no tensor result')


def transpose_source(value):
    """Prove the exact static dense contents permutation of linalg.transpose.

    Source, init and result are unencoded i8/i32 tensors with positive static
    extents. A complete i64 axis bijection maps every result extent exactly to
    its source extent. The named region consists solely of yielding its input
    scalar; init contents are overwritten and need no initialization proof.
    Invalid geometry, changed element types/encodings, dynamic/nonpositive
    extents or altered/effectful regions refuse. This is a contents relation,
    not a pointer alias: linalg.transpose actually moves data. The independent
    identity chain or zero-content proof below must justify eliminating it.
    Source operations and attribution stay intact; shape checks do not allocate
    and Python integer dimension arithmetic introduces no capacity limit.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='linalg.transpose'
            or len(owner.operands)!=2 or len(owner.results)!=1
            or value is not owner.results[0]
            or len(owner.inputs)!=1 or len(owner.outputs)!=1):return None
    source=owner.inputs[0];init=owner.outputs[0]
    if (not isinstance(source.type,TensorType)
            or not isinstance(init.type,TensorType)
            or not isinstance(value.type,TensorType)
            or str(source.type.element_type) not in ('i8','i32')
            or source.type.element_type!=value.type.element_type
            or init.type!=value.type
            or not isinstance(source.type.encoding,NoneAttr)
            or not isinstance(value.type.encoding,NoneAttr)):return None
    source_shape=list(source.type.get_shape());result_shape=list(value.type.get_shape())
    rank=len(source_shape)
    if len(result_shape)!=rank or any(d<=0 for d in source_shape+result_shape):return None
    attr=owner.properties.get('permutation',owner.attributes.get('permutation'))
    if not isinstance(attr,DenseArrayBase) or str(attr.elt_type)!='i64':return None
    permutation=list(attr.get_values())
    if (sorted(permutation)!=list(range(rank))
            or result_shape!=[source_shape[axis] for axis in permutation]):return None
    if len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1:return None
    block=owner.regions[0].block;body=list(block.ops)
    if (len(block.args)!=2
            or any(arg.type!=source.type.element_type for arg in block.args)
            or len(body)!=1 or body[0].name!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],)
            or body[0].results or body[0].regions):return None
    return source,permutation


def identity_transpose_source(value):
    """Alias a full transpose chain only after an exact identity-axis proof.

    Each adjacent transpose must satisfy transpose_source independently. Axis
    composition is traced back from the final result to the original source;
    both the complete composed permutation and exact original TensorType must
    be restored. Equal square shapes alone never prove storage identity. Any
    malformed/dynamic/encoded member or nonidentity composition refuses. No
    arbitrary transpose is stripped and no pointer offsets or storage change.
    """
    if not isinstance(value.type,TensorType):return None
    original=value;composition=list(range(len(value.type.get_shape())));seen=set()
    while getattr(value.owner,'name','')=='linalg.transpose':
        if value in seen:return None
        seen.add(value)
        relation=transpose_source(value)
        if relation is None:return None
        source,permutation=relation
        composition=[permutation[axis] for axis in composition]
        value=source
    if (seen and composition==list(range(len(composition)))
            and value.type==original.type):return value
    return None


def zero_transpose_source(value):
    """An exact i32 permutation preserves zero at every complete result cell."""
    if not isinstance(value.type,TensorType) or str(value.type.element_type)!='i32':return None
    relation=transpose_source(value)
    return relation[0] if relation is not None else None


def zero_copy_source(value):
    """Prove a full static dense i32 copy preserves every initialized zero.

    The registered linalg.copy must have exactly one input, one tensor init
    and one result, all with the same unencoded i32 TensorType and positive
    static extents. Its sole scalar block has exactly the two i32 arguments
    and a single yield of the input argument. Thus the operation overwrites
    every result cell, independent of the init's contents, with its matching
    input cell. The caller separately proves the complete input is zero.

    Dynamic/nonpositive shapes, layouts, casts, partial/mismatched signatures,
    and altered or effectful scalar bodies refuse. Rank zero covers its one
    scalar element. Shape checks use Python integers without allocating or
    truncating work. This is a contents proof only: no storage pointer alias,
    source rewrite, or change to source indices or region ownership is made.
    """
    from xdsl.dialects.linalg.ops import CopyOp
    owner=value.owner
    if (not isinstance(owner,CopyOp) or len(owner.operands)!=2
            or len(owner.inputs)!=1 or len(owner.outputs)!=1
            or len(owner.results)!=1 or value is not owner.results[0]
            or len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1):return None
    source=owner.inputs[0];init=owner.outputs[0]
    if (any(not isinstance(v.type,TensorType) for v in (source,init,value))
            or str(value.type.element_type)!='i32'
            or source.type!=value.type or init.type!=value.type
            or not isinstance(value.type.encoding,NoneAttr)
            or any(extent<=0 for extent in value.type.get_shape())):return None
    block=owner.regions[0].block;body=list(block.ops)
    if (len(block.args)!=2
            or any(arg.type!=value.type.element_type for arg in block.args)
            or len(body)!=1 or body[0].name!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],)
            or body[0].results or body[0].regions or body[0].successors):return None
    return source


def zero_broadcast_source(value):
    """Prove a static dense i32 broadcast initializes every result cell to zero.

    Input, init and result are unencoded i32 tensors with positive static
    extents, and the result type exactly equals the init type. Distinct sorted
    i64 added axes account for the complete rank increase; deleting those axes
    from the result shape restores the exact input shape. The registered named
    operation's single-block scalar region must solely yield its input element,
    so every output cell copies one input cell and the init is fully overwritten.
    The caller independently proves every input element zero. Dynamic/nonpositive
    shapes, changed element types or encodings, malformed axes and altered or
    effectful bodies refuse. This is only a contents proof: it never aliases
    storage or rewrites source operations and their ownership. Python integer
    dimension checks introduce no allocation limit or shape truncation.
    """
    from xdsl.dialects.linalg.ops import BroadcastOp
    owner=value.owner
    if (not isinstance(owner,BroadcastOp) or len(owner.operands)!=2
            or len(owner.results)!=1 or value is not owner.results[0]
            or len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1):return None
    source,init=owner.operands
    if (any(not isinstance(v.type,TensorType) for v in (source,init,value))
            or any(str(v.type.element_type)!='i32' for v in (source,init,value))
            or any(not isinstance(v.type.encoding,NoneAttr) for v in (source,init,value))
            or value.type!=init.type):return None
    source_shape=list(source.type.get_shape());result_shape=list(value.type.get_shape())
    if any(extent<=0 for extent in source_shape+result_shape):return None
    dimensions=owner.properties.get('dimensions',owner.attributes.get('dimensions'))
    if not isinstance(dimensions,DenseArrayBase) or str(dimensions.elt_type)!='i64':return None
    axes=list(dimensions.get_values());rank=len(result_shape)
    if (axes!=sorted(set(axes)) or any(axis<0 or axis>=rank for axis in axes)
            or len(source_shape)+len(axes)!=rank
            or [extent for axis,extent in enumerate(result_shape) if axis not in axes]!=source_shape):
        return None
    block=owner.regions[0].block;body=list(block.ops)
    if (len(block.args)!=2
            or any(arg.type!=value.type.element_type for arg in block.args)
            or len(body)!=1 or body[0].name!='linalg.yield'
            or tuple(body[0].operands)!=(block.args[0],)
            or body[0].results or body[0].regions):return None
    return source


def tensor_reshape_source(value):
    """Prove a static reassociation preserves the complete flat element sequence.

    Only plain dense row-major static positive i8/i32 tensors may alias storage
    through this view; equal opaque encodings do not prove a contiguous layout.
    Groups must cover each expanded dimension once in contiguous order, with
    products equal to the collapsed dimensions.
    No data is changed, generated or dropped, and the source IR stays intact.
    """
    owner=value.owner
    name=getattr(owner,'name','')
    if (name not in ('tensor.collapse_shape','tensor.expand_shape')
            or len(owner.operands)!=1 or len(owner.results)!=1
            or value is not owner.results[0]):return None
    src=owner.operands[0]
    if not isinstance(src.type,TensorType) or not isinstance(value.type,TensorType):return None
    if (str(src.type.element_type) not in ('i8','i32')
            or src.type.element_type!=value.type.element_type
            or src.type.encoding!=value.type.encoding
            or not isinstance(src.type.encoding,NoneAttr)
            or not isinstance(value.type.encoding,NoneAttr)):return None
    source_shape=list(src.type.get_shape());result_shape=list(value.type.get_shape())
    if any(d<=0 for d in source_shape+result_shape):return None
    collapsed,expanded=((result_shape,source_shape) if name=='tensor.collapse_shape'
                        else (source_shape,result_shape))
    if len(expanded)<len(collapsed):return None
    reassociation=owner.properties.get('reassociation',owner.attributes.get('reassociation'))
    if not isinstance(reassociation,ArrayAttr) or len(reassociation)!=len(collapsed):return None
    groups=[]
    for group in reassociation:
        if not isinstance(group,ArrayAttr) or not group:return None
        if any(not isinstance(dim,IntegerAttr) for dim in group):return None
        groups.append([dim.value.data for dim in group])
    if [dim for group in groups for dim in group]!=list(range(len(expanded))):return None
    for extent,group in zip(collapsed,groups):
        product=1
        for dim in group:product*=expanded[dim]
        if product!=extent:return None
    if name=='tensor.expand_shape':
        static_shape=owner.properties.get('static_output_shape',owner.attributes.get('static_output_shape'))
        if not isinstance(static_shape,DenseArrayBase) or list(static_shape.get_values())!=result_shape:return None
    return src


def zero_reshape_source(value):
    """Keep the initialization-only entrypoint restricted to i32 storage."""
    if not isinstance(value.type,TensorType) or str(value.type.element_type)!='i32':return None
    return tensor_reshape_source(value)


def tensor_shape_reshape_source(value):
    """Prove a constant-shape reshape preserves every source i8/i32 element.

    Source and result are unencoded dense i8/i32 tensors with positive static
    extents and equal complete element counts. The rank-one shape tensor is
    itself unencoded and comes directly from one arith.constant with exactly
    the result extents as its dense integer/index contents and exact type, or
    from tensor.from_elements whose operands are direct integer/index constants
    of the exact shape element type with those same complete extents.
    Dynamic/nonpositive extents, computed or malformed shapes, type/encoding
    changes and unequal counts refuse. This proves only a complete contiguous
    storage sequence; identity_tensor_source separately requires the full view
    chain to restore its original boundary type before aliasing that pointer.
    No source IR or contraction ownership changes. Products use arbitrary-
    precision integers; the proof introduces no allocation limit.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.reshape'
            or len(owner.operands)!=2 or len(owner.results)!=1
            or value is not owner.results[0] or owner.regions):return None
    source,shape=owner.operands
    if (not isinstance(source.type,TensorType)
            or not isinstance(value.type,TensorType)
            or str(source.type.element_type) not in ('i8','i32')
            or source.type.element_type!=value.type.element_type
            or not isinstance(source.type.encoding,NoneAttr)
            or not isinstance(value.type.encoding,NoneAttr)):return None
    source_shape=list(source.type.get_shape());result_shape=list(value.type.get_shape())
    if any(extent<=0 for extent in source_shape+result_shape):return None
    source_count=result_count=1
    for extent in source_shape:source_count*=extent
    for extent in result_shape:result_count*=extent
    if source_count!=result_count:return None
    if (not isinstance(shape.type,TensorType)
            or not isinstance(shape.type.encoding,NoneAttr)
            or not isinstance(shape.type.element_type,(IntegerType,IndexType))
            or list(shape.type.get_shape())!=[len(result_shape)]):return None
    constructor=shape.owner
    if (len(getattr(constructor,'results',()))!=1
            or shape is not constructor.results[0] or constructor.regions):return None
    if getattr(constructor,'name','')=='arith.constant':
        if constructor.operands:return None
        contents=constructor.properties.get('value',constructor.attributes.get('value'))
        if (not isinstance(contents,DenseIntOrFPElementsAttr)
                or contents.type!=shape.type
                or list(contents.iter_values())!=result_shape):return None
    elif getattr(constructor,'name','')=='tensor.from_elements':
        if len(constructor.operands)!=len(result_shape):return None
        for scalar,extent in zip(constructor.operands,result_shape):
            constant=scalar.owner
            if (scalar.type!=shape.type.element_type
                    or getattr(constant,'name','')!='arith.constant'
                    or constant.operands or len(constant.results)!=1
                    or scalar is not constant.results[0] or constant.regions):return None
            contents=constant.properties.get('value',constant.attributes.get('value'))
            if (not isinstance(contents,IntegerAttr) or contents.type!=scalar.type
                    or contents.value.data!=extent):return None
    else:return None
    return source


def zero_shape_reshape_source(value):
    """Keep constant-shape initialization proofs restricted to i32 storage."""
    if not isinstance(value.type,TensorType) or str(value.type.element_type)!='i32':return None
    return tensor_shape_reshape_source(value)


def zero_slice_source(value):
    """Prove a static i32 slice selects only source elements, including rank loss.

    Source and result have positive static extents, identical i32 element types
    and encodings. Nonnegative offsets and positive strides place the last
    selected index inside every source dimension. The result shape must match
    the static sizes in order after deleting only size-one dimensions, following
    upstream TensorOps.td's rank-reduction semantics; repeated unit dimensions
    are matched eagerly. Dropping a nonunit dimension or reordering dimensions
    refuses, as do dynamic operands and malformed geometry. The caller must
    independently prove all source elements zero. Each selected element remains
    zero even when its unit coordinates are omitted from the result type.
    A cropped, strided or rank-reduced zero slice provides initialization, never
    a storage alias: pointer-boundary handling retains its full-extent identity
    rule. This contents proof changes no source IR or ownership. Python integer
    bounds arithmetic cannot overflow or truncate a dimension.
    """
    owner=value.owner
    if (not isinstance(owner,tensor.ExtractSliceOp) or owner.regions
            or len(owner.operands)!=1 or len(owner.results)!=1
            or value is not owner.results[0]):return None
    source=owner.operands[0]
    if (not isinstance(source.type,TensorType) or not isinstance(value.type,TensorType)
            or str(source.type.element_type)!='i32'
            or source.type.element_type!=value.type.element_type
            or source.type.encoding!=value.type.encoding):return None
    source_shape=list(source.type.get_shape());result_shape=list(value.type.get_shape())
    rank=len(source_shape)
    if len(result_shape)>rank or any(d<=0 for d in source_shape+result_shape):return None
    indices=[]
    for key in ('static_offsets','static_sizes','static_strides'):
        attr=owner.properties.get(key,owner.attributes.get(key))
        if not isinstance(attr,DenseArrayBase) or str(attr.elt_type)!='i64':return None
        dims=list(attr.get_values())
        if len(dims)!=rank:return None
        indices.append(dims)
    offsets,sizes,strides=indices
    if (any(offset<0 for offset in offsets) or any(size<=0 for size in sizes)
            or any(stride<=0 for stride in strides)):
        return None
    result_axis=0
    for size in sizes:
        if result_axis<len(result_shape) and size==result_shape[result_axis]:
            result_axis+=1
        elif size!=1:
            return None
    if result_axis!=len(result_shape):return None
    if any(offset+(size-1)*stride>=extent
           for offset,size,stride,extent in zip(offsets,sizes,strides,source_shape)):
        return None
    return source


def zero_extract_source(value):
    """Prove a literal-coordinate extraction reads one initialized i32 zero.

    The registered pure tensor.extract reads an unencoded i32 tensor with
    positive static extents and returns exactly its scalar element type. There
    is one direct literal index constant per dimension, each statically inside
    its dimension. Rank-zero tensors admit the single scalar element with no
    coordinates. The caller independently proves the entire source is zero,
    so its selected scalar is zero without executing a tensor read on the host.

    Dynamic/nonpositive extents, encodings, non-index or computed coordinates,
    out-of-bounds coordinates and malformed/effectful operations refuse. Source
    zero obligations remain in the iterative proof worklist; no recursive
    proof, pointer alias, source rewrite or owner change is introduced. Bounds
    checks use Python integers and impose no allocation or shape truncation.
    """
    owner=value.owner
    if (not isinstance(owner,tensor.ExtractOp) or owner.regions or owner.successors
            or len(owner.results)!=1 or value is not owner.results[0]
            or not owner.operands or str(value.type)!='i32'):return None
    source,*coordinates=owner.operands
    if (not isinstance(source.type,TensorType)
            or source.type.element_type!=value.type
            or not isinstance(source.type.encoding,NoneAttr)):return None
    shape=list(source.type.get_shape())
    if any(extent<=0 for extent in shape) or len(coordinates)!=len(shape):return None
    for coordinate,extent in zip(coordinates,shape):
        constant=coordinate.owner
        if (not isinstance(coordinate.type,IndexType)
                or not isinstance(constant,arith.ConstantOp)
                or constant.operands or constant.regions or constant.successors
                or len(constant.results)!=1
                or coordinate is not constant.results[0]):return None
        literal=constant.properties.get('value',constant.attributes.get('value'))
        if (not isinstance(literal,IntegerAttr) or literal.type!=coordinate.type
                or not 0<=literal.value.data<extent):return None
    return source


def zero_insert_element_sources(value):
    """Prove scalar insertion preserves a complete dense i32 zero initializer.

    The registered pure tensor.insert must have one result exactly matching its
    dense destination, positive static extents, one i32 scalar and exactly one
    index operand per dimension. Each coordinate is a direct literal index
    constant proved inside its dimension; unknown bounds, dynamic/nonpositive
    extents, encodings, malformed signatures and any region refuse. The scalar
    is a direct literal i32 zero. The caller separately proves the destination
    entirely zero, so the selected cell and every unchanged cell remain zero.
    This content proof creates no storage alias, rewrites no source operations
    and preserves their original ownership. Arbitrary-precision bounds checks
    impose no resident-storage limit or dimension truncation.
    """
    owner=value.owner
    if (not isinstance(owner,tensor.InsertOp) or owner.regions
            or len(owner.results)!=1 or value is not owner.results[0]
            or not isinstance(value.type,TensorType)
            or str(value.type.element_type)!='i32'
            or not isinstance(value.type.encoding,NoneAttr)):return None
    shape=list(value.type.get_shape())
    if any(extent<=0 for extent in shape) or len(owner.operands)!=2+len(shape):return None
    scalar,destination,*coordinates=owner.operands
    if destination.type!=value.type or scalar.type!=value.type.element_type:return None
    constant=scalar.owner
    if (getattr(constant,'name','')!='arith.constant' or constant.operands
            or constant.regions or len(constant.results)!=1
            or scalar is not constant.results[0]):return None
    literal=constant.properties.get('value',constant.attributes.get('value'))
    if (not isinstance(literal,IntegerAttr) or literal.type!=scalar.type
            or literal.value.data!=0):return None
    for coordinate,extent in zip(coordinates,shape):
        constant=coordinate.owner
        if (not isinstance(coordinate.type,IndexType)
                or getattr(constant,'name','')!='arith.constant'
                or constant.operands or constant.regions
                or len(constant.results)!=1
                or coordinate is not constant.results[0]):return None
        literal=constant.properties.get('value',constant.attributes.get('value'))
        if (not isinstance(literal,IntegerAttr) or literal.type!=coordinate.type
                or not 0<=literal.value.data<extent):return None
    return scalar,destination


def zero_insert_sources(value):
    """Prove an i32 insertion replaces only in-bounds destination elements.

    Source, destination and result have matching element types and encodings;
    the result type exactly equals the destination type. Positive static shapes
    preserve rank. Static sizes equal the complete source shape, nonnegative
    offsets and positive strides place the last selected index inside each
    destination extent. Dynamic operands, rank reduction and malformed geometry
    refuse. The caller must independently prove both input tensors entirely zero,
    covering inserted cells and every destination cell left unchanged. This is
    only an initialization proof; insertion never becomes a storage alias.
    Python integer bounds arithmetic cannot overflow or truncate a dimension.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.insert_slice'
            or len(owner.operands)!=2 or len(owner.results)!=1
            or value is not owner.results[0]):return None
    source,destination=owner.operands
    if (not isinstance(source.type,TensorType)
            or not isinstance(destination.type,TensorType)
            or not isinstance(value.type,TensorType)
            or str(source.type.element_type)!='i32'
            or source.type.element_type!=destination.type.element_type
            or source.type.encoding!=destination.type.encoding
            or value.type!=destination.type):return None
    source_shape=list(source.type.get_shape())
    destination_shape=list(destination.type.get_shape())
    rank=len(destination_shape)
    if len(source_shape)!=rank or any(d<=0 for d in source_shape+destination_shape):return None
    indices=[]
    for key in ('static_offsets','static_sizes','static_strides'):
        attr=owner.properties.get(key,owner.attributes.get(key))
        if not isinstance(attr,DenseArrayBase) or str(attr.elt_type)!='i64':return None
        dims=list(attr.get_values())
        if len(dims)!=rank:return None
        indices.append(dims)
    offsets,sizes,strides=indices
    if (sizes!=source_shape or any(offset<0 for offset in offsets)
            or any(size<=0 for size in sizes) or any(stride<=0 for stride in strides)):
        return None
    if any(offset+(size-1)*stride>=extent
           for offset,size,stride,extent in zip(offsets,sizes,strides,destination_shape)):
        return None
    return source,destination


def zero_pad_source(value):
    """Prove an i32 pad adds only zeros around its complete source tensor.

    Positive static dense extents, nonnegative static padding and exact dimension
    arithmetic cover every result element. The index-only region must yield a
    literal i32 zero without computing from its coordinates. Its only operations
    are the yielded literal (when locally defined) and the yield; a captured
    literal is also valid upstream tensor.pad syntax. Dynamic operands, encoded
    tensors, type changes, malformed shapes and other region computations refuse.
    The caller must independently prove the original source is entirely zero;
    this is an initialization proof, never a boundary-storage alias.
    The original operations, SSA ownership and source indices stay intact.
    Python integer sums cannot overflow or truncate dimension arithmetic.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.pad' or len(owner.operands)!=1
            or len(owner.results)!=1 or value is not owner.results[0]
            or len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1):return None
    source=owner.operands[0]
    if (not isinstance(source.type,TensorType) or not isinstance(value.type,TensorType)
            or str(source.type.element_type)!='i32'
            or source.type.element_type!=value.type.element_type
            or not isinstance(source.type.encoding,NoneAttr)
            or not isinstance(value.type.encoding,NoneAttr)):return None
    source_shape=list(source.type.get_shape());result_shape=list(value.type.get_shape())
    rank=len(source_shape)
    if len(result_shape)!=rank or any(d<=0 for d in source_shape+result_shape):return None
    padding=[]
    for key in ('static_low','static_high'):
        attr=owner.properties.get(key,owner.attributes.get(key))
        if not isinstance(attr,DenseArrayBase) or str(attr.elt_type)!='i64':return None
        amounts=list(attr.get_values())
        if len(amounts)!=rank or any(amount<0 for amount in amounts):return None
        padding.append(amounts)
    if any(src+low+high!=dst for src,low,high,dst
           in zip(source_shape,*padding,result_shape)):return None
    block=owner.regions[0].block
    if len(block.args)!=rank or any(not isinstance(arg.type,IndexType) for arg in block.args):return None
    body=list(block.ops)
    if not body:return None
    terminator=body[-1]
    if (terminator.name!='tensor.yield' or len(terminator.operands)!=1
            or terminator.results or terminator.regions):return None
    scalar=terminator.operands[0];constant=scalar.owner
    if (scalar.type!=value.type.element_type or getattr(constant,'name','')!='arith.constant'
            or len(constant.operands)!=0 or len(constant.results)!=1
            or scalar is not constant.results[0] or constant.regions):return None
    attr=constant.properties.get('value',constant.attributes.get('value'))
    if not isinstance(attr,IntegerAttr) or attr.type!=value.type.element_type or attr.value.data!=0:return None
    if body!=([constant,terminator] if constant in body else [terminator]):return None
    return source


def zero_from_elements_sources(value):
    """Prove a static i32 constructor covers every result element exactly once.

    Positive static extents (including the one element of a rank-zero tensor)
    determine the exact scalar operand count. Every scalar has the precise i32
    element type; the caller independently proves each scalar a literal zero.
    Dynamic/nonpositive shapes, type changes and malformed cardinality refuse.
    The constructor creates tensor contents and never aliases boundary storage.
    Python integer products cannot overflow or truncate the element count.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.from_elements'
            or len(owner.results)!=1 or value is not owner.results[0]
            or not isinstance(value.type,TensorType)
            or str(value.type.element_type)!='i32'):return None
    shape=list(value.type.get_shape())
    if any(d<=0 for d in shape):return None
    count=1
    for extent in shape:count*=extent
    if (len(owner.operands)!=count
            or any(scalar.type!=value.type.element_type for scalar in owner.operands)):
        return None
    return tuple(owner.operands)


def zero_concat_sources(value):
    """Prove static i32 concatenation covers its result with complete inputs.

    Every operand has the same positive static rank, exact i32 element type and
    encoding as the result. A valid i64 dimension selects the only axis whose
    result extent is the exact sum of input extents; every other extent agrees.
    The caller independently proves all inputs zero. Empty operands, rank-zero
    tensors, dynamic/nonpositive extents and malformed geometry refuse. This
    content proof never introduces pointer aliases or rewrites source ownership.
    Python integer sums cannot overflow or truncate the concatenated extent.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.concat' or not owner.operands
            or len(owner.results)!=1 or value is not owner.results[0]
            or not isinstance(value.type,TensorType)
            or str(value.type.element_type)!='i32'):return None
    result_shape=list(value.type.get_shape());rank=len(result_shape)
    if rank==0 or any(d<=0 for d in result_shape):return None
    dimension=owner.properties.get('dim',owner.attributes.get('dim'))
    if not isinstance(dimension,IntegerAttr) or str(dimension.type)!='i64':return None
    axis=dimension.value.data
    if not 0<=axis<rank:return None
    concatenated=0
    for source in owner.operands:
        if (not isinstance(source.type,TensorType)
                or source.type.element_type!=value.type.element_type
                or source.type.encoding!=value.type.encoding):return None
        shape=list(source.type.get_shape())
        if len(shape)!=rank or any(d<=0 for d in shape):return None
        if any(extent!=result_shape[i] for i,extent in enumerate(shape) if i!=axis):
            return None
        concatenated+=shape[axis]
    if concatenated!=result_shape[axis]:return None
    return tuple(owner.operands)


def zero_generate_scalar(value):
    """Prove static dense i32 generation writes the same literal to every cell.

    The exact single tensor result has positive static extents and no encoding;
    there are no dynamic extent operands. Its single-block region receives one
    index argument per result dimension and ends in one i32 tensor.yield. The
    only admitted region operation before that terminator is its yielded
    arith.constant, or the region directly captures an external arith.constant.
    The caller independently proves that literal is exactly i32 zero. Thus the
    complete result is initialized, without coordinate dependence or effects.
    Other computations, effects, dynamic/nonpositive extents, encoded results,
    malformed regions and nonzero literals refuse. This creates no pointer alias
    and leaves original operations and source indices intact. Shape checks use
    arbitrary-precision integers and introduce no resident allocation limit.
    """
    owner=value.owner
    if (getattr(owner,'name','')!='tensor.generate' or owner.operands
            or len(owner.results)!=1 or value is not owner.results[0]
            or not isinstance(value.type,TensorType)
            or str(value.type.element_type)!='i32'
            or not isinstance(value.type.encoding,NoneAttr)
            or len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1):return None
    shape=list(value.type.get_shape())
    if any(extent<=0 for extent in shape):return None
    block=owner.regions[0].block
    if (len(block.args)!=len(shape)
            or any(not isinstance(arg.type,IndexType) for arg in block.args)):return None
    body=list(block.ops)
    if not body:return None
    terminator=body[-1]
    if (terminator.name!='tensor.yield' or len(terminator.operands)!=1
            or terminator.results or terminator.regions):return None
    scalar=terminator.operands[0];constant=scalar.owner
    if (scalar.type!=value.type.element_type
            or getattr(constant,'name','')!='arith.constant'
            or constant.operands or len(constant.results)!=1
            or scalar is not constant.results[0] or constant.regions):return None
    if body!=([constant,terminator] if constant in body else [terminator]):return None
    return scalar


def zero_generic_scalar(value):
    """Prove a generic initializer performs a complete literal i32 write.

    The single output/result is an identical positive-static unencoded i32
    tensor. Its symbol-free output map permutes every parallel loop dimension,
    fixing the complete loop domain and writing each result cell exactly once.
    Every input is a positive-static dense tensor with a symbol-free projected
    permutation map: mapped extents agree with the output-derived loop bounds;
    constant-zero indices are admitted only for size-one broadcast dimensions.
    Region arguments exactly match the input/output element types. The region
    contains only its yielded i32 constant and terminator, or directly captures
    an external i32 constant. No computation, effect or coordinate dependence
    is admitted. The caller checks that the literal is exactly zero.

    Reduction/partial maps, symbols, general affine indexing, dynamic/encoded
    storage and malformed scalar regions refuse. Rank-zero tensors have one
    scalar write. Checks use arbitrary-precision shape integers and introduce
    no allocation limit, pointer alias, source rewrite or ownership change.
    """
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr
    owner=value.owner
    if (getattr(owner,'name','')!='linalg.generic'
            or len(owner.results)!=1 or value is not owner.results[0]
            or not isinstance(value.type,TensorType)
            or str(value.type.element_type)!='i32'
            or not isinstance(value.type.encoding,NoneAttr)
            or len(owner.regions)!=1 or len(owner.regions[0].blocks)!=1):return None
    inputs=list(owner.inputs);outputs=list(owner.outputs)
    if len(outputs)!=1 or outputs[0].type!=value.type:return None
    shape=list(value.type.get_shape());rank=len(shape)
    if any(extent<=0 for extent in shape):return None
    maps=[amap.data for amap in owner.get_indexing_maps()]
    iterators=[iterator.data.value for iterator in owner.get_iterator_types()]
    if (len(maps)!=len(inputs)+1 or len(iterators)!=rank
            or any(iterator!='parallel' for iterator in iterators)
            or any(amap.num_dims!=rank or amap.num_symbols for amap in maps)):
        return None
    output_map=maps[-1]
    if (len(output_map.results)!=rank
            or any(not isinstance(expr,AffineDimExpr) for expr in output_map.results)):
        return None
    dimensions=[expr.position for expr in output_map.results]
    if set(dimensions)!=set(range(rank)):return None
    loop_extents={dimension:extent for dimension,extent in zip(dimensions,shape)}
    for source,amap in zip(inputs,maps):
        if (not isinstance(source.type,TensorType)
                or not isinstance(source.type.encoding,NoneAttr)):return None
        source_shape=list(source.type.get_shape())
        if (len(amap.results)!=len(source_shape)
                or any(extent<=0 for extent in source_shape)):return None
        used=set()
        for extent,expr in zip(source_shape,amap.results):
            if isinstance(expr,AffineDimExpr):
                if (expr.position not in loop_extents or expr.position in used
                        or extent!=loop_extents[expr.position]):return None
                used.add(expr.position)
            elif not isinstance(expr,AffineConstantExpr) or expr.value!=0 or extent!=1:
                return None
    block=owner.regions[0].block
    if [arg.type for arg in block.args]!=[
            source.type.element_type for source in [*inputs,*outputs]]:return None
    body=list(block.ops)
    if not body:return None
    terminator=body[-1]
    if (terminator.name!='linalg.yield' or len(terminator.operands)!=1
            or terminator.results or terminator.regions):return None
    scalar=terminator.operands[0];constant=scalar.owner
    if (scalar.type!=value.type.element_type
            or getattr(constant,'name','')!='arith.constant'
            or constant.operands or len(constant.results)!=1
            or scalar is not constant.results[0] or constant.regions):return None
    if body!=([constant,terminator] if constant in body else [terminator]):return None
    return scalar


def zero_select_sources(value):
    """Prove a scalar i32 select is zero independently of its i1 condition.

    The registered pure arith.select has exactly three operands and one scalar
    signless i32 result, with an exact scalar signless i1 condition and two
    matching i32 alternatives. Regions, successors and malformed signatures
    refuse. Both alternatives become separate zero-proof obligations: even a
    literal condition never permits skipping the other alternative. Their
    proofs use the existing iterative worklist, including shared/deep DAGs.
    This contents equivalence introduces no condition evaluation, host
    arithmetic, tensor storage alias, source rewrite or ownership change.
    """
    owner=value.owner
    if (not isinstance(owner,arith.SelectOp) or owner.regions or owner.successors
            or len(owner.operands)!=3 or len(owner.results)!=1
            or value is not owner.results[0] or value.type!=IntegerType(32)):
        return None
    condition,lhs,rhs=owner.operands
    if (condition.type!=IntegerType(1)
            or lhs.type!=value.type or rhs.type!=value.type):return None
    return (lhs,rhs)



def zero_arithmetic_sources(value):
    """Prove scalar i32 add/subtract/multiply zero from two independent zeros.

    Only registered pure arith.addi/arith.subi/arith.muli with exactly two scalar
    signless i32 operands and one identical scalar result are admitted. Zero
    plus/minus/times zero is exact zero modulo 2**32, without signed or unsigned
    overflow, so absent overflowFlags or the typed empty/NSW/NUW flag
    sets are safe. Unknown properties, attributes or overflow flags, malformed
    types/signatures, regions and successors refuse. The two existing source
    identity attributes are inert metadata and remain untouched.

    Both operands become separate obligations in the stack-safe zero worklist;
    cancellation, absorbing zero factors, literal right units or unproved
    operands never bypass either independent zero obligation.
    There are no shape/resource limits beyond this scalar signature. This
    contents equivalence emits no arithmetic, rewrites no source IR and changes
    no tensor alias, storage lifetime, source index or accelerator ownership.
    """
    owner=value.owner
    if (not isinstance(owner,(arith.AddiOp,arith.SubiOp,arith.MuliOp))
            or owner.regions or owner.successors
            or len(owner.operands)!=2 or len(owner.results)!=1
            or value is not owner.results[0] or value.type!=IntegerType(32)):
        return None
    lhs,rhs=owner.operands
    if lhs.type!=value.type or rhs.type!=value.type:return None
    if (set(owner.properties)-{'overflowFlags'}
            or set(owner.attributes)-{'merlin.source_op_index','prov.region_id'}):
        return None
    flags=owner.properties.get('overflowFlags')
    if flags is not None and (not isinstance(flags,arith.IntegerOverflowAttr)
            or not flags.data.issubset({arith.IntegerOverflowFlag.NSW,
                                      arith.IntegerOverflowFlag.NUW})):
        return None
    return (lhs,rhs)


def zero_dependencies(value):
    """Check one zero-proof node and return its immediate proof obligations.

    None refuses the node; an empty tuple proves a literal-zero root. Every
    constructor keeps its existing independent geometry, type, encoding and
    region validator. This routine never descends into the source SSA graph.
    """
    owner=value.owner
    name=getattr(owner,'name','')
    if name=='arith.select':
        return zero_select_sources(value)
    if name in ('arith.addi','arith.subi','arith.muli'):
        return zero_arithmetic_sources(value)
    if name=='linalg.transpose':
        source=zero_transpose_source(value)
        return (source,) if source is not None else None
    if name=='linalg.broadcast':
        source=zero_broadcast_source(value)
        return (source,) if source is not None else None
    if name=='linalg.copy':
        source=zero_copy_source(value)
        return (source,) if source is not None else None
    if name in ('tensor.collapse_shape','tensor.expand_shape'):
        source=zero_reshape_source(value)
        return (source,) if source is not None else None
    if name=='tensor.reshape':
        source=zero_shape_reshape_source(value)
        return (source,) if source is not None else None
    if name=='tensor.extract_slice':
        source=zero_slice_source(value)
        return (source,) if source is not None else None
    if name=='tensor.extract':
        source=zero_extract_source(value)
        return (source,) if source is not None else None
    if name=='tensor.insert_slice':
        sources=zero_insert_sources(value)
        return sources
    if name=='tensor.insert':
        sources=zero_insert_element_sources(value)
        return sources
    if name=='tensor.pad':
        source=zero_pad_source(value)
        return (source,) if source is not None else None
    if name=='tensor.from_elements':
        sources=zero_from_elements_sources(value)
        return sources
    if name=='tensor.concat':
        sources=zero_concat_sources(value)
        return sources
    if name=='tensor.generate':
        scalar=zero_generate_scalar(value)
        return (scalar,) if scalar is not None else None
    if name=='linalg.generic':
        scalar=zero_generic_scalar(value)
        return (scalar,) if scalar is not None else None
    if name=='arith.constant':
        attr=owner.properties.get('value',owner.attributes.get('value'))
        if isinstance(attr,IntegerAttr):
            return (() if (len(owner.operands)==0 and len(owner.results)==1
                           and value is owner.results[0] and str(value.type)=='i32'
                           and attr.type==value.type and attr.value.data==0) else None)
        if isinstance(attr,DenseIntOrFPElementsAttr):
            return (() if str(attr.get_element_type())=='i32'
                    and all(x==0 for x in attr.iter_values()) else None)
        return None
    if name=='tensor.splat':
        return ((owner.operands[0],) if len(owner.operands)==1
                and str(owner.operands[0].type)=='i32' else None)
    if name=='linalg.fill':
        inputs=list(owner.inputs);outputs=list(owner.outputs)
        return ((inputs[0],) if len(inputs)==1 and len(outputs)==1 and len(owner.results)==1
                and value is owner.results[0] and value.type==outputs[0].type
                and str(inputs[0].type)=='i32' else None)
    return None


def zero_initialized(value):
    """Prove complete zero initialization with a stack-safe SSA worklist.

    Every existing constructor validator supplies immediate obligations through
    zero_dependencies. Iterative storage-view resolution retains exact pointer
    alias rules. Explicit postorder frames prove each distinct normalized node
    once per invocation, so deep initializer DAGs do not grow the Python stack
    and shared producers do not cause exponential revisits. An edge to an active
    proof node is a cycle and refuses its dependent path. Memoization is local
    to this proof: mutations or separate compiler invocations cannot reuse stale
    conclusions. Empty/unbound tensors never imply zero; source operations and
    their original indices/regions remain untouched.
    """
    resolved={}
    def resolve(source):
        if source not in resolved:resolved[source]=tensor_storage_source(source)
        return resolved[source]
    root=resolve(value);memo={};active=set();work=[]
    def enter(source):
        dependencies=zero_dependencies(source)
        if dependencies is None:
            memo[source]=False
        elif not dependencies:
            memo[source]=True
        else:
            active.add(source)
            work.append((source,tuple(resolve(dep) for dep in dependencies),0))
    enter(root)
    while work:
        source,dependencies,index=work[-1]
        if index==len(dependencies):
            memo[source]=all(memo[dep] for dep in dependencies)
            active.remove(source);work.pop()
            continue
        dependency=dependencies[index]
        if dependency in active or (dependency in memo and not memo[dependency]):
            memo[source]=False
            active.remove(source);work.pop()
            continue
        work[-1]=(source,dependencies,index+1)
        if dependency not in memo:enter(dependency)
    return memo[root]


def signed_mac_body(op):
    """Match the scalar SSA DAG, independent of extension order and SSA names.

    The only admitted computation is signed i8-to-i32 extension, multiplication
    and wrapping i32 accumulation. There are exactly five scalar operations; no
    host arithmetic, casts, altered arithmetic flags or epilogues are inferred.
    """
    if op.name not in ('linalg.generic','linalg.matmul','linalg.batch_matmul'):return False
    if op.name in ('linalg.matmul','linalg.batch_matmul'):
        cast=op.properties.get('cast',op.attributes.get('cast'))
        if cast is not None and (not isinstance(cast,TypeFnAttr)
                                 or cast.data!=TypeFn.CAST_SIGNED):return False
    if len(op.body.blocks)!=1:return False
    block=op.body.block;args=list(block.args);body=list(block.ops)
    if len(args)!=3 or [str(v.type) for v in args]!=['i8','i8','i32'] or len(body)!=5:return False
    terminator=body[-1]
    if terminator.name!='linalg.yield' or len(terminator.operands)!=1:return False
    add=terminator.operands[0].owner
    if getattr(add,'name','')!='arith.addi' or add not in body:return False
    if args[2] not in add.operands:return False
    products=[v for v in add.operands if v is not args[2]]
    if len(products)!=1:return False
    mul=products[0].owner
    if getattr(mul,'name','')!='arith.muli' or mul not in body:return False
    extensions=[v.owner for v in mul.operands]
    if (len(extensions)!=2 or len(set(extensions))!=2
            or any(getattr(x,'name','')!='arith.extsi' or x not in body or len(x.operands)!=1 for x in extensions)
            or {x.operands[0] for x in extensions}!={args[0],args[1]}):return False
    if set(body)!={*extensions,mul,add,terminator}:return False
    if any(str(v.type)!='i32' for step in (*extensions,mul,add) for v in step.results):return False
    for step in (mul,add):
        flags=step.properties.get('overflowFlags')
        if flags is not None and flags.data:return False
    return True


def row_major_contraction(op):
    """Check rank-N A[B,M,K] @ W[B,K,N] -> Y[B,M,N] from affine IR.

    Loop dimensions may be numbered or ordered arbitrarily. Each map must be a
    symbol-free projected permutation with identical physical batch prefixes.
    A transposed matrix, broadcast batch, affine offset, multiple reduction,
    dynamic/zero extent or shape mismatch refuses instead of relabelling memory.
    Every input, initializer and result must have no tensor encoding: the pointer
    scheduler implements contiguous dense row-major storage and cannot infer a
    physical layout from an opaque encoding, even when logical shapes agree.
    The existing scheduler performs each nonempty batch independently, retaining
    its i32 output tiles through all K slices and covering matrix tails.
    """
    from xdsl.ir.affine import AffineDimExpr
    inputs=list(op.inputs);outputs=list(op.outputs)
    if len(inputs)!=2 or len(outputs)!=1 or len(op.results)!=1:return False
    tensors=[*inputs,outputs[0],op.results[0]]
    if any(not isinstance(v.type,TensorType) for v in tensors):return False
    if any(not isinstance(v.type.encoding,NoneAttr) for v in tensors):return False
    if [str(v.type.element_type) for v in tensors]!=['i8','i8','i32','i32']:return False
    ash,bsh,initsh,ysh=[list(v.type.get_shape()) for v in tensors]
    rank=len(ash)
    if (rank<2 or len(bsh)!=rank or len(ysh)!=rank or initsh!=ysh
            or any(d<=0 for shape in (ash,bsh,ysh) for d in shape)
            or ash[:-2]!=bsh[:-2] or ash[:-2]!=ysh[:-2]
            or ash[-1]!=bsh[-2] or ysh[-2:]!=[ash[-2],bsh[-1]]):return False
    if op.name=='linalg.batch_matmul' and rank!=3:return False
    maps=[m.data for m in op.get_indexing_maps()]
    iterators=[it.data.value for it in op.get_iterator_types()]
    if len(maps)!=3 or len(iterators)!=rank+1:return False
    actual=[]
    for amap in maps:
        if amap.num_symbols or amap.num_dims!=rank+1 or len(amap.results)!=rank:return False
        if any(not isinstance(expr,AffineDimExpr) for expr in amap.results):return False
        dims=[expr.position for expr in amap.results]
        if len(set(dims))!=rank:return False
        actual.append(dims)
    a,b,y=actual;prefix=y[:-2];m,n=y[-2:];k=a[-1]
    if (a!=prefix+[m,k] or b!=prefix+[k,n]
            or set(prefix+[m,n,k])!=set(range(rank+1))):return False
    return all(it==('reduction' if i==k else 'parallel') for i,it in enumerate(iterators))


def inventory_function(module):
    """Source-bound device regions and declarations for the provided host compiler.

    No pure host layer is implemented here. A mixed whole graph is reported as a
    composition boundary; the host provider must split it into legal device regions.
    """
    funcs=[op for op in module.body.block.ops if op.name=='func.func']
    if len(funcs)!=1 or len(funcs[0].body.blocks)!=1:raise ValueError('Require one single-block source function')
    block=funcs[0].body.block
    cb={'abi_version':'0.1','target':'gemmini','tensors':{},'commands':[],'params':{'lane_placement':[]}}
    bindings={}
    input_bindings={}
    for i,v in enumerate(block.args):
        if not isinstance(v.type,TensorType):raise ValueError('Device boundary requires tensor pointers')
        name='I'+str(i);bindings[v]=name
        input_bindings[v]=name
        cb['tensors'][name]={'shape':list(v.type.get_shape()),'dtype':str(v.type.element_type),'role':'input'}
    ret=block.last_op
    if ret.name!='func.return':raise ValueError('Missing source return')
    output_bindings={}
    for i,v in enumerate(ret.operands):
        if not isinstance(v.type,TensorType):raise ValueError('Device results require tensor pointers')
        name='Y'+str(i);bindings[v]=name
        output_bindings[identity_tensor_source(v)]=name
        cb['tensors'][name]={'shape':list(v.type.get_shape()),'dtype':str(v.type.element_type),'role':'output'}
    eligible=[]
    for index,op in enumerate(block.ops):
        if op.name=='func.return':continue
        attrs=op.attributes;region=attrs.get('prov.region_id');family=attrs.get('prov.family')
        inputs=list(getattr(op,'inputs',[]))
        # A proved literal-zero generic is initialization, even when its unused
        # inputs happen to look like an i8 contraction. Do not invent MAC work
        # or accelerator ownership for that source operation.
        zero_generic=(op.name=='linalg.generic' and len(op.results)==1
                      and zero_initialized(op.results[0]))
        device=(not zero_generic and op.name in ('linalg.generic','linalg.matmul','linalg.batch_matmul') and len(inputs)==2 and all(isinstance(v.type,TensorType) and str(v.type.element_type)=='i8' for v in inputs) and len(op.results)==1 and isinstance(op.results[0].type,TensorType) and str(op.results[0].type.element_type)=='i32')
        if device:
            eligible.append((source_index(op,index),op))
            cb['params']['lane_placement'].append({'lane':'accelerator','source_op_index':source_index(op,index),'region_id':region.data if isinstance(region,StringAttr) else '', 'reason':'i8 contraction candidate; requires checked scalar DAG and source-bound inputs'})
        elif op.name.startswith(('linalg.','math.')) or op.name=='builtin.unregistered':
            cb['params']['lane_placement'].append({'lane':'host','source_op_index':source_index(op,index),'region_id':region.data if isinstance(region,StringAttr) else '', 'reason':'Operation outside implemented i8 device region; provided host compiler owns this source operation'})
    if eligible:
        # Named unsigned casts are valid source IR, but are outside the signed
        # accelerator contract. Keep the original operation and placement;
        # refuse coverage instead of parsing it as invalid or executing the
        # named operation's default signed scalar region under an unsigned tag.
        unsigned=any(op.name in ('linalg.matmul','linalg.batch_matmul')
                     and isinstance(cast:=op.properties.get('cast',op.attributes.get('cast')),TypeFnAttr)
                     and cast.data==TypeFn.CAST_UNSIGNED for _,op in eligible)
        if unsigned:
            cb['declined']={'reason':'Named cast_unsigned integer contraction requires unsigned MAC lowering; the accelerator path implements only default or explicit cast_signed i8-to-i32 contractions','op':'source_graph'}
            return cb
        # An encoded tensor retains its accelerator eligibility and source
        # attribution, but cannot be relabelled as compact row-major storage.
        # Refuse coverage explicitly until a physical-layout ABI and matching
        # pointer lowering are implemented; the host lane cannot absorb it.
        encoded=any(not isinstance(v.type.encoding,NoneAttr)
                    for _,op in eligible
                    for v in [*op.inputs,*op.outputs,*op.results]
                    if isinstance(v.type,TensorType))
        if encoded:
            cb['declined']={'reason':'Encoded integer contraction requires explicit physical-layout ABI and lowering; dense row-major pointer scheduling admits only unencoded TensorType inputs, initializer and result','op':'source_graph'}
            return cb
        # Independent leaves may share boundary inputs, but every output must
        # be a distinct contraction result. No intermediate tensor, host result
        # or duplicate output is silently assigned a device pointer. Commands
        # retain source order while Y bindings retain the function return order.
        results=[op.results[0] for _,op in eligible]
        returned=[identity_tensor_source(v) for v in ret.operands]
        legal=(len(returned)==len(results) and len(set(returned))==len(returned)
               and set(returned)==set(results)
               and all(signed_mac_body(op) and row_major_contraction(op)
                       and all(identity_tensor_source(v) in input_bindings for v in op.inputs)
                       and all(zero_initialized(v) for v in op.outputs)
                       for _,op in eligible))
        if legal:
            for ordinal,(index,op) in enumerate(eligible):
                inputs=list(op.inputs);result=op.results[0]
                ash=list(inputs[0].type.get_shape())
                attrs={'merlin.source_op_index':index,'output_dtype':'i32'}
                if isinstance(op.attributes.get('prov.region_id'),StringAttr):attrs['prov.region_id']=op.attributes['prov.region_id'].data
                a,b=[input_bindings[identity_tensor_source(v)] for v in inputs];y=output_bindings[result]
                if len(ash)==2:
                    acc='acc' if len(eligible)==1 else 'acc'+str(ordinal)
                    cb['commands'].extend([{'opcode':'MATMUL','operands':{'lhs':a,'rhs':b,'dst':acc},'attributes':attrs},{'opcode':'COMMIT','operands':{'src':acc,'dst':y},'attributes':dict(attrs,epilogue=[])}])
                else:cb['commands'].append({'opcode':'BATCHED_MATMUL','operands':{'a':a,'w':b,'dst':y},'attributes':attrs})
            return cb
    if eligible:
        cb['declined']={'reason':'Mixed source graph requires source-bound device region decomposition by the provided host pipeline','op':'source_graph'}
    else:
        # The selected provider owns pure host layers, including layout-only
        # regions. The corrected zero-compute selector must receive the same
        # explicit composition declaration as other pure host operation families.
        cb['params']['provided_host_program']=True
    return cb
