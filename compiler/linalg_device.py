"""Typed scalar-body and affine-map interception for standalone device work.

No source operation name or provenance string selects a route. Only complete
pointwise integer expressions, signed conversions and declared numeric proofs
are admitted. Results remain memory-backed and lazy for subsequent consumers.
"""
from math import prod

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr

from merlin.llvmlower.host_tensor_codegen import tensor_shape, elem_name
from elementwise import emit_bias_add, emit_movement, emit_residual, residual_screen, _tiles


def _identity(mapping,rank):
    return (mapping.num_dims==rank and mapping.num_symbols==0 and len(mapping.results)==rank
        and all(isinstance(v,AffineDimExpr) and v.position==i for i,v in enumerate(mapping.results)))


def _finish(host,op):
    result=op.results[0]
    host.vals[result]=host.load_arg(host.ptrs[result],result.type)
    host.materialized_results.add(result)


def _constant(value):
    owner=value.owner
    if not isinstance(owner,Operation) or owner.name!='arith.constant':return None
    attr=owner.properties.get('value',owner.attributes.get('value'))
    return getattr(getattr(attr,'value',None),'data',None)


def _scalar_tree(value,args,seen=None):
    if value in args:return ('arg',args.index(value))
    constant=_constant(value)
    if constant is not None:return ('constant',constant)
    owner=value.owner
    if not isinstance(owner,Operation) or len(owner.results)!=1:return None
    if owner.name not in ('arith.extsi','arith.sitofp','arith.mulf','arith.addf',
            'math.roundeven','arith.maximumf','arith.minimumf','arith.maxnumf',
            'arith.minnumf','arith.fptosi'):
        return None
    flags=owner.properties.get('fastmath')
    if flags is not None and str(flags)!='#arith.fastmath<none>':return None
    if owner.name in ('arith.sitofp','arith.mulf','arith.addf','math.roundeven',
            'arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf') and str(value.type)!='f32':
        return None
    children=[_scalar_tree(v,args) for v in owner.operands]
    return None if any(v is None for v in children) else (owner.name,*children)


def _commutative_constant(tree,names):
    if tree is None or tree[0] not in names or len(tree)!=3:return None
    for expression,constant in ((tree[1],tree[2]),(tree[2],tree[1])):
        if constant[0]=='constant':return expression,constant[1]
    return None


def _residual_pattern(value,args):
    tree=_scalar_tree(value,list(args))
    if tree is None or tree[0]!='arith.fptosi' or str(value.type)!='i8':return None
    upper=_commutative_constant(tree[1],('arith.minimumf','arith.minnumf'))
    if upper is None or upper[1]!=127:return None
    lower=_commutative_constant(upper[0],('arith.maximumf','arith.maxnumf'))
    if lower is None or lower[1] not in (-128,0):return None
    rounded=lower[0]
    if rounded[0]!='math.roundeven' or rounded[1][0]!='arith.addf':return None
    terms=rounded[1][1:]
    scales={}
    for term in terms:
        scaled=_commutative_constant(term,('arith.mulf',))
        if scaled is None:return None
        converted,scale=scaled
        if converted[0]!='arith.sitofp':return None
        integer=converted[1]
        if integer[0]=='arith.extsi':integer=integer[1]
        if integer[0]!='arg' or integer[1] not in (0,1) or integer[1] in scales:return None
        scales[integer[1]]=float(scale)
    if set(scales)!={0,1}:return None
    return scales[0],scales[1],lower[1]==0


def _integer_sum(host,lhs,rhs,result,shape,left_type,right_type):
    """Exact accumulator-memory sum; arith.addi and ACC addition both wrap i32."""
    enc=host.enc
    sizes={"i8":1,"i32":4}
    def tile(offset,c,h,w,n):
        enc.load(enc.ptr_offset(host.ptrs[lhs],enc.mul(offset,sizes[left_type])),1<<31,h,w,
                 n*sizes[left_type],shrunk=left_type=='i8')
        enc.fence()
        enc.load(enc.ptr_offset(host.ptrs[rhs],enc.mul(offset,sizes[right_type])),3<<30,h,w,
                 n*sizes[right_type],stream=1,shrunk=right_type=='i8')
        enc.fence()
        enc.store(enc.ptr_offset(host.ptrs[result],enc.mul(offset,4)),0,h,w,n*4,dtype='i32')
        enc.fence()
    _tiles(enc,shape,tile)


def try_lower(host,op):
    if (op.name!='linalg.generic' or len(op.results)!=1 or len(op.regions)!=1
        or len(op.regions[0].blocks)!=1 or not isinstance(op.results[0].type,TensorType)
        or any(v not in host.ptrs for v in (*op.operands,*op.results))):
        return False
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if maps is None or iters is None or any('parallel' not in str(v) for v in iters):return False
    maps=[x.data for x in maps]
    shape=tensor_shape(op.results[0].type);rank=len(shape)
    if not shape or any(v<=0 for v in shape) or not _identity(maps[-1],rank):return False
    block=op.regions[0].block
    body=list(block.ops)
    if (len(block.args)!=len(op.operands) or len(maps)!=len(op.operands)
        or not body or body[-1].name!='linalg.yield' or len(body[-1].operands)!=1):return False
    if block.args[-1].uses:return False
    value=body[-1].operands[0];result=op.results[0]
    output_dtype=elem_name(result.type.element_type)
    # Exact identity/cast movement. The entire scalar program has no arithmetic.
    source_arg=value
    ext=False
    if isinstance(value.owner,Operation) and value.owner.name=='arith.extsi':
        ext=True;source_arg=value.owner.operands[0]
    if source_arg in block.args[:-1] and len(body)==(2 if ext else 1):
        index=list(block.args).index(source_arg);source=op.operands[index]
        dtype=elem_name(source.type.element_type)
        if (_identity(maps[index],rank) and tensor_shape(source.type)==shape
            and ((ext and dtype=='i8' and output_dtype=='i32') or
                 (not ext and dtype==output_dtype and dtype in ('i8','i32')))):
            emit_movement(host.enc,host.ptrs[source],host.ptrs[result],shape,
                src_dtype=dtype,output_dtype=output_dtype)
            _finish(host,op);return True
    # Read each add operand through optional signed i8-to-i32 conversion.
    owner=value.owner
    if isinstance(owner,Operation) and owner.name=='arith.addi' and output_dtype=='i32':
        sources=[];conversion_ops=[]
        for term in owner.operands:
            extension=term.owner
            if isinstance(extension,Operation) and extension.name=='arith.extsi':
                if str(term.type)!='i32':return False
                conversion_ops.append(extension);term=extension.operands[0]
            if term not in block.args[:-1]:return False
            index=list(block.args).index(term);operand=op.operands[index]
            dtype=elem_name(operand.type.element_type)
            if dtype not in ('i8','i32') or (dtype=='i8' and extension not in conversion_ops):return False
            sources.append((operand,maps[index],dtype))
        if set(body[:-1])!={owner,*conversion_ops}:return False
        for matrix,bias in ((sources[0],sources[1]),(sources[1],sources[0])):
            src,srcmap,srcdtype=matrix;b,bmap,bdtype=bias
            if (_identity(srcmap,rank) and tensor_shape(src.type)==shape and bdtype=='i32'
                and tensor_shape(b.type)==(shape[-1],) and bmap.num_dims==rank
                and bmap.num_symbols==0 and len(bmap.results)==1
                and isinstance(bmap.results[0],AffineDimExpr) and bmap.results[0].position==rank-1):
                emit_bias_add(host.enc,host.ptrs[src],host.ptrs[b],host.ptrs[result],shape,
                    src_dtype=srcdtype,output_dtype='i32')
                _finish(host,op);return True
        if all(_identity(mapping,rank) and tensor_shape(src.type)==shape for src,mapping,_ in sources):
            _integer_sum(host,sources[0][0],sources[1][0],result,shape,sources[0][2],sources[1][2])
            _finish(host,op);return True
    # Signed-i8 residual source rounds its f32 sum once and clamps explicitly.
    if (len(op.operands)==3 and output_dtype=='i8'
        and all(elem_name(v.type.element_type)=='i8' and tensor_shape(v.type)==shape
                and _identity(mapping,rank) for v,mapping in zip(op.operands[:2],maps[:2]))):
        pattern=_residual_pattern(value,block.args)
        allowed={'arith.constant','arith.extsi','arith.sitofp','arith.mulf','arith.addf',
            'math.roundeven','arith.maximumf','arith.minimumf','arith.maxnumf',
            'arith.minnumf','arith.fptosi','linalg.yield'}
        if pattern is not None and all(x.name in allowed for x in body):
            a,b,relu=pattern
            # Pure typed scalar code has an exact contract. A tolerance cannot
            # be invented from the implementation's independently rounded loads.
            try:residual_screen(a,b,0,relu)
            except (ValueError,OverflowError) as error:
                if not hasattr(host,'device_proof_gaps'):host.device_proof_gaps=[]
                host.device_proof_gaps.append({'operation':op.name,'reason':str(error)})
                return False
            emit_residual(host.enc,host.ptrs[op.operands[0]],host.ptrs[op.operands[1]],
                host.ptrs[result],shape,lhs_scale=a,rhs_scale=b,bound_lsb=0,
                output_dtype='i8',epilogue=['relu'] if relu else [])
            _finish(host,op);return True
    return False
