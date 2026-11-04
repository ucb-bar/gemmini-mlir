"""Source-proven standalone signed-i8 rectifier through accumulator readout.

The complete scalar body, types and identity maps select this route. Every
signed-i8 source value is checked against the shared exact readout proof before
emission. Source operations and other observers retain their original buffers.
"""
from functools import lru_cache
import hashlib
import json

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation

from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.llvmlower.integer_readout import derive, evaluate
from elementwise import emit_movement
from linalg_device import _constant, _identity, _finish


def _width(value):
    ty=value.type
    width=getattr(getattr(ty,'width',None),'data',None)
    return width if str(ty)==f'i{width}' and width in (8,16,32,64) else None


def _signed(value,width):
    value=int(value)&((1<<width)-1)
    return value-(1<<width) if value&(1<<(width-1)) else value


def _tree(value,source,operations):
    width=_width(value)
    if width is None:return None
    if value is source:return ('source',width)
    owner=value.owner
    if not isinstance(owner,Operation) or len(owner.results)!=1:return None
    constant=_constant(value)
    if type(constant) is int:
        operations.add(owner)
        return ('constant',width,_signed(constant,width))
    name=owner.name
    if name not in ('arith.maxsi','arith.minsi','arith.extsi','arith.trunci'):
        return None
    if owner.regions or owner.successors:return None
    if name in ('arith.maxsi','arith.minsi'):
        if len(owner.operands)!=2 or any(v.type!=value.type for v in owner.operands):return None
    else:
        if len(owner.operands)!=1:return None
        prior=_width(owner.operands[0])
        if prior is None or (name=='arith.extsi' and prior>=width) or (name=='arith.trunci' and prior<=width):return None
    children=tuple(_tree(v,source,operations) for v in owner.operands)
    if any(v is None for v in children):return None
    operations.add(owner)
    return (name,width,*children)


def _eval(tree,x):
    name,width,*children=tree
    if name=='source':return x
    if name=='constant':return children[0]
    values=[_eval(v,x) for v in children]
    if name=='arith.maxsi':return max(values)
    if name=='arith.minsi':return min(values)
    return _signed(values[0],width)


@lru_cache(maxsize=128)
def _prove(tree):
    proof=derive([1.0],-128,127,relu=True)
    if any(_eval(tree,x)!=evaluate(x,proof) for x in range(-128,128)):
        return None
    # This is a finite proof over the entire original i8 domain, not observed
    # stimulus or a change to the source's comparison tolerance.
    return dict(domain=[-128,127],checked_values=256,max_output_lsb_error=0,
        readout='signed i8 load; f32 scale 1; half-even; ReLU; saturated i8',
        shared_readout_proof_sha256=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest())


def _refuse(host,op,reason):
    if not hasattr(host,'device_proof_gaps'):host.device_proof_gaps=[]
    host.device_proof_gaps.append(dict(operation=op.name,
        source_op_index=getattr(host,'current_source_op_index',None),reason=reason))
    return False


def try_lower(host,op):
    if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
        or len(op.regions)!=1 or len(op.regions[0].blocks)!=1
        or any(not isinstance(v.type,TensorType) for v in (*op.operands,*op.results))
        or any(v not in host.ptrs for v in (*op.operands,*op.results))):return False
    block=op.regions[0].block;body=tuple(block.ops)
    # Only report activation refusals for comparison-bearing candidates; this
    # interceptor also sees unrelated generic source operations.
    if not any(v.name in ('arith.maxsi','arith.maxui','arith.maximumf','arith.maxnumf') for v in body):return False
    source,seed=op.operands;result=op.results[0]
    if str(source.type.element_type)!='i8' or str(result.type.element_type)!='i8':
        return _refuse(host,op,'standalone rectifier readout requires signed i8 source and result; full-width i32 and floating activation are not proved')
    shape=tensor_shape(result.type);rank=len(shape)
    if not shape or any(v<=0 for v in shape) or tensor_shape(source.type)!=shape or tensor_shape(seed.type)!=shape:return False
    maps=op.properties.get('indexing_maps');iters=op.properties.get('iterator_types')
    if (maps is None or len(maps)!=2 or any(not _identity(m.data,rank) for m in maps)
        or iters is None or len(iters)!=rank or any('parallel' not in str(v) for v in iters)):
        return _refuse(host,op,'standalone rectifier movement requires identity indexing and exclusively parallel source iterators')
    if (len(block.args)!=2 or block.args[1].uses or not body
        or body[-1].name!='linalg.yield' or len(body[-1].operands)!=1):return False
    used=set();tree=_tree(body[-1].operands[0],block.args[0],used)
    if tree is None or not any(v.name=='arith.maxsi' for v in used):
        return _refuse(host,op,'standalone rectifier scalar proof permits only signed max/min, signed extension, truncation and integer constants')
    # External scalar constants are permitted. Every in-region operation must
    # participate in the proven scalar result; no other effects are discarded.
    if set(body[:-1])!={v for v in used if v.parent is block}:return False
    proof=_prove(tree)
    if proof is None:
        return _refuse(host,op,'signed integer scalar expression is not exact i8 ReLU over [-128,127]')
    emit_movement(host.enc,host.ptrs[source],host.ptrs[result],shape,
        src_dtype='i8',output_dtype='i8',epilogue=['relu'])
    _finish(host,op)
    if hasattr(host,'device_receipts'):
        host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
            emission=dict(kind='standalone_signed_i8_relu',shape=list(shape),numeric_proof=proof)))
    return True
