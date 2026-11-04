"""Optional exact affine address loops for already admitted host copies."""
from math import prod
from xdsl.dialects.builtin import NoneAttr
from merlin.llvmlower.host_tensor_codegen import attr_of, tensor_shape
from merlin.llvmlower.host_layout_fusion import emit_affine_copy

# Experimental until an actual complete-region timing comparison supports it.
ENABLED = False


def try_lower(host, op):
    if not ENABLED or op.name not in ('linalg.transpose','linalg.broadcast','tensor.extract_slice'):
        return False
    if (getattr(host,'row_alignment',1) != 1 or len(op.results) != 1
            or not op.operands or op.operands[0] not in host.ptrs
            or op.results[0] not in host.ptrs):
        return False
    source, result = op.operands[0],op.results[0]
    if (not isinstance(source.type.encoding,NoneAttr)
            or not isinstance(result.type.encoding,NoneAttr)
            or source.type.element_type != result.type.element_type
            or host.ptrs[source] == host.ptrs[result]):
        return False
    src, shape = tensor_shape(source.type), tensor_shape(result.type)
    if any(n < 0 for n in (*src,*shape)):
        return False
    pitches = tuple(prod(src[i+1:]) for i in range(len(src)))
    offset = 0
    if op.name == 'linalg.transpose':
        attr = attr_of(op,'permutation')
        if attr is None: return False
        perm = tuple(map(int,attr.get_values()))
        if sorted(perm) != list(range(len(src))) or shape != tuple(src[i] for i in perm):
            return False
        strides = tuple(pitches[i] for i in perm)
    elif op.name == 'linalg.broadcast':
        attr = attr_of(op,'dimensions')
        if attr is None: return False
        dims = tuple(map(int,attr.get_values()))
        kept = tuple(i for i in range(len(shape)) if i not in dims)
        if (len(set(dims)) != len(dims) or any(i<0 or i>=len(shape) for i in dims)
                or tuple(shape[i] for i in kept) != src): return False
        strides = tuple(0 if i in dims else pitches[kept.index(i)] for i in range(len(shape)))
    else:
        if len(op.operands) != 1: return False
        attrs = [attr_of(op,key) for key in ('static_offsets','static_sizes','static_strides')]
        if any(a is None for a in attrs): return False
        offsets,sizes,steps = [tuple(map(int,a.get_values())) for a in attrs]
        if any(len(a)!=len(src) for a in (offsets,sizes,steps)): return False
        if any(o<0 or n<0 or s<=0 or n and o+(n-1)*s>=d
               for o,n,s,d in zip(offsets,sizes,steps,src)): return False
        retained=[]; cursor=0
        for i,size in enumerate(sizes):
            if cursor<len(shape) and size==shape[cursor]:
                retained.append(i); cursor+=1
            elif size!=1: return False
        if cursor!=len(shape): return False
        strides=tuple(pitches[i]*steps[i] for i in retained)
        offset=sum(o*p for o,p in zip(offsets,pitches))
    from linalg_host import _load,_store,_finish
    emit_affine_copy(host.fp.fb,shape,strides,source_offset=offset,source_elements=prod(src),
                     load=lambda i:_load(host,source,i),store=lambda i,v:_store(host,result,i,v))
    _finish(host,op)
    return True
