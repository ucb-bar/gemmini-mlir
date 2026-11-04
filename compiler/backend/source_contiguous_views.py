"""Source-bound integer layout views with identical contiguous byte order."""
from math import prod


def predecessor(op, spec, contiguous_transpose):
    """Prove the source operand of a total same-byte-order integer tensor view."""
    from xdsl.dialects.builtin import NoneAttr
    if len(op.results)!=1 or not op.operands:
        return None
    source,result=op.operands[0],op.results[0]
    a,b=spec(source),spec(result)
    if (a['dtype'] not in ('i8','i32') or b['dtype']!=a['dtype']
            or prod(a['shape'])!=prod(b['shape'])
            or any(not isinstance(v.type.encoding,NoneAttr) for v in (source,result))):
        return None
    if op.name in ('tensor.collapse_shape','tensor.expand_shape'):
        if len(op.operands)!=1:return None
        op.verify()
        return source
    if op.name=='linalg.transpose':
        proven=contiguous_transpose(op)
        return proven[0] if proven is not None else None
    if op.name=='tensor.extract_slice':
        if len(op.operands)!=1 or a!=b:return None
        def integers(key):
            attr=op.properties.get(key)
            return list(attr.get_values()) if attr is not None and hasattr(attr,'get_values') else None
        rank=len(a['shape'])
        if (integers('static_offsets')!=[0]*rank or integers('static_sizes')!=a['shape']
                or integers('static_strides')!=[1]*rank):return None
        op.verify()
        return source
    return None


def input_chain(value, root, spec, contiguous_transpose, *, blocked=()):
    """Walk only private same-block views; every intermediate has one observer."""
    from xdsl.ir import Operation
    current=value;members=[]
    for _ in range(64):
        owner=current.owner
        if (not isinstance(owner,Operation) or owner.parent is not root.parent
                or owner in blocked or len(list(current.uses))!=1):break
        if owner.name not in ('tensor.collapse_shape','tensor.expand_shape','tensor.extract_slice','linalg.transpose'):break
        source=predecessor(owner,spec,contiguous_transpose)
        if source is None:break
        members.append(owner);current=source
    return current,tuple(reversed(members))


def output_chain(value, root, spec, contiguous_transpose):
    """Retain the final observable value, stopping before a shared intermediate."""
    current=value;members=[]
    for _ in range(64):
        uses=list(current.uses)
        if len(uses)!=1:break
        owner=uses[0].operation
        if owner.parent is not root.parent or owner.name not in ('tensor.collapse_shape','tensor.expand_shape','tensor.extract_slice','linalg.transpose'):break
        source=predecessor(owner,spec,contiguous_transpose)
        if source is not current:break
        members.append(owner);current=owner.results[0]
    return current,tuple(members)
