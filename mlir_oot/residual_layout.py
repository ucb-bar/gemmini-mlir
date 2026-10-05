"""Prove a common explicit permutation around a uniform elementwise residual.

This is an ABI-preserving tensor rewrite: both flat operands and the flat
result use the same permutation. It requires a previously proved uniform
residual arithmetic contract; it never infers such a contract from a symbol.
"""
from xdsl.dialects.builtin import TensorType, NoneAttr, i8
from xdsl.ir import Operation


def shared_transpose(inputs, logical_shape):
    if len(inputs)!=2:raise ValueError('residual requires two operands')
    proofs=[]
    for value in inputs:
        op=value.owner
        if not isinstance(op,Operation) or op.name!='linalg.transpose':
            raise ValueError('residual operand lacks an explicit transpose')
        if len(op.operands)!=2 or len(op.results)!=1:raise ValueError('unexpected transpose arity')
        source=op.operands[0];before=source.type;after=value.type
        if any(not isinstance(ty,TensorType) or ty.get_element_type()!=i8 or not isinstance(ty.encoding,NoneAttr) for ty in (before,after)):
            raise ValueError('residual layout requires unencoded i8 tensors')
        a,b=tuple(before.get_shape()),tuple(after.get_shape())
        permutation=tuple(op.permutation.get_values())
        if any(x<=0 for x in (*a,*b)) or sorted(permutation)!=list(range(len(a))):
            raise ValueError('residual layout requires positive static axes and a permutation')
        if b!=tuple(logical_shape) or tuple(a[i] for i in permutation)!=b:
            raise ValueError('residual transpose shape proof differs')
        proofs.append(dict(kind='shared_explicit_transpose',permutation=list(permutation),physical_shape=list(a),logical_shape=list(b)))
    if proofs[0]!=proofs[1]:raise ValueError('residual operand permutations or shapes disagree')
    return proofs[0]
