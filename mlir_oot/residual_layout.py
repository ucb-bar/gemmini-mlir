"""Residual adapter policy around Merlin's target-independent layout proof."""
from xdsl.dialects.builtin import TensorType, i8
from merlin.llvmlower.shared_permutation import prove_shared_transpose


def shared_transpose(inputs, logical_shape):
    # The residual implementation owns these arithmetic/ABI restrictions.
    # Merlin proves only common structural layout, with no tile assumptions.
    if len(inputs) != 2:
        raise ValueError('residual requires two operands')
    if any(not isinstance(value.type, TensorType) or value.type.get_element_type() != i8
           for value in inputs):
        raise ValueError('residual layout requires unencoded i8 tensors')
    return prove_shared_transpose(inputs, logical_shape)
