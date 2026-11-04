"""Pure readout admission and terminal emission for shared tiled matmul.

This module owns no contraction or storage schedule. Narrow readout follows the
same hardware pipeline as the incumbent: accumulator scale, signed-i8 saturation,
then optional ReLU. Unsupported semantics return None before any IR is emitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from xdsl.dialects.builtin import IntegerType, i8, i32

from . import isa
from .tile_bias import BiasPreparation, has_bias


@dataclass(frozen=True)
class Readout:
    """A uniform whole-matrix accumulator readout, with a byte DMA stride."""

    element_type: IntegerType
    stride_bytes: int
    narrow: bool
    scale: float
    activation: int
    accumulator_flags: int

    def configure(self, f):
        """Configure the existing device scale/saturation/activation pipeline."""
        return isa.config_st(f, self.stride_bytes, scale=self.scale,
                             activation=self.activation)

    def store(self, f, destination, index, local, cols, rows):
        """Read a scheduled accumulator tile to a row-major typed destination."""
        # Keep evaluation/emission order identical to the incumbent raw store.
        return isa.mvout(f, f.gep(destination, index, self.element_type),
                         f.or_i(f.const(self.accumulator_flags), local), cols, rows)


def select_readout(attrs: Mapping, n: int, *,
                   bias: BiasPreparation | None = None) -> Readout | None:
    """Admit raw i32 or uniform scaled i8, optionally with final ReLU.

    A rejected descriptor emits nothing and leaves the caller's incumbent path
    available. Bias stages require a matching typed initialization descriptor.
    Column scales and pooling require other lowering paths.
    The caller retains all scratch/accumulator capacity, source-boundary and
    prepared-operand checks; this selector only proves readout semantics.
    """
    if type(n) is not int or n <= 0:
        return None
    dtype = str(attrs.get('output_dtype', 'i32'))
    if dtype not in ('i32', 'i8'):
        return None
    if attrs.get('acc_scale_by_column') is not None:
        return None
    if has_bias(attrs):
        if not isinstance(bias, BiasPreparation) or not bias.matches(attrs, n):
            return None
    elif bias is not None:
        return None
    epilogue = attrs.get('epilogue', 'none')
    stages = ([str(x).lower() for x in epilogue]
              if isinstance(epilogue, (list, tuple)) else [str(epilogue).lower()])
    neutral = ('none', 'identity', '')
    # Same stage vocabulary/order as common matmul, restricted to this module's
    # proven uniform readout. Repeated saturation/ReLU remain idempotent.
    stages = ['bias' if x == 'bias_add' else x for x in stages]
    if any(x not in neutral + ('bias', 'acc_scale', 'saturate', 'relu') for x in stages):
        return None
    active = [x for x in stages if x not in neutral]
    if 'bias' in active and bias is None:
        return None
    order = {'bias': 0, 'acc_scale': 1, 'saturate': 2, 'relu': 3}
    if (active.count('bias') > 1 or active.count('acc_scale') > 1
            or any(order[a] > order[b] for a, b in zip(active, active[1:]))):
        return None
    relu = 'relu' in stages or bool(attrs.get('relu', False))
    try:
        scale = float(attrs.get('acc_scale', 1.0))
        # The encoder serializes exactly these f32 bits. Reject values without
        # a finite representation rather than introducing a new numeric case.
        encoded = isa.f32_bits(scale)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(scale) or encoded & 0x7F800000 == 0x7F800000:
        return None
    if dtype == 'i32':
        # Preserve the incumbent raw schedule's admission and exact emission.
        if any(x != 'bias' for x in active) or relu or scale != 1.0:
            return None
        element_type, size, flags = i32, 4, 0xA0000000
    else:
        element_type, size, flags = i8, 1, 0x80000000
    stride = n * size
    if stride >= 1 << 32:
        return None
    return Readout(element_type, stride, dtype == 'i8', scale,
                   int(relu), flags)
