"""Bounded spatial residue classes for flattened convolution tiles.

The period derives only from source output width and primitive row capacity.
No source identity or empirical shape table participates in admission.
"""
from math import gcd

MAX_PERIOD = 8
MAX_EXPANDED_TAPS = 128
MAX_EMITTED_BYTES = 1 << 20


def geometry_admission(window, dim):
    """Source-only expansion bound, independent of cycles or source identity.

    The exact emitted-code cap is checked separately by the terminal consumer.
    These limits bound authored body duplication; they grant no DMA or storage
    capability and do not predict a performance improvement.
    """
    width = window.output_spatial[1]
    period = width // gcd(width, dim)
    expanded_taps = period * window.kernel[1]
    admitted = period <= MAX_PERIOD and expanded_taps <= MAX_EXPANDED_TAPS
    return dict(original_output_width=width, primitive_row_capacity=dim,
        residue_period=period, maximum_residue_period=MAX_PERIOD,
        expanded_horizontal_taps=expanded_taps,
        maximum_expanded_horizontal_taps=MAX_EXPANDED_TAPS,
        source_geometry_admitted=admitted)


def period_for(window, dim):
    admission = geometry_admission(window, dim)
    return admission['residue_period'] if admission['source_geometry_admitted'] else None


def fragments(window, builder, first, count, residue):
    """Static original-row boundaries; dynamic original image/Y coordinates."""
    oh, ow = window.output_spatial
    line = builder.op('udiv', first, ow)
    local = 0
    while local < count:
        x = (residue + local) % ow
        row = (residue + local) // ow
        length = min(count - local, ow - x)
        source_line = builder.op('add', line, row)
        yield (builder.op('udiv', source_line, oh),
               builder.op('mul', builder.op('urem', source_line, oh), window.stride[0]),
               x * window.stride[1], local, length)
        local += length
