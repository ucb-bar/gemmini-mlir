"""Direct original-column hardware readout for a single primitive M/K region.

The existing ordinary convolution schedule issues strided source RHS transfers
and narrow strided output transfers. This route avoids layout preparation when
both complete source axes fit one primitive tile. It performs no host payload
copy or arithmetic and does not capture or replace an emitter globally.
"""
from math import prod
from gemmini_readout import column_scale_runs


def select_direct_columns(window, columns, attrs, output_dtype, *, primitive):
    """Structural admission for the existing synchronous singleton-column owner.

    M and K bounds are primitive tile extents, not empirical speed cutoffs.
    Timing selection outside this bounded region remains with existing policy.
    """
    report = dict(selected=False, reason='outside one-M/one-K pointwise region')
    if str(output_dtype) not in ('i8', 'int8'):
        report['reason'] = 'source has no narrow column readout'
        return report
    if type(primitive) is not int or primitive <= 0:
        raise ValueError('direct column route requires a positive selected primitive')
    if (window.kernel != (1, 1) or window.stride != (1, 1)
            or window.dilation != (1, 1) or any(window.padding)):
        return report
    rows = window.input_shape[0] * prod(window.output_spatial)
    depth = window.input_shape[-1]
    report.update(output_rows=rows, reduction_depth=depth, columns=columns,
                  primitive=primitive, original_window_projection=True,
                  original_k_order=True, host_payload_copies=False)
    if rows > primitive or depth > primitive:
        return report
    intervals = column_scale_runs(attrs, columns, max_columns=primitive)
    if len(intervals) != columns or any(width != 1 for _, width, *_ in intervals):
        report['reason'] = 'ordinary source intervals are not independent singleton columns'
        return report
    report.update(selected=True,
        reason='complete M/K fit one primitive; original singleton hardware columns avoid host layout copies')
    return report


def try_emit_direct_conv(fn, input_ptr, weight_ptr, dst_ptr, *, input_shape,
                         weight_shape, attrs=None, bias=None, output_dtype='i32',
                         ordinary_emitter, primitive):
    """Return ordinary direct-column receipt, or None to keep caller policy."""
    from gemmini_conv import _geometry
    if not callable(ordinary_emitter):
        raise ValueError('direct columns require the ordinary emitter callback')
    attrs = dict(attrs or {})
    window, columns = _geometry(input_shape, weight_shape, attrs)
    selection = select_direct_columns(window, columns, attrs, output_dtype,
                                      primitive=primitive)
    if not selection['selected']:
        return None
    # Preserve literal source-column indices through the shared schedule.
    # Static readout avoids a host scale-table load in this bounded region.
    receipt = ordinary_emitter(fn, input_ptr, weight_ptr, dst_ptr,
        input_shape=input_shape, weight_shape=weight_shape, attrs=attrs,
        bias=bias, output_dtype=output_dtype, _static_column_readout=True)
    return dict(receipt, column_direct=selection)
