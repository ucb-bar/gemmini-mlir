"""Exact signed-byte window spans through the existing ACC movement route.

The shared window partition owns every destination byte exactly once. Each
span drains before borrowing bounded disjoint ACC tiles and after readout; no
convolution operands or accumulator values are live during preparation.
"""
from gemmini_pointwise import _ptr
import gemmini_encode as enc


def emit_horizontal_device_input(fn, window, source, destination, rows, *, maximum_panel_bytes):
    """Prepare a complete unit-window panel in disjoint caller-owned storage.

    ``window`` is the original NHWC shape with kernel/stride/dilation (1,1)
    and exact source padding. The shared packer validates the complete domain
    and byte budget before emitting any span; the caller owns both lifetimes.
    """
    from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel
    transport = make_device_span_transport(fn)
    receipt = emit_nhwc_window_panel(fn, window, source, destination, fn.const(0), rows,
        source_destination_disjoint=True, maximum_panel_bytes=maximum_panel_bytes,
        span_transport=transport)
    receipt.update(kind='device_accumulator_window_spans', host_copy_bytes=0,
        signed_i8_exact=True, zero_source=0,
        storage_completion='each span drains before ACC reuse; preparation precedes terminal configuration')
    return receipt


def make_device_span_transport(fn):
    """Return the shared packer's (kind, source, destination, count) callback.

    Source and destination are disjoint byte pointers owned by the caller.
    Unit load/readout scaling maps all signed i8 values, including -128, exactly.
    Zero spans use the same selected zero-source LOAD convention as convolution.
    """
    from gemmini_residual_schedule import selected_epoch_evidence, digest
    from merlin.llvmlower.host_builder import FnBuilder, PTR
    from xdsl.ir import SSAValue
    from merlin.compile.scheduling.movement_epochs import (
        MovementEpochLaw, plan_movement_epochs, emit_movement_epochs)
    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    if type(fn) is not FnBuilder:
        raise ValueError('device span epochs require the original typed LLVM builder')
    # Reuse the original selected MVIN/MVOUT operand, queue, progress and
    # completion evidence. No residual arithmetic or source effect is reused.
    evidence = selected_epoch_evidence()
    resources = evidence['resources']
    edge, accumulator_rows = resources['edge'], resources['accumulator_rows']
    if edge != 16 or accumulator_rows < edge:
        raise ValueError('device span movement requires the selected 16-row ACC tile')
    numeric = dict(input_dtype='signed_i8', output_dtype='signed_i8',
        load_scale_f32=1.0, readout_scale_f32=1.0, activation=0,
        exact_values=list(range(-128, 128)), zero_source=0)
    law = MovementEpochLaw(digest(evidence), digest(numeric),
        digest(dict(evidence=evidence,
            ownership='exclusive ACC before convolution initialization; disjoint fresh destination spans',
            configuration='configured then full fence; immutable during both phases')),
        True, True, True, True, True, True)
    # Finite command queue capacities also bound each constructed batch. This
    # grants neither cross-queue overlap nor same-cell concurrent accesses.
    bounds = resources['queue_entries']
    law.revalidate()

    def transport(kind, source, destination, count):
        if kind not in ('copy', 'zero') or type(count) is not int or count <= 0:
            raise ValueError('device window span requires a positive exact byte count')
        if not isinstance(destination, SSAValue) or destination.type != PTR:
            raise ValueError('device span requires its disjoint byte destination pointer')
        if kind == 'copy':
            if not isinstance(source, SSAValue) or source.type != PTR or source is destination:
                raise ValueError('copy span requires its original source pointer')
        elif source is not None:
            raise ValueError('zero span must use the selected zero source')
        plan = plan_movement_epochs(count, kind, tile=edge,
            accumulator_rows=accumulator_rows, load_issue_bound=bounds['load'],
            store_issue_bound=bounds['store'], law=law)
        control = FnBuilderControlAdapter(fn)
        enc.fence(fn)
        enc.config_load(fn, edge if kind == 'copy' else 0, scale=1.0, shrunk=True)
        enc.config_store(fn, edge, scale=1.0, activation=0)
        enc.fence(fn)  # Complete configuration before either immutable phase.

        def lower(effect, fields):
            if effect == 'complete':
                enc.fence(fn)
                return
            address = control.op('add', enc.ACC, fields['local_row'])
            if effect == 'load':
                pointer = _ptr(fn, source, fields['offset']) if kind == 'copy' else 0
                enc.mvin(fn, pointer, address, fields['rows'], fields['columns'])
            elif effect == 'store':
                enc.mvout(fn, _ptr(fn, destination, fields['offset']), address,
                          fields['rows'], fields['columns'])
            else:
                raise ValueError('unknown checked byte movement effect')

        return emit_movement_epochs(plan, control, lower)

    transport.selected_evidence = evidence
    transport.epoch_law = law
    return transport
