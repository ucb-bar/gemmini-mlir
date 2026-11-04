"""Borrow interior horizontal convolution runs; prepare only boundary rows.

The original shared horizontal emitter owns all run origins and private rows.
This adapter changes input addressing only. Merlin's checked window transporter
owns every boundary source/zero byte; the ordinary encoder owns each transfer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path

from merlin.compile.scheduling.horizontal_convolution import HorizontalConvolutionPanels
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.convolution_packing import emit_nhwc_window_panel


@dataclass(frozen=True)
class HorizontalInputAddressing:
    """Exact original source, selected transfer bounds and explicit owners."""
    horizontal: HorizontalConvolutionPanels
    source_identity: str
    facts_identity: str
    ordering_identity: str
    maximum_rows: int
    maximum_columns: int
    maximum_block_stride: int
    boundary_bytes: int
    source_destination_disjoint: bool
    private_rows_owned: bool
    # Keep the incumbent repr/identity byte-for-byte when the option is off.
    split_input_blocks: bool = field(default=False, repr=False)

    def __post_init__(self):
        p = self.horizontal
        if type(p) is not HorizontalConvolutionPanels:
            raise ValueError('horizontal input requires the checked original panel plan')
        p.__post_init__()
        if type(self.split_input_blocks) is not bool:
            raise ValueError('horizontal input split selection requires an explicit bool')
        if self.source_destination_disjoint is not True or self.private_rows_owned is not True:
            raise ValueError('horizontal input requires original allocation disjointness and private rows')
        if (p.source.window.dilation[1] != 1
                or p.source.window.input_shape[2] < p.source.window.kernel[1]
                or p.source.primitive > self.maximum_rows
                or (min(p.source.primitive, p.horizontal_columns) if self.split_input_blocks else p.horizontal_columns) > self.maximum_columns
                or p.input_block_stride > self.maximum_block_stride
                or self.boundary_bytes != p.source.primitive*p.horizontal_columns
                or not 0 < self.boundary_bytes < 1 << 40
                or self.source_identity != p.source.identity
                or any(type(s) is not str or len(s) != 64 for s in
                    (self.source_identity, self.facts_identity, self.ordering_identity))):
            raise ValueError('horizontal input source or selected extent differs')
        if self.split_input_blocks:
            self.split_plan.__post_init__()

    @property
    def split_plan(self):
        from merlin.compile.scheduling.tile_load_coalescing import TileLoadCoalescing
        p = self.horizontal
        return TileLoadCoalescing(rows=p.source.primitive, columns=p.horizontal_columns,
            tile=p.source.primitive, max_columns=p.source.primitive, local_base=0,
            block_stride=p.input_block_stride, capacity_rows=p.lhs_rows)

    @property
    def identity(self):
        if self.split_input_blocks:
            return hashlib.sha256(repr((repr(self), tuple(self.split_plan.commands()))).encode()).hexdigest()
        return hashlib.sha256(repr(self).encode()).hexdigest()

    @property
    def boundary_window(self):
        w = self.horizontal.source.window
        # Vertical position is the exact padded y already selected by original
        # ky/dilationY. Horizontal rows advance by the original output stride.
        return Window2D(w.input_shape, (1, w.kernel[1]), (1, w.stride[1]),
                        (1, 1), w.padding)

    def record(self):
        self.__post_init__()
        effects = BorrowedBufferEffects(
            'original-immutable-NHWC-direct-interior-DMA-boundary-window-noescape-v1',
            no_retained_aliases=True, no_releases=True, complete_before_return=True)
        result = dict(schema='gemmini.horizontal_input_addressing.v1',
            identity=self.identity, source_identity=self.source_identity,
            horizontal_program_identity=hashlib.sha256(repr(self.horizontal).encode()).hexdigest(),
            facts_identity=self.facts_identity, ordering_identity=self.ordering_identity,
            boundary_bytes=self.boundary_bytes,
            boundary_window=asdict(self.boundary_window),
            borrowed_effects=asdict(effects),
            source_destination_disjoint=True, private_rows_owned=True,
            interior='one original-source strided rectangle; no host preparation',
            boundary='only original boundary rows; Merlin source/zero branches write a fresh private panel',
            local_cell='local_row + segment_row + (column//primitive)*block_stride; lane=column%primitive',
            no_overlapping_device_writes=True, source_mutation=False,
            completion='real fence after each DMA and before boundary panel reuse; every callback returns complete',
            target_execution_verified=False, physical_packet_mapping_verified=False, timing=None)
        if self.split_input_blocks:
            result['split_input_blocks'] = dict(selected=True,
                mechanism='merlin.compile.scheduling.tile_load_coalescing.TileLoadCoalescing.commands',
                plan=asdict(self.split_plan), commands=tuple(self.split_plan.commands()),
                selected_max_columns=self.maximum_columns, selected_max_rows=self.maximum_rows,
                selected_max_block_stride=self.maximum_block_stride,
                source_byte='segment_pointer + slice_column; source row pitch unchanged',
                local_row='segment_local_row + slice_local_row; active row count unchanged',
                scope='both immutable interior DMA and complete boundary panel DMA',
                completion='real fence after each nonoverlapping primitive-column slice',
                physical_packet_mapping_verified=False, timing=None)
        return result


def plan_horizontal_input(horizontal, *, source_destination_disjoint=False,
                          private_rows_owned=False, split_input_blocks=False):
    """Pure admission; a missing original-owner witness returns None.

    The caller must supply the original tensor allocation owner's disjointness
    witness. Opaque pointer inequality supplies no witness. The checked shared
    horizontal emitter remains the only admitted producer of load fields.
    """
    if (type(source_destination_disjoint) is not bool or type(private_rows_owned) is not bool
            or type(split_input_blocks) is not bool):
        raise ValueError('horizontal input ownership witnesses require explicit bools')
    if source_destination_disjoint is not True or private_rows_owned is not True:
        return None
    if type(horizontal) is not HorizontalConvolutionPanels:
        raise ValueError('horizontal input requires the shared original horizontal plan')
    horizontal.__post_init__()
    w = horizontal.source.window
    if w.dilation[1] != 1 or w.input_shape[2] < w.kernel[1]:
        return None
    # Reuse the incumbent selected stream-zero extent/endpoint binding. This
    # adapter needs no zero-source instruction or destination-lane-offset law.
    from gemmini_horizontal_conv import _selected_transfer_law
    resources, law = _selected_transfer_law(Path(__file__).resolve().parent,
                                           horizontal.source.primitive)
    maximum_columns = min(resources.max_columns, resources.max_request_bytes)
    if (horizontal.source.primitive > resources.max_rows
            or (min(horizontal.source.primitive, horizontal.horizontal_columns) if split_input_blocks else horizontal.horizontal_columns) > maximum_columns
            or horizontal.input_block_stride > resources.max_block_stride):
        return None
    return HorizontalInputAddressing(horizontal, horizontal.source.identity,
        law['facts_sha256'], law['ordering_sha256'], resources.max_rows,
        maximum_columns, resources.max_block_stride,
        horizontal.source.primitive*horizontal.horizontal_columns,
        True, True, split_input_blocks)


def _check_fields(plan, fields):
    """Validate the static ABI before emitting any IR for this load."""
    p = plan.horizontal
    required = {'first_element', 'row_stride_elements', 'local_row', 'rows',
                'columns', 'block_stride_rows'}
    if (type(fields) is not dict or set(fields) != required
            or fields['columns'] != p.horizontal_columns
            or fields['block_stride_rows'] != p.input_block_stride
            or fields['row_stride_elements'] != p.source.window.stride[1]*p.source.window.input_shape[-1]):
        raise ValueError('horizontal input fields differ from the original shared emitter ABI')
    from xdsl.ir import SSAValue
    from xdsl.dialects.builtin import i64
    for key in ('first_element', 'local_row', 'rows'):
        value = fields[key]
        if type(value) is int:
            if value < 0 or (key == 'rows' and not 0 < value <= p.source.primitive):
                raise ValueError('horizontal input static field exceeds original bounds')
        elif not isinstance(value, SSAValue) or value.type != i64:
            raise ValueError('horizontal input dynamic field requires original i64 coordinates')
    if type(fields['first_element']) is int:
        first = fields['first_element']
        b, ph, pw, ci = p.padded_window.input_shape
        py, px = (first//ci//pw) % ph, (first//ci) % pw
        if (first >= b*ph*pw*ci or first % ci
                or px % p.source.window.stride[1]
                or px+p.source.window.kernel[1] > pw):
            raise ValueError('horizontal input static run leaves the original padded row')
        if (type(fields['rows']) is int
                and px+(fields['rows']-1)*p.source.window.stride[1]+p.source.window.kernel[1] > pw):
            raise ValueError('horizontal input static run leaves the original padded row')
    if type(fields['local_row']) is int:
        # Even when the active count is dynamic it is positive at the shared
        # emitter's callback. A known-invalid starting cell must fail now.
        active = fields['rows'] if type(fields['rows']) is int else 1
        last = (fields['local_row']+(p.horizontal_chunks-1)*p.input_block_stride
                +active)
        if last > p.lhs_rows:
            raise ValueError('horizontal input transfer overlaps immutable RHS')


def make_horizontal_input_loader(fn, input_ptr, admission, *, split_input_blocks=False):
    """Return ``(load(fields), receipt)`` for original horizontal input effects.

    This consumes fields emitted by emit_horizontal_convolution_panels for the
    exact admitted plan. The caller owns its invocation-private accelerator
    epoch, original output and input lifetime. Every returned load completes
    synchronously. No source or accelerator arithmetic is changed.
    """
    from merlin.llvmlower.host_builder import FnBuilder, PTR
    from xdsl.ir import SSAValue
    if type(admission) is not HorizontalInputAddressing:
        raise ValueError('horizontal input requires typed admission before mutation')
    admission.__post_init__()
    if type(split_input_blocks) is not bool or split_input_blocks != admission.split_input_blocks:
        raise ValueError('horizontal input loader split selection differs from admission')
    # Reopen the selected declaration before creating even private storage.
    current = plan_horizontal_input(admission.horizontal,
        source_destination_disjoint=admission.source_destination_disjoint,
        private_rows_owned=admission.private_rows_owned,
        split_input_blocks=split_input_blocks)
    if current != admission:
        raise ValueError('horizontal input admission or selected facts changed')
    if type(fn) is not FnBuilder or not isinstance(input_ptr, SSAValue) or input_ptr.type != PTR:
        raise ValueError('horizontal input requires original typed LLVM source pointer')
    from gemmini_support import scratch_buffer
    import gemmini_encode as enc
    p, w = admission.horizontal, admission.horizontal.source.window
    d = p.source.primitive
    b, height, width, ci = w.input_shape
    pt, pl, pb, pr = w.padding
    ph, pw = height+pt+pb, width+pl+pr
    sx, kw = w.stride[1], w.kernel[1]
    ow = w.output_spatial[1]
    boundary_window = admission.boundary_window
    if boundary_window.output_spatial != (ph, ow):
        raise ValueError('boundary window differs from original padded coordinates')
    # The fresh BSS is tiny and invocation-private. It never aliases original
    # inputs/outputs or another scratch site; loads complete before each reuse.
    panel = scratch_buffer(fn, admission.boundary_bytes, 'horizontal_input_boundary')
    control = FnBuilderControlAdapter(fn)
    receipt = admission.record()
    receipt['private_panel_owner'] = 'fresh internal scratch_buffer BSS site; one complete bounded horizontal run'

    def op(name, a, b):
        return control.op(name, a, b)

    def minimum(a, b):
        return control.select(control.cmp('ult', a, b), a, b)

    def transfer(pointer, local, rows, source_stride):
        if split_input_blocks:
            for column, columns, offset in admission.split_plan.commands():
                enc.config_load(fn, source_stride, stream=0, block_stride=p.input_block_stride)
                enc.mvin(fn, enc.pointer_offset(fn, pointer, column),
                         op('add', local, offset), rows, columns, stream=0)
                enc.fence(fn)
            return
        enc.config_load(fn, source_stride, stream=0, block_stride=p.input_block_stride)
        enc.mvin(fn, pointer, local, rows, p.horizontal_columns, stream=0)
        enc.fence(fn)

    def load(fields):
        admission.__post_init__()
        _check_fields(admission, fields)
        first, rows, local = fields['first_element'], fields['rows'], fields['local_row']
        pixel = op('udiv', first, ci)
        image = op('udiv', pixel, ph*pw)
        py = op('urem', op('udiv', pixel, pw), ph)
        px = op('urem', pixel, pw)
        x = op('udiv', px, sx)
        window_row = op('add', op('mul', op('add', op('mul', image, ph), py), ow), x)

        def boundary(start, count, maximum):
            # The library's row count is static. A retained bounded loop calls
            # its one-row transporter only for the actual boundary interval;
            # interior runs never enter this payload loop. The panel receives
            # disjoint complete rows, followed by a single completed DMA.
            if maximum == 0:
                return
            def stage():
                def row(index):
                    def copy():
                        destination = enc.pointer_offset(fn, panel,
                            op('mul', index, p.horizontal_columns))
                        origin = op('add', window_row, op('add', start, index))
                        emit_nhwc_window_panel(fn, boundary_window, input_ptr,
                            destination, origin, 1,
                            source_destination_disjoint=True,
                            maximum_panel_bytes=admission.boundary_bytes,
                            inline_memory_spans=True)
                    control.branch(control.cmp('ult', index, count), copy, lambda: None)
                fn.for_loop(0, maximum, 1, row, retain_loop=True)
                transfer(panel, op('add', local, start), count, p.horizontal_columns)
            control.branch(control.cmp('ult', 0, count), stage, lambda: None)

        def valid_y():
            left = minimum(rows, control.select(control.cmp('ult', px, pl),
                op('udiv', op('add', op('sub', pl, px), sx-1), sx), 0))
            last_start = pl+width-kw
            end = minimum(rows, control.select(control.cmp('ule', px, last_start),
                op('add', op('udiv', op('sub', last_start, px), sx), 1), 0))
            middle = op('sub', end, left)
            right = op('sub', rows, end)
            boundary(0, left, min(d, (pl+sx-1)//sx))

            def interior():
                iy = op('sub', py, pt)
                ix = op('sub', op('add', px, op('mul', left, sx)), pl)
                source = op('mul', op('add', op('mul', op('add', op('mul', image, height), iy), width), ix), ci)
                transfer(enc.pointer_offset(fn, input_ptr, source),
                         op('add', local, left), middle, sx*ci)
            control.branch(control.cmp('ult', 0, middle), interior, lambda: None)
            boundary(end, right, min(d, (pr+sx-1)//sx))

        y_inside = control.op('and', control.cmp('ule', pt, py),
                              control.cmp('ult', py, pt+height), dtype='i1')
        control.branch(y_inside, valid_y, lambda: boundary(0, rows, d))
        return receipt

    return load, receipt
