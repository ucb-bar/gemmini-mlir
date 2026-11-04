"""Bound contiguous horizontal convolution reductions in private row panels.

This is a source/layout transformation, not a target schedule or overlap law.
The caller materializes the original zero halo, binds multiblock transfers and
array fresh/stay semantics, and supplies real completion at each panel reuse.
"""

from dataclasses import dataclass
from math import prod

from .affine_gather_runs import ConvolutionGatherRuns, emit_convolution_gather_runs
from .window_transfer import CountedConvolutionPlan, Window2D


@dataclass(frozen=True)
class HorizontalConvolutionPanels:
    source: CountedConvolutionPlan
    row_blocks: int
    column_intervals: tuple[tuple[int, int], ...]
    maximum_transfer_columns: int

    def __post_init__(self):
        if type(self.source) is not CountedConvolutionPlan:
            raise ValueError("horizontal panels require the exact original convolution")
        self.source.__post_init__()
        p = self.source
        if (p.window.dilation[1] != 1 or p.row_tile != p.primitive
                or any(type(v) is not int or v <= 0 for v in (self.row_blocks, self.maximum_transfer_columns))
                or self.horizontal_columns > self.maximum_transfer_columns):
            raise ValueError("horizontal panels require contiguous bounded source columns and primitive rows")
        end = 0
        if type(self.column_intervals) is not tuple or not self.column_intervals:
            raise ValueError("horizontal panels require complete original column intervals")
        for item in self.column_intervals:
            if (type(item) is not tuple or len(item) != 2 or any(type(v) is not int for v in item)
                    or item[0] != end or not 0 < item[1] <= p.column_tile):
                raise ValueError("horizontal panel columns must partition every original output exactly")
            end += item[1]
        if end != p.output_channels:
            raise ValueError("horizontal panel columns do not cover the original output")
        if (self.operand_rows > p.operand_rows or self.accumulator_rows > p.accumulator_rows
                or max(prod(self.padded_window.input_shape), self.operand_rows * p.primitive) >= 1 << 63):
            raise ValueError("horizontal panels exceed their declared private storage")

    @property
    def padded_window(self):
        w = self.source.window
        b, h, width, ci = w.input_shape
        pt, pl, pb, pr = w.padding
        return Window2D((b, h + pt + pb, width + pl + pr, ci), w.kernel, w.stride, w.dilation)

    @property
    def horizontal_columns(self):
        return self.source.window.kernel[1] * self.source.window.input_shape[-1]

    @property
    def horizontal_chunks(self):
        return (self.horizontal_columns + self.source.primitive - 1) // self.source.primitive

    @property
    def reduction_tiles(self):
        return self.source.window.kernel[0] * self.horizontal_chunks

    @property
    def input_block_stride(self):
        return self.row_blocks * self.source.primitive

    @property
    def lhs_rows(self):
        return self.reduction_tiles * self.input_block_stride

    @property
    def rhs_base(self):
        return self.lhs_rows

    @property
    def rhs_rows(self):
        return len(self.column_intervals) * self.reduction_tiles * self.source.primitive

    @property
    def operand_rows(self):
        return self.lhs_rows + self.rhs_rows

    @property
    def accumulator_rows(self):
        return self.row_blocks * len(self.column_intervals) * self.source.primitive

    def summary(self):
        self.__post_init__()
        return dict(schema="merlin.horizontal_convolution_panels.v1", source_sha256=self.source.identity,
                    row_blocks=self.row_blocks, horizontal_columns=self.horizontal_columns,
                    reduction_tiles=self.reduction_tiles, lhs_rows=self.lhs_rows, rhs_base=self.rhs_base,
                    rhs_rows=self.rhs_rows, operand_rows=self.operand_rows,
                    accumulator_rows=self.accumulator_rows, input_block_stride=self.input_block_stride,
                    padded_input_elements=prod(self.padded_window.input_shape),
                    original_reduction_order=["kernel_y", "kernel_x", "input_channel"],
                    completion="complete RHS; stage group and bias; complete; feed/readout; complete before reuse",
                    physical_mapping_verified=False, target_execution=False, timing=None)


def plan_horizontal_convolution_panels(source, *, max_row_blocks, maximum_transfer_columns,
                                      private_rows_owned=False, column_intervals=None):
    """Select capacity-bounded complete-K panels without changing scalar exits.

    The original zero-padded NHWC input must be synchronously materialized by
    the caller. The existing window-panel byte transporter can do this with a
    1x1 window and the original padding. Its cost belongs inside the same ROI.
    No allocation, measured policy, target capability or storage grant is inferred.
    """
    if type(source) is not CountedConvolutionPlan or private_rows_owned is not True:
        raise ValueError("horizontal panels require an original source and explicit private ownership")
    source.__post_init__()
    if any(type(v) is not int or v <= 0 for v in (max_row_blocks, maximum_transfer_columns)):
        raise ValueError("horizontal panels require positive explicit transfer/unroll bounds")
    w, d = source.window, source.primitive
    if w.dilation[1] != 1 or source.row_tile != d or w.kernel[1] * w.input_shape[-1] > maximum_transfer_columns:
        return None
    if column_intervals is None:
        column_intervals = tuple((n, min(source.column_tile, source.output_channels - n))
                                 for n in range(0, source.output_channels, source.column_tile))
    if type(column_intervals) is not tuple or not column_intervals:
        raise ValueError("horizontal panels require complete original column intervals")
    chunks = w.kernel[0] * ((w.kernel[1] * w.input_shape[-1] + d - 1) // d)
    nblocks = len(column_intervals)
    pixels = w.input_shape[0] * prod(w.output_spatial)
    blocks = min(max_row_blocks, (pixels + d - 1) // d,
                 source.accumulator_rows // (nblocks * d), source.operand_rows // (chunks * d) - nblocks)
    if blocks <= 0:
        return None
    return HorizontalConvolutionPanels(source, blocks, column_intervals, maximum_transfer_columns)


def emit_horizontal_convolution_panels(plan, builder, lower, complete):
    """Emit ordinary counted effects with explicit full completion boundaries.

    Input loads write consecutive reduction blocks at input_block_stride rows.
    Each caller-bound multiblock load contains only original contiguous bytes.
    Complete must really finish all prior effects; a queue/issue acknowledgment
    is insufficient. Array reuse is declared per fresh_weights flag, and needs
    a separate target-owned source law. Every original scalar store remains.
    """
    if type(plan) is not HorizontalConvolutionPanels or not callable(complete):
        raise ValueError("checked horizontal panels and real completion are required")
    plan.__post_init__()
    identity = repr(plan)
    p, w, d = plan.source, plan.padded_window, plan.source.primitive
    gather = ConvolutionGatherRuns(w.input_shape, (w.input_shape[0], *w.output_spatial),
                                  w.stride, w.dilation, w.kernel, d)
    nblocks = len(plan.column_intervals)
    chunks = plan.horizontal_chunks
    def op(name, a, b):
        return builder.op(name, a, b)

    # Complete immutable original RHS before any input group can consume it.
    for ky in range(w.kernel[0]):
        for chunk in range(chunks):
            slot = ky * chunks + chunk
            depth = min(d, plan.horizontal_columns - chunk * d)
            for ni, (column, columns) in enumerate(plan.column_intervals):
                lower("load_weight", dict(reduction_row=ky * plan.horizontal_columns + chunk * d,
                    output_column=column, local_row=plan.rhs_base + (ni * plan.reduction_tiles + slot) * d,
                    rows=depth, columns=columns))
    complete("horizontal_rhs_ready", dict(local_row=plan.rhs_base, local_rows=plan.rhs_rows))

    def group(first_tile, row_counts):
        origins = [op("mul", op("add", first_tile, mi), d) for mi in range(len(row_counts))]
        for ky in range(w.kernel[0]):
            for mi, rows in enumerate(row_counts):
                for base, local, count in emit_convolution_gather_runs(gather, builder, origins[mi], rows):
                    def stage(base=base, local=local, count=count, mi=mi, ky=ky):
                        lower("load_horizontal_input", dict(
                            first_element=op("add", base, ky * w.dilation[0] * w.input_shape[2] * w.input_shape[-1]),
                            row_stride_elements=gather.source_row_stride,
                            local_row=op("add", ky * chunks * plan.input_block_stride + mi * d, local),
                            rows=count, columns=plan.horizontal_columns,
                            block_stride_rows=plan.input_block_stride))
                    builder.branch(builder.cmp("ult", 0, count), stage, lambda: None)
        if p.bias:
            for ni, (column, columns) in enumerate(plan.column_intervals):
                for mi, rows in enumerate(row_counts):
                    lower("initialize_bias", dict(output_row=origins[mi], output_column=column,
                        rows=rows, columns=columns, accumulator_row=(ni * plan.row_blocks + mi) * d))
        complete("horizontal_group_ready", dict(local_row=0, local_rows=plan.lhs_rows))
        for ky in range(w.kernel[0]):
            for chunk in range(chunks):
                slot = ky * chunks + chunk
                depth = min(d, plan.horizontal_columns - chunk * d)
                for ni, (column, columns) in enumerate(plan.column_intervals):
                    for mi, rows in enumerate(row_counts):
                        lower("compute", dict(output_row=origins[mi], output_column=column,
                            rows=rows, columns=columns, accumulator_row=(ni * plan.row_blocks + mi) * d,
                            lhs_row=slot * plan.input_block_stride + mi * d,
                            rhs_row=plan.rhs_base + (ni * plan.reduction_tiles + slot) * d,
                            depth=depth, first_reduction=(slot == 0), bias=p.bias, fresh_weights=(mi == 0)))
        for ni, (column, columns) in enumerate(plan.column_intervals):
            for mi, rows in enumerate(row_counts):
                lower("store", dict(output_row=origins[mi], output_column=column,
                    rows=rows, columns=columns, accumulator_row=(ni * plan.row_blocks + mi) * d))
        complete("horizontal_group_last_use", dict(local_row=0, local_rows=plan.operand_rows))

    pixels = w.input_shape[0] * prod(w.output_spatial)
    tiles, last_rows = divmod(pixels, d)
    whole, tail = divmod(tiles, plan.row_blocks)
    if whole:
        builder.loop(0, whole, 1, lambda index: group(op("mul", index, plan.row_blocks), (d,) * plan.row_blocks))
    tail_rows = (d,) * tail + ((last_rows,) if last_rows else ())
    if tail_rows:
        group(whole * plan.row_blocks, tail_rows)
    plan.__post_init__()
    if repr(plan) != identity:
        raise ValueError("original horizontal convolution source changed during emission")
    return plan.summary()


def emit_horizontal_convolution_weights(plan, builder, lower, complete):
    """Prepare this checked panel's immutable original RHS for tile consumers.

    This only binds the existing panel layout. The caller owns all private RHS
    rows until a real final completion and must preserve them between tiles.
    """
    if type(plan) is not HorizontalConvolutionPanels or not callable(complete):
        raise ValueError('horizontal tile preparation needs a checked panel and completion')
    plan.__post_init__()
    d = plan.source.primitive
    for ky in range(plan.source.window.kernel[0]):
        for chunk in range(plan.horizontal_chunks):
            slot = ky * plan.horizontal_chunks + chunk
            depth = min(d, plan.horizontal_columns - chunk * d)
            for ni, (column, columns) in enumerate(plan.column_intervals):
                lower('load_weight', dict(reduction_row=ky * plan.horizontal_columns + chunk * d,
                    output_column=column, local_row=plan.rhs_base + (ni * plan.reduction_tiles + slot) * d,
                    rows=depth, columns=columns))
    complete('horizontal_rhs_ready', dict(local_row=plan.rhs_base, local_rows=plan.rhs_rows))
    return plan.summary()


def emit_horizontal_convolution_tile(plan, builder, lower, complete, *, output_row,
                                     output_column, rows, columns, accumulator_row):
    """Emit one original row tile into caller-owned private accumulator rows.

    Consume the same horizontal reduction/gather transform as the complete-image
    emitter. The caller maps its original padded NHWC row addresses, has prepared
    these exact RHS slots, and retains real completion before ACC/RHS reuse.
    Every compute requires a fresh descriptor; no array-resident weight law is
    implied. This emits original values only; it does not publish a readout.
    """
    if (type(plan) is not HorizontalConvolutionPanels or not callable(complete)
            or type(rows) is not int or not 1 <= rows <= plan.source.primitive
            or type(columns) is not int or not 1 <= columns <= plan.source.column_tile
            or plan.row_blocks != 1):
        raise ValueError('horizontal private tile requires one checked bounded original tile')
    plan.__post_init__()
    p, w, d = plan.source, plan.padded_window, plan.source.primitive
    intervals = plan.column_intervals
    if any(c != i*p.column_tile or (n != p.column_tile and i != len(intervals)-1)
           for i,(c,n) in enumerate(intervals)):
        raise ValueError('horizontal private tiles require original regular channel intervals')
    gather = ConvolutionGatherRuns(w.input_shape, (w.input_shape[0], *w.output_spatial),
                                  w.stride, w.dilation, w.kernel, d)
    for ky in range(w.kernel[0]):
        for base, local, count in emit_convolution_gather_runs(gather, builder, output_row, rows):
            def stage(base=base, local=local, count=count, ky=ky):
                lower('load_horizontal_input', dict(
                    first_element=builder.op('add', base, ky*w.dilation[0]*w.input_shape[2]*w.input_shape[-1]),
                    row_stride_elements=gather.source_row_stride,
                    local_row=builder.op('add', ky*plan.horizontal_chunks*plan.input_block_stride, local),
                    rows=count, columns=plan.horizontal_columns,
                    block_stride_rows=plan.input_block_stride))
            builder.branch(builder.cmp('ult', 0, count), stage, lambda: None)
    if p.bias:
        lower('initialize_bias', dict(output_row=output_row, output_column=output_column,
            rows=rows, columns=columns, accumulator_row=accumulator_row))
    complete('horizontal_tile_ready', dict(local_row=0, local_rows=plan.lhs_rows))
    ni = builder.op('udiv', output_column, p.column_tile)
    for ky in range(w.kernel[0]):
        for chunk in range(plan.horizontal_chunks):
            slot = ky*plan.horizontal_chunks+chunk
            lower('compute', dict(output_row=output_row, output_column=output_column,
                rows=rows, columns=columns, accumulator_row=accumulator_row,
                lhs_row=slot*plan.input_block_stride,
                rhs_row=builder.op('add', plan.rhs_base,
                    builder.op('mul', builder.op('add', builder.op('mul', ni, plan.reduction_tiles), slot), d)),
                depth=min(d, plan.horizontal_columns-chunk*d),
                first_reduction=(slot == 0), bias=p.bias, fresh_weights=True))
    complete('horizontal_raw_tile_complete', dict(accumulator_row=accumulator_row,
                                                 rows=rows, columns=columns))
    plan.__post_init__()
    return plan.summary()
