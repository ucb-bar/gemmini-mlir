"""Source-derived input DMA at the first actual K/N/M reader.

A pending owner grants no transport-ready value. The selected target's real
scratchpad RAW dependencies order each reader after its original row writes.
Only the original full consumer drain closes the pending input lifetime.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConvolutionInputFirstUseAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    scratchpad_raw_dependencies: bool = False
    ordered_configuration: bool = False
    finite_queue_progress: bool = False
    immutable_input: bool = False
    dma_preserves_stationary_array: bool = False
    full_consumer_transport_completion: bool = False

    def require(self):
        values = asdict(self)
        if any(type(values[k]) is not str or not values[k].strip() for k in
               ("hardware_identity", "numerical_policy_identity", "qualification_identity")) or any(
            type(values[k]) is not bool or not values[k] for k in values if not k.endswith("identity")
        ):
            raise ValueError("first-use input needs actual RAW, configuration, progress, array retention and full completion admission")


class PendingConvolutionInput:
    """One original image's source rows, pending until a real consumer drain.

    First use is the lexicographically first original (KY, output Y), with
    KX=N=0 and the first CI of each existing channel transfer. Every original
    transfer is issued once, including unchanged stride or dilation holes. This initial domain retains one complete output row per M
    tile and one injective full-image ring; it allocates no additional storage.
    """

    def __init__(self, schedule, group, rows, admission, *, image):
        from .convolution_output_groups import ConvolutionOutputGroup
        from .window_transfer import ConvolutionResidentInputRows

        schedule.verify_current()
        if (type(group) is not ConvolutionOutputGroup or type(rows) is not ConvolutionResidentInputRows
                or type(admission) is not ConvolutionInputFirstUseAdmission or schedule.epoch_law is None):
            raise ValueError("pending input needs original source, complete group, ring and target law")
        group.__post_init__()
        rows.revalidate()
        admission.require()
        p = schedule.source.plan
        w = p.window
        ci = w.input_shape[-1]
        oh, ow = w.output_spatial
        live = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
        if (admission.hardware_identity != schedule.epoch_law.hardware_identity
                or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
                or rows.source_identity != schedule.source.identity or rows.window != w
                or group.output_rows != oh or group.row_blocks != oh or ow > p.row_tile
                or getattr(rows, "image_windows", 1) != 1 or live > rows.slots
                or rows.end > group.panel.rhs_base):
            raise ValueError("pending input requires the original complete image and injective single ring under its selected target")
        self.schedule, self.group, self.rows, self.admission = schedule, group, rows, admission
        self.image = image
        self.identity = schedule.identity, asdict(group), rows.record(), asdict(admission)
        self.first_uses = tuple((row, min(((ky, y) for ky in range(w.kernel[0]) for y in range(oh)
            if y * w.stride[0] + ky * w.dilation[0] == row), default=None)) for row in range(live))
        part = rows.stage_channels or ci
        if part % p.channel_tile and part < ci:
            raise ValueError("pending input transfer starts must match original counted channel readers")
        self.channel_intervals = tuple((c, min(part, ci - c)) for c in range(0, ci, part))
        self.started, self.closed = False, False
        self.visited = set()

    def _check(self):
        self.schedule.verify_current()
        self.rows.revalidate()
        self.admission.require()
        if self.identity != (self.schedule.identity, asdict(self.group), self.rows.record(), asdict(self.admission)):
            raise ValueError("pending input source, allocation or target admission changed")

    def _stage(self, builder, issue, row, channel_interval=None):
        w = self.rows.window
        h, width, ci = w.input_shape[1:]
        source_y = row - w.padding[0]
        source = builder.op("add", builder.op("mul", builder.op("add",
            builder.op("mul", self.image, h), source_y), width * ci), 0)
        self.rows.emit_stage(builder, dict(image=self.image, input_row=source_y,
            source_byte=source, cache_byte=(row % self.rows.slots) * self.rows.rows_per_source_row * self.rows.channel_tile,
            valid=builder.cmp("ult", source_y, h), row_bytes=width * ci), issue, channel_interval=channel_interval)

    def start(self, schedule, group, rows, origins, counts, builder, issue):
        self._check()
        oh, ow = rows.window.output_spatial
        if (self.started or self.closed or schedule is not self.schedule or group != self.group or rows != self.rows
                or tuple(origins) != tuple(y * ow for y in range(oh)) or counts != (ow,) * oh):
            raise ValueError("pending input must bind its exact complete original M traversal once")
        self.started = True
        for row, first in self.first_uses:
            if first is None:
                self._stage(builder, issue, row)

    def before_reader(self, builder, issue, ky, kx, channel, m):
        if not self.started or self.closed or type(m) is not int or not 0 <= m < self.group.row_blocks:
            raise ValueError("pending input reader is outside its original live M traversal")
        self.visited.add(m)
        for row, first in self.first_uses:
            if (first is not None and first[1] == m
                    and (type(ky) is not int or ky == first[0])
                    and (type(kx) is not int or kx == 0)) :
                tap = builder.op("and", builder.cmp("eq", ky, first[0]), builder.cmp("eq", kx, 0), "i1")
                for interval in self.channel_intervals:
                    condition = builder.op("and", tap, builder.cmp("eq", channel, interval[0]), "i1")
                    builder.branch(condition, lambda row=row, interval=interval:
                        self._stage(builder, issue, row, interval), lambda: None)

    def finish(self, complete, fields):
        self._check()
        if (not self.started or self.closed or self.visited != set(range(self.group.row_blocks)) or not callable(complete)):
            raise ValueError("pending input requires every original reader and a real final completion")
        complete("spatial_group_last_use", fields)
        self.closed = True

    def record(self):
        self._check()
        return dict(source_identity=self.schedule.source.identity,
            first_uses=[dict(source_row=row, first_KY_M=first) for row, first in self.first_uses],
            original_channel_transfers=self.channel_intervals,
            storage=self.rows.record(), readiness="actual target scratchpad RAW; no ready token or early drain",
            completion="real original spatial_group_last_use drain completes compute and all input DMA")


def emit_pending_input_images(schedule, group, rows, admission, builder, issue, complete, *,
        array_law, readout_law, readout_transfer_law, ready_readout, output_tiles,
        rhs_frontier=None, rhs_issue=None):
    """Ordinary counted image and original K/N/M consumer, with pending input."""
    from .convolution_output_groups import emit_rolling_convolution_output_group

    p = schedule.source.plan
    batches, _, _, ci = p.window.input_shape
    oh, ow = p.window.output_spatial
    chunks = (ci + p.channel_tile - 1) // p.channel_tile
    runs = p.window.kernel[0] * p.window.kernel[1] * chunks
    # Validate all source and selected effect premises before emitting any DMA.
    checked = PendingConvolutionInput(schedule, group, rows, admission, image=0)
    if rhs_frontier is not None:
        rhs_frontier.require(rows.rolling_plan(), runs)
        if not callable(rhs_issue):
            raise ValueError("pending input retains the actual original RHS stage emitter")

    def image_body(image):
        pending = PendingConvolutionInput(schedule, group, rows, admission, image=image)
        consumer = emit_rolling_convolution_output_group(schedule, group, builder, issue, complete,
            output_row=builder.op("mul", image, oh * ow), row_counts=(ow,) * oh,
            tile_origins=tuple(y * ow for y in range(oh)), output_tiles=output_tiles,
            input_rows=rows, array_law=array_law, readout_law=readout_law,
            readout_transfer_law=readout_transfer_law, ready_readout=ready_readout,
            defer_consumer=True, pending_input=pending)

        def after_run(run, reduction, depth):
            def stage():
                for n, (column, columns) in enumerate(rhs_frontier.next_columns):
                    local = builder.op("add", rhs_frontier.rhs_base, builder.op("mul",
                        builder.op("add", builder.op("mul", run, len(rhs_frontier.next_columns)), n), rhs_frontier.tile))
                    rhs_issue("load_weight", dict(reduction_row=reduction, output_column=column,
                        local_row=local, rows=depth, columns=columns))
            builder.branch(builder.cmp("eq", image, batches - 1), stage, lambda: None)

        consumer.advance(0, runs, after_run=None if rhs_frontier is None else after_run)
        consumer.finish()
        if not pending.closed:
            raise ValueError("pending input escaped without real full consumer completion")
        complete("input_last_use", dict(local_row=rows.base))

    builder.loop(0, batches, 1, image_body)
    return dict(schema="merlin.convolution_pending_input.v1", **checked.record(),
        original_input_rows_per_image=len(checked.first_uses), original_K_runs=runs)
