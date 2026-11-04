"""Spatial halo control adapter for the public bounded convolution issue API."""
from dataclasses import asdict

from merlin.compile.scheduling.spatial_producer import SpatialProducerPlan
from merlin.compile.scheduling.spatial_readout import prove_spatial_readout
from merlin.compile.scheduling.window_transfer import emit_convolution_load_issue_tile

def emit_pool_load_issue(plan, builder, issue, complete, *, law, peel=False):
    """Emit a checked spatial halo using bounded library load/compute epochs.

    Geometry and coverage match the public SpatialProducerPlan driver. Every
    original convolution tile uses emit_convolution_load_issue_tile, whose full
    completion boundary precedes local-row reuse; pooled stores also complete.
    """
    if type(plan) is not SpatialProducerPlan or plan.readout_mode != "single_readout":
        raise ValueError("pool load issue requires the checked single-readout spatial plan")
    plan.__post_init__()
    identity = plan.identity
    if law.source_identity != plan.producer.identity:
        raise ValueError("pool load issue law does not bind its original convolution")
    if not callable(issue) or not callable(complete):
        raise ValueError("pool load issue requires actual issue and full completion callbacks")
    if any(not callable(getattr(builder, name, None)) for name in ("op", "cmp", "select", "loop", "branch")):
        raise ValueError("pool load issue requires an integer index/control builder")

    def lower(kind, fields):
        issue(kind, fields)
        complete(kind, fields)

    p, s = plan.producer, plan.spatial
    batches = p.window.input_shape[0]
    h, w = s.input_hw
    kh, kw = s.window
    sh, sw = s.stride
    pt, pl, _, _ = s.padding
    oh, ow = s.output_hw
    # Output-border windows preserve the same clipped source samples, but must
    # not force an otherwise valid wide interior row through scalar gathers.
    interior_begin = min(ow, max(0, (pl + sw - 1) // sw))
    interior_end = min(ow, max(interior_begin, (w + pl - kw) // sw + 1))

    def op(name, a, b):
        return builder.op(name, a, b)

    def clip(value, extent):
        # Signed-negative and signed-positive boundaries are distinct. ULT
        # alone would classify negative indices as an upper-border value.
        positive = builder.select(builder.cmp("slt", value, 0), 0, value)
        return builder.select(builder.cmp("ult", positive, extent), positive, extent - 1)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda index: body(op("mul", index, tile), tile))
        if tail:
            body(whole * tile, tail)

    def image_body(image):
        def output_y_body(y):
            def output_x_body(x, output_columns):
                stripe_width = (output_columns - 1) * sw + kw
                base_x = op("sub", op("mul", x, sw), pl)
                base_y = op("sub", op("mul", y, sh), pt)

                def channel_body(column, columns):
                    def stripe_y_body(ky):
                        iy = clip(op("add", base_y, ky), h)

                        def stripe_x_body(local_x, rows):
                            raw_x = op("add", base_x, local_x)
                            last_x = op("add", raw_x, rows - 1)
                            valid = builder.op("and", builder.cmp("ult", raw_x, w), builder.cmp("ult", last_x, w), "i1")

                            def produce(ix, count, local):
                                global_row = op("add", op("mul", op("add", op("mul", image, h), iy), w), ix)
                                accumulator = op(
                                    "add", p.accumulator_row, op("add", op("mul", ky, stripe_width), local)
                                )

                                def effect(kind, fields):
                                    issue("finish_raw_tile" if kind == "store" else kind, fields)

                                emit_convolution_load_issue_tile(
                                    p,
                                    builder,
                                    effect,
                                    complete,
                                    output_row=global_row,
                                    output_column=column,
                                    rows=count,
                                    columns=columns,
                                    accumulator_row=accumulator,
                                    law=law,
                                )

                            def interior():
                                produce(raw_x, rows, local_x)

                            def boundary():
                                builder.loop(
                                    0,
                                    rows,
                                    1,
                                    lambda index: produce(
                                        clip(op("add", raw_x, index), w), 1, op("add", local_x, index)
                                    ),
                                )

                            builder.branch(valid, interior, boundary)

                        split(stripe_width, p.row_tile, stripe_x_body)

                    builder.loop(0, kh, 1, stripe_y_body)
                    output_row = op("add", op("mul", op("add", op("mul", image, oh), y), ow), x)
                    lower(
                        "store_spatial_max",
                        dict(
                            output_row=output_row,
                            output_column=column,
                            columns=columns,
                            output_pixels=output_columns,
                            accumulator_row=p.accumulator_row,
                            local_height=kh,
                            local_width=stripe_width,
                            window_height=kh,
                            window_width=kw,
                            stride_width=sw,
                            readout_tile=op("udiv", column, p.column_tile),
                        ),
                    )

                split(p.output_channels, p.column_tile, channel_body)

            if peel:
                builder.loop(0, interior_begin, 1, lambda x: output_x_body(x, 1))
                split(interior_end - interior_begin, plan.output_tile_width,
                    lambda x, count: output_x_body(op("add", x, interior_begin), count))
                builder.loop(interior_end, ow, 1, lambda x: output_x_body(x, 1))
            else:
                split(ow, plan.output_tile_width, output_x_body)

        builder.loop(0, oh, 1, output_y_body)

    builder.loop(0, batches, 1, image_body)
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError("spatial producer source or private resources changed during emission")
    coverage = dict(plan.coverage)
    if peel:
        coverage["output_x_partition"] = dict(
            left_singletons=interior_begin,
            interior_begin=interior_begin, interior_end=interior_end,
            interior_full=(interior_end-interior_begin)//plan.output_tile_width,
            interior_tail=(interior_end-interior_begin)%plan.output_tile_width,
            right_singletons=ow-interior_end,
            proof="disjoint consecutive [0,begin),[begin,end),[end,ow); every original output once")
    return dict(
        schema="submission.pool_load_issue.v1",
        completion="full device drain after each paired compute, finished raw tile and pooled readout",
        load_issue_law=asdict(law),
        boundary_peeling=peel,
        plan_sha256=identity,
        readout_proofs=(
            [prove_spatial_readout(chain) for chain in plan.readouts] if plan.readout_mode == "single_readout" else []
        ),
        readout_mode=plan.readout_mode,
        retained_readout_chains=[asdict(chain) for chain in plan.readouts],
        **coverage,
        scope="actual compact index/product emission; source binding and target execution remain separate",
    )
