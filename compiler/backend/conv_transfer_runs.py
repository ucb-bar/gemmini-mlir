"""Optional image-contiguous input runs within existing primitive M tiles.

The horizontal identity tap of a unit-stride, shape-preserving window has an
NHWC source row stride of CI even across output-row boundaries. Its valid rows
are one interval per image. A shared gather plan splits the output index space
only at image boundaries; clipping that interval supplies disjoint zero/data/
zero transfers. All other taps retain the original fragment emitter.

This changes input issue geometry only. It supplies no wide-column load law,
completion, queue independence, output tiling, or performance qualification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from merlin.compile.scheduling.affine_gather_runs import (
    ConvolutionGatherRuns, emit_convolution_gather_runs,
)

from . import isa
from .conv_spatial_panels import _Prepared, _input_fragments


def _binding(prepared):
    return sha256(json.dumps(dict(source=prepared.source.identity,
        window=asdict(prepared.source.plan.window), gather=asdict(prepared.gather),
        group=asdict(prepared.group), dim=isa.DIM, spad_rows=isa.SPAD_ROWS),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class ConvTransferRuns:
    """Source-bound index product consumed as input_fragment_emitter.

    The invocation domain is the original emitter's: 0 <= first, 0 < count <=
    DIM, first+count <= B*OH*OW; ky/kx/channel/depth remain inside the original
    KH/KW/CI loops. Columns are the original primitive channel width, including
    tails. The common encoder and original K/N completion sites are retained.
    """
    prepared: object
    binding_sha256: str
    image_gather: ConvolutionGatherRuns
    horizontal_identity_tap: int

    def revalidate(self, prepared):
        if (prepared is not self.prepared or type(prepared) is not _Prepared
                or _binding(prepared) != self.binding_sha256):
            raise ValueError('convolution transfer runs changed their admitted source/allocation')
        self.image_gather.__post_init__()
        w = prepared.source.plan.window
        batch, height, width, _ = w.input_shape
        expected = ConvolutionGatherRuns((batch, 1, height*width, 1),
            (batch, 1, height*width), (1, 1), (1, 1), (1, 1), isa.DIM)
        if (self.image_gather != expected
                or self.horizontal_identity_tap*w.dilation[1] != w.padding[1]):
            raise ValueError('convolution transfer runs changed their source index product')

    def receipt(self):
        return dict(schema='authored_conv_transfer_runs.v1',
            source_sha256=self.prepared.source.identity,
            binding_sha256=self.binding_sha256,
            shared_index_plan=asdict(self.image_gather),
            horizontal_identity_tap=self.horizontal_identity_tap,
            predicate='unit spatial stride; output H/W equal input H/W; horizontal identity tap',
            bounds='original count/depth <= DIM; original private LHS rows; original CI tail',
            completion='unchanged original input/compute and complete-K epoch',
            fallback='original source row fragments for every other tap',
            evidence='optional authored source mapping; no performance qualification')

    def emit(self, prepared, builder, first, count, ky, kx, channel, depth,
             transfer, *, origins=None):
        """Exact callable signature of conv_spatial_panels._input_fragments."""
        self.revalidate(prepared)
        if (type(count) is not int or not 0 < count <= isa.DIM
                or type(depth) is not int or not 0 < depth <= isa.DIM):
            raise ValueError('transfer runs require original bounded primitive row/channel widths')
        w = prepared.source.plan.window
        batch, height, width, ci = w.input_shape
        if type(first) is int and not 0 <= first <= batch*height*width-count:
            raise ValueError('transfer runs leave original output spatial domain')
        if any(type(index) is int and not 0 <= index < extent
                for index, extent in ((ky, w.kernel[0]), (kx, w.kernel[1]))):
            raise ValueError('transfer runs leave original kernel domain')
        if type(channel) is int and not 0 <= channel <= ci-depth:
            raise ValueError('transfer runs leave original input channel domain')

        def joined():
            tap_y = builder.op('mul', ky, w.dilation[0])
            # Saturating differences keep all interval endpoints nonnegative.
            def before(limit):
                delta = builder.select(builder.cmp('ult', tap_y, limit),
                    builder.op('sub', limit, tap_y), 0)
                return builder.select(builder.cmp('ult', delta, height), delta, height)
            valid_start = builder.op('mul', before(w.padding[0]), width)
            valid_end = builder.op('mul', before(height+w.padding[0]), width)
            # B=1 needs no image division/split: the checked M domain is inside
            # that image. Multi-image domains use the actual shared index plan.
            fragments = ((first, 0, count),) if batch == 1 else emit_convolution_gather_runs(
                self.image_gather, builder, first, count)
            for position, local, rows in fragments:
                def active(position=position, local=local, rows=rows):
                    if batch == 1:
                        image, spatial = 0, position
                    else:
                        image = builder.op('udiv', position, height*width)
                        spatial = builder.op('urem', position, height*width)
                    def clip(endpoint):
                        delta = builder.select(builder.cmp('ult', spatial, endpoint),
                            builder.op('sub', endpoint, spatial), 0)
                        return builder.select(builder.cmp('ult', delta, rows), delta, rows)
                    prefix, end = clip(valid_start), clip(valid_end)
                    def issue(zero, start, length):
                        def nonempty():
                            physical = 0
                            if not zero:
                                source_pixel = builder.op('sub', builder.op('add',
                                    builder.op('add', spatial, start),
                                    builder.op('mul', tap_y, width)), w.padding[0]*width)
                                physical = builder.op('add', builder.op('mul',
                                    builder.op('add', builder.op('mul', image, height*width), source_pixel), ci), channel)
                            transfer(dict(zero=zero, first_element=physical,
                                local_row=builder.op('add', prepared.group.lhs_row(0),
                                    builder.op('add', local, start)),
                                rows=length, columns=depth, row_stride_elements=ci))
                        builder.branch(builder.cmp('ult', 0, length), nonempty, lambda: None)
                    issue(True, 0, prefix)
                    issue(False, prefix, builder.op('sub', end, prefix))
                    issue(True, end, builder.op('sub', rows, end))
                builder.branch(builder.cmp('ult', 0, rows), active, lambda: None)

        builder.branch(builder.cmp('eq', kx, self.horizontal_identity_tap), joined,
            lambda: _input_fragments(prepared, builder, first, count, ky, kx,
                channel, depth, transfer, origins=origins))


def plan_transfer_runs(prepared):
    """Read-only optional admission; unsupported geometry keeps original input.

    The shared image gather is an output-index plan, not a padded payload
    allocation. Final source addresses are derived from the original window.
    """
    if type(prepared) is not _Prepared:
        return None
    prepared.gather.__post_init__()
    prepared.group.__post_init__()
    w = prepared.source.plan.window
    batch, height, width, _ = w.input_shape
    oh, ow = w.output_spatial
    horizontal, remainder = divmod(w.padding[1], w.dilation[1])
    if (w.stride != (1, 1) or (oh, ow) != (height, width)
            or remainder or horizontal >= w.kernel[1]
            or prepared.group.panel.tile != isa.DIM
            or prepared.group.lhs_row(0)+isa.DIM > prepared.group.panel.rhs_base):
        return None
    index_plan = ConvolutionGatherRuns((batch, 1, height*width, 1),
        (batch, 1, height*width), (1, 1), (1, 1), (1, 1), isa.DIM)
    return ConvTransferRuns(prepared, _binding(prepared), index_plan, horizontal)


def lower_transfer_runs(ctx, ifm, rhs, dst, attributes):
    """Consume one optional product, or return None before target mutation.

    Unsupported source geometry retains the caller's existing uniform-store
    route. The actual product is passed to the common terminal encoder.
    """
    from .conv_spatial_panels import _prepare, _emit_prepared
    prepared = _prepare(ctx, ifm, rhs, dst, attributes, allow_wide=True)
    product = plan_transfer_runs(prepared)
    if product is None:
        return None
    product.revalidate(prepared)
    emitted = _emit_prepared(ctx, ifm, rhs, dst, prepared,
        trial_mode='uniform_stores', input_fragment_emitter=product.emit)
    if not emitted or emitted['source_sha256'] != prepared.source.identity:
        raise ValueError('convolution transfer run product was not emitted')
    receipt = dict(product.receipt(), emitted=emitted)
    ctx.metadata.setdefault('convolution_transfer_runs', []).append(receipt)
    return receipt
