"""Source-bound complete-K RHS ownership for private spatial pool blocks.

The library's resident-panel planner owns physical row legality. The canonical
window tile generator still supplies every source product; this adapter removes
only its repeated immutable weight transfers and projects their original K
coordinates to the retained panel. No tensor payload is materialized on host.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

from merlin.compile.scheduling.affine_gather_runs import plan_convolution_resident_panel
from merlin.compile.scheduling.block_schedule import Geometry


@dataclass(frozen=True)
class ResidentSpatialRHS:
    source_identity: str
    producer: object
    panel: object

    @classmethod
    def prepare(cls, plan):
        plan.__post_init__()
        p = plan.source.producer
        w = p.window
        # Counted plans represent a small CI as a primitive tail; the
        # physical retained-panel pitch still comes from the primitive.
        if (p.channel_tile != min(p.primitive, w.input_shape[-1])
                or p.row_tile > p.primitive):
            return None
        panel = plan_convolution_resident_panel(
            (*w.kernel, w.input_shape[-1], p.output_channels),
            Geometry(p.primitive, p.operand_rows, None, p.accumulator_rows, True),
            private_rows_owned=True, max_column_blocks=1)
        if panel is None or p.lhs_row + p.row_tile > panel.rhs_base:
            return None
        result = cls(plan.identity, p, panel)
        result.require(plan)
        return result

    def require(self, plan):
        self.panel.__post_init__()
        if self.source_identity != plan.identity or self.producer != plan.source.producer:
            raise ValueError('resident RHS differs from private pooled source')
        p = self.producer
        if (self.panel.column_blocks != 1 or self.panel.tile != p.primitive
                or p.lhs_row + p.row_tile > self.panel.rhs_base
                or self.panel.rhs_base + self.panel.rhs_rows > p.operand_rows):
            raise ValueError('resident RHS overlaps private LHS or exceeds operand rows')

    def phase(self, builder, lower, complete, column, columns):
        p = self.producer
        ci = p.window.input_shape[-1]
        kh, kw = p.window.kernel
        chunks = self.panel.channel_tiles
        op = builder.op

        def row(reduction):
            slot = op('add', op('mul', op('udiv', reduction, ci), chunks),
                      op('udiv', op('urem', reduction, ci), p.channel_tile))
            return op('add', self.panel.rhs_base, op('mul', slot, p.primitive))

        def y(ky):
            def x(kx):
                def channel(chunk, depth):
                    reduction = op('add', op('mul', op('add', op('mul', ky, kw), kx), ci),
                                   op('mul', chunk, p.channel_tile))
                    lower('load_weight', dict(reduction_row=reduction, output_column=column,
                        local_row=row(reduction), rows=depth, columns=columns))
                whole, tail = divmod(ci, p.channel_tile)
                if whole: builder.loop(0, whole, 1, lambda i: channel(i, p.channel_tile))
                if tail: channel(whole, tail)
            builder.loop(0, kw, 1, x)
        builder.loop(0, kh, 1, y)
        complete('resident_rhs_ready', {})
        pending = None

        def effects(kind, fields):
            nonlocal pending
            if kind == 'load_weight':
                if pending is not None:
                    raise ValueError('resident RHS descriptor pairing was lost')
                pending = row(fields['reduction_row'])
            elif kind == 'compute':
                if pending is None:
                    raise ValueError('resident RHS product lacks canonical source K')
                lower(kind, dict(fields, rhs_row=pending))
                pending = None
            else:
                if pending is not None:
                    raise ValueError('resident RHS unmatched descriptor crosses another effect')
                lower(kind, fields)
        return effects

    def emit_tile_effects(self, builder, lower, **coordinates):
        """Reuse shared affine row fragments and disjoint zero/data DMA bands.

        The immutable producer source defines both the logical gather and every
        original K product. Source reduction order and private ACC coordinates
        are unchanged; shared fragments replace only the input transfer cover.
        """
        from types import SimpleNamespace
        from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns
        from .conv_spatial_panels import (
            _gather_coordinates, _gather_bands, _emit_gather_bands,
        )
        p = self.producer
        w = p.window
        shape = w.input_shape
        pt, pl, pb, pr = w.padding
        gather = ConvolutionGatherRuns(
            (shape[0], shape[1] + pt + pb, shape[2] + pl + pr, shape[3]),
            (shape[0], *w.output_spatial), w.stride, w.dilation, w.kernel, p.primitive)
        prepared = SimpleNamespace(source=SimpleNamespace(plan=p), gather=gather,
            group=SimpleNamespace(lhs_row=lambda index: p.lhs_row))
        common = dict(coordinates)
        p.__post_init__()
        if (any(type(common.get(key)) is not int or common[key] <= 0
                for key in ('rows', 'columns'))
                or common['rows'] > p.row_tile or common['columns'] > p.column_tile):
            raise ValueError('resident pooled producer rectangle exceeds its primitive source tile')
        if p.bias: lower('initialize_bias', common)
        fragments = _gather_coordinates(prepared, builder,
            common['output_row'], common['rows'])
        ci = shape[-1]
        kh, kw = w.kernel
        op = builder.op
        def y(ky):
            def x(kx):
                bands = _gather_bands(prepared, builder, fragments, ky, kx)
                def channel(chunk, depth):
                    start = op('mul', chunk, p.channel_tile)
                    _emit_gather_bands(prepared, builder, bands, start, depth,
                        lambda fields: lower('load_input', fields))
                    reduction = op('add', op('mul', op('add', op('mul', ky, kw), kx), ci), start)
                    lower('load_weight', dict(reduction_row=reduction,
                        output_column=common['output_column'], local_row=p.rhs_row,
                        rows=depth, columns=common['columns']))
                    first = op('and', builder.cmp('eq', ky, 0), builder.cmp('eq', kx, 0), 'i1')
                    first = op('and', first, builder.cmp('eq', start, 0), 'i1')
                    lower('compute', dict(common, lhs_row=p.lhs_row, rhs_row=p.rhs_row,
                        depth=depth, first_reduction=first, bias=p.bias))
                full, tail = divmod(ci, p.channel_tile)
                if full: builder.loop(0, full, 1, lambda chunk: channel(chunk, p.channel_tile))
                if tail: channel(full, tail)
            builder.loop(0, kw, 1, x)
        builder.loop(0, kh, 1, y)
        lower('store', common)

    def receipt(self):
        return dict(source_sha256=self.source_identity, panel=asdict(self.panel),
            identity=hashlib.sha256(json.dumps(asdict(self.panel), sort_keys=True).encode()).hexdigest(),
            input_cover='shared affine original-row fragments with disjoint zero/data/zero bands',
            weight_load_scope='once per N tile across all images and halo blocks',
            physical_layout='injective original kernel/channel slot times primitive row pitch',
            completion='complete weight preparation and complete phase last use',
            host_payload_operations=0)
