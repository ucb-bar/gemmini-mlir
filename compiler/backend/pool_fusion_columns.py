"""Optional source-exact scalar-column pooled readouts.

Each distinct-scale column is produced in accumulator lane zero. The original
positive-f32/RNE/activation/clamp readout is monotone in that column, so the
existing complete-i32 spatial proof moves its maximum without combining scales
or rounding boundaries. No offset-lane or changing-configuration overlap law is
assumed: every original pooled store completes before the next column phase.
"""
from dataclasses import dataclass
from math import isfinite

from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain, prove_spatial_readout
from . import isa


@dataclass(frozen=True)
class ColumnSpatialReadout:
    scales: tuple[float, ...]
    relu: bool
    chains: tuple[SpatialReadoutChain, ...]

    @classmethod
    def prepare(cls, attrs, columns, *, automatic):
        if automatic or attrs.get('pool_column_readout', 'off') != 'on':
            raise ValueError('column pooling requires explicit optional scalar-column selection')
        ep = tuple(attrs.get('epilogue', ()))
        expected = tuple(stage for stage in ('bias_add', 'acc_scale', 'relu', 'maxpool') if stage in ep)
        if (ep != expected or 'acc_scale' not in ep or 'maxpool' not in ep
                or attrs.get('requant_shift') is not None or 'acc_scale' in attrs):
            raise ValueError('column pooling needs the exact original ordered scale/readout chain')
        scales = tuple(map(float, attrs['acc_scale_by_column']))
        if len(scales) != columns or any(not isfinite(s) or s <= 0 for s in scales):
            raise ValueError('column pooling spatial proof requires positive finite original scales')
        relu = bool(attrs.get('relu') or 'relu' in ep)
        q = tuple(RoundedReadout((s,), relu) for s in scales)
        chains = tuple(SpatialReadoutChain(stage, RoundedReadout((1.0,)), stage) for stage in q)
        for chain in chains:
            prove_spatial_readout(chain)
        return cls(tuple(stage.scales[0] for stage in q), relu, chains)

    def emit_config(self, ctx, column, *, stride, **pool_fields):
        # Shared target selector emits only literal commands and column-index
        # control flow; it loads no configuration table or tensor payload.
        from .dense_readout import _c, _emit_column_config
        funct, rs1, _ = isa.config_st(stride, activation=int(self.relu), **pool_fields)
        words = tuple(isa.config_st(stride, scale=s)[2] for s in self.scales)
        _emit_column_config(ctx, _c(ctx.fb, column), funct, rs1, words)

    def receipt(self):
        return dict(schema='backend.column_spatial_readout.v1',
            scales=list(self.scales), activation=int(self.relu), producer_tile_columns=1,
            accumulator_lane=0, readout_proofs=[prove_spatial_readout(chain) for chain in self.chains],
            numeric_pipeline='source signed i32 including bias; original f32 scale; RNE; activation; signed i8 clamp; maxpool',
            maximum_law='positive finite readout monotone separately in each original column; no cross-column pooling',
            configuration='shared literal CONFIG_ST column selector; no tensor/configuration-table loads',
            completion='real raw-result and pooled-store drains before next column phase or storage reuse',
            automatic_selection=False, host_payload_operations=0, timing='UNKNOWN')
