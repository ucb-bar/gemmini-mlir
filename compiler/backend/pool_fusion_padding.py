"""Source-bound minimum raw halo for a private monotone pooled producer.

The original producer exists only at valid spatial coordinates. Each original
pool window is nonempty, its declared i8 padding is -128, and its readout Q is
monotone. Q(INT32_MIN) <= Q(v) for every original signed i32 producer value v.
Thus adding raw INT32_MIN at a missing coordinate changes neither maximum,
even when Q(INT32_MIN) is greater than -128 (zero scale or ReLU). This adapter
does not apply to completely padded windows or separately observable tensors.
"""
from dataclasses import dataclass

from merlin.compile.scheduling.spatial_readout import prove_spatial_readout


@dataclass(frozen=True)
class MinimumSpatialHalo:
    source_identity: str
    raw_value: int = -(1 << 31)

    @classmethod
    def prepare(cls, plan):
        plan.__post_init__()
        if not any(plan.source.spatial.padding):
            return None
        result = cls(plan.identity)
        result.require(plan)
        return result

    def require(self, plan):
        plan.__post_init__()
        source = plan.source
        s = source.spatial
        if (self.source_identity != plan.identity or self.raw_value != -(1 << 31)
                or s.accumulator_bits != 32 or s.output_bits != 8
                or s.pad_value != -128 or source.readout_mode != 'single_readout'):
            raise ValueError('minimum raw halo is not bound to this exact signed source/readout')
        # Source producer validation proves every Y/X window has a valid cell,
        # via the first-window end and last-window start in each monotone axis.
        # Its readout chain retains the exact rounding/saturation pipeline.
        for chain in source.readouts:
            prove_spatial_readout(chain)

    def receipt(self):
        return dict(schema='backend.minimum_spatial_halo.v1',
            source_sha256=self.source_identity, raw_i32_value=self.raw_value,
            original_pool_pad_i8=-128,
            proof='each original window nonempty; monotone Q(INT32_MIN)<=Q(v); minimum halo cannot change max',
            source_products='all valid producer coordinates preserved; only duplicate border products removed',
            physical_mapping='disjoint missing ACC cells loaded from immutable i32 literal; never overwrite valid cells',
            completion='real full drain after block preparation before pooled store',
            host_payload_operations=0, timing='UNKNOWN')
