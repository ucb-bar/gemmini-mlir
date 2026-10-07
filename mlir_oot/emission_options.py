"""Immutable, complete target emission facts for checked same-family replacement.

These records carry scheduling/ABI facts only. Constructor validation remains
responsible for resources and numeric legality; no record grants eligibility or
profitability. A new keyword without a corresponding field refuses replacement
instead of silently disappearing between compiler factories.
"""

from dataclasses import dataclass, fields, replace
from inspect import Parameter, signature

from merlin.llvmlower.segmented_matrix_view import SegmentedRows

from .readout_store_plan import PairedReadoutPlan


@dataclass(frozen=True, kw_only=True)
class GemmEmissionOptions:
    prefetch_b_rows: tuple[int, int] | None = None
    resident_a_load_tiles: int = 1
    input_view: SegmentedRows | None = None
    cached_b_resource_capacity: bool = False
    stationary_b_tail_before_last_full: bool = False
    cached_a_output_blocks: bool = False


@dataclass(frozen=True, kw_only=True)
class FlatConvEmissionOptions:
    wide_a: bool = False
    separate_b_bank: bool = False
    band_rows: int | None = None
    virtual_padding: bool = False
    pingpong_b: bool = False
    loop_spatial: bool = False
    store_plan: PairedReadoutPlan | None = None


@dataclass(frozen=True, kw_only=True)
class ResidentConvEmissionOptions:
    rows_per_tile: int = 1
    loop_channels: bool = False
    prefetch_b: bool = False
    weight_base: int | None = None
    source_stride: bool = False
    row_residue: bool = False
    compact_commands: bool = False
    weight_issue_tiles: int | None = None
    flat_spatial_planes: bool = False
    store_plan: PairedReadoutPlan | None = None
    tail_before_last_full: bool = False


@dataclass(frozen=True, kw_only=True)
class ResidentStripeEmissionOptions:
    stripe_rows: int | None = None
    compact_inner_commands: bool = False
    compact_reduction_commands: bool = False


def _names(record_type, constructor):
    """Close the complete declared API, including defaults, before any clone."""
    params = tuple(signature(constructor).parameters.values())
    if len(params) < 2 or params[0].name != "self":
        raise ValueError("emission constructor must declare self and source shape")
    if params[1].kind not in (
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError("emission constructor requires a positional source shape")
    rest = params[2:]
    if any(p.kind != Parameter.KEYWORD_ONLY for p in rest):
        raise ValueError("emission options require an explicit keyword-only API")
    members = fields(record_type)
    if [p.name for p in rest] != [f.name for f in members]:
        raise ValueError("emission options do not cover the complete constructor API")
    if any(p.default != f.default for p, f in zip(rest, members)):
        raise ValueError("emission options defaults differ from the constructor API")
    return tuple(f.name for f in members)


class EmissionOptionsMixin:
    @property
    def emission_options(self):
        """Snapshot effective current facts, including legacy public attributes.

        Records are immutable. Taking a fresh snapshot also preserves existing
        direct attribute users; replacement never depends on a stale constructor
        copy. Derived allocations/IR builders are deliberately absent.
        """
        cls = type(self)
        record_type = cls.__dict__.get("emission_options_type")
        if record_type is None:
            raise ValueError("selected family has no declared emission options")
        names = _names(record_type, cls.__init__)
        aliases = cls.__dict__.get("emission_option_attributes", {})
        if any(name not in names for name in aliases):
            raise ValueError("emission option attribute alias is not declared")
        return record_type(
            **{name: getattr(self, aliases.get(name, name)) for name in names}
        )

    def with_emission_options(self, *, shape=None, **changes):
        """Preserve all facts, change explicit fields, then revalidate the emitter.

        Unknown fields refuse. The original generator/options/IR are not mutated.
        Cross-family layout selection requires its own explicit legality proof.
        """
        current = self.emission_options
        try:
            selected = replace(current, **changes)
        except TypeError as error:
            raise ValueError("unknown emission option replacement") from error
        cls = type(self)
        owner = cls.__dict__.get("emission_shape_attribute")
        if owner is None:
            raise ValueError("emission family has no declared source shape")
        source = getattr(self, owner) if shape is None else shape
        candidate = cls(
            source, **{f.name: getattr(selected, f.name) for f in fields(selected)}
        )
        # Constructor normalization is permitted (e.g. complete-band default or
        # compact loops imply channel loops), while all requests are validated.
        _ = candidate.emission_options
        return candidate
