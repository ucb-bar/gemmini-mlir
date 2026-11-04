"""Explicit, provisional choice over the existing convolution panel products.

Uniform stores remain the default.  The fragmented affine option is an
author-selected experiment, not an assessed shape policy or a learned cutoff.
Both products retain the shared source, storage, epoch and terminal encoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod

from merlin.transforms.pipeline_policy import (
    PipelinePolicy, PipelineVariant, PolicyFacts, RegisteredPass,
    ResourcePredicate, execute_policy,
)
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from . import isa
from .conv_spatial_panels import _prepare, _emit_prepared

LEVEL = 'authored_conv_spatial_source'
OUTPUT = 'source_bound_conv_spatial_product'
OPTIONS = ('uniform_stores', 'provisional_affine_fragmented')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _binding(ctx, names, prepared):
    """Bind original semantics and the actual shared allocation, not case IDs."""
    return _digest(dict(
        tensors=[(name, tuple(ctx.spec(name)['shape']), ctx.spec(name)['dtype'])
                 for name in names],
        source=prepared.source.identity, gather=asdict(prepared.gather),
        group=asdict(prepared.group), epoch=asdict(prepared.epoch),
        attributes=prepared.attributes,
        rhs_pitch=prepared.rhs_pitch, output_pitch=prepared.output_pitch,
        zero_seed=prepared.zero_seed, ordering=prepared.ordering,
    ))


@dataclass(frozen=True)
class ConvSpatialProduct:
    """An actual admitted source/panel/epoch product consumed through emit().

    Replanning is read-only and detects changed source or resources before the
    common lowerer emits anything. The lowerer keeps its original independent
    admission and actual command encoding; this policy grants no target law.
    """
    ifm: str
    rhs: str
    dst: str
    attributes_json: str
    physical_views: object
    allow_wide: bool
    trial_mode: str
    prepared: object
    binding_sha256: str

    def _names(self):
        bias = self.prepared.attributes.get('bias')
        return (self.ifm, self.rhs, self.dst) + ((bias,) if bias else ())

    def revalidate(self, ctx):
        if self.trial_mode not in ('uniform_stores', 'affine_origins'):
            raise ValueError('convolution regional product changed its registered mode')
        if _binding(ctx, self._names(), self.prepared) != self.binding_sha256:
            raise ValueError('convolution regional product changed its source or allocation')
        current = _prepare(ctx, self.ifm, self.rhs, self.dst,
            json.loads(self.attributes_json), self.physical_views,
            allow_wide=self.allow_wide)
        if current is None or _binding(ctx, self._names(), current) != self.binding_sha256:
            raise ValueError('convolution regional product no longer matches current admission')

    def emit(self, ctx):
        self.revalidate(ctx)
        result = _emit_prepared(ctx, self.ifm, self.rhs, self.dst,
            self.prepared, trial_mode=self.trial_mode)
        if (not result or result['source_sha256'] != self.prepared.source.identity
                or result['trial_mode'] != self.trial_mode
                or result['rhs_rows'] != self.prepared.group.panel.rhs_rows
                or result['rhs_base'] != self.prepared.group.panel.rhs_base):
            raise ValueError('convolution regional emission differs from its selected product')
        return result


@dataclass(frozen=True)
class _RegionalSource:
    binding_sha256: str
    shapes: tuple
    element_types: tuple
    resources: tuple


def regional_pipeline_policy():
    return PipelinePolicy((
        PipelineVariant('affine_origins', ('conv_spatial_affine_origins',), (
            ResourcePredicate('explicit_affine_request', 1),
            ResourcePredicate('kernel_3x3', 1),
            ResourcePredicate('full_input_channel_tile', 1),
            ResourcePredicate('full_output_channel_tile', 1),
            ResourcePredicate('gather_fragments', 4),
        )),
        PipelineVariant('uniform_stores', ('conv_spatial_uniform_stores',)),
    ), fallback='uniform_stores')


def plan_conv_regional(ctx, ifm, rhs, dst, attributes, *,
        option='uniform_stores', physical_views=None, allow_wide=True):
    """Return (source-bound product or None, selection receipt), without emission.

    Only an explicit provisional option can select affine origins. Fragmentation
    is the shared gather's actual bounded primitive-row extent. It is an
    unassessed performance hypothesis; independent storage/source legality still
    comes from _prepare. Missing boundary and channel-neighbor evidence prevents
    enabling this region by default.
    """
    if option not in OPTIONS:
        raise ValueError('convolution regional option must be one of '+repr(OPTIONS))
    prepared = _prepare(ctx, ifm, rhs, dst, attributes, physical_views,
                        allow_wide=allow_wide)
    if prepared is None:
        return None, dict(schema='authored_conv_regional.v1', option=option,
                          refusal='original spatial-panel admission refused')
    names = (ifm, rhs, dst) + ((prepared.attributes['bias'],)
        if prepared.attributes.get('bias') else ())
    binding = _binding(ctx, names, prepared)
    p = prepared.source.plan
    pixels = prod(prepared.gather.output_spatial)
    active_rows = min(isa.DIM, pixels)
    resources = tuple(sorted(dict(
        dim=isa.DIM, spad_rows=isa.SPAD_ROWS, acc_rows=isa.ACC_ROWS,
        explicit_affine_request=int(option == 'provisional_affine_fragmented'),
        kernel_3x3=int(p.window.kernel == (3, 3)),
        full_input_channel_tile=int(p.window.input_shape[-1] >= isa.DIM),
        full_output_channel_tile=int(p.output_channels >= isa.DIM),
        gather_fragments=prepared.gather.max_fragments(active_rows),
        gather_fragment_bound_rows=active_rows,
        spatial_tiles=(pixels+isa.DIM-1)//isa.DIM,
        reduction_tiles=prepared.group.panel.reduction_tiles,
        resident_column_blocks=prepared.group.panel.column_blocks,
        actual_rhs_rows=prepared.group.panel.rhs_rows,
        actual_accumulator_rows=prepared.group.panel.accumulator_rows,
    ).items()))
    source = _RegionalSource(binding,
        tuple(tuple(ctx.spec(n)['shape']) for n in names),
        tuple(ctx.spec(n)['dtype'] for n in names), resources)
    # This names an authored resource view, not hardware qualification.
    context = PipelineContext(resources='authored_conv_resources:'+_digest(resources))
    def verify_source(value):
        if type(value) is not _RegionalSource or value != source:
            raise ValueError('convolution policy source changed')
    session = PipelineSession(source, context=context, ir_level=LEVEL,
        fingerprint=lambda value: _digest(asdict(value)),
        verify_ir=verify_source)
    facts = PolicyFacts(session.stamp(), shapes=source.shapes,
        element_types=source.element_types, layouts=('nhwc', 'row_major', 'row_major')
            + (('row_major',) if len(names) == 4 else ()),
        operations=frozenset({'convolution'}), resources=resources)
    candidates = {mode: ConvSpatialProduct(ifm, rhs, dst,
        json.dumps(prepared.attributes, sort_keys=True, allow_nan=False),
        physical_views, allow_wide, mode, prepared, binding)
        for mode in ('uniform_stores', 'affine_origins')}
    registry = {}
    for mode, product in candidates.items():
        key = 'conv_spatial_'+mode
        def action(current, product=product):
            if current.module != source:
                raise ValueError('convolution policy cannot reuse a different source product')
            product.revalidate(ctx)
            return {OUTPUT: product}
        registry[key] = RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    selected = execute_policy(session, regional_pipeline_policy(), registry, facts, output=OUTPUT)
    return selected.artifact, dict(schema='authored_conv_regional.v1', option=option,
        qualification='explicit provisional option; not assessed_shape_policy qualification',
        evidence_guard='no automatic affine promotion; incomplete boundary/channel-neighbor pairs',
        source_sha256=prepared.source.identity, binding_sha256=binding,
        policy=selected.receipt)


def lower_conv_regional(ctx, ifm, rhs, dst, attributes, **options):
    """Consume the chosen actual product, retaining selection and emitted receipts."""
    product, selection = plan_conv_regional(ctx, ifm, rhs, dst, attributes, **options)
    if product is None:
        return False
    emitted = product.emit(ctx)
    receipt = dict(selection, emitted=emitted)
    ctx.metadata.setdefault('convolution_regional', []).append(receipt)
    return receipt
