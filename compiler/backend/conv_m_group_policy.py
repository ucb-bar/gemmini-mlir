"""Provisional resource selection of existing across-M ready-store epochs.

Capacity and checked source products establish legality. The structural region
below is a frozen performance hypothesis, not an assessed shape policy. Full-K
RHS residency already belongs to the incumbent: this route groups ready stores,
without claiming additional weight reuse or a new hardware ordering law.
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
from .conv_transfer_regional import plan_transfer_regional
from .conv_weight_panels import ConvWeightPanelEpochs, plan_weight_panels

LEVEL = 'authored_conv_m_group_choice'
OUTPUT = 'checked_conv_m_group_product'
OPTIONS = ('incumbent', 'resource', 'stride1_resource')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class _Source:
    binding: str
    resources: tuple


def m_group_pipeline_policy():
    """No fitted dimensional cutoff: select only an actual larger legal group."""
    return PipelinePolicy((
        PipelineVariant('m_groups', ('conv_m_group_product',), tuple(
            ResourcePredicate(name, 1) for name in (
                'explicit_resource_option', 'uniform_i8_3x3',
                'aligned_channels', 'ordinary_window', 'group_grows',
                'checked_m_group_product'))),
        PipelineVariant('incumbent', ('conv_m_group_incumbent',)),
    ), fallback='incumbent')


def _facts(ctx, incumbent, candidate, option):
    prepared = incumbent.regional.prepared
    p = prepared.source.plan
    w = p.window
    ci = w.input_shape[-1]
    attrs = prepared.attributes
    # Storage/epoch legality is intentionally not inferred from these features.
    # Actual candidate revalidation owns it. No per-column or raw-i32 expansion
    # is promoted from the narrow uniform-i8 timing evidence.
    return tuple(sorted(dict(
        explicit_resource_option=int(option in ('resource', 'stride1_resource')),
        uniform_i8_3x3=int(w.kernel == (3, 3) and
            ctx.spec(incumbent.regional.dst)['dtype'] == 'i8' and
            attrs.get('acc_scale_by_column') is None and
            attrs.get('requant_shift') is None),
        aligned_channels=int(ci % isa.DIM == 0 and p.output_channels % isa.DIM == 0),
        ordinary_window=int(w.dilation == (1, 1) and
            w.stride in (((1, 1),) if option == 'stride1_resource' else ((1, 1), (2, 2))) and w.padding in ((0, 0, 0, 0), (1, 1, 1, 1))),
        group_grows=int(candidate is not None and
            candidate.prepared.group.row_blocks > prepared.group.row_blocks),
        checked_m_group_product=int(candidate is not None),
        dim=isa.DIM, acc_rows=isa.ACC_ROWS, spad_rows=isa.SPAD_ROWS,
        spatial_tiles=(prod(prepared.gather.output_spatial)+isa.DIM-1)//isa.DIM,
        original_m_slots=prepared.group.row_blocks,
        selected_m_slots=candidate.prepared.group.row_blocks if candidate else prepared.group.row_blocks,
        resident_n_slots=prepared.group.panel.column_blocks,
        complete_rhs_rows=prepared.group.panel.rhs_rows,
    ).items()))


@dataclass(frozen=True)
class ConvMGroupChoice:
    """Retain and consume the exact checked alternative selected by the policy."""
    incumbent: object
    candidate: ConvWeightPanelEpochs | None
    selected: str
    option: str
    resources: tuple
    policy_receipt_json: str

    @property
    def product(self):
        return self.candidate if self.selected == 'm_groups' else self.incumbent

    def revalidate(self, ctx):
        if self.option not in OPTIONS or self.selected not in ('m_groups', 'incumbent'):
            raise ValueError('unknown convolution M-group policy choice')
        self.incumbent.revalidate(ctx)
        if self.candidate is not None:
            if type(self.candidate) is not ConvWeightPanelEpochs or self.candidate.original is not self.incumbent:
                raise ValueError('M-group choice is not bound to its exact incumbent')
            self.candidate.revalidate(ctx)
        if _facts(ctx, self.incumbent, self.candidate, self.option) != self.resources:
            raise ValueError('M-group source/resource predicate changed')
        enabled = all(dict(self.resources)[name] for name in (
            'explicit_resource_option', 'uniform_i8_3x3', 'aligned_channels',
            'ordinary_window', 'group_grows', 'checked_m_group_product'))
        if (self.selected == 'm_groups') != enabled:
            raise ValueError('M-group selection differs from its frozen predicate')

    def emit(self, ctx):
        self.revalidate(ctx)
        emitted = self.product.emit(ctx)
        self.revalidate(ctx)
        receipt = dict(schema='authored_conv_m_group_policy.v1',
            selected=self.selected, option=self.option,
            qualification='provisional structural region; not assessed_shape_policy',
            evidence=('development trial41 plus67; stride1-only refinement awaits fresh validation'
                if self.option == 'stride1_resource' else
                'trial41 plus67 development; resource option has mixed stride2 result'),
            mechanism='across-M ready-store epochs; incumbent already retains complete RHS',
            policy=json.loads(self.policy_receipt_json), resources=dict(self.resources),
            source_sha256=self.incumbent.regional.prepared.source.identity,
            emitted=emitted)
        ctx.metadata.setdefault('convolution_m_group_policy', []).append(receipt)
        return receipt


def plan_conv_m_groups(ctx, ifm, rhs, dst, attributes, *, option='resource'):
    """Return an immutable real product choice, or None before any emission.

    ``incumbent`` selects the unchanged transfer-regional product. ``resource``
    is an explicit provisional selection over the frozen structural region;
    unsupported source geometry retains that same incumbent. ``stride1_resource``
    adds only a unit-stride restriction, developed after the mixed stride2
    validation result of resource; it is a separate provisional option. Independent common
    source/storage/epoch admissions remain authoritative for both products.
    """
    if option not in OPTIONS:
        raise ValueError('convolution M-group option must be one of '+repr(OPTIONS))
    candidate = plan_weight_panels(ctx, ifm, rhs, dst, attributes)
    incumbent = candidate.original if candidate else plan_transfer_regional(ctx, ifm, rhs, dst, attributes)
    if incumbent is None:
        return None
    incumbent.revalidate(ctx)
    if candidate is not None:
        candidate.revalidate(ctx)
    resources = _facts(ctx, incumbent, candidate, option)
    source = _Source(incumbent.regional.binding_sha256, resources)
    def verify(value):
        if type(value) is not _Source or value != source:
            raise ValueError('M-group policy source changed')
    session = PipelineSession(source,
        context=PipelineContext(resources='authored_m_group_resources:'+_digest(resources)),
        ir_level=LEVEL, fingerprint=lambda value:_digest(asdict(value)), verify_ir=verify)
    facts = PolicyFacts(session.stamp(), operations=frozenset({'convolution'}), resources=resources)
    registry = {}
    for key, product in (('conv_m_group_product', candidate), ('conv_m_group_incumbent', incumbent)):
        def action(current, product=product):
            if current.module != source or product is None:
                raise ValueError('M-group policy selected an unavailable source product')
            product.revalidate(ctx)
            return {OUTPUT: product}
        registry[key] = RegisteredPass(PassSpec(key, 0, produces=(OUTPUT,),
            mutates_ir=False, ir_level=LEVEL), action)
    selected = execute_policy(session, m_group_pipeline_policy(), registry, facts, output=OUTPUT)
    choice = ConvMGroupChoice(incumbent, candidate,
        'm_groups' if candidate is not None and selected.artifact is candidate else 'incumbent',
        option, resources, json.dumps(selected.receipt, sort_keys=True))
    choice.revalidate(ctx)
    return choice


def lower_conv_m_groups(ctx, ifm, rhs, dst, attributes, *, option='resource'):
    """Emit selected real shared product; None is an atomic admission refusal."""
    product = plan_conv_m_groups(ctx, ifm, rhs, dst, attributes, option=option)
    return None if product is None else product.emit(ctx)


def lower_regional_m_groups(ctx, ifm, rhs, dst, attributes, *, option='resource'):
    """Compiler pre-dispatch hook: leave the incumbent entirely untouched.

    A None result performs planning only, with no metadata or terminal mutation;
    the compiler continues its existing spatial/other route. A selected M-group
    consumes its checked product and reports both policy and actual emission.
    """
    choice = plan_conv_m_groups(ctx, ifm, rhs, dst, attributes, option=option)
    if choice is None or choice.selected != 'm_groups':
        return None
    return choice.emit(ctx)
