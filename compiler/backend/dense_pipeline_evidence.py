"""Offline custody and identity admission for existing dense timing feedback.

Native result pins are preserved under their original names. A capsule digest
is not an input/reference digest, and an ELF digest is not a lowered-IR digest.
Missing original evaluation identities therefore leave the retained incumbent.
This module neither benchmarks nor constructs a retrospective validation split.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path

from merlin.transforms.evaluated_exploration import CompletedObservation, EvaluationRequest
from merlin.transforms.pass_exploration import ExplorationGeometry
from merlin.transforms.pipeline_policy import PolicyFacts
from merlin.transforms.shape_policy import ShapeCaseFeedback


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class NativePipelineObservation:
    alternative: str
    case: str
    response_sha256: str
    detail_sha256: str
    numerical_status: str
    native_status: str | None
    cycles: int | None
    complete_event_accounting: bool
    unresolved_events: bool
    elf_sha256: str | None
    facts_sha256: str | None
    model_sha256: str | None
    scope_sha256: str | None
    comparison_domain_sha256: str | None
    capsule_sha256: str | None
    timing_qualified: bool
    failure_category: str | None

    @property
    def complete_modeled_window(self):
        return (self.native_status in ('PROVISIONAL', 'QUALIFIED')
            and self.complete_event_accounting and not self.unresolved_events
            and type(self.cycles) is int and self.cycles > 0)


def load_native_observation(response_path, *, case, alternative, workspace='.'):
    """Read one original response and its hash-bound detail, without inference."""
    if alternative not in ('retained', 'segmented'):
        raise ValueError('unknown dense pipeline alternative')
    root = Path(workspace).resolve()
    path = Path(response_path)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    response = json.loads(raw)
    rows = [r for r in response.get('cases', ()) if r.get('capsule') == case]
    if len(rows) != 1:
        raise ValueError('original response must contain exactly one requested case')
    link = rows[0].get('detail', {})
    target = (root / link.get('path', '')).resolve()
    if not target.is_relative_to(root) or not _pin(link.get('sha256')):
        raise ValueError('detail requires a workspace-contained path and original digest')
    detail_raw = target.read_bytes()
    if sha256(detail_raw).hexdigest() != link['sha256']:
        raise ValueError('original result detail digest mismatch')
    detail = json.loads(detail_raw)
    if detail.get('capsule') != case:
        raise ValueError('detail belongs to another case')
    p = detail.get('performance') or {}
    outcome = (p.get('feedback_brief') or {}).get('outcomes') or {}
    numerical = outcome.get('numerical_status', 'unknown')
    if numerical not in ('pass', 'fail', 'unknown'):
        numerical = 'unknown'
    cycles = p.get('estimated_whole_soc_cycles')
    if type(cycles) is not int or cycles <= 0:
        cycles = None
    return NativePipelineObservation(alternative, case, sha256(raw).hexdigest(),
        link['sha256'], numerical, p.get('status'), cycles,
        p.get('complete_event_accounting') is True,
        bool(p.get('unresolved_events', True)), p.get('elf_sha256'),
        p.get('facts_sha256'), p.get('model_sha256'), p.get('scope_sha256'),
        p.get('comparison_domain_sha256'), p.get('capsule_sha256'),
        p.get('timing_qualified') is True, detail.get('failure_category'))


@dataclass(frozen=True)
class OriginalPipelineBinding:
    """Owner-held pre-evaluation artifacts; never synthesized from native rows.

    Requests must name the actual executed ELF bytes. The explicit native model
    and comparison-domain pins bind this adapter's native feedback producer;
    they do not replace the requests' independently selected provider identity.
    Facts and implementation pin belong to the original authored pass registry.
    """
    requests: tuple[tuple[str, EvaluationRequest], ...]
    geometry: ExplorationGeometry
    facts: PolicyFacts
    implementation_sha256: str
    native_model_sha256: str
    native_comparison_domain_sha256: str

    def __post_init__(self):
        if (type(self.requests) is not tuple or len(self.requests) != 2
                or {name for name, _ in self.requests} != {'retained', 'segmented'}
                or any(type(r) is not EvaluationRequest for _, r in self.requests)
                or type(self.geometry) is not ExplorationGeometry
                or type(self.facts) is not PolicyFacts
                or not all(_pin(v) for v in (self.implementation_sha256,
                    self.native_model_sha256, self.native_comparison_domain_sha256))):
            raise ValueError('original paired requests, facts and native domain pins required')


@dataclass(frozen=True)
class PipelineEvidencePair:
    observations: tuple[NativePipelineObservation, ...]
    feedback: ShapeCaseFeedback | None
    refusal: tuple[str, ...]

    def receipt(self):
        return dict(schema='dense_pipeline_native_evidence.v1',
            admitted=self.feedback is not None, refusal=self.refusal,
            observations=tuple(asdict(row) for row in self.observations),
            physical_qualification_inferred=False, validation_partition_inferred=False)


MISSING_ORIGINAL_BINDING = (
    'original paired ELF EvaluationRequests with source/ABI/inputs/reference/library/provider/ROI/context identities',
    'original source-stamped PolicyFacts and geometry',
    'original selected registry/target implementation digest',
    'original native model/comparison-domain binding',
)


def load_pipeline_evidence(retained_response, segmented_response, *, case,
                           workspace='.', binding=None):
    """Admit exact native model observations or retain an explicit refusal.

    The optional binding must already exist from the original experiment. This
    function cannot reconstruct missing pins from current source or shape names.
    Failed/unknown native results remain failed/unknown CompletedObservations;
    the shared assessor decides their eligibility, never a filtered winner list.
    """
    rows = tuple(load_native_observation(path, case=case, alternative=name,
        workspace=workspace) for name, path in
        (('retained', retained_response), ('segmented', segmented_response)))
    if binding is None:
        return PipelineEvidencePair(rows, None, MISSING_ORIGINAL_BINDING)
    if type(binding) is not OriginalPipelineBinding:
        raise ValueError('use the original typed evidence binding')
    binding.__post_init__()
    requests = dict(binding.requests)
    missing, completed = [], []
    for row in rows:
        request = requests[row.alternative]
        identity = request.identity
        checks = (
            (request.program.format == 'elf' and request.program_sha256 == row.elf_sha256,
                'executed ELF bytes differ from original request'),
            (identity.basis == 'modeled_roi', 'native model feedback requires modeled ROI identity'),
            (identity.facts_sha256 == row.facts_sha256, 'selected hardware facts differ'),
            (identity.scope_sha256 == row.scope_sha256, 'complete-call scope differs'),
            (binding.native_model_sha256 == row.model_sha256, 'native model differs'),
            (binding.native_comparison_domain_sha256 == row.comparison_domain_sha256,
                'native comparison domain differs'),
        )
        missing.extend(row.alternative+': '+reason for ok, reason in checks if not ok)
        if all(ok for ok, _ in checks):
            completed.append((row.alternative, CompletedObservation(identity,
                request.program_sha256, row.detail_sha256, row.numerical_status,
                'complete' if row.complete_modeled_window else 'unknown',
                row.complete_modeled_window, row.cycles)))
    if missing:
        return PipelineEvidencePair(rows, None, tuple(missing))
    try:
        feedback = ShapeCaseFeedback(binding.geometry, binding.facts,
            tuple(completed), binding.implementation_sha256)
    except ValueError as error:
        return PipelineEvidencePair(rows, None, (str(error),))
    return PipelineEvidencePair(rows, feedback, ())


def assess_pipeline_evidence(pairs, *, design, predicates=None):
    """Use only admitted original identities and the caller's frozen design.

    Incomplete rows are reported, never retroactively assigned to held-out sets.
    The shared assessor retains the incumbent for missing roster/boundary data.
    """
    from .dense_pipeline_policy import assess_dense_pipeline_policy
    pairs = tuple(pairs)
    if any(type(pair) is not PipelineEvidencePair for pair in pairs):
        raise ValueError('assessment needs typed original evidence pairs')
    feedback = tuple(pair.feedback for pair in pairs if pair.feedback is not None)
    policy = assess_dense_pipeline_policy(feedback, design=design, predicates=predicates)
    return policy, tuple(pair.receipt() for pair in pairs)
