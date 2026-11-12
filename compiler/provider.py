"""Adapt actual Gemmini region emission to Merlin's installed compile service."""

from __future__ import annotations

import hashlib

name = "gemmini"


def _implementation_identity():
    """Pin active OOT and shared selected-invocation owners, excluding archived SDK.

    This is a compiler source receipt, not complete environment attribution or
    independent implementation verification.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    paths = [p for p in root.iterdir() if p.is_file() and p.suffix in (".py", ".json")]
    paths += [p for p in (root / "backend").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    import importlib

    for module_name in ("merlin.compiler.backend.selected", "merlin.compiler.backend.sourceproposal",
                        "merlin.compiler.pipeline", "merlin.compiler.invocation"):
        module = importlib.import_module(module_name)
        digest.update(module_name.encode())
        digest.update(b"\0")
        digest.update(Path(module.__file__).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _recipe(pattern, proof):
    import json

    from merlin.compiler.backend.sourceproposal import LoweringRecipe

    numerical = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return LoweringRecipe(
        _implementation_identity(), "gemmini.single-dense-direct.v1",
        (("pattern", pattern), ("readout_proof_sha256", numerical.hexdigest()),
         ("coverage", "complete-single-block-and-ordered-live-tensor-ports"),
         ("resources", "backend-local-legality-checks; no complete device resource envelope")),
        (("llvm-dialect", "region"), ("json", "command-plan"), ("json", "provider-record")), "logical_tensor_v1",
    )


def _proposals(source, plans, *, resource: str, memory_domain: str):
    """Offer implemented dense patterns without selecting a program partition.

    Existing source matchers supply both the raw contraction and its supported
    complete readout. Their numerical records are compiler claims; Merlin checks
    them independently against the caller-selected target contract.
    """
    import json

    from merlin.compiler.backend.sourceproposal import SourceProposal, proposal_boundary, source_numerical_identity
    from merlin.compiler.dialects.lowering.global_plan import CycleInterval, RegionAlternative

    from .backend.source_groups import _dependencies

    source.revalidate()
    indices = {operation: index for index, operation in enumerate(source.operations)}
    offers = []
    for plan in plans:
        root_index = indices.get(plan.root)
        if root_index is None:
            continue
        bare, _ = _dependencies(plan.root, [plan.root])
        variants = [("raw", "dense", bare, None)]
        if plan.output is not plan.root.results[0] and plan.diagnostic.get("cut") == "requant":
            variants.insert(0, ("fused", "dense-requant", plan.operations, plan.diagnostic))
        for label, pattern, operations, proof in variants:
            if any(operation not in indices for operation in operations):
                continue
            internal = {value for operation in operations for value in operation.results}
            if any(value in internal for value in plan.root.operands[:2]):
                # The actual extracted dense callback requires explicit operand
                # ports; literal-state materialization is not admitted here.
                continue
            nodes = tuple(sorted(indices[operation] for operation in operations))
            # Noncontiguous regions require movement/effect analysis beyond this
            # first protocol route. Retain the ordinary unselected source path.
            if nodes != tuple(range(nodes[0], nodes[-1] + 1)):
                continue
            inputs, outputs = proposal_boundary(source, nodes)
            metadata = (
                ("pattern", pattern),
                ("scratch_scope", "additional host tensor scratch; accelerator local resources checked by backend"),
            )
            if proof is not None:
                metadata += (("numerical_proof", json.dumps(
                    proof, sort_keys=True, separators=(",", ":"), allow_nan=False
                )),)
            alternative = RegionAlternative(
                id=f"{name}_{label}_{root_index}", nodes=nodes, implementation=pattern, placement=resource,
                cycles=CycleInterval.unknown("provider has no comparable full-region cost model"),
                inputs=inputs, outputs=outputs, metadata=tuple(sorted(metadata)),
            )
            offers.append(SourceProposal(
                alternative, name, resource, memory_domain, "logical_tensor_v1",
                source.stamp.source_sha256, tuple(source.operation_ordinals[index] for index in nodes),
                source_numerical_identity(source, nodes),
                lowering=_recipe(pattern, proof),
            ))
    source.revalidate()
    return tuple(offers)


def propose(source, *, resource: str, memory_domain: str):
    """Preserve the ordinary unpaged source proposal API."""
    from .backend.source_groups import _contraction_plans

    source.revalidate()
    return _proposals(source, _contraction_plans(source.module), resource=resource, memory_domain=memory_domain)


def describe():
    """Implemented matcher/recipe scope, independently of hardware capability."""
    from merlin.compiler.backend.discovery import SourcePattern

    return (SourcePattern("gemmini.dense-and-readout.v1", ("linalg.generic", "linalg.matmul"),
                          "gemmini.single-dense-direct.v1", 2),)


def discovery_identity():
    return _implementation_identity()


def propose_page(source, context, *, cursor=None):
    """Match a bounded roster using Merlin's existing frontier and bound cursor.

    One actual root matcher session may contain several existing theorem calls.
    The source/tensor bounds constrain that session's scope; this trusted Python
    entrypoint does not impose a wall-time or memory limit on those calls.
    """
    from itertools import islice
    from math import prod

    from merlin.compiler.planning.frontiers import discover
    from merlin.compiler.backend.discovery import ProposalContext, ProposalPage

    from .backend.source_groups import _contraction_plans

    source.revalidate()
    if (type(context) is not ProposalContext or context.provider != name
            or context.source_sha256 != source.stamp.source_sha256
            or context.patterns != describe() or context.implementation_sha256 != discovery_identity()):
        raise ValueError("proposal context differs from source or active provider implementation")
    budget = context.budget
    alternatives_per_anchor = context.patterns[0].alternatives_per_anchor
    proof_sessions_per_anchor = 1 + alternatives_per_anchor
    if (budget.max_candidates < alternatives_per_anchor
            or budget.max_proof_sessions < proof_sessions_per_anchor):
        raise ValueError("insufficient provider page budget: one atomic raw/fused root needs "
                         f"{alternatives_per_anchor} proposal slots and {proof_sessions_per_anchor} proof sessions")
    members = tuple(islice(source.module.walk(), budget.max_source_operations + 1))
    if len(members) > budget.max_source_operations:
        raise ValueError("proposal discovery source operation budget exceeded")
    # Constant decoding and numerical vector work are never admitted for an
    # arbitrarily large tensor, even when the returned page would be tiny.
    oversized = any(prod(shape) > budget.max_tensor_elements for op in members
                    for value in (*op.operands, *op.results)
                    if (shape := getattr(value.type, "get_shape", lambda: ())()))
    # Reserve one provider proof session plus two independent admission sessions
    # per root. A full root's raw/fused alternatives always travel together.
    roots_per_page = min(budget.max_match_visits, budget.max_proof_sessions // proof_sessions_per_anchor,
                         budget.max_candidates // alternatives_per_anchor)
    report = discover(source.module, capability_contract={}, numerical_policy_sha256=context.numerical_policy_sha256,
                      max_candidates=roots_per_page, cursor=cursor,
                      max_region_operations=budget.max_region_operations,
                      provider_roots=tuple(dict.fromkeys(op for pattern in context.patterns
                                                        for op in pattern.root_operations)),
                      discovery_context=context.record())
    offers, reasons, sessions = [], [], 0
    for row in report['candidates']:
        root = members[row['detail']['root_ordinal']]
        if oversized:
            reasons.append({'anchor_ordinal': row['detail']['root_ordinal'], 'status': 'budget_exhausted',
                            'category': 'tensor_element_budget',
                            'reason': 'source tensor exceeds scoped numerical work bound'})
            continue
        sessions += 1
        try:
            plans = _contraction_plans(source.module, roots=(root,))
            found = _proposals(source, plans, resource=context.resource, memory_domain=context.memory_domain)
        except ValueError as exc:
            reasons.append({'anchor_ordinal': row['detail']['root_ordinal'], 'status': 'refused',
                            'category': 'unsupported_implementation', 'reason': str(exc)})
            source.revalidate()
            continue
        if not found:
            reasons.append({'anchor_ordinal': row['detail']['root_ordinal'], 'status': 'no_match',
                            'category': 'no_implemented_source_pattern'})
        for proposal in found:
            if len(proposal.alternative.nodes) > budget.max_region_operations:
                reasons.append({'alternative': proposal.alternative.id, 'status': 'budget_exhausted',
                                'category': 'region_operation_budget'})
            else:
                offers.append(proposal)
        for plan in plans:
            if plan.diagnostic.get('refusals') or plan.diagnostic.get('reason'):
                reasons.append({'anchor_ordinal': row['detail']['root_ordinal'], 'status': 'refused',
                                'category': 'readout_implementation', 'detail': plan.diagnostic})
    source.revalidate()
    if context.implementation_sha256 != discovery_identity() or context.patterns != describe():
        raise ValueError("provider implementation changed during proposal discovery")
    report['provider_reasons'] = reasons
    report['work'] = {'source_operation_scope': len(members), 'matcher_visits': sessions,
                      'provider_proof_sessions': sessions, 'returned_proposals': len(offers),
                      'reserved_shared_admission_sessions': len(offers), 'bounds': context.record()['budget'],
                      'session_scope': ('one root matcher and its existing theorem calls; '
                                        'bounded whole source and tensor sizes'),
                      'worker_time_memory_limits': 'not_enforced_by_this_trusted_in_process_hook'}
    report['scope_incomplete'] = bool(report['next_cursor']) or any(r['status'] == 'budget_exhausted' for r in reasons)
    return ProposalPage(tuple(offers), report)


def validate(request) -> None:
    """Reject deployment promises that this region-only emitter cannot fulfill."""
    if request.source.kind != "mlir":
        raise ValueError("Gemmini region emission requires admitted MLIR source")
    options = request.options
    if options.system is not None or options.platform is not None or options.harts is not None:
        raise ValueError("Gemmini region emission does not build a system/platform image or schedule harts")
    if options.parameters:
        from .callable import selected_tools

        selected_tools(options.parameters)


def compile(request):
    """Return actual LLVM-dialect and command products, with no runtime admission."""
    return _compile(request)


def validate_selected(request, selected):
    """Accept the exact chosen implementation or refuse before creating output."""
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    from .backend.program import _single_device_region
    from .backend.source_groups import _contraction_plans

    validate(request)
    selected.revalidate(request)
    proposal = selected.admitted.proposal
    pattern = dict(proposal.lowering.guards).get("pattern")
    if proposal.provider != name or proposal.alternative.implementation != pattern:
        raise ValueError("selected implementation is not owned by this provider")
    module = parse_mlir_text(request.source.text)
    plans = _contraction_plans(module)
    if len(plans) != 1:
        raise ValueError("selected implementation has no exact dense source match")
    plan = plans[0]
    realized = "dense-requant" if plan.output is not plan.root.results[0] else "dense"
    proof = plan.diagnostic if realized == "dense-requant" else None
    if pattern != realized or proposal.lowering != _recipe(realized, proof):
        raise ValueError("selected implementation/pipeline/guards changed; rediscover before lowering")
    if _single_device_region(request.source.text) is None:
        raise ValueError("selected implementation cannot lower the complete outlined source and ABI")


def compile_selected(request, selected):
    from dataclasses import replace

    validate_selected(request, selected)
    result = _compile(request, selected=selected)
    validate_selected(request, selected)
    return replace(result, metadata={**result.metadata,
                                    "selected_lowering": selected.acknowledgment(request, result)})


def _compile(request, *, selected=None):
    from merlin.compiler.artifact import Artifact, Compilation

    from .api import compile_source

    validate(request)
    buffer = None
    if selected is not None:
        from .backend.program import _single_device_region

        buffer = _single_device_region(request.source.text)
        if buffer is None:
            raise ValueError("selected direct recipe no longer matches; explicit replanning required")
    product = compile_source(request.source.text, output=request.output, selected_buffer=buffer)
    extra, metadata = (), {}
    if request.options.parameters:
        from .callable import materialize, selected_tools

        extra, admission = materialize(product, request.source.text, selected_tools(request.options.parameters))
        metadata = {"scope": "native_region_inputs", "c_call_admission": admission}
    return Compilation(
        directory=product.directory,
        backend="gemmini",
        source_sha256=hashlib.sha256(request.source.text.encode("utf-8")).hexdigest(),
        artifacts=(
            Artifact.from_file(product.artifact, kind="llvm-dialect", role="region"),
            Artifact.from_file(product.commands, kind="json", role="command-plan"),
            Artifact.from_file(product.manifest, kind="json", role="provider-record"),
        ) + extra,
        metadata={"scope": "compiled_region", "execution": "not_run", "qualification": "not_established", **metadata},
    )
