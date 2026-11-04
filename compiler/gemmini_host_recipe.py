"""Source-bound transport for explicitly admitted shared host choices.

This module never grants permissions. The ordinary build owner revalidates the
request against its original sealed software profile before realizing it.
"""
from __future__ import annotations

import hashlib


def _sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def attach_host_request(command_buffer, emitted_source, *, software_bytes,
                        entry='kernel_entry', index_bits=None, choices=None):
    """Attach a checked request to the exact terminal artifact, or omit it.

    ``software_bytes`` must be the selected original public software declaration;
    it is diagnostic input here, never a replacement for build-owner admission.
    ``choices`` uses HostOptimizationRequest fields, excluding source_sha256.
    No choices and missing original permission both preserve the command buffer.
    Invalid explicit choices under an admitted policy raise rather than degrade.
    Returns JSON-compatible selection evidence, not realization evidence.
    """
    import yaml
    from merlin.llvmlower.host_optimization_request import (
        HostOptimizationRequest, HostOptimizationSelection,
        context_from_software, prepare_host_optimization,
    )
    if 'host_optimization_request' in command_buffer:
        raise ValueError('host recipe cannot replace an existing source-bound request')
    if not isinstance(software_bytes, bytes):
        raise TypeError('host recipe requires exact original software bytes')
    semantic_sha = hashlib.sha256(software_bytes).hexdigest()
    software = yaml.safe_load(software_bytes)
    context = context_from_software(
        software, semantic_sha256=semantic_sha, target='gemmini', entry=entry,
        index_bits=index_bits, allow_missing=True,
    )
    evidence = dict(schema='gemmini.host_recipe.v1',
                    source_sha256=_sha(emitted_source),
                    semantic_sha256=semantic_sha,
                    realization='not_measured_by_emitter')
    if not context.permitted_stages or not context.permitted_optimizations:
        return dict(evidence, status='omitted', reason='original_host_permissions_absent')
    if not choices:
        return dict(evidence, status='omitted', reason='no_explicit_choices')
    if 'source_sha256' in choices:
        raise ValueError('host recipe binds the actual emitted source itself')
    request = HostOptimizationRequest(source_sha256=_sha(emitted_source), **choices)
    selection = HostOptimizationSelection(context, request)
    prepared = prepare_host_optimization(emitted_source, selection, finished_llvm_only=True)
    if prepared.report()['status'] != 'selected':
        return dict(evidence, status='omitted', reason='no_enabled_choices')
    # The original ordinary build producer owns application and object evidence.
    # Keeping only the source-bound request avoids shipping replacement authority.
    command_buffer['host_optimization_request'] = request.record()
    return dict(evidence, status='requested', selection_sha256=selection.identity(),
                selected_optimizations=prepared.report()['selected_optimizations'])


def selected_math_recipe(policy, symbols):
    """Describe an explicitly selected runtime recipe consumed by the linker.

    None/native retain native symbol names. Portable policies return existing
    shared wrapper symbols plus the exact source/flags that the ordinary object
    producer must compile/link. This function performs no policy selection and
    makes no claim that merely describing a recipe realizes it.
    """
    from merlin.runtime.host_math import host_math_recipe, runtime_symbol_bindings
    policy = 'native' if policy is None else policy
    recipe = host_math_recipe(policy)
    bindings = ({symbol: symbol for symbol in symbols} if policy == 'native'
                else runtime_symbol_bindings(policy, symbols))
    return dict(schema='gemmini.host_math_recipe.v1', policy=policy,
                symbol_bindings=bindings,
                source_sha256=_sha(recipe.source) if recipe.source is not None else None,
                compile_flags=list(recipe.compile_flags), link_flags=list(recipe.link_flags),
                realization='requires_original_linker_build_host_math')
