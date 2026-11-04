"""Consume Merlin's exact preparation sharing before device extraction.

The ordinary host pass sees device calls only after extraction. Running the same
checked tensor pass here lets the actual source consumers share preparation
results, including preparations which will themselves become device regions.
This is invocation-local SSA reuse; it grants no persistent weight lifetime.
"""
from __future__ import annotations

import hashlib
import io
from math import prod


def source_text(module):
    """Print the actual transformed module, suitable for a caller's artifact."""
    from xdsl.printer import Printer

    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _identity(text):
    return hashlib.sha256(text.encode()).hexdigest()


def share_preparations(module):
    """Rewrite live source uses with the library's exact same-block CSE.

    Call before discovering regions or collecting operation ordinals. The caller
    retains the original source bytes as its ABI/task-plan authority. The returned
    receipt describes this module mutation, not a target qualification or timing
    claim. Unsupported operations, returned preparation cones, strict floating
    environments and unknown effect/layout permissions retain the library route.
    """
    from merlin.llvmlower.tensor_provenance_cse import deduplicate_tensor_preparation

    before_text = source_text(module)
    before = tuple(module.walk())
    # Keep the public tensor boundary and actual return SSA values unchanged.
    functions = tuple((op, op.function_type) for op in before if op.name == 'func.func')
    returns = tuple((op, tuple(op.operands)) for op in before if op.name == 'func.return')
    snapshots = {
        op: dict(source_index=index, operation=op.name,
                 result_types=[str(value.type) for value in op.results],
                 provenance={key: str(value) for key, value in op.attributes.items()
                             if key.startswith('prov.')})
        for index, op in enumerate(before)
    }
    rewrites = deduplicate_tensor_preparation(module)
    if any(op.function_type != signature for op, signature in functions):
        raise ValueError('exact preparation sharing changed a public function type')
    if any(tuple(op.operands) != values for op, values in returns):
        raise ValueError('exact preparation sharing changed a returned tensor identity')
    after = tuple(module.walk())
    surviving = set(after)
    after_text = source_text(module)
    return dict(
        schema='submission.source_preparation.v1',
        implementation='merlin.llvmlower.tensor_provenance_cse.deduplicate_tensor_preparation',
        source_sha256=_identity(before_text), output_sha256=_identity(after_text),
        source_identity_format='generic printed live module; caller retains original exact source bytes',
        source_operations=len(before), output_operations=len(after), rewrites=rewrites,
        removed=[snapshots[op] for op in before if op not in surviving],
        shared=[dict(source_index=index, operation=op.name,
                     provenance={key: str(value) for key, value in op.attributes.items()
                                 if key.startswith('prov.')})
                for index, op in enumerate(before)
                if op in surviving and 'prov.cse_sources' in op.attributes],
        scope='same sequential block, exact SSA operands and typed preparation body',
        consumption='original source operands replaced before device region extraction',
        public_function_types='unchanged', returned_tensor_identities='unchanged',
        persistent_weight_cache=False, measured_cycles=None)


def _materializations(plans):
    """Static target output footprint, not DMA traffic or elapsed cycles."""
    total = 0
    for plan in plans:
        typ = plan.output.type
        width = typ.element_type.bitwidth
        if width % 8:
            raise ValueError('source preparation admission needs byte-addressed device outputs')
        total += prod(typ.get_shape()) * (width // 8)
    return dict(device_regions=len(plans), device_output_bytes=total)


def select_preparations(module, planner):
    """Apply sharing only when it does not increase either static device cost.

    The library pass can expose a shared intermediate that splits an otherwise
    private composed slice region. Preview the real region planner on an IR clone
    to retain that incumbent route when its materializations would increase.
    The accepted rewrite is then run on the original module; clone operations
    never enter callbacks, operands, provenance or source operation ordinals.
    """
    incumbent = planner(module)
    before = _materializations(incumbent)
    preview = module.clone()
    try:
        report = share_preparations(preview)
        after = _materializations(planner(preview))
    finally:
        preview.drop_all_references()
    admitted = all(after[key] <= before[key] for key in before)
    admission = dict(
        before=before, proposed=after, accepted=admitted,
        rule='nonincreasing device region count and materialized output bytes',
        status='provisional_static_structural_admission', measured_cycles=None)
    if not admitted:
        report.update(applied=False, admission=admission,
                      proposed_output_sha256=report['output_sha256'],
                      output_sha256=report['source_sha256'], consumption='unchanged incumbent source')
        for key in ('rewrites', 'removed', 'shared', 'output_operations'):
            report['proposed_' + key] = report[key]
        report.update(rewrites={}, removed=[], shared=[], output_operations=report['source_operations'])
        return incumbent, report
    actual = share_preparations(module)
    if (actual['source_sha256'], actual['output_sha256'], actual['rewrites']) != (
            report['source_sha256'], report['output_sha256'], report['rewrites']):
        raise ValueError('source preparation preview differs from live rewrite')
    selected = planner(module)
    if _materializations(selected) != after:
        raise ValueError('source preparation preview differs from live device materializations')
    actual.update(applied=bool(actual['rewrites']), admission=admission)
    return selected, actual


def source_composition_request(preferences, *, enabled):
    """Set only the existing native source owner's preparation preference.

    A packaging caller may serialize this mapping to source_composition.json
    *before* submitting its immutable package snapshot. This deliberately does
    not write files during compilation or infer that the native owner applied
    the request. Its owner must rederive legality and provide execution evidence.
    The local nonincreasing-materialization guard is a separate candidate path.
    """
    if type(enabled) is not bool:
        raise ValueError('preparation source preference must be Boolean')
    if not isinstance(preferences, dict) or preferences.get('schema') != 'merlin.source-composition-preferences.v1':
        raise ValueError('preparation request needs the existing source composition schema')
    if any(type(preferences.get(key)) is not bool for key in ('shared_input_siblings', 'prepack_weights')):
        raise ValueError('preparation request needs existing Boolean composition fields')
    if ('share_exact_tensor_preparations' in preferences
            and type(preferences['share_exact_tensor_preparations']) is not bool):
        raise ValueError('existing preparation source preference must be Boolean')
    result = dict(preferences)
    result['share_exact_tensor_preparations'] = enabled
    return result
