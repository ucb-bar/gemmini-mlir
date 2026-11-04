"""Source-only realization of an explicit immutable RHS preparation request.

Text-only compilation cannot pre-apply stored weight bytes: the library's offline
prepacker requires a bound immutable manifest, payload and regenerated caller.
This adapter instead shares proven identical immutable RHS layouts within the
original invocation, before its host/device partition. It retains the complete
logical pointer ABI and provides original operation identities to attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from io import StringIO
from pathlib import Path

from xdsl.ir import Operation
from xdsl.printer import Printer
from model_regions import integer_contractions
from source_preparation import prepare_source_layouts as _prepare_layouts, _owner
from merlin.llvmlower import tensor_preparation_identity
from merlin.transforms import tensor_views


@dataclass(frozen=True)
class HostPreparation:
    module: object
    witnesses: tuple[dict, ...]
    source_ordinals: tuple[int, ...]
    aliases: tuple[dict, ...]
    report: dict


def requested(*, root=None):
    """Read only the explicit source-composition preference; absence is off."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    path = root / 'source_composition.json'
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError('source_composition.json must be a direct regular file of at most 8 KB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate source composition option: ' + key)
            result[key] = value
        return result
    request = json.loads(path.read_text(), object_pairs_hook=unique)
    if (not isinstance(request, dict)
            or request.get('schema') != 'merlin.source-composition-preferences.v1'
            or type(request.get('prepack_weights')) is not bool):
        raise ValueError('source composition requires its explicit Boolean prepack_weights preference')
    return request['prepack_weights']


def _text(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def prepare_host_preparations(module, *, prepack_weights=None, entry='forward', protected_ordinals=(),
                              share_exact_tensor_preparations=False):
    """Return an original-indexed source product, suitable before region cutting.

    All fields consumed by LayoutPreparation clients are provided unchanged.
    Caller-requested protected writers and every original returned writer stay
    present. Only layout operations in actual integer-contraction RHS cones may
    be removed, unless general preparation sharing is separately requested.
    Arithmetic remains separate source work.
    No weight contents, mutable-memory aliases or cross-invocation residency are
    inferred from the RHS role.
    """
    enabled = requested() if prepack_weights is None else prepack_weights
    if type(enabled) is not bool:
        raise ValueError('prepack_weights must be an explicit Boolean')
    if type(share_exact_tensor_preparations) is not bool:
        raise ValueError('share_exact_tensor_preparations must be an explicit Boolean')
    module.verify()
    owner = _owner(module, entry)
    original_ops = tuple(owner.body.block.ops)
    original = _text(module)
    digest = hashlib.sha256(original.encode()).hexdigest()
    report = {'schema': 'merlin.source_rhs_preparation.v1', 'requested': enabled,
              'source_sha256': digest, 'output_sha256': digest,
              'realization': 'off', 'removed_source_operations': 0,
              'offline_prepack': 'not_selected', 'scope': 'one original tensor SSA invocation'}
    if not enabled:
        return HostPreparation(module, (), tuple(range(len(original_ops))), (), report)
    if entry != 'forward':
        raise ValueError('RHS preparation currently requires the original forward source entry')
    contractions = integer_contractions(module)
    roots = {record.operation for record in contractions}
    cone, pending = set(), [record.operation.inputs[1] for record in contractions]
    while pending:
        value = pending.pop()
        producer = value.owner
        if (not isinstance(producer, Operation) or producer.parent is not owner.body.block
                or producer in roots or producer in cone):
            continue
        cone.add(producer)
        pending.extend(producer.operands)
    protected = set(protected_ordinals)
    if any(type(index) is not int or not 0 <= index < len(original_ops) for index in protected):
        raise ValueError('protected ordinals must index original forward operations')
    if not share_exact_tensor_preparations:
        protected.update(index for index, op in enumerate(original_ops) if op not in cone)
    result = _prepare_layouts(module, entry=entry, protected_ordinals=protected)
    returned = tuple(next(op for op in owner.body.block.ops if op.name == 'func.return').operands)
    prepared_owner = _owner(result.module, entry)
    prepared_return = next(op for op in prepared_owner.body.block.ops if op.name == 'func.return')
    if (owner.function_type != prepared_owner.function_type
            or tuple(value.type for value in returned) != tuple(value.type for value in prepared_return.operands)):
        raise ValueError('source RHS preparation changed original input/output ABI')
    returned_writers = {original_ops.index(value.owner) for value in returned if value.owner in original_ops}
    if returned_writers - set(result.source_ordinals):
        raise ValueError('source RHS preparation erased an original returned writer')
    if _text(module) != original:
        raise ValueError('source RHS preparation mutated original source')
    report.update({
        'realization': ('combined_layout_reuse' if share_exact_tensor_preparations
                        else 'immutable_rhs_layout_reuse') if result.aliases else 'unchanged',
        'removed_source_operations': len(result.aliases),
        'output_sha256': hashlib.sha256(_text(result.module).encode()).hexdigest(),
        'shared_general_preparations': share_exact_tensor_preparations,
        'removed_rhs_operations': sum(original_ops[alias['source_ordinal']] in cone for alias in result.aliases),
        'original_pointer_abi': str(owner.function_type),
        'original_returned_writer_ordinals': sorted(returned_writers),
        'rhs_contraction_ordinals': [record.source_ordinal for record in contractions],
        'rhs_preparation_ordinals': sorted(original_ops.index(op) for op in cone),
        'offline_prepack': 'refused_missing_immutable_weight_manifest_payload_and_regenerated_caller',
        'refusal': None if result.aliases else 'no_duplicate_admitted_rhs_layout_preparation',
        'mechanisms': ['source_preparation.prepare_source_layouts',
                       'merlin.transforms.tensor_views.canonicalize_tensor_views',
                       'merlin.llvmlower.tensor_preparation_identity.canonical_tensor_read_view'],
        'library_sha256': {Path(component.__file__).name:
                          hashlib.sha256(Path(component.__file__).read_bytes()).hexdigest()
                          for component in (tensor_views, tensor_preparation_identity)},
    })
    return HostPreparation(result.module, result.witnesses, result.source_ordinals, result.aliases, report)
