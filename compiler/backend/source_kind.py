"""Select the declared interface grammar or a structurally parsed tensor entry.

Classification does not add provenance, numerical attributes, or device eligibility.
The tensor frontend remains responsible for the semantics of each operation.
"""
from __future__ import annotations

import re


def classify_source(text: str) -> str:
    # The fixed interface grammar has custom syntax outside the tensor parser.
    # Inspect its actual module declaration, excluding comments and strings from
    # the separate mixed-function check. Do not infer a grammar from filenames.
    uncommented = re.sub(r'"(?:[^"\\]|\\.)*"|//[^\n]*|/\*.*?\*/',
                         lambda m: m[0] if m[0].startswith('"') else '', text,
                         flags=re.S)
    tokens = re.sub(r'"(?:[^"\\]|\\.)*"', '""', uncommented)
    declaration = re.match(r'\s*(?:builtin\.)?module\s+attributes\s*\{([^}]*)\}', tokens)
    if declaration and re.search(r'\bmerlin_iface\.(?:version|abi_version|target)\s*=', declaration[1]):
        if re.search(r'\bfunc\.func\b', tokens) or re.search(r'"func\.func"\s*\(', uncommented):
            raise ValueError('ambiguous source mixes declared merlin_iface grammar and func.func')
        return 'merlin_iface'

    # This is the same selected parser and unique-entry boundary used by the
    # source plan. A provenance marker is unnecessary and grants no semantics.
    from merlin.targetgen.oot_starterkit.plan import _parsed_source
    from merlin.targetgen.contract import linalg_iface as li
    from xdsl.dialects.builtin import TensorType

    try:
        _, block, operations, returned = _parsed_source(text.encode('utf-8'))
    except Exception as exc:
        raise ValueError(f'unsupported or malformed tensor source: {exc}') from exc
    function = block.parent_op()
    if list(function.parent_op().body.block.ops) != [function]:
        raise ValueError('tensor source supports only its unique entry at module scope')
    boundary = [*block.args, *returned.operands]
    if not boundary or any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError('tensor source requires tensor-typed entry arguments and results')
    for value in boundary:
        if any(dim < 0 for dim in li._shape(value.type)):
            raise ValueError('tensor source requires static entry shapes')
    if any(li._op_name(op).startswith('merlin_iface.') for root in operations for op in root.walk()):
        raise ValueError('ambiguous source mixes tensor entry and merlin_iface operations')
    return 'linalg-on-tensors'
