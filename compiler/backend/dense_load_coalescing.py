"""Checked refusal boundary for unqualified dense wide-load addressing.

The selected declarations expose MVIN packing and dependency footprints. They
do not establish the exact wide byte-to-local-lane mapping required by shared
``propose_tile_load_coalescing`` and ``admit_panel_tile_loads``. Until that
independent law exists this adapter returns the identical counted program.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter
from merlin.sched.ir import Kernel, Loop, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain


MISSING_PREMISES = (
    'selected exact copy: source_byte + row*source_pitch_bytes + column*element_bytes',
    'selected local mapping: row_base + floor(column/DIM)*block_stride + row, lane column%DIM',
    'decoded MVIN row/column limits, independent of register packet slot widths',
    'selected load-channel pitch/block-stride capture and complete numerical/write effects',
)
REFUSAL = 'selected target lacks independently qualified exact wide-load addressing'


def _facts_digest(facts):
    if type(facts) is not dict or not isinstance(facts.get('facts'), dict):
        raise ValueError('dense load coalescing requires the actual RTL facts document')
    # This is document custody only, not a physical qualification or file pin.
    try:
        data = json.dumps(facts, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError('dense load coalescing facts must be finite JSON data') from exc
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class DenseLoadCoalescingRefusal:
    kernel_sha256: str
    facts_document_sha256: str
    adapter_identity: tuple[str, str, int]
    static_nodes: int
    dynamic_lhs_loads: int
    dynamic_rhs_loads: int
    refusal: str = REFUSAL
    missing_premises: tuple[str, ...] = MISSING_PREMISES
    applied: bool = False

    def revalidate(self, kernel, adapter, facts):
        if (type(kernel) is not Kernel or type(adapter) is not DenseLoopAdapter
                or kernel.digest() != self.kernel_sha256
                or _facts_digest(facts) != self.facts_document_sha256
                or (adapter.configuration_identity, adapter.numerical_policy_identity,
                    adapter.index_bits) != self.adapter_identity
                or self.applied is not False or self.refusal != REFUSAL
                or self.missing_premises != MISSING_PREMISES):
            raise ValueError('dense coalescing refusal source, facts or admission boundary changed')

    def as_dict(self):
        return dict(asdict(self), schema='dense_load_coalescing_refusal.v1',
            emission='identical original kernel; no wide loads or completion changes',
            evidence='packing and dependency footprints do not establish exact copy semantics',
            facts_identity='canonical JSON custody digest, not an independent physical qualification')


def select_dense_load_coalescing(kernel, adapter, facts, *, max_static_nodes=4096):
    """Return ``(original_kernel, immutable_refusal)`` without any mutation.

    Call on the actual checked dense Kernel before terminal lowering. This
    performs bounded counted-IR and actual load-codec checks; it neither
    enumerates dynamic commands nor constructs a synthetic address law. The
    original complete source/epoch result must still own actual emission.
    """
    if (type(kernel) is not Kernel or type(adapter) is not DenseLoopAdapter
            or type(max_static_nodes) is not int or max_static_nodes <= 0):
        raise ValueError('dense load coalescing requires typed counted IR, adapter and a finite bound')
    if check_structure(kernel):
        raise ValueError('dense load coalescing requires a well-formed original kernel')
    adapter.__post_init__()
    check_integer_domain(kernel, adapter.index_bits)
    facts_identity = _facts_digest(facts)
    original_identity = kernel.digest()
    nodes = 0
    counts = {'lhs': 0, 'rhs': 0}

    def walk(body, multiplicity):
        nonlocal nodes
        for statement in body:
            nodes += 1
            if nodes > max_static_nodes:
                raise ValueError('dense load coalescing exceeds its static inspection budget')
            if type(statement) is Loop:
                walk(statement.body, multiplicity*statement.extent)
                continue
            if statement.instr == 'merlin_dense_panel_drain':
                continue
            operation = adapter.project(statement)
            if type(operation) is Load:
                role = {'input': 'lhs', 'weight': 'rhs', 'lhs': 'lhs', 'rhs': 'rhs'}.get(operation.role)
                if role is None or adapter.encode(operation) != statement:
                    raise ValueError('dense load coalescing needs a complete actual load-codec round trip')
                counts[role] += multiplicity

    walk(kernel.body, 1)
    proof = DenseLoadCoalescingRefusal(original_identity, facts_identity,
        (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.index_bits),
        nodes, counts['lhs'], counts['rhs'])
    proof.revalidate(kernel, adapter, facts)
    return kernel, proof
