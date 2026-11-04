"""Read-only terminal command cleanup proposals, with explicit missing authority.

The shared cleanup verifier proves only a typed effect-stream property. Selected
Gemmini dispatch facts do not establish that CONFIG is a persistent pure write,
or that every consumer captures that state before a redundant command disappears.
Consequently this module never removes actual instructions. No caller-supplied
Boolean, source CSE result, or shared proposal receipt grants execution authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from merlin.compile.scheduling.effect_cleanup import Effect, cleanup_indices, verify_cleanup

_PACKAGE = Path(__file__).resolve().parent
_ASM = re.compile(r'\.insn r 0x7b, 3, ([0-9]+), x0, \$0, \$1')
_MISSING = (
    'Selected CONFIG_EX/LD/ST persistent-write idempotence: no enqueue, pulse, '
    'acknowledgement, status, memory or completion effect beyond the decoded state.',
    'Selected consumer configuration capture and clobber law, including per-stream '
    'load state and ordered pending consumers; identical register values alone are insufficient.',
)


def _constant(value):
    owner = value.owner
    literal = getattr(owner, 'properties', {}).get('value')
    if (getattr(owner, 'name', None) != 'llvm.mlir.constant'
            or not isinstance(literal, IntegerAttr)
            or not isinstance(value.type, IntegerType) or value.type.width.data != 64):
        return None
    return literal.value.data & ((1 << 64) - 1)


def _proposal_effect(op):
    # These are projections for proposal inspection only, never target laws.
    if type(op) is llvm.InlineAsmOp:
        asm = op.asm_string.data.strip()
        if asm == 'fence' and not op.operands:
            return Effect('barrier')
        match = _ASM.fullmatch(asm)
        if match and len(op.operands) == 2 and op.constraints.data == 'r,r,~{memory}':
            funct = int(match[1])
            if funct == 0:
                a, b = map(_constant, op.operands)
                if a is None or b is None:
                    return Effect('unknown')
                selector = a & 3  # Selected ConfigCmdType (rs1 bits 0:2).
                if selector in (0, 1, 2):
                    # Retain ALL packed bits. Never compare a partially decoded
                    # command or infer equality from SSA names / source attrs.
                    stream = (a >> 3) & 3 if selector == 1 else 0
                    return Effect('configuration', f'config:{selector}:{stream}', (a, b))
            if funct in (1, 2, 3, 4, 5, 6, 14):
                return Effect('activity')
        # FLUSH, custom assembly and unknown CONFIG selectors reset knowledge.
        return Effect('unknown')
    # A closed LLVM pure-operation vocabulary; all CPU memory/control/unknown
    # operations reset configuration knowledge. No walk across block boundaries.
    if op.name in ('llvm.mlir.constant', 'llvm.add', 'llvm.sub', 'llvm.mul',
                   'llvm.and', 'llvm.or', 'llvm.xor', 'llvm.shl', 'llvm.lshr',
                   'llvm.ashr', 'llvm.icmp', 'llvm.select', 'llvm.zext',
                   'llvm.sext', 'llvm.trunc', 'llvm.ptrtoint', 'llvm.inttoptr',
                   'llvm.getelementptr', 'llvm.mlir.addressof'):
        if not op.regions and not op.successors:
            return Effect('pure')
    return Effect('unknown')


def audit_command_cleanup(module, *, maximum=65536):
    """Inspect original final-LLVM blocks; return checked, unexecuted proposals."""
    reports = []
    for parent in module.walk():
        for region in parent.regions:
            for block in region.blocks:
                ops = tuple(block.ops)
                if not any(type(op) is llvm.InlineAsmOp for op in ops):
                    continue
                effects = list(map(_proposal_effect, ops))
                fences = [i for i, effect in enumerate(effects) if effect.kind == 'barrier']
                # Preserve first/last completion in EVERY block, stronger than
                # preserving function entry/exit; completion ABI is untouched.
                for i in {fences[0], fences[-1]} if fences else ():
                    effects[i] = Effect('barrier', protected=True)
                effects = tuple(effects)
                kept = cleanup_indices(effects, maximum=maximum)
                checked = verify_cleanup(effects, kept, maximum=maximum)
                removed = [dict(index=i, kind=e.kind, state=e.state, value=e.value)
                           for i, e in enumerate(effects) if i not in set(kept)]
                reports.append(dict(block=len(reports), **checked, proposed_omissions=removed,
                                    effect_sha256=hashlib.sha256(repr(effects).encode()).hexdigest()))
    selected = {}
    for filename in ('rtl_facts.json', 'COMMAND_ORDERING_SELECTION.json', 'target_contract.yaml'):
        selected[filename] = hashlib.sha256((_PACKAGE/'data'/filename).read_bytes()).hexdigest()
    return dict(schema='gemmini.terminal_cleanup_audit.v1', blocks=reports,
                proposed_omissions=sum(len(b['proposed_omissions']) for b in reports),
                selected_inputs=selected, actual_removed_operations=0,
                execution_qualification='NOT_ESTABLISHED', missing_target_laws=list(_MISSING),
                status='REFUSED_MISSING_TARGET_LAW')


def cleanup_commands(module, *, enabled=False, maximum=65536):
    """Default-off no-op. Enabling refuses instead of executing unqualified proposals."""
    if type(enabled) is not bool:
        raise ValueError('cleanup enablement must be Boolean')
    report = audit_command_cleanup(module, maximum=maximum)
    if enabled:
        raise ValueError('command cleanup refused: ' + ' '.join(_MISSING))
    return report
