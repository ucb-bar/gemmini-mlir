"""Stated semantic refusals for operations outside selected hardware routes.

A refusal is a successful compiler response containing ``declined``, as defined
by the public OOT package generator. Compiler errors remain exceptions; this
module does not turn arbitrary lowering failures into hardware claims.
"""
from __future__ import annotations

# Selected target_contract.yaml runner.chipyard_rocc.readout declarations.
# These describe named source stages, not a claim that float scaling implements
# the integer shift/rounding operation named ``requant``.
_READOUT_STAGES = {'i8': frozenset({'acc_scale', 'relu', 'maxpool'}),
                   'i32': frozenset()}
_CONTRACTION_STAGES = frozenset({'bias_add', 'bias'})


def classify_refusal(program: dict) -> dict | None:
    """Return a hardware-evidenced decline, or leave ordinary lowering intact.

    Only the named integer requant stage has an established unsupported
    semantic contract here. Other unimplemented stages still raise their real
    compiler errors rather than being mislabeled unsupported by hardware.
    """
    if program.get('target') != 'gemmini':
        return None
    for command in program.get('commands', []):
        if command.get('opcode') not in {'COMMIT', 'CONV2D'}:
            continue
        attrs = command.get('attributes') or {}
        stages = attrs.get('epilogue') or []
        dtype = attrs.get('output_dtype', 'i32')
        if not isinstance(stages, (list, tuple)) or dtype not in _READOUT_STAGES:
            continue
        for stage in stages:
            if stage != 'requant':
                continue
            admitted = _READOUT_STAGES[dtype] | _CONTRACTION_STAGES
            if stage in admitted:
                continue
            return {
                'op': command['opcode'],
                'stage': stage,
                'reason': (
                    f"selected {dtype} readout and contraction-scoped routes do not "
                    "apply the source integer 'requant' stage; float accumulator "
                    "scaling does not establish its integer shift/rounding semantics"
                ),
                'evidence': (
                    'target_contract.yaml runner.chipyard_rocc.readout: '
                    "i8 epilogue_capability applies acc_scale/relu/maxpool; "
                    'i32 applies no readout stages; stage_routes admit bias_add/bias '
                    'only at accumulator_seed'
                ),
            }
    return None


def declined_buffer(program: dict, refusal: dict) -> dict:
    """A schema-valid stated decline containing no executable command stream."""
    return {
        'abi_version': program.get('abi_version', '0.1'),
        'target': program['target'],
        'backend': 'authored-explicit-gemmini',
        'tensors': {},
        'commands': [],
        'params': {},
        'resources': {},
        'metrics_requested': [],
        'declined': refusal,
    }
