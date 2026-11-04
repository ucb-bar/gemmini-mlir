"""Exact admission and bounded proof for signed half-up requantization.

The selected target declares `acc_scale`, `relu` and `maxpool` i8 stages, but
not `requant`. A numerical coincidence cannot override that stage admission.
For a future source-qualified route this module compares complete monotone
output transitions, not sampled inputs, before mapping half-up to RNE scaling.
No tensor arithmetic emitted by this module executes on the CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from merlin.llvmlower.requantization import quantized


@dataclass(frozen=True)
class RequantDecision:
    status: str
    reason: str
    shift: int
    accumulator_min: int
    accumulator_max: int
    scale: float | None = None
    witness: dict | None = None
    adapter_attrs: dict | None = None


def _source(value, shift, relu):
    # This is the public Tensor.requant semantic: add the positive half unit,
    # then signed arithmetic shift. Compile-time numeric proof only.
    out = value if shift == 0 else (value + (1 << (shift-1))) >> shift
    return max(0 if relu else -128, min(127, out))


def prove_half_up_readout(shift, lo, hi, *, relu=False):
    """Prove source half-up equals hardware f32 RNE for every integer in range.

    Returns a complete proof record, including a concrete witness on failure.
    Saturation makes each function a monotone step function with <=255 steps;
    equality of every transition and the initial value proves the whole domain.
    """
    if type(shift) is not int or not 0 <= shift <= 31:
        raise ValueError('source requant shift must be an integer in [0,31]')
    if type(lo) is not int or type(hi) is not int or not -(1 << 31) <= lo <= hi < (1 << 31):
        raise ValueError('requant proof needs an explicit signed-i32 accumulator interval')
    scale = 2.0 ** -shift
    source = lambda x: _source(x, shift, relu)
    target = lambda x: quantized(x, (scale,), relu)
    def threshold(fn, output):
        a, b = lo, hi+1
        while a < b:
            mid = (a+b)//2
            if fn(mid) >= output:
                b = mid
            else:
                a = mid+1
        return a
    witness = None
    if source(lo) != target(lo):
        witness = lo
    if witness is None:
        for output in range((0 if relu else -128)+1, 128):
            a, b = threshold(source, output), threshold(target, output)
            if a != b:
                witness = min(a,b)
                break
    return {'exact': witness is None, 'shift': shift, 'scale': scale,
            'accumulator_min': lo, 'accumulator_max': hi, 'relu': bool(relu),
            'proof': 'complete signed-i8 monotone output transition comparison',
            'source_rounding': 'half_up', 'target_rounding': 'half_even',
            'witness': None if witness is None else {
                'accumulator': witness, 'source': source(witness), 'hardware': target(witness)}}


def selected_i8_stages():
    """Read only the public selected source stage declaration."""
    import yaml
    package = Path(__file__).resolve().parent
    packaged = package/'data'/'target_contract.yaml'
    path = packaged if packaged.is_file() else package.parent/'target_contract.yaml'
    contract = yaml.safe_load(path.read_text())
    routes = contract['runner']['chipyard_rocc']['readout']['epilogue_capability']
    return frozenset(stage for route in routes if route['selector'] == 'i8'
                     for stage in route.get('applies', ()))


def decide_requant(attrs, *, lo, hi, declared_stages=None):
    """Return a concrete refusal, implementation gap or exact readout adapter.

    `declared_stages` may only come from the caller's selected source contract;
    omitting it reads this launch's public target contract. A successful numeric
    proof does not grant hardware admission by itself. Other scale stages must
    retain their original order and are never silently replaced here.
    """
    attrs = dict(attrs)
    shift = attrs.get('requant_shift', attrs.get('shift', 0))
    if type(shift) is not int:
        raise ValueError('requantization needs a static original integer shift')
    relu = bool(attrs.get('relu', False) or attrs.get('activation') in (1,'relu','ReLU')
                or 'relu' in attrs.get('epilogue', ()))
    proof = prove_half_up_readout(shift, lo, hi, relu=relu)
    stages = selected_i8_stages() if declared_stages is None else frozenset(declared_stages)
    common = dict(shift=shift, accumulator_min=lo, accumulator_max=hi,
                  scale=proof['scale'], witness=proof['witness'])
    if 'requant' not in stages:
        return RequantDecision('declared_refusal',
            "selected i8 stage routes do not declare requant; applies="+','.join(sorted(stages)), **common)
    scales = [attrs[key] for key in ('scale','acc_scale','output_scale','acc_scale_by_column') if key in attrs]
    if any(not isinstance(s,(int,float)) or float(s) != 1.0 for s in scales):
        return RequantDecision('implementation_gap',
            'ordered source scaling before/after half-up requant needs a separate complete transition proof', **common)
    if not proof['exact']:
        return RequantDecision('implementation_gap',
            'source half-up differs from hardware half-even at the reported accumulator; retain the original stage', **common)
    mapped = {k:v for k,v in attrs.items() if k not in ('shift','requant_shift','scale','acc_scale','output_scale','acc_scale_by_column')}
    mapped['epilogue'] = [stage for stage in attrs.get('epilogue', ()) if stage != 'requant']
    mapped['acc_scale'] = proof['scale']
    return RequantDecision('proved_acc_scale',
        'declared route and complete bounded integer-output transition proof permit hardware scaling',
        adapter_attrs=mapped, **common)


def source_admission(commands):
    """Return a stated source-contract refusal reason, otherwise None.

    Accept a command-buffer mapping, a command list, or frontend operation
    descriptions. This screens only the selected unsupported `requant` stage;
    unimplemented admitted stages remain compiler work and are never refused.
    Numeric attrs alone do not create a stage absent from the source program.
    """
    if isinstance(commands, dict):
        commands = commands.get('commands', commands.get('operations', ()))
    stages = selected_i8_stages()
    if 'requant' in stages:
        return None
    for index, command in enumerate(commands):
        opcode = str(command.get('opcode', command.get('op', ''))).upper()
        attrs = command.get('attributes', command.get('attrs', {})) or {}
        epilogue = attrs.get('epilogue', ())
        if isinstance(epilogue, str):
            epilogue = (epilogue,)
        if opcode in ('REQUANT', 'REQUANTIZE') or 'requant' in epilogue:
            return (f"command {index} {opcode}: selected source contract does not declare "
                    "a readout or contraction-scoped route applying stage 'requant'; "
                    "i8 stages apply " + ','.join(sorted(stages)))
    return None
