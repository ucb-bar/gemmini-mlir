"""Conservative terminal schedule recording and local configuration normalization.

Install on a FnBuilder before device lowering. The adapter preserves instruction
order, fences and effects. It only removes exact, constant configuration repeats
within one LLVM basic block, with invalidation at unknown hardware effects.
Recorded dependence edges express required ordering, not proved target completion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from xdsl.dialects import llvm
from merlin.sched.ir import Call, Const, Kernel
from merlin.sched.check.config import check_config

_INSN = re.compile(r'^\.insn r (?:123|0x7[bB]),\s*3,\s*(\d+),\s*x0,\s*\$0,\s*\$1$')
_NAMES = {0:'CONFIG',1:'MVIN2',2:'MVIN',3:'MVOUT',4:'COMPUTE_FLIP',
          5:'COMPUTE_STAY',6:'PRELOAD',7:'FLUSH',14:'MVIN3',23:'MVOUT_SPAD'}


@dataclass(frozen=True)
class Axis:
    name: str
    start: int
    stop: int
    step: int


@dataclass(frozen=True)
class Effect:
    resource: str
    access: str
    address: int | str | None = None


@dataclass(frozen=True)
class Event:
    index: int
    block: int
    instruction: str
    axes: tuple[Axis, ...]
    effects: tuple[Effect, ...]
    depends_on: tuple[int, ...]
    # Fences provide explicit synchronization points. These are issue indices
    # covered by the fence, not hardware completion-token names.
    completion_inputs: tuple[int, ...] = ()
    removed: bool = False


def _constant(value):
    owner = getattr(value, 'owner', None)
    if getattr(owner, 'name', None) != 'llvm.mlir.constant':
        return None
    attr = owner.properties.get('value')
    try:
        return attr.value.data & ((1 << 64) - 1)
    except AttributeError:
        return None


def _domain(rs1):
    if rs1 is None:
        return None
    cmd = rs1 & 3
    if cmd == 0:
        return 'ex.strides' if rs1 & 128 else 'ex'
    if cmd == 1:
        state = (rs1 >> 3) & 3
        return f'ld{state}' if state < 3 else None
    if cmd == 2:
        return 'st'
    return None


class TerminalSchedule:
    """Builder adapter; `.events` and `.snapshot()` expose the normalized trace.

    The snapshot is compiler evidence, not a complete execution schedule proof:
    memory effects intentionally cover entire address spaces, and cross-block
    branch conditions and loop-carried completion remain unchecked.
    """
    def __init__(self, builder, *, normalize=True):
        self.builder = builder
        self.normalize = bool(normalize)
        self.events = []
        self.axes = []
        self.removed_configs = 0
        self._original_add = builder.add
        self._original_loop = builder.for_loop
        self._block = None
        self._blocks = {}
        self._live = {}
        self._last_write = {}
        self._readers = {}
        self._pending = []
        self._calls = {}
        self._ssa = {}
        self._axis_count = 0

    def _enter_block(self):
        block = self.builder.blk
        if block is not self._block:
            self._block = block
            self._live.clear()
            self._last_write.clear()
            self._readers.clear()
            # Do not assign branch-local instructions completion meaning outside
            # that block. The LLVM CFG retains the actual cross-block control.
            self._pending.clear()
        if block not in self._blocks:
            self._blocks[block] = len(self._blocks)
        return self._blocks[block]

    def _address(self, value):
        const = _constant(value)
        if const is not None:
            return const
        if value not in self._ssa:
            self._ssa[value] = f'ssa_{len(self._ssa)}'
        return self._ssa[value]

    def _record(self, block, name, effects=(), *, removed=False, fence=False, call=None):
        index = len(self.events)
        dependencies = set()
        for effect in effects:
            prior = self._last_write.get(effect.resource)
            if prior is not None:
                dependencies.add(prior)
            if effect.access != 'read':
                dependencies.update(self._readers.get(effect.resource, ()))
        if any(effect.resource == 'device' for effect in effects):
            dependencies.update(self._pending)
        completion = tuple(self._pending) if fence else ()
        dependencies.update(completion)
        event = Event(index, block, name, tuple(self.axes), tuple(effects),
                      tuple(sorted(dependencies)), completion, removed)
        self.events.append(event)
        if not removed:
            for effect in effects:
                if effect.access == 'read':
                    self._readers.setdefault(effect.resource, set()).add(index)
                else:
                    self._last_write[effect.resource] = index
                    self._readers[effect.resource] = set()
            if fence:
                self._pending.clear()
            else:
                self._pending.append(index)
            if call is not None:
                self._calls.setdefault(block, []).append(call)
        return event

    def add(self, op):
        block = self._enter_block()
        if not isinstance(op, llvm.InlineAsmOp):
            if isinstance(op, llvm.CallOp):
                self._record(block, 'UNKNOWN_CALL', [Effect('device', 'readwrite')])
                self._live.clear()
            return self._original_add(op)
        asm = op.asm_string.data.strip()
        if asm == 'fence' or asm.startswith('fence '):
            self._record(block, 'FENCE', [Effect('device', 'readwrite')], fence=True,
                         call=Call('FENCE', ()))
            self._live.clear()
            return self._original_add(op)
        match = _INSN.fullmatch(asm)
        if not match or len(op.operands) != 2:
            self._record(block, 'UNKNOWN_ASM', [Effect('device', 'readwrite')])
            self._live.clear()
            return self._original_add(op)
        funct = int(match.group(1))
        rs1, rs2 = op.operands
        c1, c2 = _constant(rs1), _constant(rs2)
        name = _NAMES.get(funct, f'UNKNOWN_{funct}')
        if funct == 0:
            domain = _domain(c1)
            signature = (c1, c2) if c1 is not None and c2 is not None else None
            redundant = (self.normalize and domain is not None and signature is not None
                         and self._live.get(domain) == signature)
            effects = [Effect('config.' + (domain or 'unknown'), 'write')]
            sets = ((domain, Const((c1 << 64) | c2)),) if signature is not None and domain else ()
            self._record(block, name, effects, removed=redundant,
                         call=Call(name, (), sets=sets))
            if redundant:
                self.removed_configs += 1
                return op
            if domain in ('ex', 'ex.strides'):
                self._live.pop('ex', None)
                self._live.pop('ex.strides', None)
            if domain is None:
                self._live.clear()
            elif signature is None:
                self._live.pop(domain, None)
            else:
                self._live[domain] = signature
            return self._original_add(op)
        domains = {1:('ld1',),2:('ld0',),14:('ld2',),3:('st',),
                   4:('ex',),5:('ex',),6:('ex',)}.get(funct, ())
        effects = [Effect('config.' + d, 'read') for d in domains]
        if funct in (1, 2, 14):
            space = 'accumulator' if c2 is not None and c2 & (1 << 31) else 'local'
            # Dynamic local addresses may designate either address space.
            effects.extend([Effect('dram', 'read', self._address(rs1)),
                            Effect(space, 'write', self._address(rs2))])
            if c2 is None:
                effects.append(Effect('accumulator', 'write'))
        elif funct in (3, 23):
            effects.extend([Effect('local', 'read', self._address(rs2)),
                            Effect('accumulator', 'read', self._address(rs2)),
                            Effect('dram', 'write', self._address(rs1))])
        elif funct in (4, 5, 6):
            effects.extend([Effect('local', 'readwrite'), Effect('accumulator', 'readwrite')])
        else:
            effects.append(Effect('device', 'readwrite'))
            self._live.clear()
        assumes = tuple((d, Const((self._live[d][0] << 64) | self._live[d][1]))
                        for d in domains if d in self._live)
        self._record(block, name, effects, call=Call(name, (), assumes=assumes))
        return self._original_add(op)

    def for_loop(self, start, stop, step, body, *, retain_loop=False):
        axis = Axis(f'axis_{self._axis_count}', start, stop, step)
        self._axis_count += 1
        def wrapped(iv):
            self.axes.append(axis)
            try:
                body(iv)
            finally:
                self.axes.pop()
        return self._original_loop(start, stop, step, wrapped, retain_loop=retain_loop)

    def kernels(self):
        """Block-local generic schedule views; LLVM CFG/axis trace remains authoritative."""
        return tuple(Kernel(f'terminal_block_{block}', (), tuple(self._calls.get(block, ())))
                     for block in self._blocks.values())

    def config_reports(self):
        """Partial generic config checks; missing/dynamic assumptions remain unmodeled."""
        return tuple(check_config(kernel) for kernel in self.kernels())

    def snapshot(self):
        return {'schema': 'gemmini.terminal_schedule.v1',
                'removed_configs': self.removed_configs,
                'events': [asdict(event) for event in self.events],
                'unchecked': ['cross_block_dependencies', 'loop_carried_completion',
                              'physical_placement', 'dynamic_config_values',
                              'target_completion_semantics']}


def install(builder, *, normalize=True):
    """Install exactly once before lowering; return the reusable schedule adapter."""
    existing = getattr(builder, 'terminal_schedule', None)
    if existing is not None:
        if existing.normalize != bool(normalize):
            raise ValueError('terminal schedule already installed with another mode')
        return existing
    schedule = TerminalSchedule(builder, normalize=normalize)
    builder.add = schedule.add
    builder.for_loop = schedule.for_loop
    builder.terminal_schedule = schedule
    return schedule
