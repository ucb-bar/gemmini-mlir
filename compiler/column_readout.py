"""Exact post-producer column readout through Gemmini accumulator DMA.

This opt-in route consumes a completed, contiguous, raw i32 matrix and writes
all original i8 columns. The counted schedule, source numerical proofs, byte
mapping and conservative accumulator ownership are explicit. It makes no cycle
or target-execution claim and is not enabled by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct

from isa import ACC, DIM
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.llvmlower.integer_readout import derive, evaluate
from merlin.llvmlower.requantization import f32, quantized
from merlin.sched.ir import (Call, Const, Kernel, Loop, TensorArg, Var,
                             check_structure, concretize, instances, select)


def _bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _geometry():
    facts = json.loads(Path(__file__).with_name('facts.json').read_text())['facts']
    array = facts['arrays'][0]
    spaces = {space['name']: space for space in facts['memory_address_spaces']}
    acc = spaces['accumulator']
    if array['rows'] != DIM or array['cols'] != DIM or acc['lanes'] != DIM or acc['lane_bits'] != 32:
        raise ValueError('column readout requires selected signed-i32 sixteen-lane accumulator geometry')
    return array['rows'], acc['row_bytes'], acc['bytes']


def _activation(value):
    if isinstance(value, str):
        value = () if value in ('none', '') else (value,)
    if type(value) is bool:
        value = ('relu',) if value else ()
    if not isinstance(value, (tuple, list)) or tuple(value) not in ((), ('relu',)):
        raise ValueError('column readout activation must be empty or one relu')
    return bool(value)


@lru_cache(maxsize=4096)
def _numerical_proof(scale_bits, relu):
    scale = struct.unpack('<f', struct.pack('<I', scale_bits))[0]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('exact column readout requires finite positive f32 scales')
    # Positive scaling commutes with the target's accumulator-domain ReLU.
    # The source threshold partition is complete over signed i32; each hardware
    # result is monotone, so matching every transition and predecessor proves
    # all original values, including float-conversion and saturation boundaries.
    proof = derive((scale,), relu=relu)
    points = {-(1 << 31), (1 << 31) - 1}
    for threshold in proof['thresholds']:
        points.update(x for x in (threshold - 1, threshold) if -(1 << 31) <= x < 1 << 31)
    for value in points:
        hardware = quantized(max(value, 0) if relu else value, (scale,), False)
        if hardware != evaluate(value, proof):
            raise ValueError('column readout target/source ordering differs')
    return dict(scale_bits=scale_bits, exact=True, domain=[-(1 << 31), (1 << 31) - 1],
                checked_partition_endpoints=len(points),
                rounding='f32 convert, f32 multiply, half-even integer narrowing',
                activation_equivalence='positive scale commutes with accumulator ReLU',
                source_proof_sha256=hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest())


def _kernel(m, n, rows):
    blocks = (m + rows - 1) // rows
    row = Var('row_block') * rows
    column = Var('column')
    height = select('row_block', blocks - 1, (m - 1) % rows + 1, rows)
    logical = row * n + column
    return Kernel('gemmini_column_readout', (
        TensorArg('raw', (m, n), 'i32', 'read'),
        TensorArg('output', (m, n), 'i8', 'write'),
    ), (Loop('column', n, (
        Call('configure_scale', (('index', column),)),
        Loop('row_block', blocks, (
            Call('reload_i32', (('byte_offset', logical * 4), ('rows', height),
                                ('columns', Const(1)), ('acc_row', Const(0)))),
            Call('complete', ()),
            Call('store_i8', (('byte_offset', logical), ('rows', height),
                              ('columns', Const(1)), ('acc_row', Const(0)))),
            Call('complete', ()),
        )),
    )),))


@lru_cache(maxsize=64)
def _readout_groups(scale_bits, group_runs):
    """Partition source columns, merging only bit-identical adjacent scales."""
    if not group_runs:
        return tuple((column, 1) for column in range(len(scale_bits)))
    result = []
    start = 0
    while start < len(scale_bits):
        stop = start + 1
        while stop < len(scale_bits) and scale_bits[stop] == scale_bits[start]:
            stop += 1
        for column in range(start, stop, DIM):
            result.append((column, min(DIM, stop - column)))
        start = stop
    return tuple(result)


def _planned_kernel(m, n, rows, scale_bits, group_runs):
    groups = _readout_groups(scale_bits, group_runs)
    # This retains the incumbent schedule byte-for-byte when no two adjacent
    # source columns can share a readout, including all-distinct vectors.
    if not group_runs or all(width == 1 for _, width in groups):
        return _kernel(m, n, rows)
    blocks = (m + rows - 1) // rows
    row = Var('row_block') * rows
    height = select('row_block', blocks - 1, (m - 1) % rows + 1, rows)
    body = []
    # Consecutive groups of the same width have an affine source-column map.
    # Keep them counted instead of expanding one instruction site per column;
    # irregular run tails start a new compact segment, with no runtime tables.
    first = 0
    while first < len(groups):
        width = groups[first][1]
        end = first + 1
        while end < len(groups) and groups[end][1] == width:
            end += 1
        group = Var('readout_group_' + str(first))
        index = group + first
        column = group * width + groups[first][0]
        logical = row * n + column
        body.append(Loop(group.name, end - first, (
            Call('configure_scale', (('index', index),)),
            Loop('row_block', blocks, (
                Call('reload_i32', (('byte_offset', logical * 4), ('rows', height),
                                    ('columns', Const(width)), ('acc_row', Const(0)))),
                Call('complete', ()),
                Call('store_i8', (('byte_offset', logical), ('rows', height),
                                  ('columns', Const(width)), ('acc_row', Const(0)))),
                Call('complete', ()),
            )),
        )))
        first = end
    return Kernel('gemmini_grouped_column_readout', (
        TensorArg('raw', (m, n), 'i32', 'read'),
        TensorArg('output', (m, n), 'i8', 'write'),
    ), tuple(body))


@dataclass(frozen=True)
class ColumnReadoutPlan:
    m: int
    n: int
    scale_bits: tuple[int, ...]
    relu: bool
    rows_per_reload: int
    accumulator: AddressWindow
    kernel: Kernel
    checked_kernel_sha256: str
    group_runs: bool = False

    def revalidate(self):
        if type(self.m) is not int or type(self.n) is not int or min(self.m, self.n) <= 0:
            raise ValueError('column readout requires positive static matrix extents')
        if self.m * self.n * 4 >= 1 << 63 or self.n * 4 >= 1 << 32:
            raise ValueError('column readout offsets/strides exceed selected command and i64 fields')
        if type(self.scale_bits) is not tuple or len(self.scale_bits) != self.n or type(self.relu) is not bool:
            raise ValueError('column readout requires one literal f32 scale per original column')
        edge, row_bytes, capacity = _geometry()
        self.accumulator.__post_init__()
        if self.rows_per_reload != edge or self.accumulator != AddressWindow('accumulator', 0, edge * row_bytes):
            raise ValueError('column readout requires its exact private complete-row accumulator window')
        if self.accumulator.start + self.accumulator.size > capacity:
            raise ValueError('column readout accumulator ownership exceeds selected capacity')
        errors = check_structure(self.kernel)
        if errors:
            raise ValueError('invalid column readout kernel: ' + '; '.join(errors))
        # Exact canonical symbolic byte/effect form proves the complete strided
        # partition: (row_block*edge + lane)*N + column, each once, bounds M*N.
        if type(self.group_runs) is not bool:
            raise ValueError('column readout grouping must be an explicit bool')
        groups = _readout_groups(self.scale_bits, self.group_runs)
        covered = 0
        for first, width in groups:
            if first != covered or not 1 <= width <= edge:
                raise ValueError('column groups do not cover source columns in order')
            if any(bits != self.scale_bits[first] for bits in self.scale_bits[first:first+width]):
                raise ValueError('column group crosses a source f32 scale boundary')
            covered += width
        if covered != self.n:
            raise ValueError('column groups dropped original output columns')
        expected = _planned_kernel(self.m, self.n, edge, self.scale_bits, self.group_runs)
        if self.kernel != expected or self.kernel.digest() != self.checked_kernel_sha256:
            raise ValueError('column readout byte mapping/effects differ from the checked plan')
        for bits in set(self.scale_bits):
            if type(bits) is not int or not 0 <= bits < 1 << 32:
                raise ValueError('column readout scale must be literal uint32 bits')
            _numerical_proof(bits, self.relu)
        return self

    @property
    def identity(self):
        payload = dict(m=self.m, n=self.n, bits=self.scale_bits, relu=self.relu,
                       kernel=self.checked_kernel_sha256, accumulator_bytes=self.accumulator.size,
                       group_runs=self.group_runs)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def report(self):
        self.revalidate()
        blocks = (self.m + self.rows_per_reload - 1) // self.rows_per_reload
        groups = _readout_groups(self.scale_bits, self.group_runs)
        return dict(schema='authored.gemmini_column_readout.v1', identity=self.identity,
                    kernel_sha256=self.checked_kernel_sha256, shape=[self.m, self.n],
                    load_bytes=self.m*self.n*4, store_bytes=self.m*self.n,
                    reloads=len(groups)*blocks, readouts=len(groups)*blocks,
                    dynamic_scale_configurations=len(groups), accumulator_bytes=self.accumulator.size,
                    run_grouping=self.group_runs, source_column_groups=[list(group) for group in groups],
                    baseline_kernel_sha256=_kernel(self.m,self.n,self.rows_per_reload).digest(),
                    baseline_reload_count=self.n*blocks,
                    grouped_scale_equality='exact uint32 f32 bit identity across every contiguous group',
                    source_columns_preserved=True, source_i32_unmodified=True,
                    scale_transport='constant_i32_dispatch literal command fields; no table loads',
                    completion='full fence after each reload and each readout before local reuse',
                    numerical_proofs=[_numerical_proof(bits,self.relu) for bits in sorted(set(self.scale_bits))],
                    target_execution='not_established', performance='not_measured')

    def emit(self, builder, raw_ptr, out_ptr):
        return emit_plan(builder, self, raw_ptr, out_ptr)


def plan_column_readout(m, n, scales, activation=(), *, group_runs=False):
    """Plan original per-column i32->i8 readout; no pooling or scale folding."""
    if type(m) is not int or type(n) is not int or min(m, n) <= 0:
        raise ValueError('column readout requires positive static matrix extents')
    scales = tuple(scales)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scales):
        raise ValueError('column scales must be declared numeric constants')
    if len(scales) != n:
        raise ValueError('one source scale is required per original output column')
    bits = tuple(_bits(f32(value)) for value in scales)
    edge, row_bytes, _ = _geometry()
    if type(group_runs) is not bool:
        raise ValueError('column readout grouping must be an explicit bool')
    kernel = _planned_kernel(m, n, edge, bits, group_runs)
    return ColumnReadoutPlan(m, n, bits, _activation(activation), edge,
        AddressWindow('accumulator', 0, edge*row_bytes), kernel, kernel.digest(), group_runs).revalidate()


class _IndexBridge:
    def __init__(self, builder): self.builder = builder
    def op(self, opcode, lhs, rhs): return getattr(self.builder, opcode)(lhs, rhs)
    def cmp(self, predicate, lhs, rhs): return self.builder.icmp(predicate, lhs, rhs)
    def select(self, condition, lhs, rhs): return self.builder.select(condition, lhs, rhs)
    def loop(self, start, stop, step, callback):
        with self.builder.counted_loop(start, stop, step) as index:
            callback(index)


def emit_plan(builder, readout_plan, raw_ptr, out_ptr):
    """Emit a checked plan; caller supplies disjoint contiguous logical buffers.

    A full entry drain transfers the private accumulator window from the raw
    producer. Reload completion precedes readout; store completion precedes any
    overwrite or scale change. No asynchronous ordering law is assumed.
    """
    if type(readout_plan) is not ColumnReadoutPlan:
        raise ValueError('typed column readout plan required')
    p = readout_plan.revalidate()
    if raw_ptr == out_ptr:
        raise ValueError('column readout source and destination must be disjoint')
    from merlin.compile.tuning_knobs import knob
    mode = knob('column_readout_epochs', default='off', values=('off', 'disjoint_rows'),
                root=Path(__file__).resolve().parent)
    emitted_kernel, epoch_report = p.kernel, None
    if mode == 'disjoint_rows' and p.m > DIM:
        from column_readout_epochs import epoch_kernel
        emitted_kernel, epoch_report = epoch_kernel(p.m, p.n, p.rows_per_reload,
                                                    p.scale_bits, p.group_runs)
    builder.fence()
    builder.config_ex(activation=int(p.relu))
    builder.config_ld(p.n * 4, shrink=False, id=0)
    dispatch_bits = tuple(p.scale_bits[first] for first, _ in _readout_groups(p.scale_bits, p.group_runs))
    def lower(name, attrs):
        if name == 'configure_scale':
            builder.constant_i32_dispatch(dispatch_bits, attrs['index'],
                lambda bits: builder.config_st(p.n, activation=int(p.relu), scale_bits=bits))
        elif name == 'reload_i32':
            builder.mvin(builder.ptr_offset(raw_ptr, attrs['byte_offset']), builder.add(ACC, attrs['acc_row']),
                         cols=attrs['columns'], rows=attrs['rows'], id=0)
        elif name == 'store_i8':
            builder.mvout(builder.ptr_offset(out_ptr, attrs['byte_offset']), builder.add(ACC, attrs['acc_row']),
                          cols=attrs['columns'], rows=attrs['rows'])
        elif name == 'complete':
            builder.fence()
        else:
            raise ValueError('unrecognized column readout effect ' + name)
    emission = emit_index_program(emitted_kernel, _IndexBridge(builder), lower)
    builder.fence()
    return {**p.report(), 'emission': emission, 'epoch_selection': mode,
            'emitted_kernel_sha256': emitted_kernel.digest(), 'epoch_admission': epoch_report}


def emit_column_readout(builder, raw_ptr, out_ptr, m, n, scales, activation=(), *, group_runs=False):
    """Minimal hook after a raw-i32 producer; returns its checked receipt."""
    return emit_plan(builder, plan_column_readout(m, n, scales, activation, group_runs=group_runs), raw_ptr, out_ptr)


def execute_plan(readout_plan, raw):
    """Host-only development checker for the actual typed schedule's effects.

    This function is never called by the compiler emitter or generated kernel.
    It checks strided DMA offsets, local ownership, each output write exactly
    once, and complete column-specific source numeric behavior.
    """
    p = readout_plan.revalidate()
    if len(raw) != p.m*p.n or any(type(x) is not int or not -(1<<31) <= x < 1<<31 for x in raw):
        raise ValueError('schedule execution requires exactly M*N signed-i32 values')
    output, writes = [None]*len(raw), [0]*len(raw)
    local, pending, scale, loaded = [], False, None, False
    dispatch_bits = tuple(p.scale_bits[first] for first, _ in _readout_groups(p.scale_bits, p.group_runs))
    for call, env in instances(p.kernel):
        attrs = concretize(call, env)
        if call.instr == 'configure_scale':
            if pending:
                raise ValueError('scale changed before readout completed')
            scale = struct.unpack('<f',struct.pack('<I',dispatch_bits[attrs['index']]))[0]
        elif call.instr == 'reload_i32':
            if pending or attrs['byte_offset'] % 4 or not 1 <= attrs['columns'] <= p.rows_per_reload or attrs['acc_row'] != 0:
                raise ValueError('invalid accumulator reload ownership or byte mapping')
            index = attrs['byte_offset']//4
            local = [[raw[index + lane*p.n + column] for column in range(attrs['columns'])]
                     for lane in range(attrs['rows'])]
            loaded, pending = True, True
        elif call.instr == 'complete':
            pending = False
        elif call.instr == 'store_i8':
            if (pending or not loaded or len(local) != attrs['rows'] or scale is None
                    or any(len(row) != attrs['columns'] for row in local)):
                raise ValueError('readout consumed an incomplete or mismatched reload')
            for lane,values in enumerate(local):
                for column,value in enumerate(values):
                    index = attrs['byte_offset'] + lane*p.n + column
                    output[index] = quantized(max(0,value) if p.relu else value,(scale,),False)
                    writes[index] += 1
            loaded, pending = False, True
    if pending or any(count != 1 for count in writes):
        raise ValueError('column readout did not publish every original output exactly once')
    return output
