"""Untimed fixed-partition/source-observer and actual ELF layout analysis.

This experiment changes no compiler policy, source, golden, or qualified ELF.
The 22-factor screen reuses the FIRST captured input, not 22 independent taps.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import struct
import subprocess
import time

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)

W = Path(__file__).resolve().parent
C = Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006')
Y = Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006')
P = C/'out/artifacts/probes/source-expression-interval-table-20261006'
NORMAL = C/'out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic'
T = Y/'out/artifacts/probes/source-expression-interval-table-20261006/normal_whole_v3'
CONTROL = Y/'out/artifacts/probes/tiny-rectangular-whole-20261006/whole'
CAPTURE = Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006/capture')
B = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
GCC = Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(name, value):
    path = W/name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')

effects = IntervalEffectContract(True, True, True, True, True)
module = parse_mlir_text((NORMAL/'typed_prepacket.generic.mlir').read_text())
proofs, refusals = find_closed_scalar_i8_observers(module, effects=effects)
assert len(proofs) == 22 and not refusals
for proof in proofs:
    validate_closed_scalar_observer(proof)
assert len({p.expression.canonical_sha256 for p in proofs}) == 1
expression = proofs[0].expression
plan = json.loads((P/'source_plan.json').read_text())
assert proofs[0].quant_factor_bits == plan['consumer']['quant_factor_bits']

lib = ctypes.CDLL(str(P/'native_oracle.so'))
for name in ('source_array', 'quant_array'):
    getattr(lib, name).argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]

def source(x):
    x = np.ascontiguousarray(x, np.float32)
    result = np.empty_like(x)
    lib.source_array(x.size, x.ctypes.data, result.ctypes.data)
    return result

def quant(x):
    x = np.ascontiguousarray(x, np.float32)
    result = np.empty(x.shape, np.int8)
    lib.quant_array(x.size, x.ctypes.data, result.ctypes.data)
    return result

def finish(v, up, q):
    with np.errstate(all='ignore'):
        return quant((v*up).astype(np.float32)*q)

a, sa, b, sb = [np.load(CAPTURE/(name+'.npy')) for name in ('a', 'scale_a', 'b', 'scale_b')]
dequant = np.array(0x3c9a3f89, np.uint32).view(np.float32)
x = ((a.astype(np.float32)*dequant).astype(np.float32)*sa).astype(np.float32).ravel()
up = ((b.astype(np.float32)*dequant).astype(np.float32)*sb).astype(np.float32).ravel()
original = source(x)
factors = [np.array(p.quant_factor_bits, np.uint32).view(np.float32) for p in proofs]
references = [finish(original, up, q) for q in factors]
assert np.array_equal(references[0], np.load(CAPTURE/'expected.npy').ravel())
raw = x.view(np.uint32)
inputs_hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in (a, sa, b, sb)]

# This declaration precedes any result collection; widths are resource choices,
# not production defaults. No cycle labels are read or fitted by this script.
declaration = dict(
    schema='source_interval_partition_predeclaration_v1',
    started_utc=datetime.now(timezone.utc).isoformat(),
    widths=[16, 17, 18, 19, 20],
    budgets_bytes=[(1 << n)*8 for n in range(16, 21)],
    same_source_canonical_sha256=expression.canonical_sha256,
    same_source_wide_cartesian_enclosure=True,
    scope='Original complete FIRST M8/5632-column input; all22 typed source quant factors on this same input as a constants stress screen. Later groups input distributions UNKNOWN.',
    no_hardware_or_cycle_measurement=True,
)
write('declaration.json', declaration)

rows = []
for bits in declaration['widths']:
    started = time.monotonic()
    table = build_source_interval_table(expression, effects=effects, leading_bits=bits, max_table_bytes=(1 << bits)*8)
    path = W/f'table_{bits}.bin'
    assert not path.exists()
    path.write_bytes(table.data)
    values = np.frombuffer(table.data, '<f4').reshape(-1, 2)
    indices = raw >> np.uint32(32-bits)
    bounds = values[indices]
    valid = (bounds[:, 0] <= bounds[:, 1]) & np.isfinite(x) & np.isfinite(up)
    assert np.all((original[valid] >= bounds[valid, 0]) & (original[valid] <= bounds[valid, 1]))
    safe = np.where(valid[:, None], bounds, np.float32(1))
    factor_rows = []
    for ordinal, (proof, q, reference) in enumerate(zip(proofs, factors, references)):
        low, high = finish(safe[:, 0], up, q), finish(safe[:, 1], up, q)
        accepted = valid & (low == high)
        carriers = np.where(accepted, bounds[:, 0], original).astype(np.float32)
        actual = finish(carriers, up, q)
        assert np.array_equal(actual, reference)
        factor_rows.append(dict(
            source_binding_index=ordinal,
            quant_factor_bits=proof.quant_factor_bits,
            accepted=int(accepted.sum()),
            source_replays=int((~accepted).sum()),
            original_i8_exact=True,
            original_i8_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            actual_i8_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            unobserved_carrier_changes=int(np.count_nonzero(carriers.view(np.uint32) != original.view(np.uint32))),
        ))
    # The table is aligned64 in the executed candidate. Counts describe requested
    # offsets, not resident cache lines, physical transactions, or cache misses.
    rows.append(dict(
        leading_bits=bits,
        table_bytes=len(table.data),
        valid_cells=table.valid_cells,
        invalid_cells=(1 << bits)-table.valid_cells,
        table_sha256=table.sha256,
        original_first_M8_elements=x.size,
        original_first_M8_valid_cells=int(valid.sum()),
        unique_cells=int(np.unique(indices).size),
        requested_64B_regions=int(np.unique(indices//8).size),
        requested_4KiB_regions=int(np.unique(indices//512).size),
        logical_table_read_bytes=x.size*8,
        source_replay_arithmetic_counts_first_factor={key: value*factor_rows[0]['source_replays'] for key, value in Counter(s.opcode for s in expression.steps).items()},
        source_fallback_retained=True,
        runtime_mode_guard_retained=True,
        source_oracle_shared_finish_cost='Source preparation, final2 roundedFMUL/clamp/RNE/store remain; no measured runtime cost inferred from acceptance.',
        all22_factors_same_first_inputs=factor_rows,
        elapsed_native_analysis_seconds=time.monotonic()-started,
    ))
    print(json.dumps({k: rows[-1][k] for k in ('leading_bits', 'table_bytes', 'valid_cells', 'unique_cells', 'requested_64B_regions', 'requested_4KiB_regions')}, separators=(',', ':')) + ' first_replay=' + str(factor_rows[0]['source_replays']), flush=True)

assert inputs_hashes == [hashlib.sha256(v.tobytes()).hexdigest() for v in (a, sa, b, sb)]
record = dict(
    schema='source_interval_partition_untimed_analysis_v1',
    rows=rows,
    effects=vars(effects),
    original_first_M8_source_and_i8_exact_all_widths=True,
    all22_factor_same_fixture_i8_exact=True,
    later_groups_actual_inputs_available=False,
    universally_proved_cells='Unmodified generic source interval evaluator and fixed raw-word partition, every source operation rounded binary32 under explicitRNE; all members covered by Cartesian proof. Fixture enclosure checks are additional diagnostics, not proof.',
    actual_cycles='UNKNOWN for16/17/18/19.20-bit2033 is an independent held whole negative, not training.',
    promotion=False,
    token_usage_available=False,
    pins={str(p): sha(p) for p in [Path(__file__), W/'declaration.json', NORMAL/'typed_prepacket.generic.mlir', NORMAL/'census.json', P/'native_oracle.so', P/'source_activation.ll', P/'source_quantize.ll', P/'source_plan.json', C/'src/merlin/llvmlower/source_expression_interval.py', *(CAPTURE/(n+'.npy') for n in ('a', 'scale_a', 'b', 'scale_b', 'expected')), *W.glob('table_*.bin')]},
)
write('parameter_analysis.json', record)

def elf_layout(path):
    with path.open('rb') as f:
        ident = f.read(16)
        assert ident[:6] == b'\x7fELF\x02\x01'
        h = struct.unpack('<HHIQQQIHHHHHH', f.read(48))
        shoff, entsize, count, names_index = h[5], h[10], h[11], h[12]
        assert entsize == 64
        f.seek(shoff)
        sections = [struct.unpack('<IIQQQQIIQQ', f.read(64)) for _ in range(count)]
        f.seek(sections[names_index][4])
        names = f.read(sections[names_index][5])
        def string(data, i):
            return data[i:data.find(b'\0', i)].decode()
        named = []
        for s in sections:
            if s[2] & 2:
                named.append(dict(name=string(names, s[0]), type=s[1], flags=s[2], address=s[3], file_offset=s[4], bytes=s[5], alignment=s[8]))
        symbols = {}
        for s in sections:
            if s[1] != 2:
                continue
            strings = sections[s[6]]
            f.seek(strings[4])
            texts = f.read(strings[5])
            f.seek(s[4])
            raw_symbols = f.read(s[5])
            for off in range(0, len(raw_symbols), 24):
                name, info, other, index, address, size = struct.unpack_from('<IBBHQQ', raw_symbols, off)
                text = string(texts, name)
                if text in {'_bss_start', '_bss_end', '_stack_top', 'MERLIN_WEIGHTS_BASE', 'MERLIN_STACK_BYTES', 'source_interval_table', 'malloc', 'free', 'merlin_malloc', 'forward', '_start', 'arena_cur', 'arena_used', 'arena_top'} or text.startswith('forward.extracted.531'):
                    symbols[text] = dict(address=address, bytes=size, section=index, info=info)
    return dict(path=str(path), sections=named, selected_symbols=symbols)

layouts = {'control2004': elf_layout(CONTROL/'model.elf'), 'candidate2033': elf_layout(T/'target_v2/model.elf')}
cmd = [str(GCC.with_name('riscv64-unknown-elf-objdump')), '-dr', str(B/'malloc.o')]
assembly = subprocess.check_output(cmd, text=True)
(W/'unchanged_allocator.disasm').write_text(assembly)
control_link = json.loads((T/'target_v2/controlled_link.json').read_text())
assert control_link['only_changed_object'] == 'model.o'
original_objects = control_link['baseline_objects']
candidate_objects = control_link['candidate_objects']
assert original_objects[str(B/'malloc.o')] == candidate_objects[str(B/'malloc.o')] == sha(B/'malloc.o')
write('layout_analysis.json', dict(
    schema='actual2004_2033_elf_layout_analysis_v1',
    layouts=layouts,
    only_changed_object='model.o',
    allocator_object_byteidentical=True,
    allocator_object_sha256=sha(B/'malloc.o'),
    allocator_actual_ISA_path=str(W/'unchanged_allocator.disasm'),
    dynamic_input_output_pointer_addresses='UNKNOWN: existing actual strict log does not print pointer identities. No new run or instrumentation.',
    memory_cause='UNKNOWN. Readonlytable/text and bss/stackVMA changes are observations, not attribution of single-runwhole regression.',
    common_address_padded_control_design='Hold text/helper layout and all allocator/runtime/weights/input preparation fixed; pad immutable rodata to same8MiB andalign64 in control, verify sections/stack/heap pointers then compare. Diagnostic notbuilt/admitted, not newtraininglabel.',
    pins={str(p): sha(p) for p in [Path(__file__), W/'unchanged_allocator.disasm', B/'malloc.o', T/'target_v2/controlled_link.json', T/'physical_table_closure.json', Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/baremetal/spike/model_link.ld'), Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/baremetal/spike/merlin_malloc.c')]},
    elf_identity_from_qualified_link=dict(control=control_link['baseline_elf_sha256'], candidate=control_link['elf_sha256']),
))
