"""Recheck the owned ZIP's physical routines and recorded timing definitions.

This is a retained experiment, not a compiler selector. A PC histogram counts
retired CPU instructions, and does not measure accelerator busy time.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004')
ELF = ROOT / 'out/reference_q1013/exo-nofsm-t3-f.elf'
DIS = ROOT / 'out/reference_q1013/resnet50.dis'
HIST = Path('/scratch/agustin/tmp/gemmini-reference-parity-20261005/out/reference_nofsm_diagnostic/original.histogram')
UART = ROOT / 'docs/perf_records/q1013_diagnostic_reference_firesim.uart'

def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

symbols = []
for line in subprocess.run(['readelf', '-sW', str(ELF)], check=True, capture_output=True, text=True).stdout.splitlines():
    fields = line.split()
    if len(fields) >= 8 and fields[3] == 'FUNC' and fields[2].isdigit() and int(fields[2]):
        start, size, name = int(fields[1], 16), int(fields[2]), fields[7]
        if name.startswith(('fx_', 'glue_')):
            symbols.append((start, start + size, name))

instructions = {}
for line in DIS.read_text().splitlines():
    address, separator, body = line.partition(':')
    if separator and body.split():
        try:
            instructions[int(address.strip(), 16)] = body.split()[0]
        except ValueError:
            pass

counts = Counter()
per = defaultdict(Counter)
unknown = 0
for line in HIST.read_text().splitlines():
    fields = line.split()
    if len(fields) != 2:
        continue
    try:
        pc, count = int(fields[0], 16), int(fields[1])
    except ValueError:
        continue
    owners = [s for s in symbols if s[0] <= pc < s[1]]
    if not owners:
        continue
    owner = max(owners, key=lambda s: (s[1] - s[0], s[2]))[2]
    mnemonic = instructions.get(pc)
    if mnemonic is None:
        unknown += count
        continue
    counts[mnemonic] += count
    per[owner][mnemonic] += count

float_prefixes = ('fadd.', 'fsub.', 'fmul.', 'fdiv.', 'fsqrt.', 'fmadd.', 'fmsub.', 'fnmadd.', 'fnmsub.', 'fcvt.', 'fmin.', 'fmax.')
floating = {k: v for k, v in counts.items() if k.startswith(float_prefixes)}
whole, bracket, other = 22387449, 22347798, 12066
result = {
    'schema': 'owned_reference_host_work_census_v1',
    'scope': 'Physical fx_/glue_ function PC union in the owned original ELF; not a pure host/accelerator cycle split, not a source-model numerical equivalence proof.',
    'pins': [pin(p) for p in (ELF, DIS, HIST, UART, Path(__file__))],
    'physical_function_count': len(per),
    'retired_instructions_in_function_union': sum(counts.values()),
    'unmapped_selected_instructions': unknown,
    'floating_arithmetic_and_conversion_instructions': floating,
    'mnemonics': dict(counts.most_common()),
    'per_function': {k: {'retired_instructions': sum(v.values()), 'floating_arithmetic_and_conversion': {m: n for m, n in v.items() if m.startswith(float_prefixes)}} for k, v in sorted(per.items())},
    'stock1876': {
        'whole_cycles': whole,
        'bracketed_cycles': bracket,
        'unbracketed_cycles': whole - bracket,
        'unbracketed_percent': (whole - bracket) * 100 / whole,
        'other_cycles': other,
        'other_is_device_gap': True,
        'other_witness': {'start_timer': '0x8009e586', 'glue_gap_call': '0x8009e5a4', 'fence': '0x8009e5a8', 'end_timer': '0x8009e5ac', 'delta_into_s8': '0x8009e5b0', 'printf_other_argument': '0x8009e800', 'glue_gap': '0x8009c794', 'fx_gap': '0x80096c16'},
        'legacy_comparison_other_plus_unbracketed_cycles': other + whole - bracket,
        'pure_cpu_cycles': None,
        'pure_accelerator_busy_cycles': None,
        'limitation': 'Primitive and layer intervals include CPU dispatch/address/command issue, transfers, waits and device work. Other is the timed device global average pool, not host-only work.'
    },
    'source_to_device_findings': [
        'Reference input is already i8; source-qualified compiler model accepts f32 NCHW and must preserve its quantization.',
        'Reference stem halo padding uses device load/store commands; pooling uses the hardware store path.',
        'Reference residual adds use an identity matmul with real D and device transfer/store scaling.',
        'Reference global average pool uses an all-ones integer matmul; FC is also a device matmul.',
        'Reference current1992 comparison found three host stride-2 projection materializations that the existing segmented device-input binding can remove.',
        'Current source-qualified integer mean remains a host reduction and can feed a device integer-sum provider with the existing exact consumer certificate/replay retained.'
    ],
    'no_foreign_folder_access': True,
    'no_cpu_percentage_inferred_from_retired_instructions': True,
}
output = ROOT / 'docs/perf_records/q1013_host_work_coverage_recheck_20261006.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('physical_function_count', 'retired_instructions_in_function_union', 'unmapped_selected_instructions', 'floating_arithmetic_and_conversion_instructions')}))
