"""Actual complete successor PC costs by disjoint linked function extent.

These counts conserve all retired instructions. Shared callees are charged to
their own addresses, never to a guessed caller. No FPGA latency/utilization is
inferred from the unordered functional histogram.
"""
from bisect import bisect_right
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess

from mlir_oot.cpu_opcode_census import instruction_index
from mlir_oot.executed_features import parse_pc_histogram

BASE = Path(__file__).resolve().parent
EXECUTION = BASE / 'smol_endpoint_dag_whole_strict'
WORK = BASE / 'smol_endpoint_whole_retirement'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


WORK.mkdir(exist_ok=False)
qualification_path = EXECUTION / 'qualification.json'
qualification = json.loads(qualification_path.read_text())
assert qualification['status'] == 'PASS' and qualification['bitwise_exact']
for path, digest in qualification['pins'].items():
    assert sha(path) == digest, path
elf = Path(qualification['target_elf'])
histogram_path = EXECUTION / 'spike.stderr'
histogram = parse_pc_histogram(histogram_path.read_text())
index = instruction_index(elf.read_bytes())
readelf = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-readelf')
symbols_text = subprocess.check_output([str(readelf), '-sW', str(elf)], text=True)
symbols_path = WORK / 'actual_symbols.txt'; symbols_path.write_text(symbols_text)
extents = {}
for line in symbols_text.splitlines():
    fields = line.split()
    if len(fields) != 8 or fields[3] != 'FUNC' or fields[6] in ('UND', 'ABS'):
        continue
    start = int(fields[1], 16)
    size = int(fields[2], 16 if fields[2].startswith('0x') else 10)
    if size == 0:
        continue
    extents.setdefault((start, start + size), set()).add(fields[-1])
ordered = sorted(extents)
# A sweep creates exact address intervals. Equal-size aliases share a bucket;
# partially overlapping extents remain explicitly ambiguous.
events = {}
for extent in ordered:
    start, end = extent
    events.setdefault(start, []).append((1, extent))
    events.setdefault(end, []).append((-1, extent))
boundaries = sorted(events)
active = set(); intervals = []
for i, start in enumerate(boundaries):
    for sign, extent in events[start]:
        if sign == 1:
            active.add(extent)
        else:
            active.remove(extent)
    if i + 1 < len(boundaries):
        intervals.append((start, boundaries[i + 1], tuple(sorted(active))))
starts = [row[0] for row in intervals]
counts = Counter(); classes = {}; touched = Counter(); unknown = Counter()
for pc, count in histogram.items():
    position = bisect_right(starts, pc) - 1
    owners = ()
    if position >= 0 and pc < intervals[position][1]:
        owners = intervals[position][2]
    if len(owners) != 1:
        unknown['ambiguous_overlapping_function' if owners else 'no_nonzero_function_extent'] += count
        continue
    extent, = owners
    if pc not in index:
        raise ValueError('executed function PC is not an ELF instruction boundary')
    counts[extent] += count
    touched[extent] += index[pc][1]
    classes.setdefault(extent, Counter())[index[pc][0]] += count
assert sum(counts.values()) + sum(unknown.values()) == sum(histogram.values())
rows = [dict(symbols=sorted(extents[extent]), start=extent[0], end=extent[1],
    static_bytes=extent[1] - extent[0], retired_instructions=count,
    touched_instruction_bytes=touched[extent], cpu_opcode_classes=dict(classes[extent]))
    for extent, count in counts.most_common()]
record = dict(schema='root_smol_endpoint_actual_whole_function_retirement_v1',
    status='conserved_actual_function_address_counts', elf_sha256=sha(elf),
    histogram_sha256=sha(histogram_path), unique_executed_pcs=len(histogram),
    total_retired_instructions=sum(histogram.values()), assigned_instructions=sum(counts.values()),
    unassigned_instructions=dict(unknown), rows=rows,
    functional_model_window_cycles=qualification['spike_functional_cycles'],
    stock_whole_cycles='UNKNOWN', scope='Complete executable including startup/output/harness; each distinct function extent is exclusive of shared callees. Aliases retained together. No caller attribution, model-time partition, host/accelerator percentage or hardware cycle price.',
    pins={str(p): sha(p) for p in [qualification_path, elf, histogram_path,
        readelf, symbols_path, Path(__file__)]})
(WORK / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(total_retired=record['total_retired_instructions'],
    unassigned=record['unassigned_instructions'],
    top=[dict(symbols=row['symbols'], instructions=row['retired_instructions']) for row in rows[:18]])))
