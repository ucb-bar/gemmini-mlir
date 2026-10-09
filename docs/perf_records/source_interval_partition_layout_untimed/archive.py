"""Supplement original sealed table records; never replace their gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

W = Path(__file__).resolve().parent
Y = W.parents[3]
OUT = Y/'docs/perf_records/source_interval_partition_layout_untimed'
assert not OUT.exists()
OUT.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
analysis = json.loads((W/'parameter_analysis.json').read_text())
layout = json.loads((W/'layout_analysis.json').read_text())
for receipt in (analysis, layout):
    for path, digest in receipt['pins'].items():
        assert sha(path) == digest, path
old = Path('/scratch/agustin/tmp/gemmini-packed-rhs-current-20261006/out/artifacts/probes/source-expression-interval-table-20261006/normal_whole_v3/explicit_policy_reclosure/table.bin')
assert sha(old) == analysis['rows'][-1]['table_sha256']
raw = Path('/scratch/agustin/tmp/firesim-golden-recovery-20261005/job2033_verified.json')
hw = json.loads(raw.read_text())
assert hw['job_id'] == 2033 and hw['kernel_cycles'] == 424921379 and hw['control_job'] == 2004
assert hw['elf_sha256'] == layout['elf_identity_from_qualified_link']['candidate']
for key in ('control2004', 'candidate2033'):
    s = layout['layouts'][key]['selected_symbols']
    assert s['MERLIN_WEIGHTS_BASE']['address'] == 0x84100000
    assert s['MERLIN_STACK_BYTES']['address'] == 0x1000000
asm = (W/'unchanged_allocator.disasm').read_text().split('<malloc>:', 1)[1]
assert 'lui\ta5,0x87b' in asm and 'slli\ta5,a5,0x9' in asm
facts = dict(
    actual_allocator_base=0x87b000 << 9,
    actual_allocator_base_hex=hex(0x87b000 << 9),
    allocator_alignment=64,
    allocation_order_and_dynamic_pointers='Not instrumented; same fixed base does not prove every allocation pointer unchanged.',
    source_live_output_type='22 typed tensor<1x8x5632xi8> helpers,991232logical i8 carrier bytes per complete model. Source dimensions/shapes derived from immutable typedIR, not ELF inference.',
    source_first_helper_input_element_bytes=dict(a_i32=8*5632*4, b_i32=8*5632*4, scale_a_f32=5632*4, scale_b_f32=5632*4, output_i8=8*5632),
    first_helper_temporaries='Physical compiler temporaries/frame offsets UNKNOWN in this analysis; source element bytes are not allocated/live traffic.',
    same_first_fixture_22_factor_caveat='Each source factor is real typedIR; latergroup operand distributions are not this first fixture. Do not multiply its acceptance/cache counts into a whole prediction.',
    cache_inference='Requested region cardinality is not cachemiss/physicalDRAM traffic, and M2 warm timing does not price normalM8 or cold22helper whole composition.',
)
copied = {}
for path in [W/'analyze.py', W/'declaration.json', W/'parameter_analysis.json', W/'layout_analysis.json', W/'unchanged_allocator.disasm', Path(__file__), raw]:
    target = OUT/path.name
    shutil.copy2(path, target)
    assert sha(target) == sha(path)
    copied[str(target)] = sha(target)
record = dict(
    schema='source_interval_partition_layout_archive_v1',
    finished_utc=datetime.now(timezone.utc).isoformat(),
    hypothesis='Reduce immutable table and requested region footprints using coarser fixed binary32 partitions while retaining source-wide proof and original exacti8 observer; inspect actual linkVMA confounds before attributing whole cost.',
    ownership='Generic source proof/table/observer Merlin; actualRV64 ELF/link/allocator facts OOT. Experiment width selection explicit, no automaticpolicy.',
    actual_change='Untimed tables16/17/18/19 generated; original20bit tablebyteidentical. No new native whole/targetELF/GSIM/stock execution and no existing artifact mutation.',
    facts=facts,
    parameter_rows=[{k: r[k] for k in ('leading_bits','table_bytes','valid_cells','invalid_cells','unique_cells','requested_64B_regions','requested_4KiB_regions')} | dict(first_M8_replays=r['all22_factors_same_first_inputs'][0]['source_replays'], all22_factors_replays_range=[min(v['source_replays'] for v in r['all22_factors_same_first_inputs']),max(v['source_replays'] for v in r['all22_factors_same_first_inputs'])]) for r in analysis['rows']],
    whole_heldout=dict(candidate_job=2033, cycles=hw['kernel_cycles'], control_job=2004, control_cycles=422018733, fractional_regression=hw['kernel_cycles']/422018733-1, original_full256000_digest_exact=True, repetitions=1, result='REJECTED performance promotion; measuredchampion unchanged. Memorycause UNKNOWN. Held wholevalidation, nevertrain.'),
    before_after_gate='All5widths universalcells retain unmodifiedproof; complete first45056originali8 and22actualsourcefactors on samefixture exact. Finalwhole20bit originalaccuracyPASS butcostnegative.',
    candidate_selection='UNKNOWN until model ranks fallback/sourcefinish/tabletraffic/layout and appropriate wholecorrectness gates. No implicit16/19bitselection or admission.',
    token_usage_available=False,
    original_receipt_immutable=True,
    pins=analysis['pins'] | layout['pins'] | copied | {str(old): sha(old),str(raw):sha(raw)},
)
(Y/'docs/perf_records/source_interval_partition_layout_untimed.json').write_text(json.dumps(record,indent=2)+'\n')
print('PARTITION_LAYOUT_ARCHIVED',len(record['pins']),flush=True)
