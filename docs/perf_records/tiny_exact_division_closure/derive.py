"""Reconcile pinned typed source division domains and unchanged target PCs."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

W = Path(__file__).resolve().parent
O = W.parents[3]
P = W.parent / 'tiny-norm1926-pc-histogram-20261006'
S = Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006/source_division_census.json')
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
profile = json.loads((P / 'receipt.json').read_text())
assert profile['status'] == 'pass' and profile['full_original256000_digest_match']
assert sha(P / 'histogram.log') == profile['histogram_sha256']
census = json.loads(S.read_text())
hist = {int(a, 16): int(b) for a, b in re.findall(r'^([0-9a-f]+) (\d+)$', (P / 'histogram.log').read_text(), re.M)}
rows = []
for line in (P / 'disassembly.txt').read_text().splitlines():
    m = re.match(r'\s*([0-9a-f]+):\s+fdiv\.s\s+(.+)', line)
    if m and 0x800697fa <= int(m[1], 16) < 0x800d5040:
        rows.append(dict(pc=m[1], retired_count=hist.get(int(m[1], 16), 0), operands=m[2]))
counts = Counter(r['retired_count'] for r in rows)
assert counts == {22528: 44, 256: 176}
assert sum(r['retired_count'] for r in rows) == 1036288
summary = census['summary']
assert summary['reciprocal_three_rounded_mul_to_i8_clamp']['logical_divisions'] == 44 * 22528
assert summary['live_float_or_other_DAG']['logical_divisions'] == 176 * 256
assert not any(r['direct_division_integer_observer'] for r in census['rows'])
result = dict(
    schema='tiny_exact_division_source_machine_census_v1',
    status='pass',
    ELF_sha256=profile['ELF_sha256'],
    full_original256000_digest_match=True,
    forward_executed_fdiv_s=1036288,
    forward_fdiv_s_pc_count=220,
    source_summary=summary,
    machine_pc_count_by_retired_count={str(k): v for k, v in counts.items()},
    direct_quotient_integer_observer_eligible_sites=0,
    hot_row_shared_integer_observer_eligible_sites=0,
    rows=rows,
    source_machine_relation='Exact total/family count reconciliation under the pinned actual source and final ELF; machine PCs have no debug-line SSA identity. No individual PC-to-source operation identity is asserted.',
    closure='The 991232 preDown divisions are per-element 1/d followed by three individually rounded f32 multiplies. The 45056 softmax divisions share row denominators but escape as live f32. Constant /2048 normalization divisions disappear upstream. A quotient-to-RNE threshold cannot replace either live DAG.',
    removed_cycles_bound=None,
    cost_scope='PC census measures retired instructions, not hardware division latency. Older1880 profile1901 charges127539158cycles to22 complete preDown host gaps including all arithmetic, loads/stores and dispatch context; it is not a division-only or1926 profile.',
    prior_screen='The previously archived certified two-Newton f32-only reciprocal observer already avoids f64, passed original45056values, and lost134.99% retired instructions. Its small GSIM pair remained incomplete and negative; not rerun.',
    result='No direct quotient-to-integer transform installed. Pivot to measured source-exact code generation/loop organization in the dominant host section.',
    production_policy_enabled=False,
    pins={str(p): sha(p) for p in [S, P/'receipt.json', P/'census.json', P/'histogram.log', P/'disassembly.txt', P/'source_contractions.json']},
    token_usage_available=False,
)
(W / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
archive = O / 'docs/perf_records/tiny_exact_division_closure'
archive.mkdir(exist_ok=False)
for name, p in [('derive.py', Path(__file__)), ('source_division_census.json', S), ('receipt.json', W/'receipt.json')]:
    (archive/name).write_bytes(p.read_bytes())
(O/'docs/perf_records/tiny_exact_division_closure_journey.json').write_text(json.dumps({
    'hypothesis':'Row-shared exact quotient-to-integer thresholds could remove hot scalar division.',
    'ownership':'Typed arithmetic and consumer closure Merlin; target execution census OOT experimental evidence.',
    'actual_emitted_change':'None: actual source and final1926 ELF inspected unchanged.',
    'before_after_metric':'1036288 forward divisions before; zero eligible direct or row-shared integer observers. No transformed after measurement.',
    'gate':'Pinned typed source, unchanged-ELF full256000digest functional histogram, exact220PC/total/family reconciliation.',
    'decision':'Reject direct quotient-to-RNE substitution for this source. Existing reciprocal-chain negative remains retained.',
    'receipt_path':str(W/'receipt.json'), 'receipt_sha256':sha(W/'receipt.json'),
    'hardware_cycle_claim':None, 'token_usage_available':False,
}, indent=2)+'\n')
print('DIVISION_CLOSURE_EXACT', len(rows), sum(r['retired_count'] for r in rows))
