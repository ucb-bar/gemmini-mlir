"""Independent full successor target correctness; functional counters are not FPGA timings."""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'smol_endpoint_dag_whole_strict'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


admission_path = WORK / 'root_admission.json'
admission = json.loads(admission_path.read_text())
packet_path = Path(admission['source_packet'])
assert sha(packet_path) == admission['source_packet_sha256'] == '3cb1b2d77e89302cb56d82ed3d99c25edc6c472b1524d1e9fa4dae7cdba056ec'
packet = json.loads(packet_path.read_text())
assert len(packet['pins']) == admission['source_pins_revalidated'] == 191
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
terminal_path = WORK / 'terminal.json'
terminal = json.loads(terminal_path.read_text())
assert terminal['returncode'] == 0 and terminal['argv'] == admission['argv']
assert terminal['argv'][3] == '7200s' and '-m0x80000000:0x400000000' in terminal['argv']
elf = Path(admission['elf']); engine = Path(admission['engine'])
assert sha(elf) == packet['target_elf_sha256'] == admission['elf_sha256'] == terminal['elf_sha256']
assert sha(engine) == admission['engine_sha256']
stdout = WORK / 'spike.stdout'; histogram = WORK / 'spike.stderr'
assert sha(stdout) == terminal['stdout_sha256'] and sha(histogram) == terminal['histogram_sha256']
source_admission = WORK.parent / 'smol_endpoint_whole_stock/source_admission.json'
source = json.loads(source_admission.read_text())
parser = Path(source['protocol']['parser_file'])
assert sha(parser) == source['pins'][str(parser)]
spec = importlib.util.spec_from_file_location('sealed_smol_endpoint_protocol', parser)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
golden = Path(source['protocol']['golden'])
report = module.verify_protocol(stdout.read_text(), np.load(golden, allow_pickle=False))
audit = audit_elf(elf.read_bytes())
assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
record = dict(schema='root_smol_endpoint_dag_whole_target_qualification_v1', status='PASS',
    source_pins_revalidated=191, target_elf=str(elf), target_elf_sha256=sha(elf),
    original_elements=1600, original_gate=admission['original_gate'],
    original_raw_output_sha256=source['protocol']['expected_sha256'], bitwise_exact=True,
    rank_mismatch=0, native_source_validation=packet['native']['bitwise_mismatches'] == 0,
    terminal=terminal, final_nofsm=audit, spike_functional_cycles=report['cycles'],
    whole_firesim_cycles='UNKNOWN', stock_job=2113, stock_capacity_proven=False,
    scope='Independent actual complete successor target run, all1600 original f32 words exact. Functional counters do not establish FPGA timing. No target group counters, physical capacity or allocator highwater inferred from native instrumentation.',
    pins={str(p): sha(p) for p in [packet_path, admission_path, terminal_path, stdout,
        histogram, elf, engine, source_admission, parser, golden, Path(__file__),
        WORK.parent / 'run_smol_endpoint_dag_whole_strict.py']})
(WORK / 'qualification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status='PASS', original1600_bitwise_exact=True,
    spike_functional_cycles=report['cycles'], elapsed_seconds=terminal['elapsed_seconds'],
    whole_firesim_cycles='UNKNOWN', stock_job=2113)))
