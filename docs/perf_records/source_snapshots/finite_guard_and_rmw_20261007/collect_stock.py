"""Close immutable admissions against queue-owned terminal observations."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import numpy as np

from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parent
TINY = Path('/scratch/agustin/tmp/gemmini-tiny-finite-domain-20261007')
RESIDUAL = Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007')
ALIAS = 'alveo_u250_firesim_gemmini_rocket_stock'
HWDB = '5946d30df49231256bd6f5d0f31bdbc03e574990cda611fa5cf64c831b57c126'
BIT = '6bfb72e3d69d0bed14eaa8db9fd5faf53f82780ef9ecded6b2d49983dfcbc229'
TAR = 'a9a190b9fc26d577b1e8af0e3b46a94c8efaf2976f7fca6650236e0fd6d4eca1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')

def common(job_id, folder, packet_path, pin_key):
    job = Path(f'/scratch/firesim_queue/jobs/{job_id}')
    with sqlite3.connect('file:/scratch/firesim_queue/queue.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        terminal = dict(db.execute('SELECT id,state,phase,exit_code,submitted_at,started_at,ended_at FROM jobs WHERE id=?', (job_id,)).fetchone())
    assert (terminal['state'], terminal['phase'], terminal['exit_code']) == ('DONE', 'DONE', 0)
    spec = json.loads((job / 'runworkload-full.json').read_text())
    preflight = json.loads((folder / 'preflight/firesim_preflight.json').read_text())
    submit = json.loads((folder / 'queue_submission.json').read_text())
    packet = json.loads(packet_path.read_text())
    assert sha(packet_path) == submit['packet_sha256']
    pins = packet[pin_key]
    assert len(pins) == submit['pins_revalidated']
    for path, digest in pins.items():
        assert sha(path) == digest, path
    assert spec['job_id'] == job_id and spec['user'] == 'agustin'
    assert spec['hw_config'] == preflight['hardware_key'] == ALIAS
    assert spec['stage_from'] == preflight['elf']
    assert spec['hwdb_config_artifact'] == preflight['hwdb_artifact']
    assert spec['hwdb_config_artifact_sha256'] == preflight['hwdb_artifact_sha256'] == sha(spec['hwdb_config_artifact']) == HWDB
    assert preflight['bitstream_sha256'] == sha(preflight['bitstream_path']) == TAR
    elf = Path(spec['stage_from'])
    staged_elf = job / 'simulation/sim_slot_0/merlin-golden-nofsm-probe0-probe.elf'
    elf_digest = sha(elf)
    assert elf_digest == preflight['elf_sha256']
    staged_digest = sha(staged_elf) if staged_elf.is_file() else None
    if staged_digest is None:
        staged_record = json.loads((folder / 'actual_staged_identity.json').read_text())
        assert staged_record['job_id'] == job_id and staged_record['observed_before_teardown']
        staged_digest = staged_record['objects']['elf']['sha256']
    assert staged_digest == elf_digest
    assert preflight['elf_nofsm_audit']['elf_sha256'] == elf_digest
    assert preflight['elf_nofsm_audit']['status'] == 'pass'
    audit = audit_elf(elf.read_bytes())
    assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
    uart = job / 'simulation/sim_slot_0/uartlog'
    raw = uart.read_bytes()
    text = raw.decode().replace('\r', '')
    assert '*** PASSED ***' in text and 'COMMAND_EXIT_CODE="0"' in text
    archived_uart = folder / f'stock{job_id}_uart.txt'
    archived_uart.write_bytes(raw)
    provenance = {str(path): sha(path) for path in [packet_path, folder / 'queue_submission.json', folder / 'preflight/firesim_preflight.json', archived_uart, Path(__file__), job / 'runworkload-full.json']}
    provenance[str(elf)] = elf_digest
    result = dict(job=terminal, hardware_alias=ALIAS, hwdb_artifact_sha256=HWDB, bitstream_tar_sha256=TAR, nofsm_audit=audit, source_pins_revalidated=len(pins), actual_staged_elf_sha256=elf_digest, elapsed_queue_seconds=terminal['ended_at'] - terminal['started_at'], pins=provenance, token_usage_available=False, token_usage=None)
    return result, packet, text

def tiny():
    folder = ROOT / 'tiny_finite_scale_whole_stock'
    packet_path = TINY / 'docs/perf_records/finite_scale_guard_whole.json'
    result, packet, text = common(2076, folder, packet_path, 'pins')
    staged_path = folder / 'actual_staged_identity.json'
    staged = json.loads(staged_path.read_text())
    assert staged['job_id'] == 2076 and staged['observed_before_teardown'] and staged['phase'] == 'RUNNING'
    assert staged['objects']['elf']['sha256'] == result['actual_staged_elf_sha256']
    assert staged['objects']['bitstream']['sha256'] == BIT
    adapter_path = Path(packet['standard_reference_validation'])
    adapter = json.loads(adapter_path.read_text())
    assert sha(adapter_path) == packet['standard_reference_validation_sha256']
    assert adapter['status'] == 'pass' and adapter['spike_full_output_match'] and adapter['torch_allclose']
    assert adapter['elf_sha256'] == result['actual_staged_elf_sha256']
    assert adapter['torch_atol'] == .03125 and adapter['torch_rtol'] == .02
    reference, golden = Path(adapter['reference_path']), Path(adapter['torch_golden_path'])
    assert sha(reference) == adapter['reference_sha256'] and sha(golden) == adapter['torch_golden_sha256']
    values, torch_values = np.load(reference, allow_pickle=False), np.load(golden, allow_pickle=False)
    assert values.shape == torch_values.shape and values.size == 256000
    assert values.dtype == np.float32 and np.isfinite(values).all() and np.isfinite(torch_values).all()
    assert np.allclose(values, torch_values, atol=.03125, rtol=.02, equal_nan=False)
    raw_digest = hashlib.sha256(values.astype('<f4', copy=False).tobytes(order='C')).hexdigest()
    assert raw_digest == adapter['spike_output_sha256']
    assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([0-9a-f]{64})$', text, re.M) == [('256000', '1024000', raw_digest)]
    assert text.splitlines().count('DONE') == 1 and text.splitlines().count('METRIC memref_rank_mismatch 0') == 1
    assert re.findall(r'^METRIC build_hash (\S+)$', text, re.M) == [adapter['metrics']['build_hash']]
    metrics = re.findall(r'^METRIC cycles (\d+)$', text, re.M)
    assert len(metrics) == 1
    cycles = int(metrics[0])
    assert packet['all155devicebindings'] and packet['unchanged_nonmodel_leaves'] == 11 and packet['baseline2070_ELF_byteidentical']
    result.update(schema='tiny_finite_broadcast_guard_stock2076_whole_terminal_v1', status='verified_whole_model_observation', staged_identity=staged, cycles=cycles, control_job=2070, control_cycles=394765577, saving_cycles=394765577-cycles, saving_fraction=(394765577-cycles)/394765577, remaining_300M_gap=cycles-300000000, original_output_gate=dict(count=256000, raw_sha256=raw_digest, source_bitexact=True, torch_atol=.03125, torch_rtol=.02, torch_allclose=True), scope='Complete original eight-token22-layer TinyLlama first output, unchanged155targetbindings and11nonmodel leaves. Both immutable scans and all source/table/products/continuations/finish included. One stock whole observation; no precision-policy change or claim of pure utilization.')
    for path in [staged_path, adapter_path, reference, golden]:
        result['pins'][str(path)] = sha(path)
    save(folder / 'stock2076_terminal.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'cycles', 'saving_cycles', 'saving_fraction', 'remaining_300M_gap', 'elapsed_queue_seconds']}), flush=True)

def rmw():
    folder = ROOT / 'acc_identity_rmw_stock'
    packet_path = RESIDUAL / 'out/acc_identity_rmw_v1/qualification.json'
    result, packet, text = common(2077, folder, packet_path, 'flat_file_pins')
    windows = re.findall(r'^ACC_RMW_COUNTER repeat=(\d+) cycles=(\d+)$', text, re.M)
    assert len(windows) == 2 and [int(i) for i, _ in windows] == [0, 1] and all(int(c) > 0 for _, c in windows)
    assert text.splitlines().count('ACC_RMW_PASS values=256 repeats=2 identity_scale=1 accumulator_rmw=1') == 1
    assert not re.search(r'^ACC_RMW_FAIL', text, re.M)
    result.update(schema='acc_identity_rmw_stock2077_terminal_v1', status='capability_protocol_pass_staged_bit_identity_unavailable', complete_windows_cycles=[int(c) for _, c in windows], exact_values=256, repeats=2, adjacent_guard_bytes=128, bitstream_stage_sha256=None, limitation='Terminal collection followed teardown; actual staged bitstream was not preserved before its removal. Immutable one-entry HWDB/archive and actual retained staged ELF identities are verified. A forthcoming complete residual capsule must preserve the actual staged bitstream; this receipt is not used as a fully sealed performance champion.', scope='Unit-scale signedi32 ACC overwrite/accumulate/fullMVOUT capability, not a complete residual schedule or network speedup.')
    save(folder / 'stock2077_terminal.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'complete_windows_cycles', 'elapsed_queue_seconds']}), flush=True)

if __name__ == '__main__':
    tiny()
    rmw()
