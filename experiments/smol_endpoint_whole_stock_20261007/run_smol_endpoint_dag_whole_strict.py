"""Complete whole target execution with an evidence-based longer bound."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007')
WORK = Path(__file__).resolve().parent/'smol_endpoint_dag_whole_strict'
PACKET = ROOT/'docs/perf_records/prepared_endpoint_dag_normal_whole_qualification.json'
ELF = ROOT/'out/artifacts/probes/prepared-endpoint-normal/build_v3/model.elf'
ENGINE = Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike')
EXPECTED = 'f90b8c7e8d6c1ed668a0bb4eaf7918caaa2dc8e3fe4f93b6c14143f995889f07'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

WORK.mkdir(exist_ok=False)
packet = json.loads(PACKET.read_text())
assert len(packet['pins']) == 191
for path, digest in packet['pins'].items():
    assert sha(path) == digest, path
assert sha(ELF) == packet['target_elf_sha256'] == EXPECTED
assert packet['native']['bitwise_mismatches'] == 0
argv = ['timeout', '--signal=TERM', '--kill-after=15s', '7200s', str(ENGINE),
        '-g', '--extension=gemmini', '--isa=rv64gc', '-m0x80000000:0x400000000', str(ELF)]
save(WORK/'root_admission.json', dict(schema='root_smol_endpoint_dag_whole_target_admission_v1',
    status='native_source_closed_target_pending', source_packet=str(PACKET), source_packet_sha256=sha(PACKET),
    source_pins_revalidated=191, elf=str(ELF), elf_sha256=EXPECTED, engine=str(ENGINE), engine_sha256=sha(ENGINE),
    argv=argv, original_gate=dict(elements=1600, atol=0.03125, rtol=0.02), hardware_admission=False,
    bound_basis='Prepared-constants predecessor a90f target completed all1600 original source words in4366.498 seconds. Fresh endpoint-DAG successor has191 closed pins and independent native1600 exact output; no predecessor target pass transferred.7200s is a functional run bound, not a hardware performance forecast; no predecessor pass transferred to successor',
    prior_timeout=str(Path(__file__).resolve().parent/'smol_normal_whole_strict/terminal.json'),
    scope='Actual complete normal48 source target, original1600 final gate. Explicit16GiB Spike map is functional only; final stock memory binding and whole eligibility remain pending'))
started = time.monotonic()
with (WORK/'spike.stdout').open('wb') as stdout, (WORK/'spike.stderr').open('wb') as stderr:
    result = subprocess.run(argv, stdout=stdout, stderr=stderr)
terminal = dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-started, argv=argv,
    elf_sha256=sha(ELF), stdout_sha256=sha(WORK/'spike.stdout'), histogram_sha256=sha(WORK/'spike.stderr'))
save(WORK/'terminal.json', terminal)
print(json.dumps(terminal), flush=True)
assert result.returncode == 0, 'Whole target gate unresolved; retain terminal and outputs'
