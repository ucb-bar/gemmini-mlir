"""Unambiguous mcycle protocol; frozen normal-control and exact-row bodies."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
from mlir_oot.no_fsm_audit import audit_elf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/exact_row_clock_pair'
OLD = Path('/scratch/agustin/tmp/gemmini-polynomial-pair-provider-20261006/out/frontier_i64_allocated_pair_v2')
LINK_BASE = Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    source = (OLD / 'driver.c').read_text()
    assert source.count('csrr %0,mcycle') == 1 and 'minstret' not in source
    assert source.count('WORKSPACE_GROUP_INSTRUCTIONS') == 1
    selected = source.replace('WORKSPACE_GROUP_INSTRUCTIONS', 'WORKSPACE_GROUP_CYCLES')
    (OUT / 'driver.c').write_text(selected)
    old_build = json.loads((OLD / 'candidate/build.json').read_text())
    command = [x.replace(str(OLD / 'driver.c'), str(OUT / 'driver.c')).replace(str(OLD / 'driver.o'), str(OUT / 'driver.o')) for x in old_build['compile_driver']]
    subprocess.run(command, check=True)
    link_original = json.loads((LINK_BASE / 'build.json').read_text())['link']
    records = {}
    for arm, body in [('control', ROOT / 'out/sparse_dyadic_group/control/target_numeric/provider.o'), ('candidate', ROOT / 'out/exact_row_group/candidate/target_numeric/provider.o')]:
        dest = OUT / arm
        dest.mkdir()
        link = [str(body) if x == str(LINK_BASE / 'target_numeric/provider.o') else str(OUT / 'driver.o') if x == str(OLD / 'driver.o') else str(dest / 'model.elf') if x == str(LINK_BASE / 'model.elf') else x for x in link_original]
        subprocess.run(link, check=True, capture_output=True)
        audit = audit_elf((dest / 'model.elf').read_bytes())
        assert audit['status'] == 'pass'
        (dest / 'nofsm.json').write_text(json.dumps(audit, indent=2) + '\n')
        run = ['timeout', '--signal=TERM', '--kill-after=15s', '600s', '/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike', '--isa=rv64gc', '--extension=gemmini', str(dest / 'model.elf')]
        start = time.time()
        with (dest / 'stdout').open('w') as stdout, (dest / 'stderr').open('w') as stderr:
            result = subprocess.run(run, stdout=stdout, stderr=stderr)
        records[arm] = dict(link=link, provider_object=str(body), provider_sha256=sha(body), elf_sha256=sha(dest / 'model.elf'), command=run, returncode=result.returncode, seconds=time.time()-start)
        (dest / 'terminal.json').write_text(json.dumps(records[arm], indent=2) + '\n')
        assert result.returncode == 0
        print(arm, (dest / 'stdout').read_text(), flush=True)
    (OUT / 'build.json').write_text(json.dumps(dict(driver_compile=command, original_driver_sha256=sha(OLD / 'driver.c'), driver_sha256=sha(OUT / 'driver.c'), change='Only printed counter label; tick remains explicit mcycle CSR', records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
