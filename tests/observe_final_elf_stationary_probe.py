"""Close an isolated request census on an already qualified complete ELF.

The two observational providers must leave complete output and PC execution
unchanged. This is a functional census, not a hardware timing prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def provider_pins(engine):
    build = json.loads((engine / 'build.json').read_text())
    rows = [*build['outputs'].values(), build['hook'], build['builder']]
    for row in rows:
        if sha(row['path']) != row['sha256']:
            raise ValueError('observational provider artifact changed')
    return rows


def run(args):
    qualified = json.loads(args.qualification.read_text())
    if qualified['elf_sha256'] != sha(args.elf):
        raise ValueError('candidate differs from qualified final ELF')
    if not qualified['spike_full_output_match']:
        raise ValueError('complete original source gate is absent')
    reference = np.load(qualified['reference_path'], allow_pickle=False)
    if reference.dtype != np.float32 or reference.size != qualified['output_elements']:
        raise ValueError('original reference shape/type differs')
    original = reference.astype('<f4', copy=False).tobytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != qualified['spike_output_sha256']:
        raise ValueError('original reference digest differs')
    audit = audit_elf(args.elf.read_bytes())
    if audit['status'] != 'pass':
        raise ValueError('complete executable contains disallowed custom ISA')
    if not args.reclose:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        raise ValueError('existing census directory is absent')
    providers = {}
    for name, engine in [('previous', args.previous_engine), ('stationary', args.engine)]:
        artifact_pins = provider_pins(engine)
        out = args.output / name
        if not args.reclose:
            out.mkdir()
        command = [str(engine / 'spike'), '-g',
                   '--extlib=' + str(engine / 'libgemmini_telemetry.so'),
                   '--extension=gemmini', '--isa=rv64gc',
                   '-m' + args.memory, str(args.elf)]
        env = dict(os.environ)
        for key in ('MERLIN_GEMMINI_TELEMETRY_SCOPES',):
            env.pop(key, None)
        env.update(MERLIN_GEMMINI_TELEMETRY=str(out / 'operands.json'),
                   MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC='1')
        elapsed, returncode = None, None
        if not args.reclose:
            start = time.monotonic()
            with (out / 'stdout').open('wb') as stdout, (out / 'histogram').open('wb') as stderr:
                result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                        timeout=args.wall_cap)
            elapsed, returncode = time.monotonic() - start, result.returncode
            if returncode:
                raise ValueError('isolated functional census did not finish')
        text = (out / 'stdout').read_text().replace('\r', '')
        if text.splitlines().count('DONE') != 1 or text.splitlines().count('METRIC memref_rank_mismatch 0') != 1:
            raise ValueError('complete numeric execution did not close')
        rows = [line.split() for line in text.splitlines() if line.startswith('OUT ')]
        if len(rows) != 1:
            raise ValueError('sampled output record ambiguous')
        printed = int(rows[0][1])
        if not 0 < printed <= reference.size or len(rows[0]) != printed + 2:
            raise ValueError('printed output word extent differs')
        words = np.asarray([int(v) for v in rows[0][2:]], dtype='<u4')
        if words.tobytes() != original[:printed * 4]:
            raise ValueError('any printed original prefix word changed')
        if f'OUT_SHA256 f32le {reference.size} {len(original)} {digest}' not in text.splitlines():
            raise ValueError('complete target output digest differs')
        retirement = re.findall(r'^METRIC cycles (\d+)\s*$', text, re.M)
        if len(retirement) != 1:
            raise ValueError('functional retirement marker ambiguous')
        providers[name] = {
            'command': command, 'exit_code': returncode,
            'elapsed_seconds': elapsed, 'functional_roi_retirement_proxy': int(retirement[0]),
            'printed_prefix_words_independently_exact': printed,
            'all_output_words_validated_by_complete_digest': reference.size,
            'provider_artifacts': artifact_pins,
            'pins': {k: pin(out / k) for k in ('stdout', 'histogram', 'operands.json')},
        }
        print(json.dumps({'provider':name,'elapsed_seconds':elapsed,'original_words_exact':reference.size}),flush=True)
    a, b = args.output / 'previous', args.output / 'stationary'
    for name in ('stdout', 'histogram'):
        if (a / name).read_bytes() != (b / name).read_bytes():
            raise ValueError('observational provider altered target PC execution or output')
    old = json.loads((a / 'operands.json').read_text())
    new = json.loads((b / 'operands.json').read_text())
    keys = ('preload_real_stationary_b','preload_retained_stationary_b','preload_unknown_mode')
    stripped = dict(new)
    stripped['schema'] = old['schema']
    stripped['rows'] = [{k:v for k,v in row.items() if k not in keys} for row in new['rows']]
    if stripped != old:
        raise ValueError('any original operand census field changed')
    feature_keys = (*keys,'padded_compute_rows','padded_mac_slots','requested_load_bytes','requested_store_bytes','unknown_dma_commands')
    features = {key:sum(row[key] for row in new['rows']) for key in feature_keys}
    if features['preload_unknown_mode'] or features['unknown_dma_commands']:
        raise ValueError('actual request signature has unknown configured state')
    record = {
        'schema':'qualified_final_elf_stationary_request_census_v1','status':'PASS',
        'providers':providers,'features':features,'output_words':reference.size,
        'original_output_digest':digest,'nofsm':audit,
        'previous_stdout_histogram_all_old_census_fields_identical':True,
        'reclose_existing_files_without_new_execution':args.reclose,
        'execution_receipt_scope':'Existing-file mode does not infer exit code or elapsed time; these stay UNKNOWN. Complete DONE/output/PC/census comparisons are independently reclosed.',
        'cycle_prediction':'UNKNOWN; request counts are not durations or physical traffic',
        'scope':'Actual compiled final ELF, functional full-source output gate. No target instruction or production simulator changed. Nominal padded work and logical bytes only; no shape-invariant floors or whole-model price.',
        'pins':{'qualification':pin(args.qualification),'elf':pin(args.elf),
                'reference':pin(qualified['reference_path']),'script':pin(__file__)},
    }
    with (args.output / 'request_census.json').open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'PASS','features':features}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--previous-engine',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--memory',default='0x80000000:0x400000000')
    p.add_argument('--wall-cap',type=int,default=180)
    p.add_argument('--reclose',action='store_true')
    run(p.parse_args())
