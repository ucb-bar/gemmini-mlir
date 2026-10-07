"""Seal exact 2101 semantic-object diagnostic and actual linked boundary joins."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from mlir_oot.golden_device_profile import parse_profile
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent
OWN = WORK.parents[1]
BASE = Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal/controlled2095')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, v):
    (WORK/name).write_text(json.dumps(v, indent=2)+'\n')

def main():
    manifest=json.loads((WORK/'profile_manifest.json').read_text())
    validation=json.loads((WORK/'qualification.json').read_text())
    native=json.loads((WORK/'native_qualification.json').read_text())
    assert native['quality_pass'] and native['exact_equal'] and native['elements']==1000
    golden=np.load(BASE.parent/'normal/capture/golden.npy')
    actual=np.load(WORK/'native/output.npy')
    assert np.array_equal(golden.view(np.uint32),actual.view(np.uint32))
    assert sha(WORK/'model.elf')==validation['elf_sha256']
    assert sha(manifest['accepted_elf'])==sha(WORK/'reproduced_control.elf')==manifest['accepted_elf_sha256']
    for p,digest in manifest['semantic_object_sha256'].items(): assert sha(p)==digest
    profile=parse_profile((WORK/'spike.log').read_text(),manifest)
    assert profile==json.loads((WORK/'spike_profile.json').read_text())
    assert [e[1] for e in profile['events']]==list(range(71))
    audit=audit_elf((WORK/'model.elf').read_bytes())
    assert audit==json.loads((WORK/'nofsm_audit.json').read_text()) and audit['status']=='pass'
    # Revalidate that the link adds only the timer object and explicit wraps.
    original=json.loads((BASE/'candidate/linker_argv.json').read_text())
    linked=json.loads((WORK/'link_profile.argv.json').read_text())
    clean=[a for a in linked if a!=str(WORK/'profile.o') and not a.startswith('-Wl,--wrap=')]
    assert clean[:-1]==original[:-1]
    assert clean[-1]==str(WORK/'model.elf')
    wraps=[a.removeprefix('-Wl,--wrap=') for a in linked if a.startswith('-Wl,--wrap=')]
    assert Counter(wraps)==Counter(['merlin_run_multi','htif_exit',*[r['symbol'] for r in manifest['boundaries']]])
    nm={}
    for label, elf in [('baseline',Path(manifest['accepted_elf'])),('profile',WORK/'model.elf')]:
        text=subprocess.run([str(LLVM/'llvm-nm'),'-P','--defined-only',str(elf)],check=True,capture_output=True,text=True).stdout
        (WORK/(label+'_defined_symbols.txt')).write_text(text)
        nm[label]={}
        for line in text.splitlines():
            parts=line.split()
            if len(parts)>=3 and parts[1] in ('T','t'):
                assert parts[0] not in nm[label]
                nm[label][parts[0]]=dict(address=int(parts[2],16),size=int(parts[3],16) if len(parts)>3 else None)
    wrappers=['__wrap_'+r['symbol'] for r in manifest['boundaries']]
    disasm=subprocess.run([str(LLVM/'llvm-objdump'),'-d','--disassemble-symbols='+','.join(wrappers),str(WORK/'model.elf')],check=True,capture_output=True,text=True).stdout
    (WORK/'wrapper_disassembly.txt').write_text(disasm)
    blocks={name:body for name,body in re.findall(r'^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)',disasm,re.M|re.S)}
    joins=[]
    for row in manifest['boundaries']:
        sym=row['symbol'];wrapper='__wrap_'+sym
        assert sym in nm['baseline'] and sym in nm['profile'] and wrapper in nm['profile']
        # Includes AUIPC/JALR far calls; actual disassembler resolves the target.
        calls=re.findall(r'\b(?:jal|jalr)\s+[^\n]*<([^>]+)>',blocks[wrapper])
        assert calls==[sym],(wrapper,calls)
        selected=row.get('selected_implementation',{})
        kernel=selected.get('kernel')
        if kernel: assert kernel in nm['profile'] and kernel in nm['baseline']
        joins.append(dict(id=row['id'],source_callee=row.get('source_symbol'),host_undefined=row['symbol'] if row['category']!='classifier_primitive' else None,declared_void_pointer_arity=row['pointer_arity'],baseline_entry=nm['baseline'][sym],profile_entry=nm['profile'][sym],profile_wrapper=nm['profile'][wrapper],wrapper_calls_actual_selected_boundary=True,selected_implementation_symbol=selected.get('symbol'),selected_kernel=kernel,selected_kernel_entry=nm['profile'].get(kernel),exact_executed_boundary_count=1))
    save('actual_entry_join.json',dict(schema='current2101_actual_boundary_join_v1',boundaries=joins,caller_source=manifest['source_path'],caller_llvm=manifest['host_llvm_path'],source_order_exact=True,all71_wrap_to_real_calls_exact=True,alias_scope='original public residual0 ABI resolves selected key9 implementation; names alone do not prove its arithmetic; original semantic object hashes and stock2101 authority retained'))
    paths=set(WORK.iterdir())
    paths={p for p in paths if p.is_file() and p.name not in ('seal.log','sealed_qualification.json')}
    paths.update(map(Path,manifest['semantic_object_sha256']))
    paths.update(map(Path,native['input_pins']))
    paths.update(Path(manifest[k]) for k in ['catalog_path','source_path','host_llvm_path','classifier_shim_source_path','accepted_elf'])
    paths.add(Path(manifest['current_selected_metadata_catalog']['path']))
    paths.add(Path(manifest['current_selected_metadata_catalog']['source_path']))
    paths.add(Path(manifest['controlled_predictor_key_binding']['selected_manifest']['path']))
    paths.add(BASE/'controlled_link.json')
    paths.add(BASE/'qualification/spike_validation.json')
    paths.add(BASE/'candidate/linker_argv.json')
    paths.update([OWN/'mlir_oot/golden_device_profile.py',OWN/'mlir_oot/no_fsm_audit.py',OWN/'tests/fused_whole_model_probe.py'])
    paths.update([WORK/'native/validation.json',WORK/'native/output.npy',WORK/'native/reference.c',WORK/'native/model.o',WORK/'native/model.so'])
    paths.update([BASE/'qualification/build_view/quant_hoist_args.json',BASE/'qualification/build_view/quant_hoist_values.npz'])
    paths.update(p for p in (BASE.parent/'normal/capture').iterdir() if p.is_file())
    for argv in [original,linked]:
        paths.update(Path(a) for a in argv if a.startswith('/') and Path(a).is_file())
    for tool in ['llvm-nm','llvm-objdump']: paths.add(LLVM/tool)
    paths.add(Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'))
    # Bind actual candidate partial-link graph leaves without replacing it from
    # freshly generated metadata. Parent receipt retains deeper source proofs.
    graph=json.loads((BASE/'controlled_link.json').read_text())
    for key in ['partial_link_graph','segmented_partial_link_graph']:
        for node in graph[key]:
            if node.get('arm')=='candidate':
                paths.update(Path(a) for a in node['argv'] if a.startswith('/') and Path(a).is_file())
    for p in list(paths):
        if p.suffix=='.o':
            paths.update(q for q in [p.with_suffix('.ll'),p.with_suffix('.c')] if q.is_file())
    parent_receipt=OWN/'docs/perf_records/current_dense_stationary_tail_2095_whole_qualification.json'
    paths.add(parent_receipt)
    assert all(p.is_file() for p in paths)
    pins=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)]
    record=dict(schema='current2101_boundary_profile_qualified_v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),status='qualified_diagnostic_ready_for_root_review',baseline_job=2101,baseline_stock_cycles=28728702,baseline_elf=manifest['accepted_elf'],baseline_elf_sha256=manifest['accepted_elf_sha256'],profile_elf=str(WORK/'model.elf'),profile_elf_sha256=sha(WORK/'model.elf'),profile_manifest=str(WORK/'profile_manifest.json'),profile_manifest_sha256=sha(WORK/'profile_manifest.json'),profile_parser=str(OWN/'mlir_oot/golden_device_profile.py'),profile_parser_sha256=sha(OWN/'mlir_oot/golden_device_profile.py'),boundaries=71,semantic_object_identity=True,baseline_link_byte_exact=True,all19_current_dense_and_both_flat_winners_retained=True,fresh_native_original1000_exact=True,strict_spike_original1000_exact=True,final_elf_zero_fsm=True,typed_source_host_undefined_actual_entry_join=True,profile_conserved=True,spike_profile=profile,original_accuracy_gate=validation['original_accuracy_gate'],hardware_scope='FireSimGemminiRocketConfig stock; root submission pending',scope='boundary diagnostic; callbacks include CPU adapters, issue, DMA, waits and readout; gaps include CPU/profiler. Pure accelerator busy/CPU utilization UNKNOWN. Native checks unchanged source stand-ins, wrapper target-only.',stock_cycles=None,instrumentation_overhead_cycles=None,preliminary_refusal='v1 metadata path dict treated as string; stopped after exact baseline link, before instrumentation. Corrected explicit path/hash admission, no original code change.',token_usage_available=False,pins=pins)
    save('sealed_qualification.json',record)
    print(json.dumps(dict(elf=record['profile_elf'],sha256=record['profile_elf_sha256'],pins=len(pins),boundary_count=71,original1000_exact=True,spike_profile={k:v for k,v in profile.items() if k!='events'})))

if __name__=='__main__': main()
