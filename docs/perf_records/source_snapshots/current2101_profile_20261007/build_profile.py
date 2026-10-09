"""Diagnostic profile of immutable, accepted 2101 objects; no semantic rebuild."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_device_profile import emit, parse_profile
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent
BASE = Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal/controlled2095')
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
RUNTIME = Path('/scratch/agustin/tmp/merlin-resnet-qualified-runtime-20261005/merlin/runtime')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (WORK / name).write_text(json.dumps(value, indent=2) + '\n')


def run(argv, log, timeout=600):
    save(log + '.argv.json', list(map(str, argv)))
    with (WORK / (log + '.log')).open('w') as stream:
        result = subprocess.run(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{log} failed ({result.returncode}); see retained log')


def main():
    argv = json.loads((BASE / 'candidate/linker_argv.json').read_text())
    base_elf = Path(argv[-1])
    assert sha(base_elf) == '1aa4fa5a921153d10d951d3acea89b3e2d6ba1bba1f3f97fc02718ab749bf3d3'
    objects = [Path(x) for x in argv if x.endswith('.o')]
    inputs = {str(p): sha(p) for p in objects}
    control = WORK / 'reproduced_control.elf'
    run([*argv[:-1], str(control)], 'reproduce_control')
    assert sha(control) == sha(base_elf), 'baseline linker does not reproduce accepted bytes'

    catalog_path = BASE / 'qualification/build_view/device_catalog/device_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    source = Path(catalog['source_snapshot'])
    assert sha(source) == catalog['source_sha256']
    calls = [op.callee.root_reference.data for op in parse_module(source.read_text()).walk() if op.name == 'func.call']
    routes = []
    for group, category in [('fused_requantizations', 'convolution_and_requant_adapter'), ('residual_additions', 'residual_adapter'), ('guarded_mean_additions', 'global_mean_adapter')]:
        routes.extend(dict(symbol=r['symbol'], category=category, source_region=r.get('region', r.get('source_region')), shape=r.get('shape'), schedule=r.get('schedule'), kernel=r['kernel']) for r in catalog[group])
    routes.append(dict(symbol=catalog['pooled_stem']['symbol'], category='stem_pool_adapter', shape=catalog['pooled_stem']['shape']))
    by_name = {r['symbol']: r for r in routes}
    assert len(by_name) == len(routes) == 70
    assert Counter(calls) == Counter({s: 1 for s in by_name}), 'source call coverage changed'
    host = next(p for p in objects if p.name == 'model.o')
    llvm = host.parent / 'host_llvm/model.linked.ll'
    llvm_text = llvm.read_text()
    declarations = {}
    for line in llvm_text.splitlines():
        if line.startswith('declare ') and '@_mlir_ciface_gemmini_' in line:
            m = re.fullmatch(r'declare dso_local void @([A-Za-z_0-9]+)\(([^()]*)\) local_unnamed_addr #\d+', line)
            assert m, f'unsupported declared ABI: {line}'
            args = m.group(2).split(', ')
            assert len(args) in (4, 5) and args == ['ptr noundef'] * len(args)
            declarations[m.group(1)] = len(args)
    assert set(declarations) == {'_mlir_ciface_' + s for s in by_name}
    nm = subprocess.run([str(LLVM / 'llvm-nm'), '--undefined-only', str(host)], check=True, capture_output=True, text=True).stdout
    assert {line.split()[-1] for line in nm.splitlines() if '_mlir_ciface_gemmini_' in line} == set(declarations)
    boundaries = [dict(by_name[s], id=i, source_symbol=s, symbol='_mlir_ciface_' + s, pointer_arity=declarations['_mlir_ciface_' + s]) for i, s in enumerate(calls)]
    assert len(catalog['kernels']) == 1
    classifier = catalog['kernels'][0]['symbol']
    shim_source = next(p for p in objects if p.name == 'device_catalog_shim.o').with_suffix('.c')
    assert f'extern void {classifier}(void *lhs, void *rhs, void *out);' in shim_source.read_text()
    boundaries.append(dict(id=len(boundaries), symbol=classifier, pointer_arity=3, category='classifier_primitive'))
    symbols = [(row['symbol'], row['pointer_arity']) for row in boundaries]
    manifest = dict(schema='current2101_adapter_profile_v1', scope='diagnostic boundary timers; callbacks include CPU adapter/issue/DMA/waits; gaps include CPU and profiler overhead', boundaries=boundaries, expected_device_calls=71, expected_symbol_calls={str(row['id']): 1 for row in boundaries}, catalog_path=str(catalog_path), catalog_sha256=sha(catalog_path), source_path=str(source), source_sha256=sha(source), host_llvm_path=str(llvm), host_llvm_sha256=sha(llvm), classifier_shim_source_path=str(shim_source), classifier_shim_source_sha256=sha(shim_source), accepted_elf=str(base_elf), accepted_elf_sha256=sha(base_elf), semantic_object_sha256=inputs, baseline_reproduced_sha256=sha(control), controlled_native_alias_binding=catalog['controlled_native_alias_binding'], controlled_source_domain_binding=catalog['controlled_source_domain_binding'], baseline_unprofiled_firesim_cycles=28728702)
    # These public boundaries follow the actual frozen caller, not a renamed
    # normal-route source. Record both current implementation facts and the
    # controlled public alias relation without substituting the host graph.
    normal_catalog_path = BASE.parent / 'normal/build_direct/device_catalog/device_catalog.json'
    selected_catalog = json.loads(normal_catalog_path.read_text())
    selected = {}
    for group in ['fused_requantizations', 'residual_additions', 'guarded_mean_additions']:
        for row in selected_catalog[group]:
            name = row.get('original_symbol', row.get('source_symbol', row['symbol']))
            # A source_symbol for the accepted segmented implementation can name
            # its original source, while the current caller names the selected ABI.
            if row['symbol'] in by_name:
                name = row['symbol']
            assert name not in selected
            selected[name] = row
    selected[selected_catalog['pooled_stem']['symbol']] = selected_catalog['pooled_stem']
    assert set(selected) == set(by_name)
    key_binding = catalog['controlled_predictor_key_binding']
    key_manifest_path = Path(key_binding['selected_manifest']['path'])
    key_manifest = json.loads(key_manifest_path.read_text())
    assert sha(key_manifest_path) == key_binding['selected_manifest']['sha256']
    assert key_manifest['routes'][0]['original_symbol'] in selected
    selected[key_manifest['routes'][0]['original_symbol']] = key_manifest['routes'][0]
    manifest['current_selected_metadata_catalog'] = dict(path=str(normal_catalog_path), sha256=sha(normal_catalog_path), source_path=selected_catalog['source_snapshot'], source_sha256=selected_catalog['source_sha256'], scope='selected strategy metadata; frozen caller source remains catalog_path')
    for row in boundaries[:-1]:
        row['selected_implementation'] = selected[row['source_symbol']]
    manifest['controlled_predictor_key_binding'] = key_binding
    manifest['controlled_link_receipt'] = dict(path=str(BASE/'controlled_link.json'),sha256=sha(BASE/'controlled_link.json'))
    manifest['baseline_unprofiled_stock_job'] = 2101
    save('profile_manifest.json', manifest)
    c = WORK / 'profile.c'
    c.write_text(emit(symbols))
    obj = WORK / 'profile.o'
    run([*argv[:7], '-I', RUNTIME / 'baremetal/spike', '-I', RUNTIME / 'c', '-c', c, '-o', obj], 'compile_profile')
    end = argv.index('-lm')
    profile_elf = WORK / 'model.elf'
    link = [*argv[:end], str(obj), '-Wl,--wrap=merlin_run_multi', '-Wl,--wrap=htif_exit', *['-Wl,--wrap=' + s for s, _ in symbols], *argv[end:-1], str(profile_elf)]
    run(link, 'link_profile')
    assert inputs == {str(p): sha(p) for p in objects}, 'semantic object changed during link'
    audit = audit_elf(profile_elf.read_bytes())
    save('nofsm_audit.json', audit)
    assert audit['status'] == 'pass'
    run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike', '--extension=gemmini', '--isa=rv64gc', '-m0x80000000:0x80000000', profile_elf], 'spike', timeout=1200)
    log = (WORK / 'spike.log').read_text().replace('\r', '')
    assert re.search(r'^DONE$', log, re.M)
    assert re.search(r'^METRIC memref_rank_mismatch 0$', log, re.M)
    output = re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([a-f0-9]{64})$', log, re.M)
    assert len(output) == 1
    validation = json.loads((BASE / 'qualification/spike_validation.json').read_text())
    golden = Path(validation['torch_golden_path'])
    reference = Path(validation['reference_path'])
    assert sha(golden) == sha(reference) == validation['torch_golden_sha256']
    g = np.load(golden)
    r = np.load(reference)
    assert g.shape == r.shape and np.array_equal(g, r)
    raw = hashlib.sha256(g.astype('<f4').tobytes()).hexdigest()
    assert output[0] == ('1000', '4000', raw)
    profile = parse_profile(log, manifest)
    assert [row[1] for row in profile['events']] == list(range(71)), 'execution call order differs from bound source'
    save('spike_profile.json', profile)
    save('qualification.json', dict(schema='current2101_profile_qualification_v1', semantic_objects_unchanged=True, baseline_link_byte_exact=True, original_accuracy_gate=dict(atol=0.0, rtol=0.0, elements=1000, raw_output_sha256=raw, torch_golden_sha256=sha(golden), native_reference_sha256=sha(reference), pass_gate=True), elf_sha256=sha(profile_elf), final_elf_nofsm=True, all_71_declared_boundary_calls=True, profile_conservation=True, profile_scope=manifest['scope'], spike_cycles_are_retired_instructions=True, spike_log_sha256=sha(WORK / 'spike.log'), firesim_result=None))
    print(json.dumps(dict(elf_sha256=sha(profile_elf), boundary_count=71, exact_output=True, zero_fsm=True, spike_profile={k:v for k,v in profile.items() if k != 'events'})))


if __name__ == '__main__':
    main()
