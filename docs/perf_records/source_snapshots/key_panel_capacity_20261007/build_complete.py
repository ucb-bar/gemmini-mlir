"""Capacity-derived key batching on original input and independent pair tails."""
from pathlib import Path
import copy, hashlib, json, subprocess
import numpy as np
from merlin.llvmlower import quantized_affine_pair as pair
from merlin.perf.layer_bench import build_program
from xdsl.dialects.builtin import StringAttr
from mlir_oot.captured_residual_bundle import compile_adapter
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.rectifier_residual_catalog import adapter, kernel
from mlir_oot.no_fsm_audit import audit_elf

WORK = Path(__file__).resolve().parent / 'qualified_v2'
WORK.mkdir(exist_ok=False)
LLVM = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
TOOLS = Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin')
PROFILE = Path('/scratch/agustin/tmp/gemmini-current2101-profile-20261007/out/current2101_profile/profile_manifest.json')
OLD = Path('/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_original_ranked_v1')
SOURCE_PACKET = OLD / 'stock_packet.json'

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2) + '\n')

def run(argv):
    return subprocess.run(list(map(str, argv)), capture_output=True, text=True, check=True)

packet = json.loads(SOURCE_PACKET.read_text())
assert sha(SOURCE_PACKET) == '80a2dac306d9f96ff94011c263182355adb1954f012671a34c8f8fac7ee753d8'
for path, digest in packet['flat_file_pins'].items():
    assert sha(path) == digest, path
manifest = json.loads(PROFILE.read_text())
route = manifest['boundaries'][5]['selected_implementation']
assert route['panel_batch'] == 4 and route['proof']['indicator_family'] == 'predictor_key'
assert manifest['baseline_unprofiled_stock_job'] == 2101
original_kernel = Path(route['compilation']['compiler_argv'][-1][-1])
original_adapter = Path(route['adapter_compilation']['compiler_argv'][-1])
assert sha(original_kernel) == route['compilation']['object_sha256']
assert sha(original_adapter) == route['adapter_compilation']['object_sha256']
capture = json.loads((OLD / 'recipe.json').read_text())['source_capture']
assert capture['all1000_original_f32_bits_exact'] and capture['elements'] == route['m'] * route['n']
assert sha(OLD / 'a.bin') == capture['lhs_sha256'] and sha(OLD / 'b.bin') == capture['rhs_sha256']
table = pair.source_table(**route['proof']['source'])
assert hashlib.sha256(table.tobytes()).hexdigest() == route['proof']['source_table_sha256']
cases = []
for label, m, n, a, b in [
    ('original', route['m'], route['n'], np.frombuffer((OLD / 'a.bin').read_bytes(), np.int8), np.frombuffer((OLD / 'b.bin').read_bytes(), np.int8)),
    ('independent_all_pairs_tail', 1040, 128,
     ((np.arange(1040 * 128, dtype=np.int64) % 65536) // 256 - 128).astype(np.int8),
     (np.arange(1040 * 128, dtype=np.int64) % 256 - 128).astype(np.int8)),
]:
    folder = WORK / label
    folder.mkdir(exist_ok=False)
    count = m * n
    expected = table[a.astype(np.int16) + 128, b.astype(np.int16) + 128]
    assert len(a) == len(b) == len(expected) == count
    objects = []
    arms = []
    for arm, factor in [(0, 4), (1, 16)]:
        armfolder = folder / f'impl{factor}'
        armfolder.mkdir()
        selected = copy.deepcopy(route)
        selected.update(m=m, n=n, panel_batch=factor)
        module = kernel(selected)
        function, = module.body.block.ops
        function.properties['sym_name'] = StringAttr(selected['kernel'])
        compiled = compile_module(module, LLVM, armfolder / 'device')
        source = armfolder / 'adapter.c'
        source.write_text(adapter(selected))
        ac = compile_adapter(source, armfolder / 'adapter.o', LLVM)
        if label == 'original' and factor == 4:
            assert compiled['object_sha256'] == sha(original_kernel), 'Control kernel must reproduce actual current2101'
            assert ac['object_sha256'] == sha(original_adapter), 'Control adapter must reproduce actual current2101'
        name = f'capacity{factor}_kernel'
        ko, ao = armfolder / 'rebound_kernel.o', armfolder / 'rebound_adapter.o'
        run([TOOLS / 'riscv64-unknown-elf-objcopy', '--redefine-sym', selected['kernel'] + '=' + name,
             armfolder / 'device/kernel.o', ko])
        run([TOOLS / 'riscv64-unknown-elf-objcopy', '--redefine-sym', selected['kernel'] + '=' + name,
             '--redefine-sym', '_mlir_ciface_' + selected['symbol'] + '=' + ('exact_residual' if arm == 0 else 'rectified_residual'),
             armfolder / 'adapter.o', ao])
        objects.extend([ko, ao])
        arms.append(dict(arm=arm, factor=factor, compilation=compiled, adapter=ac,
                         complete_plan=json.loads(module.attributes['gemmini.key_rectified_plan'].data)))
    fixture = folder / 'fixture.S'
    assembly = '.section .rodata\n'
    for name, array in [('a', a), ('b', b), ('check_a', a), ('check_b', b), ('expected', expected)]:
        target = folder / (name + '.bin')
        target.write_bytes(array.tobytes())
        assembly += f'.balign 64\n.global fixture_{name}\nfixture_{name}:\n.incbin "{target}"\n'
    fixture.write_text(assembly)
    # Frozen ranked harness times all adapter setup/private scratch/commands,
    # DDR store/reload and final fence. Validation remains outside the window.
    harness = (OLD / 'probe.c').read_text().replace('#define N 802816', f'#define N {count}').replace('#define M 12544', f'#define M {m}').replace('#define K 64', f'#define K {n}')
    (folder / 'probe.c').write_text(harness)
    (folder / 'benchmark_buffer.h').write_bytes((OLD / 'benchmark_buffer.h').read_bytes())
    program = build_program([folder / 'probe.c', fixture, *objects], folder / 'arm0', target='gemmini',
        extra_cflags=['-I' + str(folder), '-O2', '-fno-fast-math', '-ffp-contract=off'], max_loaded_bytes=None)
    raw = program.elf.read_bytes()
    sections = run([TOOLS / 'riscv64-unknown-elf-readelf', '-SW', program.elf]).stdout
    selector, = [line.split(']', 1)[1].split() for line in sections.splitlines()
                 if ']' in line and line.split(']', 1)[1].split()[:1] == ['.selector']]
    offset = int(selector[3], 16)
    assert raw[offset:offset + 8] == bytes(8)
    altered = bytearray(raw)
    altered[offset] = 1
    arm1 = folder / 'arm1/layer.elf'
    arm1.parent.mkdir()
    arm1.write_bytes(altered)
    assert [i for i, (x, y) in enumerate(zip(raw, altered, strict=True)) if x != y] == [offset]
    for arm, elf in enumerate([program.elf, arm1]):
        audit = audit_elf(elf.read_bytes())
        assert audit['status'] == 'pass' and not audit['forbidden'] and not audit['unknown']
        result = subprocess.run([str(TOOLS / 'spike'), '--extension=gemmini', '--isa=rv64gc',
                                 '-m0x80000000:0x400000000', str(elf)], capture_output=True, text=True, timeout=300)
        (elf.parent / 'spike.stdout').write_text(result.stdout)
        (elf.parent / 'spike.stderr').write_text(result.stderr)
        assert result.returncode == 0 and f'RECTIFIED_RANKED_PASS arm{arm} all{count}' in result.stdout
        arms[arm].update(elf=str(elf), elf_sha256=sha(elf), strict_stdout_sha256=sha(elf.parent / 'spike.stdout'),
                         nofsm_audit=audit, returncode=result.returncode)
        print(json.dumps(dict(strict=label, arm=arm, elements=count, factor=arms[arm]['factor'])), flush=True)
    case = dict(label=label, m=m, n=n, elements=count, complete_cost=True, arms=arms,
                same_elf_except_selector=True, selector_offset=offset,
                all65536_pairs_covered=label == 'independent_all_pairs_tail',
                original_capture=label == 'original', original_gate='all source i8 words exact,4096 guards, immutable inputs, ranked descriptors, fflags')
    save(folder / 'qualification.json', case)
    cases.append(case)
pins = {str(path): sha(path) for path in [PROFILE, SOURCE_PACKET, OLD / 'recipe.json', original_kernel, original_adapter, Path(__file__)]}
for path in WORK.rglob('*'):
    if path.is_file() and path.name != 'build.log':
        pins[str(path)] = sha(path)
save(WORK / 'qualification.json', dict(schema='capacity_derived_key_batch_complete_qualification_v1', status='PASS',
     cases=cases, source_capture=capture, control_actual2101_kernel_and_adapter_byte_identical=True,
     default_factor_unchanged=4, chosen_alternative_factor=16, pins=pins,
     scope='Target-only resource-derived batch legality; original arithmetic/chunk/store/RMW completion and owner lifetime unchanged. No new numeric policy, model selection or performance claim.'))
