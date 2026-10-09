"""Explicit Gemmini bare-metal harness recipe and pinned GSIM command provider.

No legacy backend imports, default project paths, model routing, or command
buffer code generation. Merlin owns build/link orchestration, engine receipt
validation, process deadlines, and output parsing. This package owns the
Gemmini software ABI and GSIM command-line protocol.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen import gsim_emulator

TARGET = 'gemmini'
GSIM_EMU_ENV = 'MERLIN_GEMMINI_GSIM_EMU'
register(BackendInfo(TARGET, TargetClass.NPU, BackendKind.KERNEL, __name__))


def _path(variable, *, directory=False):
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f'explicit {variable} is required')
    path = Path(value).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'{variable} does not resolve to an existing path')
    return path


def harness_build_recipe():
    """Same curated RV64GC harness ABI, with caller-selected tool/source paths."""
    compiler = _path('MERLIN_RISCV_GCC')
    root = _path('MERLIN_GEMMINI_HARNESS_DIR', directory=True)
    common = root / 'riscv-tests/benchmarks/common'
    raw = os.environ.get('MERLIN_GEMMINI_LOAD_ADDRESS')
    if raw is None:
        raise ValueError('explicit MERLIN_GEMMINI_LOAD_ADDRESS is required')
    address = int(raw, 0)
    if address <= 0 or address >= 1 << 64 or address % 4096:
        raise ValueError('load address must be a positive page-aligned unsigned64 address')
    sources = (common / 'syscalls.c', common / 'crt.S')
    link = common / 'test.ld'
    if not all(p.is_file() for p in (*sources, link)):
        raise ValueError('curated harness lacks CRT, syscalls, or link template')
    return HarnessBuildRecipe(
        compiler=compiler,
        include_roots=(root / 'riscv-tests', root / 'riscv-tests/env', root, common),
        support_sources=sources, link_script=link, load_address=address,
        cflags=('-DPREALLOCATE=1', '-DMULTITHREAD=1', '-mcmodel=medany', '-std=gnu99',
                '-O2', '-ffast-math', '-fno-common', '-fno-builtin-printf',
                '-fno-tree-loop-distribute-patterns', '-march=rv64gc', '-Wa,-march=rv64gc',
                '-DID_STRING=', '-DPRINT_TILE=0', '-nostdlib', '-nostartfiles', '-static', '-DBAREMETAL=1'),
        ldflags=('-lm', '-lgcc'),
    )


def gsim_backdoor_env():
    return {'MERLIN_GSIM_LOADMEM': '1'}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _engine():
    # Explicit path is mandatory; do not inherit a sibling checkout's engine home.
    _path(GSIM_EMU_ENV)
    return gsim_emulator.citation(TARGET, env_var=GSIM_EMU_ENV)


@dataclass(frozen=True)
class GsimCommand:
    argv: tuple[str, ...]
    env_overrides: tuple[tuple[str, str], ...]
    evidence_json: str

    def revalidate(self):
        evidence = json.loads(self.evidence_json)
        if _canonical(_engine()) != _canonical(evidence['engine']):
            raise ValueError('selected GSIM engine provenance changed')
        if list(self.argv) != evidence['argv'] or list(map(list, self.env_overrides)) != evidence['env_overrides']:
            raise ValueError('GSIM command or environment differs from pinned evidence')
        for path, digest in evidence['file_pins'].items():
            if _sha(path) != digest:
                raise ValueError('GSIM command input changed: ' + path)
        return {'status': 'unchanged', 'command_sha256': hashlib.sha256(self.evidence_json.encode()).hexdigest()}


def prepare_gsim_command(elf, *, expected_elf_sha256, expected_engine_provenance, max_cycles=None):
    """Construct only: the shared bounded runner owns execution and wall time."""
    if type(max_cycles) is not int or not 0 < max_cycles < 1 << 63:
        raise ValueError('explicit positive bounded integer max_cycles is required')
    path = Path(elf).resolve()
    if _sha(path) != expected_elf_sha256:
        raise ValueError('ELF differs from expected bytes')
    engine = _engine()
    if (engine.get('available') is not True or engine.get('refused') is not False
            or engine.get('flavour') != 'binary' or engine.get('receipt_status') != 'bound'):
        raise ValueError('GSIM requires an available binary with a bound build receipt')
    if _canonical(engine) != _canonical(expected_engine_provenance):
        raise ValueError('GSIM engine differs from expected provenance')
    emulator = Path(engine['path']).resolve()
    if not os.access(emulator, os.X_OK) or _sha(emulator) != engine['binary_sha256']:
        raise ValueError('GSIM executable differs from receipt')
    receipt = Path(engine['receipt']['receipt_path']).resolve()
    if _sha(receipt) != engine['receipt']['receipt_sha256']:
        raise ValueError('GSIM build receipt changed')
    interpreter = Path(sys.executable).resolve()
    # This generated GSIM harness keeps circuit state on its process stack.
    # The isolated prelude execs the engine; deadlines remain the caller's.
    prelude = ('import os,resource,sys\n'
               'try: resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))\n'
               'except (ValueError,OSError): pass\n'
               'os.execv(sys.argv[1],sys.argv[1:])\n')
    argv = (str(interpreter), '-I', '-S', '-c', prelude, str(emulator), str(path),
            f'+max-cycles={max_cycles}', f'+loadmem={path}')
    pins = (path, emulator, receipt, interpreter, Path(__file__).resolve(),
            Path(__file__).parent / 'contracts/target_contract.yaml',
            Path(__file__).parent / 'provider.yaml', Path(gsim_emulator.__file__).resolve())
    evidence = dict(schema='gemmini_gsim_harness_command_v1', argv=list(argv), env_overrides=[],
                    engine=engine, file_pins={str(p): _sha(p) for p in pins},
                    max_cycles=max_cycles, scope='Command construction only; no numeric or performance verdict')
    command = GsimCommand(argv, (), _canonical(evidence))
    command.revalidate()
    return command
