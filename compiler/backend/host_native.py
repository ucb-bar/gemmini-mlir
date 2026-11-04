"""Build the actual mixed host/device product, with explicit runtime suppliers.

This adapter emits no arithmetic. It translates the existing complete program,
uses the caller's selected CPU flags, and lets the native linker establish closure.
An object build alone never claims an executable or numerical qualification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json
import subprocess

from .host_runtime import WholeProgramError, WholeProgramResult


def _pin(path: Path) -> dict:
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def build_native_product(
    result: WholeProgramResult,
    module: str | Path,
    *,
    compiler: str | Path,
    translator: str | Path,
    nm: str | Path,
    cflags: Sequence[str],
    runtime_objects: Sequence[str | Path] = (),
    linker: str | Path | None = None,
    linker_script: str | Path | None = None,
    entry_symbol: str | None = None,
) -> dict:
    """Compile RV64 host code and optionally link caller-supplied runtime objects.

    The caller owns all CPU, startup, memory-map and runtime choices. Passing
    runtime objects requires all three link choices; no missing supplier is
    replaced by a stub. Failures persist the exact command and stderr before
    raising. Only an ELF with no undefined symbols counts as linked.
    """
    module = Path(module)
    work = result.routed_source.parent / 'native'
    work.mkdir(exist_ok=True)
    report_path = work / 'native-product.json'
    report = dict(schema='submission.whole_program_native.v1',
        source=_pin(Path(result.stats['adapter_original_source'])),
        input_module=_pin(module), source_abi=result.stats['adapter_abi'],
        lowered_abi=result.stats['adapter_lowered_abi'], cpu_flags=list(cflags),
        object_status='pending', link_status='not_requested',
        numerical_status='not_run', runtime_objects=[], commands=[])

    def persist():
        report_path.write_text(json.dumps(report, indent=2))
        result.stats['adapter_native_product'] = dict(path=str(report_path.resolve()), **report)

    def run(argv):
        argv = list(map(str, argv))
        done = subprocess.run(argv, capture_output=True, text=True, timeout=900)
        report['commands'].append(dict(argv=argv, returncode=done.returncode,
            stderr=done.stderr[-6000:]))
        persist()
        if done.returncode:
            raise WholeProgramError('native product build failed: ' + done.stderr[-6000:])
        return done.stdout

    if not any(flag.startswith('-march=') for flag in cflags) or not any(flag.startswith('-mabi=') for flag in cflags):
        raise WholeProgramError('native product requires explicit selected CPU and ABI flags')
    wants_link = bool(runtime_objects) or any(x is not None for x in (linker, linker_script, entry_symbol))
    if wants_link and (not runtime_objects or linker is None or linker_script is None or entry_symbol is None):
        raise WholeProgramError('native link requires runtime objects, linker, linker script and entry symbol together')
    if wants_link and (not entry_symbol.isascii() or not entry_symbol.isidentifier()):
        raise WholeProgramError('invalid native runtime entry symbol')
    report['tools'] = {name: _pin(Path(value)) for name, value in (
        ('compiler', compiler), ('translator', translator), ('nm', nm))}
    native_ir, obj = work / 'program.ll', work / 'program.o'
    try:
        run([translator, '--mlir-to-llvmir', module.resolve(), '-o', native_ir])
        run([compiler, *cflags, '-ffreestanding', '-fno-vectorize', '-fno-slp-vectorize',
            '-fstack-usage', '-c', native_ir, '-o', obj])
        report['object'] = _pin(obj)
        report['object_status'] = 'pass'
        undefined = run([nm, '--undefined-only', '--format=posix', obj])
        report['object_undefined_symbols'] = sorted(line.split()[0] for line in undefined.splitlines() if line.strip())
        defined = run([nm, '--defined-only', '--extern-only', '--format=posix', obj])
        report['object_defined_symbols'] = sorted(line.split()[0] for line in defined.splitlines() if line.strip())
        expected = set(result.stats['adapter_lowered_abi'][key] for key in ('entry_symbol', 'completion_symbol'))
        if not expected <= set(report['object_defined_symbols']):
            raise WholeProgramError('native object lost the complete-program entry or completion definition')
        stack = obj.with_suffix('.su')
        if stack.is_file():
            report['stack_usage'] = dict(**_pin(stack), rows=stack.read_text().splitlines())
        if wants_link:
            report['link_status'] = 'pending'
            report['runtime_objects'] = [_pin(Path(p)) for p in runtime_objects]
            report['linker'] = _pin(Path(linker))
            report['linker_script'] = _pin(Path(linker_script))
            elf = work / 'program.elf'
            run([linker, '--no-undefined', '-T', Path(linker_script).resolve(), '-e', entry_symbol,
                obj, *(Path(p).resolve() for p in runtime_objects), '-o', elf])
            unresolved = run([nm, '--undefined-only', '--format=posix', elf])
            if unresolved.strip():
                raise WholeProgramError('linked native program retains undefined runtime symbols')
            report['executable'] = _pin(elf)
            report['runtime_entry_symbol'] = entry_symbol
            report['link_status'] = 'pass'
    except Exception as exc:
        if report['object_status'] != 'pass':
            report['object_status'] = 'fail'
        elif wants_link:
            report['link_status'] = 'fail'
        report['failure'] = str(exc)
        persist()
        raise
    persist()
    return report
