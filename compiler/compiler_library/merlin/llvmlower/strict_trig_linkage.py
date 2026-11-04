"""Explicit link dependencies for LLVM's strict binary32 sin/cos lowering.

This adapter selects no runtime, library search path, numerical policy or host
permission. An original caller may bind its already selected native math archive.
Absent that selection, emission produces an unresolved dependency contract only.
The optional object/link audits use explicit selected tools and actual products;
link supplier evidence alone is not source numerical equivalence.
"""
from pathlib import Path
import subprocess

from merlin.common.digest import sha256_file
from merlin.targetgen.host_linkage_contract import validate_linkage_contract, SUPPLIER
from .link_supplier_trace import trace_symbol_flags, observe_link_suppliers

SCHEMA = 'merlin.strict_f32_trig_dependencies.v1'
_SYMBOLS = {'cosf': 'cos', 'sinf': 'sin'}


def requirements(symbols):
    symbols = list(symbols)
    if not symbols or symbols != sorted(set(symbols)) or any(s not in _SYMBOLS for s in symbols):
        raise ValueError('strict trig dependencies require a canonical nonempty sinf/cosf roster')
    return {
        'schema': SCHEMA,
        'symbols': symbols,
        'declarations': [dict(source_operation='math.'+_SYMBOLS[s],
                              llvm_intrinsic='llvm.'+_SYMBOLS[s]+'.f32',
                              external_symbol=s, result='f32', arguments=['f32'],
                              calling_convention='ccc', fastmath=[]) for s in symbols],
        'supplier_status': 'unresolved',
        'scope': 'strict LLVM source lowering; selected supplier and numerical equivalence not established',
    }


def _checked(requirement):
    if not isinstance(requirement, dict) or requirement != requirements(requirement.get('symbols', [])):
        raise ValueError('malformed or changed strict trig dependency contract')
    return requirement


def _identity(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('explicit canonical file required')
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def inspect_object(requirement, obj, *, nm, run=subprocess.run):
    """Observe an actual object's unresolved symbols using the selected nm.

    No tool discovery or fallback is performed. A compiler-induced replacement
    symbol (such as sincosf) requires its own contract rather than being hidden.
    """
    _checked(requirement)
    before, tool = _identity(obj), _identity(nm)
    argv = [tool['path'], '--undefined-only', '--format=posix', before['path']]
    result = run(argv, capture_output=True, text=True, timeout=60)
    if result.returncode or len(result.stdout.encode()) > 65536:
        raise ValueError('selected object symbol inspector failed or returned oversized output')
    undefined = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[1] != 'U':
            raise ValueError('unsupported selected object symbol inspector output')
        undefined.append(fields[0])
    expected = set(requirement['symbols'])
    if not expected <= set(undefined):
        raise ValueError('emitted object does not retain every declared trig dependency')
    if set(undefined) & {'sin', 'cos', 'sincos', 'sincosf'}:
        raise ValueError('emitted object uses an uncontracted math replacement symbol')
    if before != _identity(obj) or tool != _identity(nm):
        raise ValueError('object or selected inspector changed during inspection')
    return dict(schema='merlin.strict_f32_trig_object.v1', dependency=requirement,
                object=before, inspector=tool, argv=argv,
                undefined_symbols=sorted(set(undefined)), supplier_status='unresolved')


def bind_selected_archive(requirement, contract, archive):
    """Bind the original caller's exact native archive; grant no host permission.

    This only supplies explicit archive inputs and diagnostic trace flags. The
    caller retains its original linker, CPU/ABI options, libc and runtime flags.
    It neither adds -lm search nor selects/compiles a portable implementation.
    """
    _checked(requirement)
    contract = validate_linkage_contract(contract)
    identity = _identity(archive)
    if (contract['supplier'] != SUPPLIER or contract['symbols'] != requirement['symbols']
            or contract['archive_sha256'] != identity['sha256']):
        raise ValueError('selected native archive does not match the original supplier contract')
    return dict(schema='merlin.strict_f32_trig_archive.v1', dependency=requirement,
                contract=contract, archive=identity,
                link_inputs=[identity['path']],
                diagnostic_flags=list(trace_symbol_flags(requirement['symbols'])),
                supplier_status='selected_not_observed')


def observe_selected_link(binding, *, argv, input_identities, result):
    """Bind successful actual-link diagnostics through Merlin's shared checker."""
    if not isinstance(binding, dict):
        raise ValueError('selected strict trig archive binding required')
    reproduced = bind_selected_archive(binding['dependency'], binding['contract'], binding['archive']['path'])
    if reproduced != binding or result.returncode != 0 or list(map(str, result.args)) != list(map(str, argv)):
        raise ValueError('link or selected math archive differs from the bound invocation')
    matching = [item for item in input_identities if item.get('path') == binding['archive']['path']]
    if matching != [binding['archive']]:
        raise ValueError('actual link inputs do not retain the selected archive identity')
    selected = {s: Path(binding['archive']['path']) for s in binding['dependency']['symbols']}
    observed = observe_link_suppliers(argv=list(map(str, argv)), input_identities=input_identities,
                                      expected_suppliers=selected, stderr=result.stderr)
    return dict(schema='merlin.strict_f32_trig_link.v1', dependency=binding['dependency'],
                supplier_status='observed_selected_link', observation=observed,
                scope='observed native symbol resolution; source numerical equivalence not established')
