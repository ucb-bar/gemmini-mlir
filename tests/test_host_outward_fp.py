"""Actual instruction rounding fields and activation order of the CPU capability."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from mlir_oot.host_outward_fp import emit_fixed_outward_f64_header


@pytest.mark.parametrize('name', ['', '2bad', 'bad-name', 'x;bad', '\u00e9'])
def test_explicit_c_identifier_required(name):
    with pytest.raises(ValueError, match='identifier'):
        emit_fixed_outward_f64_header(name=name, host_isa='rv64gc')


def test_no_implicit_cpu_policy():
    with pytest.raises(ValueError, match='capability'):
        emit_fixed_outward_f64_header(name='proof', host_isa='unknown')


def test_narrowing_requires_explicit_boolean():
    with pytest.raises(ValueError, match='boolean'):
        emit_fixed_outward_f64_header(name='proof', host_isa='rv64gc', narrow_f32=1)


def test_narrowing_is_disabled_by_default():
    assert 'fcvt.s.d' not in emit_fixed_outward_f64_header(name='proof', host_isa='rv64gc')


@pytest.fixture(scope='module')
def tools():
    cc = os.environ.get('MERLIN_RISCV_CC') or shutil.which('riscv64-unknown-elf-gcc')
    if not cc:
        pytest.skip('explicit RV64 CPU compiler required')
    objdump = os.environ.get('MERLIN_RISCV_OBJDUMP') or str(Path(cc).with_name('riscv64-unknown-elf-objdump'))
    if not Path(objdump).is_file():
        pytest.skip('matching CPU disassembler required')
    return cc, objdump


def compile_probe(tmp_path, tools, prefix=''):
    cc, objdump = tools
    (tmp_path / 'capability.h').write_text(
        emit_fixed_outward_f64_header(name='proof', host_isa='rv64gc'))
    (tmp_path / 'probe.c').write_text(prefix + '''
#include "capability.h"
double up(double a,double b){return MERLIN_F64_OUTWARD_ADD_UP(a,b);}
double down(double a,double b){return MERLIN_F64_OUTWARD_ADD_DOWN(a,b);}
double product(double a,double b){return MERLIN_F64_OUTWARD_MUL_UP(a,b);}
''')
    result = subprocess.run([cc, '-std=c11', '-O2', '-march=rv64gc', '-mabi=lp64d',
                             '-c', str(tmp_path / 'probe.c'), '-o', str(tmp_path / 'probe.o')],
                            capture_output=True, text=True)
    return result, objdump


def test_actual_object_has_static_rounding_and_no_rounding_csr(tmp_path, tools):
    result, objdump = compile_probe(tmp_path, tools)
    assert result.returncode == 0, result.stderr
    listing = subprocess.check_output([objdump, '-d', str(tmp_path / 'probe.o')], text=True)
    instructions = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].endswith(':') and len(fields[1]) in (4, 8):
            try:
                word = int(fields[1], 16)
            except ValueError:
                continue
            instructions.append((word, fields[2], line))
    arithmetic = [(word, name) for word, name, _ in instructions if name in ('fadd.d', 'fmul.d')]
    assert [(name, (word >> 12) & 7) for word, name in arithmetic] == [
        ('fadd.d', 3), ('fadd.d', 2), ('fmul.d', 3)]
    assert all(not name.startswith('csr') for _, name, _ in instructions)


@pytest.mark.parametrize('prefix,message', [
    ('#define MERLIN_ORDERED_FMA_BOUNDS_H\n', 'must precede'),
    ('#define MERLIN_F64_OUTWARD_ADD_UP(a,b) ((a)+(b))\n', 'already selected'),
])
def test_late_or_duplicate_activation_refuses(tmp_path, tools, prefix, message):
    result, _ = compile_probe(tmp_path, tools, prefix)
    assert result.returncode != 0
    assert message in result.stderr


def test_actual_directed_conversion_encodings(tmp_path, tools):
    cc, objdump = tools
    (tmp_path / 'capability.h').write_text(
        emit_fixed_outward_f64_header(name='proof', host_isa='rv64gc', narrow_f32=True))
    (tmp_path / 'probe.c').write_text('''#include "capability.h"
float down(double x){return MERLIN_F32_OUTWARD_FROM_F64_DOWN(x);}
float up(double x){return MERLIN_F32_OUTWARD_FROM_F64_UP(x);}
''')
    subprocess.run([cc, '-std=c11', '-O2', '-march=rv64gc', '-mabi=lp64d',
                    '-c', str(tmp_path / 'probe.c'), '-o', str(tmp_path / 'probe.o')], check=True)
    listing = subprocess.check_output([objdump, '-d', str(tmp_path / 'probe.o')], text=True)
    encoded = []
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] == 'fcvt.s.d':
            encoded.append((int(fields[1], 16) >> 12) & 7)
    assert encoded == [2, 3]
    assert 'csr' not in listing
