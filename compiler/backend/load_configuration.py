"""Decode selected Gemmini load setters for the generic LLVM CFG cleanup.

Only CONFIG_LD is a removable write.  MVIN consumes but never changes the load
registers.  Fences leave their state intact.  Calls, unknown inline assembly,
unknown configuration operands and other configuration families forget state.
Pool/store and execute setters are deliberately not treated as pure writes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from merlin.compile.scheduling.configuration import ConfigurationEffects
from merlin.compile.scheduling.llvm_configuration import eliminate_llvm_configuration
from . import isa


def _integer(value):
    owner = value.owner
    if getattr(owner, 'name', None) != 'llvm.mlir.constant':
        return None
    attribute = owner.properties.get('value')
    value = getattr(getattr(attribute, 'value', None), 'data', None)
    return (int(value) & ((1 << 64) - 1)) if type(value) is int else None


def _admission():
    record = json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    order = record['ordering']
    digest = hashlib.sha256(json.dumps(order, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (record['base_facts_sha256'] != 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or digest != 'ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or order['same_queue']['load']['release'] != 'issue'):
        raise ValueError('load configuration cleanup requires selected ordered load lane')
    config = next(row for row in order['funct_queue'] if 'by_code' in row)
    if config['by_code'].get('1') != 'load':
        raise ValueError('CONFIG_LD is not assigned to the selected load lane')
    return {'facts_sha256': record['base_facts_sha256'], 'ordering_sha256': digest,
            'persistent_state': 'CONFIG_LD channel registers; MVIN reads and does not write them',
            'ordering': 'selected load queue releases in issue order'}


def _project(op):
    keep = ConfigurationEffects(ordered=True)
    if op.name == 'llvm.call' or op.regions:
        return None
    if op.name != 'llvm.inline_asm':
        return keep
    assembly = op.asm_string.data.strip()
    if assembly == 'fence':
        return keep
    match = re.fullmatch(r'\.insn r 123, 3, (\d+), x0, \$0, \$1', assembly)
    if match is None:
        return None
    funct = int(match.group(1))
    if funct != isa.FUNCT['config']:
        return keep if funct in isa.FUNCT.values() else None
    rs1, rs2 = map(_integer, op.operands)
    if rs1 is None or rs1 & 3 != 1 or rs2 is None:
        return None
    channel = (rs1 >> 3) & 3
    if channel > 2:
        return None
    # The complete setter pair conservatively represents every written field;
    # other setters invalidate it rather than assuming field disjointness.
    return ConfigurationEffects(writes=((f'load{channel}.rs1', rs1), (f'load{channel}.stride', rs2)),
                                ordered=True, pure_write=True)


def cleanup(function):
    admission = _admission()
    result = eliminate_llvm_configuration(function, _project)
    return dict(result, admission=admission, mechanism='exact_load_configuration_cfg')
