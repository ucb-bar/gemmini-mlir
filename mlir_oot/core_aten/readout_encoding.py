"""Readout encoding bound to the synced contract and selected RTL facts."""

from functools import lru_cache
from pathlib import Path

import yaml

from merlin.targetgen.rocc.semantics import bundle_layouts, derived_readout_bits, field_slot
from merlin.targetgen.rtl.facts import load_facts


@lru_cache(maxsize=1)
def contract():
    return yaml.safe_load((Path(__file__).resolve().parents[2] /
                           'merlin-support/contracts/target_contract.yaml').read_text())


def slots():
    return bundle_layouts(load_facts(contract()['name']))


def field(bundle, name):
    result = field_slot(slots().get(bundle), name)
    if result is None:
        raise ValueError(f'UNKNOWN readout encoding: {bundle}.{name}')
    return result


def pack(bundle, **values):
    word = 0
    for name, value in values.items():
        offset, width = field(bundle, name)
        if type(value) is not int or not 0 <= value < 1 << width:
            raise ValueError(f'{bundle}.{name} does not fit its derived field')
        word |= value << offset
    return word


def unpack(word, bundle, name):
    offset, width = field(bundle, name)
    return (word >> offset) & ((1 << width) - 1)


def code(table, name):
    return next(int(k) for k, v in contract()['encoding'][table].items() if v == name)


def flags():
    return derived_readout_bits(contract()['encoding']['addr_len'], contract=contract())


def activation(relu):
    from ..tables import isa
    roles = contract()['rocc_operand_roles']['classes']['CONFIG_ST']['fields']
    return next(row['equals'] for row in roles if row['name'] == 'relu') if relu else isa.NO_ACTIVATION


def config_st(stride, act, multiplier_bits, pool=None):
    return (code('semantic_class', 'CONFIG'),
            pack('ConfigMvoutRs1', cmd_type=code('config_subtype', 'CONFIG_ST'), activation=act, **(pool or {})),
            pack('ConfigMvoutRs2', stride=stride, acc_scale=multiplier_bits))


def supports_narrow_path(path):
    """A caller's numeric proof must describe this device's actual scalar ABI."""
    abi = contract()['readout_semantics']['scalar_abi']
    stages = {s['kind']: s for s in path['readout']}
    convert, scale, requant = (stages.get(kind, {}) for kind in ('convert', 'scale', 'requant'))
    return (path['accumulator']['dtype'] == abi['accumulator_dtype'] and
            convert.get('to') == abi['scale_dtype'] and scale.get('dtype') == abi['scale_dtype'] and
            requant.get('to') == abi['output_dtype'] and
            requant.get('clamp') == [abi['clamp_min'], abi['clamp_max']])


def runtime_scale_facts(raw_facts, path):
    """Re-derive transport from selected facts, provider roles, layout and scalar ABI."""
    from merlin.targetgen.composition_facts import derive_runtime_readout_scale

    result = derive_runtime_readout_scale(facts=raw_facts, contract=contract(), output_path=path)
    if result.get('status') != 'derived' or not result.get('evidence'):
        raise ValueError('UNKNOWN runtime readout scale: ' + result.get('reason', 'missing evidence'))
    value = result.get('value')
    expected = dict(schema='merlin.runtime_readout_scale.v1', dtype='f32', granularity='per_command',
                    operand='rs2', bundle='ConfigMvoutRs2', field='acc_scale', offset=32, width=32,
                    encoding='ieee754_binary32_bits')
    if (value != expected or type(value.get('offset')) is not int
            or type(value.get('width')) is not int):
        raise ValueError('unsupported runtime readout scale transport')
    if field('ConfigMvoutRs2', 'acc_scale') != (value['offset'], value['width']):
        raise ValueError('runtime scale transport differs from selected register layout')
    return value


def runtime_scale_slot():
    """The selected register field and ABI used by runtime CONFIG_ST emission."""
    roles = contract()['rocc_operand_roles']['classes']['CONFIG_ST']['fields']
    selected = [row for row in roles if row['name'] in ('acc_scale', 'acc_scale_bits')]
    if not selected or any(row.get('as', 'f32') != 'f32' for row in selected) or any(
                           (row.get('operand'), row.get('bundle'), row.get('field')) !=
                           ('rs2', 'ConfigMvoutRs2', 'acc_scale') for row in selected):
        raise ValueError('UNKNOWN runtime scale CONFIG_ST role')
    if contract()['readout_semantics']['scalar_abi']['scale_dtype'] != 'f32':
        raise ValueError('runtime scale requires the f32 scalar ABI')
    slot = field('ConfigMvoutRs2', 'acc_scale')
    if slot != (32, 32):
        raise ValueError('runtime scale requires the whole binary32 CONFIG_ST field')
    return slot
