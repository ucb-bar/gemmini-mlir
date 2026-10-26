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


def config_st(stride, act, multiplier_bits):
    return (code('semantic_class', 'CONFIG'),
            pack('ConfigMvoutRs1', cmd_type=code('config_subtype', 'CONFIG_ST'), activation=act),
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
