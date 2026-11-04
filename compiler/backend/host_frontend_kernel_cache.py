"""Exact typed-function identity for code sharing across source call sites.

Only local SSA spelling, the entry symbol and provenance metadata are normalized.
Operand positions, types, constants, maps, scalar bodies and other attributes stay
in the identity. This never caches tensor values or establishes effect premises.
"""
from __future__ import annotations

import hashlib


def typed_device_function_identity(source: str) -> str:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(source)
    functions = [op for op in module.body.block.ops if op.name == 'func.func']
    if len(functions) != 1:
        raise ValueError('device code sharing requires one complete typed function')
    functions[0].properties['sym_name'] = StringAttr('shared_device_kernel')
    for operation in module.walk():
        for attributes in (operation.attributes, operation.properties):
            for name in tuple(attributes):
                if name.startswith('prov.'):
                    del attributes[name]
        for result in operation.results:
            result.name_hint = None
        for region in operation.regions:
            for block in region.blocks:
                for argument in block.args:
                    argument.name_hint = None
    module.verify()
    return hashlib.sha256(text(module, generic=True).encode()).hexdigest()
