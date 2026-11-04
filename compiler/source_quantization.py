"""Exact original quant_ext seams through Merlin's upstream decomposition."""
from __future__ import annotations


def _name(op):
    name = getattr(op, 'op_name', None)
    return name.data if op.name == 'builtin.unregistered' and name is not None else op.name


def needs_quantization_preparation(module):
    return any(_name(op).startswith('quant_ext.') for op in module.walk())


def prepare_host_quantization(module):
    """Clone and decompose captured quantizers without changing their boundaries.

    The selected public pass computes reciprocal scale once, then multiply,
    round-to-even, add zero point and clamp in the captured precision/order.
    It handles supported dequantizer granularities too. Source identities and
    existing emission ownership attributes are retained on replacement trees.
    Unsupported extensions fail explicitly before entering upstream MLIR.
    """
    from merlin.llvmlower.passes_xdsl import lower_quant_ext

    prepared = module.clone()
    def retain_source(original, replacements, result):
        source = {key: value for key, value in original.attributes.items()
                  if key.startswith(('prov.', 'merlin.'))}
        for replacement in replacements:
            for nested in replacement.walk():
                for key, value in source.items():
                    nested.attributes.setdefault(key, value)
    lower_quant_ext(prepared, on_rewrite=retain_source)
    remaining = [_name(op) for op in prepared.walk() if _name(op).startswith('quant_ext.')]
    if remaining:
        raise ValueError('unsupported original quantization extension: ' + ', '.join(sorted(set(remaining))))
    prepared.verify()
    return prepared
