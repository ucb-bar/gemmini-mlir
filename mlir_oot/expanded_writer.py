"""Provider-owned bridge for explicitly contracted expanded-memref writers.

The caller supplies source-bound result identity and complete-write facts. This
module adapts the existing expanded C shim; it does not infer kernel semantics
from matching tensor shapes or automatically select catalog entries.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedWriterContract:
    symbol: str
    borrowed_symbol: str
    result_argument: int
    fully_written_arguments: tuple[int, ...]
    allocation_alignment: int


def transform(module, contracts):
    from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers

    contracts = tuple(contracts)
    for contract in contracts:
        for name in (contract.symbol, contract.borrowed_symbol):
            if not name or not (name[0].isascii() and (name[0].isalpha() or name[0] == '_')) or any(
                not (ch.isascii() and (ch.isalnum() or ch == '_')) for ch in name
            ):
                raise ValueError('C identifier required for expanded bridge')
    return rewrite_fresh_tensor_writers(module, [
        FreshTensorWriterContract(c.symbol, c.result_argument, c.fully_written_arguments,
                                  c.borrowed_symbol, c.allocation_alignment,
                                  declaration_abi='expanded_memref',
                                  wrapper_symbol=c.symbol + '__fresh_tensor_result') for c in contracts
    ])


def shim(routes):
    """Match the provider's intptr_t expanded C ABI, discarding its result."""
    ranks = sorted({rank for r in routes for rank in [*r['argument_ranks'], r['result_rank']]})
    if any(type(rank) is not int or rank <= 0 for rank in ranks):
        raise ValueError('positive ranked expanded descriptors required')
    code = '#include <stdint.h>\n'
    for rank in ranks:
        code += (f'typedef struct {{ void *allocated,*aligned; intptr_t offset; '
                 f'intptr_t sizes[{rank}],strides[{rank}]; }} writer_memref_{rank};\n')
    for route in routes:
        types = []
        values = []
        args = []
        for i, rank in enumerate(route['argument_ranks']):
            args.append(f'writer_memref_{rank}*a{i}')
            types.extend(['void*', 'void*', 'intptr_t'] + ['intptr_t'] * (2 * rank))
            values.extend([f'a{i}->allocated', f'a{i}->aligned', f'a{i}->offset'])
            values.extend(f'a{i}->sizes[{j}]' for j in range(rank))
            values.extend(f'a{i}->strides[{j}]' for j in range(rank))
        code += (f"extern writer_memref_{route['result_rank']} {route['symbol']}({','.join(types)});\n"
                 f"void _mlir_ciface_{route['borrowed_symbol']}({','.join(args)}) {{\n"
                 f" (void){route['symbol']}({','.join(values)});\n}}\n")
    return code
