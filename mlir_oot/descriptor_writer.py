"""Fresh tensor results for explicitly contracted ranked-descriptor adapters.

The caller proves each selected adapter fully writes its designated arguments
and returns the passed result descriptor. A borrowed void bridge discards that
descriptor, leaving upstream MLIR responsible for the fresh allocation's lifetime.
No output identity is inferred from matching tensor types.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptorWriterContract:
    symbol: str
    result_argument: int
    fully_written_arguments: tuple[int, ...]


def transform(module, contracts):
    from merlin.llvmlower.fresh_tensor_writer import (
        FreshTensorWriterContract, rewrite_fresh_tensor_writers,
    )

    contracts = tuple(contracts)
    for contract in contracts:
        name = contract.symbol
        if not name or not (name[0].isalpha() or name[0] == '_') or any(
            not (ch.isascii() and (ch.isalnum() or ch == '_')) for ch in name
        ):
            raise ValueError('C identifier required for descriptor bridge')
    return rewrite_fresh_tensor_writers(module, [
        FreshTensorWriterContract(c.symbol, c.result_argument,
                                  c.fully_written_arguments,
                                  c.symbol + '__borrowed_write', 64)
        for c in contracts
    ])


def shim(routes):
    """Emit a 64-bit descriptor-pointer bridge; never an expanded-memref adapter."""
    code = '#include <stdint.h>\n#if UINTPTR_MAX != UINT64_MAX\n#error 64-bit descriptor ABI required\n#endif\n'
    for route in routes:
        count = len(route['argument_ranks'])
        args = ','.join('void*a' + str(i) for i in range(count))
        types = ','.join('void*' for _ in range(count + 1))
        values = ','.join('a' + str(i) for i in range(count))
        code += (f"extern void _mlir_ciface_{route['symbol']}({types});\n"
                 f"void _mlir_ciface_{route['borrowed_symbol']}({args}) {{\n"
                 f" uintptr_t ignored_result[{3 + 2 * route['result_rank']}];\n"
                 f" _mlir_ciface_{route['symbol']}(ignored_result,{values});\n}}\n")
    return code
