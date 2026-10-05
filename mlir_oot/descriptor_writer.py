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
    from xdsl.dialects import bufferization, func, memref
    from xdsl.dialects.builtin import MemRefType, NoneAttr, TensorType, UnitAttr
    from xdsl.ir import Block, Region

    contracts = tuple(contracts)
    selected = {c.symbol: c for c in contracts}
    if len(selected) != len(contracts):
        raise ValueError('duplicate descriptor writer contract')
    declarations = {o.sym_name.data: o for o in module.body.block.ops if o.name == 'func.func'}
    plans = []
    # Validate the complete selection before editing any declaration.
    for name, contract in selected.items():
        if not name or not (name[0].isalpha() or name[0] == '_') or any(
            not (ch.isascii() and (ch.isalnum() or ch == '_')) for ch in name
        ):
            raise ValueError('C identifier required for descriptor bridge')
        decl = declarations.get(name)
        if decl is None or decl.body.blocks or 'llvm.emit_c_interface' not in decl.attributes:
            raise ValueError('contract requires a bodyless ranked C-interface adapter')
        types, outputs = tuple(decl.function_type.inputs), tuple(decl.function_type.outputs)
        if len(outputs) != 1 or any(
            not isinstance(t, TensorType) or any(d <= 0 for d in t.get_shape())
            or t.encoding != NoneAttr() for t in types
        ):
            raise ValueError('one result and static unencoded tensor arguments required')
        attrs = decl.properties.get('arg_attrs')
        if attrs is None or len(attrs) != len(types):
            raise ValueError('complete positional access policy required')
        access = [getattr(a.data.get('bufferization.access'), 'data', None) for a in attrs]
        if any(x not in ('read', 'write') for x in access):
            raise ValueError('read-only or fully written arguments required')
        writers = tuple(i for i, a in enumerate(access) if a == 'write')
        if contract.fully_written_arguments != writers:
            raise ValueError('full-write contract disagrees with declaration access')
        result_index = contract.result_argument
        if type(result_index) is not int or result_index not in writers or types[result_index] != outputs[0]:
            raise ValueError('result identity disagrees with writable argument type')
        borrowed = name + '__borrowed_write'
        if borrowed in declarations:
            raise ValueError('borrowed bridge symbol collision')
        calls = [o for o in module.walk() if o.name == 'func.call' and o.callee.root_reference.data == name]
        if not calls or any(
            getattr(o.arguments[i].owner, 'name', None) != 'tensor.empty'
            or sum(1 for _ in o.arguments[i].uses) != 1 for o in calls for i in writers
        ):
            raise ValueError('every writable call argument must be a sole-use tensor.empty')
        plans.append((decl, contract, types, attrs, access, borrowed, len(calls)))
    report = []
    for decl, contract, types, attrs, access, borrowed, call_count in plans:
        buffers = [MemRefType(t.get_element_type(), t.get_shape()) for t in types]
        block = Block(arg_types=types)
        arguments = []
        for i, typ in enumerate(buffers):
            if access[i] == 'read':
                op = bufferization.ToBufferOp.build(
                    operands=[block.args[i]], result_types=[typ], properties={'read_only': UnitAttr()}
                )
            else:
                op = memref.AllocOp.get(typ.element_type, shape=typ.get_shape(), alignment=64)
            block.add_op(op)
            arguments.append(op.results[0])
        block.add_op(func.CallOp(borrowed, arguments, []))
        result = bufferization.ToTensorOp(arguments[contract.result_argument], restrict=True, writable=True)
        block.add_ops([result, func.ReturnOp(result.tensor)])
        wrapper = func.FuncOp(contract.symbol, (types, tuple(decl.function_type.outputs)),
                              Region(block), visibility='private', arg_attrs=attrs)
        wrapper.attributes.update({k: v for k, v in decl.attributes.items() if k != 'llvm.emit_c_interface'})
        external = func.FuncOp(borrowed, (buffers, []), Region(), visibility='private', arg_attrs=attrs)
        external.attributes['llvm.emit_c_interface'] = UnitAttr()
        decl.parent.insert_ops_before([external, wrapper], decl)
        decl.parent.erase_op(decl)
        report.append(dict(symbol=contract.symbol, calls=call_count,
                           fresh_writers=list(contract.fully_written_arguments),
                           result_argument=contract.result_argument, borrowed_symbol=borrowed,
                           argument_ranks=[len(t.get_shape()) for t in types],
                           result_rank=len(types[contract.result_argument].get_shape())))
    module.verify()
    return report


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
