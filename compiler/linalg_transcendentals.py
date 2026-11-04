"""Standard LLVM lowering of strict source f32 trigonometric operations.

LLVM owns argument reduction and native libm selection. No approximation,
reassociation, cross-operation identity, or alternative runtime policy is chosen.
The caller remains responsible for linking the original selected math supplier.
"""
from xdsl.dialects import arith, llvm, math
from xdsl.dialects.builtin import Float32Type


def try_lower_scalar(host, op, ins):
    classes = {'math.sin': math.SinOp, 'math.cos': math.CosOp}
    if op.name not in classes or type(op) is not classes[op.name]:
        return False, None
    if (len(op.operands) != 1 or len(op.results) != 1 or len(ins) != 1
            or not isinstance(op.operands[0].type, Float32Type)
            or op.results[0].type != op.operands[0].type or op.regions or op.successors
            or set(op.properties) - {'fastmath'}
            or any(not name.startswith('prov.') for name in op.attributes)):
        return False, None
    flags = op.properties.get('fastmath')
    if flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data):
        return False, None
    name = op.name.split('.')[1]
    result = host.fp.fb.add(llvm.CallIntrinsicOp('llvm.'+name+'.f32', ins,
                                                [op.results[0].type])).results[0]
    symbols = getattr(host, 'required_host_math_symbols', None)
    if symbols is None:
        symbols = host.required_host_math_symbols = set()
    symbols.add(name+'f')
    return True, result


def dependency_contract(host):
    """Typed unresolved link requirements for the actually emitted intrinsics."""
    from merlin.llvmlower.strict_trig_linkage import requirements
    symbols = sorted(getattr(host, 'required_host_math_symbols', ()))
    return requirements(symbols) if symbols else None
