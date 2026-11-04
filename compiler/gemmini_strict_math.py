"""Source scalar math through Merlin's ordinary native CPU math ABI.

The common native pipeline runs convert-math-to-libm before math-to-LLVM.
These f32 calls use that same symbol/ABI selection. No portable approximation
recipe or new runtime is selected here. The selected toolchain supplies libm.
FpBuilder's approximations remain available to separately admitted callers;
this mixin is for the source host lane without approximation permission.
"""
from xdsl.dialects import builtin, llvm


class StrictSourceMath:
    def fconst(self, value):
        # Python float keys equate the two zeros. Keep the source negative-zero
        # literal separate while preserving ordinary finite constant caching.
        import math
        if value == 0.0 and math.copysign(1.0,value) < 0:
            key = ('source_negative_zero',)
            cached = self._fconsts.get(key)
            if cached is None:
                operation = llvm.ConstantOp(builtin.FloatAttr(-0.0,builtin.f32),builtin.f32)
                self.fb.entry.add_op(operation)
                cached = self._fconsts[key] = operation.results[0]
            return cached
        return super().fconst(value)

    def source_extremum(self, name, lhs, rhs):
        """Keep arith's NaN propagation and signed-zero distinctions explicit."""
        intrinsic = {
            'arith.maximumf':'llvm.maximum',
            'arith.minimumf':'llvm.minimum',
            'arith.maxnumf':'llvm.maxnum',
            'arith.minnumf':'llvm.minnum',
        }[name]
        if lhs.type != builtin.f32 or rhs.type != builtin.f32:
            raise ValueError('source float extremum requires f32 scalar operands')
        return self.fb.add(llvm.CallIntrinsicOp(
            intrinsic,[lhs,rhs],[builtin.f32])).results[0]

    def fneg(self, value):
        return self.fb.add(llvm.FNegOp(value)).results[0]

    def round_near_even(self, value):
        """Native fixed ties-to-even rounding, including the source zero sign."""
        if value.type != builtin.f32:
            raise ValueError('source roundeven requires an f32 scalar operand')
        return self.fb.add(llvm.CallIntrinsicOp(
            'llvm.roundeven', [value], [builtin.f32])).results[0]

    def _native_math(self, symbol, *arguments):
        if any(value.type != builtin.f32 for value in arguments):
            raise ValueError('source native math requires f32 scalar operands')
        declarations = getattr(self.fb, '_gemmini_globals', None)
        if declarations is None:
            declarations = self.fb._gemmini_globals = []
        signature = llvm.LLVMFunctionType([builtin.f32]*len(arguments),builtin.f32)
        existing = [op for op in declarations
                    if isinstance(op,llvm.FuncOp) and op.sym_name.data == symbol]
        if existing:
            if len(existing) != 1 or existing[0].function_type != signature or existing[0].body.blocks:
                raise ValueError('source native math symbol has a conflicting definition')
        else:
            declarations.append(llvm.FuncOp(symbol,signature,linkage=llvm.LinkageAttr('external')))
        return self.fb.add(llvm.CallOp(symbol,*arguments,return_type=builtin.f32)).results[0]

    def exp(self, value):
        return self._native_math('expf',value)

    def erf(self, value):
        return self._native_math('erff',value)

    def log(self, value):
        return self._native_math('logf',value)

    def tanh(self, value):
        return self._native_math('tanhf',value)

    def sin(self, value):
        return self._native_math('sinf',value)

    def cos(self, value):
        return self._native_math('cosf',value)

    def powf(self, lhs, rhs):
        return self._native_math('powf',lhs,rhs)
