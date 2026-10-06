# Exact finite bound conversion

`ExactBoundConversionContract` is an explicit portable capability for binary64
to binary32 floor and ceiling. Unlike a general outward interval hook, each
result must be the nearest representable value in its specified direction and
must preserve signed zero. Source rounding cannot change; arithmetic must be
nontrapping and exception flags unobserved. The default is no capability.

The source attention emitter accepts `exact_bound_conversion=contract` and
requires separately provided `MERLIN_F32_EXACT_FLOOR_FROM_F64` and
`MERLIN_F32_EXACT_CEIL_FROM_F64` hooks. Merlin contains no CPU instructions.
Provider/compiler/ABI evidence belongs to the selected host provider.

Only the original bound finish, admitted product-row finish and admitted
separable-radius finish consume the hooks. Their existing contracts prove
finite endpoints in the binary32 finite range. Overflow/nonfinite unknown
inputs retain existing rejection before this private operation. Original
source arithmetic, polynomial conversions, bounds, ordered replay and output
observations are unchanged.

For finite in-range x, casting in a supported rounding mode followed by the
original exact comparison and conditional adjacent-value correction equals
floor32(x) or ceil32(x). The new capability computes those same results directly.
Both paths preserve a signed zero input. The domain excludes overflow, where
an arbitrary cast followed by a nonfinite-preserving adjacency helper would
not establish this identity. The hook is not an authorization to omit those
producer domain proofs.

Qualification includes exact rational oracles, all five target rounding modes,
subnormal and normal boundaries, zeros, ties, refused NaN/infinity/out-of-range
values, actual static rounding fields and unchanged ambient rounding. Default
provider object and final ELF identity, complete source group and unchanged
original model gate are independent checks. Instruction counts are not cycles.
