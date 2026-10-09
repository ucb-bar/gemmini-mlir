# Exact absolute products for ordered FMA bounds

The optional `prepare_absolute_products` policy computes two exact integer
products under the existing primitive callback contract. The first reconstructs
the signed real dot S. After its last signed use, the canonical private radix
planes are converted to absolute magnitude and reused to reconstruct
T = sum(abs(a[i] * b[i])). Seven-bit magnitude digits share each coefficient's
sign, so taking their magnitudes preserves the represented absolute value.
This identity does not hold for arbitrary balanced-digit encodings.

Every source row must independently match its reconstruction and have point
envelopes. The current private encoder flags and exact pointers establish that
condition during mandatory widening. Unknown or nonpoint rows keep the original
checked norm path. The source plan's existing weighted-prefix proof makes every
integer reconstruction and power-of-two scaling exact in binary64. An arbitrary
precomputed center or absolute array does not supply that proof.

Stable RNE, gradual underflow and unobserved nontrapping exceptions are required.
The existing gamma and subnormal budgets enclose the original increasing-K
binary32 FMA reduction by gamma_k * T + eta_k. A uniform maximum T closes every
source prefix and final endpoint below `FLT_MAX`; invalid or overflowing arrays
refuse the prepared plan. Exact floor/ceiling capabilities are optional and
retain their separate finite-domain contract.

The original center, source operands and source envelopes stay unchanged.
Only private signed planes whose last signed product use has completed may be
mutated. Integer/readout scratch is fully initialized for the new products and
reused; a separate absolute-center array adds eight bytes per maximum product
cell. Actual compiled workspace queries remain authoritative. Complete callback
writes and synchronous completion are required before CPU consumption. The
next encoder fully overwrites the planes before their next signed use.

The policy is default off and requires canonical integer reconstruction plus
encoder equality proofs. Softmax producer-span composition currently refuses
because its successful-writer coverage does not yet include this new producer.
No target instructions, model identities or automatic schedule selection are
introduced. Extra products, plane conversion, reconstruction, allocation and
all remaining replay must be included in complete cost measurements.

The first original complete-group screen preserves observed consumer outputs
and reduces replay, but increases retired instructions by 7.79%. Hardware cycles
are unknown. This is an experimental bound strategy, not a promoted gain.
