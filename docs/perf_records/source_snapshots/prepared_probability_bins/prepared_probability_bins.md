# Prepared probability bins

`emit_source_attention_frontier(..., prepare_probability_bins=True)` snapshots
both rounded BF16 endpoint values before the probability ambiguity test. Exact
source replay refreshes the snapshot before publication. The original interval,
source denominator, replay decision and output semantics remain unchanged.

For finite ordered endpoints whose rounded bits agree, the source binary64
midpoint, binary32 conversion and BF16 conversion must produce that same bin by
monotonicity. Different signed-zero bits, nonfinite or unordered endpoints use
the original midpoint expression. This requires stable source rounding and
nontrapping, unobserved exception flags; it is not an approximate policy.

The option defaults off. The snapshot is a private value, never a cached pointer
or a lifetime claim about mutable interval storage. A changed refinement grammar
is refused. Tests cover all BF16 anchors and nearby binary32 tie values, both
endpoint orders, nonfinite values, signed zeros and four native rounding modes,
plus complete independent source-group mask/shape/refusal tests.

Target execution and complete-group cost qualification are separate from this
portable proof. No hardware performance claim follows from fewer conversions.
