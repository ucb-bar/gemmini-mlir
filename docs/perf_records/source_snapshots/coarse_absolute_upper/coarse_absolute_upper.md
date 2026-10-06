# Coarse positive absolute-product envelope

This default-off producer accepts an explicit positive magnitude precision from
one through six bits. It requires the existing canonical three signed radix-128
planes, exact source/reconstruction equality, point envelopes, integer center
reconstruction and private nonaliasing scratch. Unknown source rows keep the
checked norm path. The producer does not choose a precision from model identity,
observed values or accuracy results.

For each canonical coefficient N, let s=21-b and U=ceil(abs(N)/2^s).
Then 0<=U<=2^b<=64 and abs(N)<=2^s U. One degree-zero integer
product returns sum U_a U_b. Static admission K*2^(2b)<=INT32_MAX
prevents every positive accumulator prefix from overflowing. Scaling this result
by 2^(2s) and the existing power-of-two row steps gives T_upper, an upper
bound on the absolute source dot, not necessarily its exact value. The unchanged
signed center S is exact. The source ordered-FMA radius is gamma_K*T_upper+eta_K;
its uniform finite-prefix admission and all checked fallbacks remain mandatory.

Plane zero is overwritten only after all original signed products finish.
Magnitude construction reads all three original digits for each independent
index before overwriting that index. B retains its existing plane/K/column
layout. The callback writes every i32 readout before use; a distinct double
upper array remains live through bound consumption. Source, reconstruction,
envelopes and signed centers are not overwritten. Callback failures refuse the
entire provider before output publication.

Power-of-two scaling is exact in binary64: the positive i32 result has at most
31 significant bits, while the admitted binary32 encoder step exponents are
within [-169,107]. Pair scaling and the explicit precision shift stay well
inside the binary64 normal finite range. Binary32 overflow is independently
refused by the source-prefix envelope.

The extra plane conversion, callback, readback, workspace initialization,
scaling, proof checks and any increased exact replay belong inside the complete
measurement. No performance or whole-model acceptance is implied by this proof.
