# Current ResNet intervals against the owned reference

Stock profile **1919** instruments the exact current **1903** object set. Its
36,138,975 forward cycles conserve 30,678,196 device-wrapper cycles and 5,460,779
host-gap cycles across all 70 calls. The model metric is 36,139,403 cycles;
the original uninstrumented 1903 metric is **36,102,704**, so instrumentation
adds **36,699** metric cycles. The tail within the host total is 80,533 cycles.
All original 1,000 output words remain exact. The final executable audit has
zero FSM instructions.

| Geometry class | Reference 1876 | Current profile 1919 | Interval difference |
|---|---:|---:|---:|
| Pointwise | 9,907,412 | 11,315,574 | +1,408,162 |
| Spatial 3×3 | 8,723,089 | 11,968,480 | +3,245,391 |
| Residual | 2,192,393 | 5,470,706 | +3,278,313 |
| Stem and pool | 1,083,057 | 1,444,368 | +361,311 |
| Classifier | 429,781 | 479,068 | +49,287 |
| Host gaps / reference other plus uncounted | 51,717 | 5,460,779 | +5,409,062 |
| Forward total | **22,387,449** | **36,138,975** | **+13,751,526** |

This is a geometry comparison across different numeric and timing contracts.
The reference has different input, weights, quantization scales and classifier
narrowing. Its layer intervals can include CPU epilogues. Our primitive wrappers
exclude descriptor checks and CPU readout; those appear before subsequent calls.
The reference enters with int8 input; our immutable f32 NCHW input requires host
quantization and packing. The differences above are not causal savings estimates.
Reference residual timing exists only as an aggregate, so no per-reference
residual timing is invented. Three reference pointwise stride-2 projections are
classified by geometry rather than their printed `conv` timer label.

## Current priorities

The seven largest current host gaps account for **5,289,822** cycles:

| Interval preceding | Current cycles |
|---|---:|
| Stem | 2,192,819 |
| matmul_26 | 1,150,086 |
| Projection matmul_14 | 614,837 |
| matmul_49 | 563,376 |
| Classifier | 344,814 |
| Projection matmul_27 | 286,309 |
| Projection matmul_46 | 137,581 |

These intervals include every intervening CPU operation and timer overhead.
The `matmul_26` and `matmul_49` gaps contain CPU integer readout after the preceding
spatial calls; dedicated host/readout studies must establish their exact split.
The projection gaps contain physical layout copies. The target accepts a proved
segmented source view in the separately qualified default-off provider; copies
must stay until the actual producer type, alias lifetime and selected consumer
contract close. These interval totals are bounds on opportunities, not measured
savings from eliminating an individual operation.

Six spatial calls with output geometry M=196, K=2304, N=256 have a combined
**1,490,776** positive interval difference. One is the stride-2 transition and
five are subsequent stride-1 calls. This supports continuing general layout,
resource-aware residency and compact command-loop work; the source arithmetic
and reduction order remain immutable. Their reference/current shape match
does not establish numerical equivalence.

Residuals remain a separate 3,278,313-cycle aggregate difference. The existing
39-chunk lower bound applies to one affine/readout family. A complete source
certificate now permits a five-chunk shared producer with two readouts and an
exact joint decoder. Its original first-layer capsule improves 3.37% on GSIM,
including both stores and the entire decoder scan. A single-readout alternative
uses a certificate-derived output guard and original ordered-f32 source replay.
Full-domain and complete original-input timing gates decide admission; neither
result can be added to this stock profile or transferred to other coefficients.

## Evidence and reproduction

[Machine-readable current alignment](perf_records/q1013_1903_current_profile_alignment.json)
contains all 70 source-bound events, 54 paired convolution/classifier geometries,
current schedules, the largest host/device intervals, stock bitstream identity,
and 319 reclosed artifact pins. It rehashes qualification, actual executable,
UART, linked objects, component leaves and preserved staged-identity observations.
The simulator removed staged executable/bitstream bytes at teardown; that
limitation is recorded separately from the retained observations.

Run [the analysis driver](../tests/current_reference_alignment_probe.py) with
explicit `--archive`, `--qualification`, `--catalog`, `--pairing`, `--reference`,
`--reference-staged-elf`, `--reference-staged-bitstream` and `--output` paths.
It independently parses the actual UART using the shared existing profile parser
and refuses failed terminal status, changed artifact hashes, hardware mismatch,
changed source geometry, incomplete call coverage or interval conservation.
Earlier 1850/1899 receipts retain their historical measurements; only source
geometry correspondence is reused.
