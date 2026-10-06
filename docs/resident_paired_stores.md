# Exact paired stores for flat resident input

`GoldenResidentConv(..., flat_spatial_planes=True, store_plan=plan)` accepts the existing typed `PairedReadoutPlan`. The default remains the original three-pointer producer. Other resident layouts refuse paired stores.

Eligibility uses the existing complete ordered-binary32 readout certificate and conservative signed-byte sum-of-products range. Every reduction prefix must fit i32, and the entire possible accumulator interval must lie inside the certificate. Captured values do not establish the range.

The raw paired ABI is `A:i8*, B:i8*, first:i8*, second:i8*`. For each completed N panel, all first-output wide byte stores and all second-output wide byte stores precede any reuse of its accumulator cells. The final fence precedes return. Source reduction order, input residency, B slots and computation geometry are unchanged.

The existing ranked adapter requires two complete disjoint byte buffers, both disjoint from the immutable inputs. The existing source preparation changes only a sole-use fresh uninitialized scratch allocation to its selected byte-write ABI. Both allocations remain live through the exact decoder; live i32 storage is never reinterpreted. The original four-tensor declaration/result ownership and decoder remain unchanged.

Normal compilation composes the existing explicit `flat_resident_planes=True` option with `readout_pair_policy='source_proven'`. Selection uses typed padding/stride/resource facts. Unsupported shapes or absent proofs retain the prior schedule. No workload or symbol selects the compiler strategy.

Weight-packet selection can independently constrain its admitted input layouts.
`issue_resident_weight_packets(..., include_flat_planes=False)` and normal
`build(..., resident_weight_issue_flat_planes=False)` retain an already selected
flat spatial-plane schedule. Channel-plane schedules still receive the requested
packet transform. The CLI spelling is
`--resident-weight-issue-channel-planes-only`. The default remains `True`, with
the same emitted code and manifest fields as before. This is an explicit plan
constraint; exclusion does not imply that the flat packet schedule is illegal
or slower. Measured profitability and whole-model composition remain separate
gates.

`test_resident_paired_stores.py` and the shared resident CFG probe prove non-square shapes, spatial/channel tails, both output extents, source cells, increasing K, accumulator lifetimes, final fence, conservative range refusal and unchanged default IR. `gsim_paired_resident_conv_probe.py` measures the complete producer, both stores and exact decoder, then compares every output byte, the complete second primitive output, both buffers' guards and all input bytes at common addresses. GSIM timing is a capsule metric, not a whole FireSim prediction.
