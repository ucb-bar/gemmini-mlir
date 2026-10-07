# Preserve the complete target emission plan

Target emitter factories must preserve every typed constructor option unless the
factory explicitly replaces it. `GemmEmissionOptions`, `FlatConvEmissionOptions`,
`ResidentConvEmissionOptions`, and `ResidentStripeEmissionOptions` are immutable
records of those facts. Their field names and defaults are checked against the
complete public constructor signature before replacement. An unrecorded new
keyword, undeclared subclass, or changed default refuses replacement.

`with_emission_options` takes a current snapshot, uses `dataclasses.replace`,
and invokes the existing resource-checked constructor. It keeps typed input-view
and store-plan objects intact. Unknown fields and invalid layouts fail without
changing the source emitter. Constructor normalization, such as complete-band
defaults or compact loops requiring channel loops, remains explicit.

Ordinary same-family DMA, B-slot, tail, command-loop, paired-readout and segmented
input factories use this API. Cross-family selection retains its separate typed
layout/resource proof. Decision receipts are not emission options and do not
grant eligibility or profitability.

The 2095 and 2101 source-bound normal recipes each reproduce all 52 primitive
and adapter objects, the requant aggregate and native oracle exactly. All three
actual accepted segmented-input primitives, adapters, aggregate and oracle also
reproduce exactly. Both flat and dense winning tail options survive the same
factories. This change has no performance claim or automatic policy change.

Focused constructor, family, failure, clone and existing schedule tests pass
132 checks. The first validation driver stopped after the binary comparisons
on a runtime tuple versus serialized JSON list. Its original bytes and logs
are retained; read-only JSON-normalized reclosure avoids repeating compilation.
Earlier test fixture refusals are retained as well.

The sealed source/object/test receipts are recorded in
`docs/perf_records/emission_option_survival_qualification.json`.
