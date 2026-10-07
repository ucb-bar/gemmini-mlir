# Normal stationary-B tail placement

The explicit `resident_tail_before_last_full` option derives eligibility from the
selected typed resident-flat spatial family, complete A lifetime, accumulator
capacity, two or more full DIM tiles, and one short tile. It moves that short tile
after the first real-B compute while retaining a full tile as the final stationary
compute. Each output retains increasing source K, the same preload/compute multiset,
all DMA/store order and complete readout/fence behavior. Default objects are unchanged.

The current source-bound normal route changes only the two independently qualified
flat spatial producers. Audit names identify their original sources; they do not
select the production strategy. Both selected object bodies and actual executed
entry counts are checked in the final controlled ELF, in addition to the full
original 1,000-word exact output gate.

## Rejected new option propagation attempt

An initial new-option build admitted the tail modifier before paired readout selection.
The paired producer factory omitted this newly introduced schedule field. The source
recipe's decision metadata claimed admission but the generated paired object remained
unchanged. This attempt is retained as a rejection, not a measured implementation.
The qualified route applies tail placement after readout selection and forwards the
field in the factory. A choose-to-actual-IR regression and final object/entry closure
check both the direct API and ordinary source-bound route.

A general compiler infrastructure follow-up should represent target schedule choices
in an immutable typed schedule record. Readout selection should replace only the store
plan (for example with `dataclasses.replace`) while preserving every schedule field.
The same obligation applies to other normal-binding clones. Metadata preservation
alone is insufficient: actual emitted command/object closure remains authoritative.
This packet contains the narrow qualified fix; it does not perform that broader refactor.

## Measurement scope

The original paired spatial producer improves in stock FireSim from 890,568 to
809,631 cycles (jobs 2093/2094). The second, direct i8 producer improves separately
in pinned GSIM from 713,094 to 621,674 cycles. Its 25,088 outputs, 4,096 guards and
2,384,384 immutable operand bytes pass. An independent non-square/channel-tail
paired fixture also passes and improves in GSIM. Neither result supplies a whole-model
forecast. The fresh normal build and the controlled current-2090 link both pass
native and production Spike against all original words, with no FSM instructions.
The controlled link retains all other device, host, runtime and weight objects.
